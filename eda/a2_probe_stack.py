"""Probes on hidden states and a sparse blend of the channels — all on the CPU, over the shared folds.

    python -I eda/a2_probe_stack.py <run> <job> [--tag bench] [--pca 128] [--save NAME]

Reads the outputs of a measurement run on training rows (artifacts/runs/<run>/<run>/<job>/):
    <tag>_diag.npy     log probabilities of the codes without training
    <tag>_diag_h.npy   hidden state at the decision position
    <tag>_fac.npy, <tag>_ans.npy, <tag>_fac_h.npy   check of the final answer (if present)

Channels (each cross-fitted over the shared folds, nested where one channel builds on another):
    Z   code ranking of the untrained model
    P   task prior (artifacts/oof_prior_question_v1.npy)
    H   linear probe on the hidden state (standardisation, principal components, multinomial regression)
    N   NONE signal from the answer check (margin yes − no; optionally a probe on the hidden state of the check)
Blend:  log p_k = a·Z_k + b·P_k + c·H_k + [k = NONE]·(d·N + e), weights fitted per fold on the log loss of the other
folds (few parameters instead of a freely fitted stacking model).

No leave-one-out; aggregates go to artifacts/run_summaries/<run>__<job>__stack.json.

The blend helpers `floor_probs`, `fit_mix` and `apply_mix` are also used by the deployment code.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

COMP = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(COMP), str(COMP / "src")]

import labels  # noqa: E402
import metric  # noqa: E402
from vlm import common as C  # noqa: E402

EPS = 1e-9


def log_softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(1, keepdims=True)
    return z - np.log(np.exp(z).sum(1, keepdims=True))


def clean_logp(lp: np.ndarray) -> np.ndarray:
    """Log probabilities over all classes; dropped candidates share the remaining mass of the row."""
    from vlm.engine import probs_from_logp

    return np.log(np.clip(probs_from_logp(lp), 1e-12, 1.0))


def fit_probe(Xtr, ytr, Xte, n_classes: int, pca: int, C_reg: float) -> np.ndarray:
    """Multinomial regression on standardised principal components; returns log probabilities over all classes."""
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression

    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    A, B = (Xtr - mu) / sd, (Xte - mu) / sd
    k = min(pca, A.shape[0] - 1, A.shape[1])
    p = PCA(n_components=k, random_state=0).fit(A)
    A, B = p.transform(A), p.transform(B)
    s = A.std(0) + 1e-6
    clf = LogisticRegression(C=C_reg, max_iter=400, tol=1e-4).fit(A / s, ytr)
    out = np.full((len(Xte), n_classes), np.log(EPS))
    out[:, clf.classes_] = np.log(np.clip(clf.predict_proba(B / s), EPS, 1))
    return log_softmax(out)


def crossfit_probe(X, y, folds, n_classes, pca, C_reg) -> np.ndarray:
    """OOF predictions of the probe over the folds that occur in `folds`."""
    out = np.zeros((len(y), n_classes))
    for k in np.unique(folds):
        tr, va = folds != k, folds == k
        out[va] = fit_probe(X[tr], y[tr], X[va], n_classes, pca, C_reg)
    return out


def floor_probs(P: np.ndarray, eps: float, allowed: np.ndarray) -> np.ndarray:
    """Mix a channel with the uniform distribution over the allowed classes: (1 − eps) · P + eps · U.

    Reason: a channel that is almost always confident (the reasoning channel: median top probability 0.99) acts as a veto in
    the log-linear mean — when it is wrong, its near-zero probability pushes the correct code out of the first three, and the
    order of the alternatives depends on its noisy remainder. The floor limits this; eps = 0 leaves P unchanged."""
    if not eps:
        return P
    u = np.where(allowed, 1.0, 0.0)
    u = u / u.sum()
    return (1.0 - eps) * P + eps * u[None, :]


def fit_mix(chans: list[np.ndarray], none_feats: np.ndarray | None, y: np.ndarray) -> np.ndarray:
    """Blend weights on log loss (L-BFGS, few parameters)."""
    from scipy.optimize import minimize

    n_c = len(chans)
    n_n = 0 if none_feats is None else none_feats.shape[1]

    def nll(theta):
        z = sum(theta[i] * chans[i] for i in range(n_c))
        z = z.copy()
        if n_n:
            z[:, 0] = z[:, 0] + none_feats @ theta[n_c : n_c + n_n] + theta[-1]
        lp = log_softmax(z)
        return -lp[np.arange(len(y)), y].mean() + 1e-3 * float(np.sum(theta[n_c:] ** 2))

    x0 = np.concatenate([np.full(n_c, 1.0 / n_c), np.zeros(n_n + (1 if n_n else 0))])
    res = minimize(nll, x0, method="L-BFGS-B", bounds=[(0, 4)] * n_c + [(-10, 10)] * (len(x0) - n_c))
    return res.x


def apply_mix(theta, chans, none_feats) -> np.ndarray:
    n_c = len(chans)
    z = sum(theta[i] * chans[i] for i in range(n_c)).copy()
    if none_feats is not None:
        z[:, 0] = z[:, 0] + none_feats @ theta[n_c : n_c + none_feats.shape[1]] + theta[-1]
    return np.exp(log_softmax(z))


def report(name: str, y, p, rows_ref, masks) -> dict:
    rows = metric.map3_rows_from_proba(y, p)
    pb = metric.paired_bootstrap(rows, rows_ref)
    d = {"map3": round(float(rows.mean()), 4), "sigma": round(float(metric.bootstrap_sigma(rows)), 4),
         "logloss": round(float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-6, 1)).mean()), 4),
         "delta_vs_prior": round(pb["delta"], 4), "sigma_delta": round(pb["sigma"], 4), "z": round(pb["z"], 2)}
    for m_name, m in masks.items():
        if m.sum() >= 30:
            d[m_name] = round(float(rows[m].mean()), 4)
    print(f"  {name:<34} MAP@3 {d['map3']:.4f} ± {d['sigma']:.4f} | vs. prior {d['delta_vs_prior']:+.4f} ± {d['sigma_delta']:.4f} (z {d['z']:+.1f}) | "
          f"NONE {d.get('NONE', float('nan')):.3f} errors {d.get('error rows', float('nan')):.3f} | log loss {d['logloss']:.3f}")
    return d


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run")
    ap.add_argument("job")
    ap.add_argument("--tag", default="bench")
    ap.add_argument("--pca", type=int, default=128)
    ap.add_argument("--save", help="name under which the blended OOF matrix is saved (artifacts/oof_<name>.npy)")
    a = ap.parse_args()
    root = C.find_data_root()
    train = C.load_table(root, "train")
    folds_all = C.load_folds(len(train))
    classes = labels.load_classes(root / "competition_data" / "taxonomy_v1.csv")
    n_classes = len(classes)
    y_all = labels.encode(train["root_code"], classes)
    d = COMP / "artifacts" / "runs" / a.run / a.run / a.job
    meta = d / f"{a.tag}_meta.json"
    if meta.is_file() and json.loads(meta.read_text()).get("table") != "train":
        raise SystemExit("This output does not belong to training rows — this script only evaluates on labelled rows.")
    idx = np.load(d / f"{a.tag}_idx.npy")
    y, folds = y_all[idx], folds_all[idx]
    prior = np.log(np.clip(np.load(COMP / "artifacts" / "oof_prior_question_v1.npy")[idx], EPS, 1))
    rows_prior = metric.map3_rows_from_proba(y, np.exp(prior))
    masks = {"NONE": y == 0, "error rows": y != 0, "part A": idx < 358, "part B": idx >= 358}
    print(f"{a.run}/{a.job}/{a.tag}: {len(idx)} rows, folds {np.bincount(folds).tolist()}")
    out: dict = {"n": int(len(idx))}
    out["prior"] = report("P  task prior", y, np.exp(prior), rows_prior, masks)

    Z = clean_logp(np.load(d / f"{a.tag}_diag.npy"))
    out["zeroshot"] = report("Z  untrained model", y, np.exp(Z), rows_prior, masks)

    # NONE signals from the answer check
    none_cols, none_names = [], []
    for probe in ("fac", "ans"):
        f = d / f"{a.tag}_{probe}.npy"
        if f.is_file():
            lp = np.load(f)
            m = lp[:, 0] - lp[:, 1]
            none_cols.append(np.where(np.isfinite(m), m, 0.0))
            none_names.append(probe)
            fac = (train["final_answer_correct"].iloc[idx].str.lower() == "true").to_numpy()
            out[f"{probe}_acc"] = round(float(((m > 0) == fac).mean()), 4)
            print(f"  answer check {probe}: accuracy against final_answer_correct {out[f'{probe}_acc']:.4f}, against NONE {((m > 0) == (y == 0)).mean():.4f}")
    N = np.column_stack(none_cols) if none_cols else None

    # probe on the hidden state: nested, so that the blend only sees OOF values of the probe
    H = None
    hf = d / f"{a.tag}_diag_h.npy"
    if hf.is_file():
        X = np.load(hf).astype(np.float32)
        best = None
        for C_reg in (0.01, 0.03, 0.1, 0.3):
            oof = crossfit_probe(X, y, folds, n_classes, a.pca, C_reg)
            ll = -oof[np.arange(len(y)), y].mean()
            r = metric.map3_rows_from_proba(y, np.exp(oof)).mean()
            print(f"    probe C={C_reg}: MAP@3 {r:.4f}, log loss {ll:.3f}")
            if best is None or ll < best[0]:
                best = (ll, C_reg, oof)
        H = best[2]
        out["probe_C"] = best[1]
        out["probe"] = report(f"H  probe (C={best[1]}, best by log loss)", y, np.exp(H), rows_prior, masks)

    # blends, weights per fold fitted on the other folds
    def crossfit_mix(chans, none_feats, nested_probe: bool):
        p = np.zeros((len(y), n_classes))
        thetas = []
        for k in np.unique(folds):
            tr, va = folds != k, folds == k
            ch_tr, ch_va = [c[tr] for c in chans], [c[va] for c in chans]
            if nested_probe:  # refit the probe cross-wise within the other folds for the training rows
                Xh = np.load(hf).astype(np.float32)
                inner = crossfit_probe(Xh[tr], y[tr], folds[tr], n_classes, a.pca, out["probe_C"])
                outer = fit_probe(Xh[tr], y[tr], Xh[va], n_classes, a.pca, out["probe_C"])
                ch_tr, ch_va = ch_tr + [inner], ch_va + [outer]
            th = fit_mix(ch_tr, None if none_feats is None else none_feats[tr], y[tr])
            p[va] = apply_mix(th, ch_va, None if none_feats is None else none_feats[va])
            thetas.append([round(float(v), 3) for v in th])
        return p, thetas

    combos = [("P+N  prior + answer check", [prior], N, False), ("Z+P  model + prior", [Z, prior], None, False),
              ("Z+P+N  plus answer check", [Z, prior], N, False)]
    if H is not None:
        combos += [("Z+P+H  plus probe", [Z, prior], None, True), ("Z+P+H+N  everything", [Z, prior], N, True)]
    best_p = None
    for name, chans, nf, nested in combos:
        if nf is None and "N" in name.split()[0].split("+"):
            continue
        p, thetas = crossfit_mix(chans, nf, nested)
        key = name.split()[0]
        out[key] = report(name, y, p, rows_prior, masks)
        out[key]["weights_per_fold"] = thetas
        best_p = p
    if a.save and best_p is not None:
        if len(idx) != len(y_all) or not (idx == np.arange(len(y_all))).all():
            raise SystemExit("--save requires all training rows in table order (otherwise the matrix would be misaligned).")
        from kaggle_core import artifacts as A
        from kaggle_core.folds import folds_hash

        A.save_oof(a.save, best_p, np.zeros((0, n_classes)), {"folds_hash": folds_hash(folds_all), "model_type": "stack", "source": f"{a.run}/{a.job}/{a.tag}",
                                                              "cv_score": float(metric.map3_from_proba(y, best_p))}, COMP / "artifacts")
        print(f"  OOF of the last blend saved: artifacts/oof_{a.save}.npy (with meta)")
    dst = COMP / "artifacts" / "run_summaries"
    dst.mkdir(parents=True, exist_ok=True)
    (dst / f"{a.run}__{a.job}__stack.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
