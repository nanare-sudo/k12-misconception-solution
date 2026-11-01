"""Assemble the fold-wise measurements of a model into one OOF matrix over all 1,500 training rows.

    python -I eda/a6_assemble_oof.py <name> <run>/<job>/<tag>[/<probe>] [...] [--fill <channel> [--only <file.npy>]]
    python -I eda/a6_assemble_oof.py <name> --mean --fill <channel> <draw>[@<file.npy>] [<draw>[@<file.npy>] ...]

Every source provides rows (idx) and log-probabilities; together they must cover every training row exactly once.
With `--fill <channel>` the coverage may be incomplete: rows without a measurement take the probabilities from
artifacts/oof_<channel>.npy (assembled channel of a cascade, e.g. the reasoning channel only on the rows that the
channel without reasoning does not already consider error-free). `--only` additionally restricts the measurements to
the table positions of the file (path relative to the data folder), so that a fold that was measured completely is
assembled exactly like the others and like the test side.

With `--mean` every source is ONE draw of the same channel (several runs of one draw joined with "+", e.g. one per
fold): a row gets the arithmetic mean of the probabilities of all draws that cover it (every reasoning text is a draw;
the mean integrates over the reasoning). `@<file.npy>` restricts a draw to the table positions of the file (otherwise
--only applies), so that e.g. the first draw gets all rows with reasoning and every further draw only the rows of the
narrow gate, exactly as on the test side.

Written via kaggle_core.artifacts.save_oof:
    artifacts/oof_<name>.npy        probabilities (1500 x 76), dropped candidates with the remaining mass
    artifacts/oof_<name>_h.npy      hidden state at the decision position, if stored
    artifacts/meta_<name>.json      fold hash, sources, MAP@3 with σ
Training rows only (the meta file of every source is checked).
"""
import json
import sys
from pathlib import Path

import numpy as np

COMP = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(COMP), str(COMP / "src")]
import labels  # noqa: E402
import metric  # noqa: E402
from kaggle_core import artifacts as A  # noqa: E402
from kaggle_core.folds import folds_hash  # noqa: E402
from vlm import common as C  # noqa: E402
from vlm.engine import probs_from_logp  # noqa: E402


def mean_draws(draws, root, only, P, folds, y, absent):
    """Mean of the probabilities over several draws; writes into P and returns the number of draws per row."""
    n = len(y)
    S = np.zeros_like(P); cnt = np.zeros(n, int)
    for draw in draws:
        dspec, _, rows_file = draw.partition("@")
        rows_file = rows_file or only
        keep_rows = None if rows_file is None else np.load(rows_file if Path(rows_file).is_absolute() else root / rows_file)
        got = np.zeros(n, bool)
        for spec in dspec.split("+"):
            parts = spec.split("/"); run, job, tag = parts[:3]; probe = parts[3] if len(parts) > 3 else "diag"
            d = COMP / "artifacts" / "runs" / run / run / job
            meta = d / f"{tag}_meta.json"
            if meta.is_file() and json.loads(meta.read_text()).get("table") != "train":
                raise SystemExit(f"{spec}: no output for training rows")
            idx = np.load(d / f"{tag}_idx.npy"); lp = np.load(d / f"{tag}_{probe}.npy")
            done = d / f"{tag}_{probe}.done.npy"
            if done.is_file() and not np.load(done).all():
                keep = np.load(done); idx, lp = idx[keep], lp[keep]
            if keep_rows is not None:
                keep = np.isin(idx, keep_rows); idx, lp = idx[keep], lp[keep]
            if got[idx].any():
                raise SystemExit(f"{draw}: {int(got[idx].sum())} rows appear twice within one draw")
            S[idx] += probs_from_logp(lp, absent=absent); cnt[idx] += 1; got[idx] = True
        rows = np.where(got)[0]
        print(f"  draw {dspec[:90]}{'…' if len(dspec) > 90 else ''}: {len(rows)} rows (per fold {[int(got[folds == k].sum()) for k in range(5)]})"
              + (f", restricted to {rows_file}" if rows_file else ""))
    cov = cnt > 0
    P[cov] = S[cov] / cnt[cov, None]
    for k in sorted(set(cnt[cov].tolist())):
        m = cnt == k
        print(f"  {k} draw(s): {int(m.sum())} rows (per fold {[int(m[folds == f].sum()) for f in range(5)]}), MAP@3 of the mean there {metric.map3_rows_from_proba(y[m], P[m]).mean():.4f}")
    return cnt


def main():
    args = sys.argv[1:]
    fill = None
    if "--fill" in args:
        k = args.index("--fill")
        fill = args[k + 1]
        args = args[:k] + args[k + 2:]
    only = None
    if "--only" in args:
        k = args.index("--only")
        only = args[k + 1]
        args = args[:k] + args[k + 2:]
        if fill is None:
            raise SystemExit("--only exists only together with --fill")
    mean = "--mean" in args
    if mean:
        args.remove("--mean")
        if fill is None:
            raise SystemExit("--mean exists only together with --fill")
    name, specs = args[0], args[1:]
    root = C.find_data_root(); train = C.load_table(root, "train"); folds = C.load_folds(len(train))
    classes = labels.load_classes(root / "competition_data" / "taxonomy_v1.csv")
    y = labels.encode(train["root_code"], classes); n = len(y)
    absent = ~labels.seen_mask(train["root_code"], classes)
    P = np.zeros((n, len(classes))); H = None; seen = np.zeros(n, int)
    if mean:
        cnt = mean_draws(specs, root, only, P, folds, y, absent)
        seen = (cnt > 0).astype(int)
        specs = [f"mean:{s}" for s in specs]
    for spec in ([] if mean else specs):
        parts = spec.split("/"); run, job, tag = parts[:3]; probe = parts[3] if len(parts) > 3 else "diag"
        d = COMP / "artifacts" / "runs" / run / run / job
        meta = d / f"{tag}_meta.json"
        if meta.is_file() and json.loads(meta.read_text()).get("table") != "train":
            raise SystemExit(f"{spec}: no output for training rows")
        idx = np.load(d / f"{tag}_idx.npy"); lp = np.load(d / f"{tag}_{probe}.npy")
        done = d / f"{tag}_{probe}.done.npy"
        if done.is_file() and not np.load(done).all():
            if fill is None:
                raise SystemExit(f"{spec}: measurement incomplete ({int((~np.load(done)).sum())} rows missing)")
            keep = np.load(done); idx, lp = idx[keep], lp[keep]
        if only is not None:
            keep = np.isin(idx, np.load(only if Path(only).is_absolute() else root / only)); idx, lp = idx[keep], lp[keep]
        P[idx] = probs_from_logp(lp, absent=absent); seen[idx] += 1
        hf = d / f"{tag}_{probe}_h.npy"
        if hf.is_file() and fill is None:
            h = np.load(hf)
            H = np.zeros((n, h.shape[1]), np.float16) if H is None else H
            H[idx] = h
        r = metric.map3_rows_from_proba(y[idx], P[idx])
        print(f"  {spec}: {len(idx)} rows (folds {sorted(set(folds[idx].tolist()))}), MAP@3 {r.mean():.4f}")
    if (seen > 1).any() or (fill is None and not (seen == 1).all()):
        raise SystemExit(f"coverage incomplete: {int((seen == 0).sum())} rows missing, {int((seen > 1).sum())} twice")
    if fill is not None:
        miss = seen == 0
        P[miss] = np.load(COMP / "artifacts" / f"oof_{fill}.npy")[miss]
        H = None  # hidden states of two channels do not mix
        print(f"  --fill {fill}: {int(miss.sum())} of {n} rows from oof_{fill}.npy (per fold {[int(miss[folds == k].sum()) for k in range(5)]})")
        specs = list(specs) + [f"fill:oof_{fill}"]
    rows = metric.map3_rows_from_proba(y, P)
    cv = float(rows.mean()); sigma = float(metric.bootstrap_sigma(rows))
    A.save_oof(name, P, np.zeros((0, len(classes))), {"folds_hash": folds_hash(folds), "model_type": "vlm", "sources": specs,
                                                      "cv_score": cv, "cv_sigma": sigma, "feature_set": "image+task"}, COMP / "artifacts")
    if H is not None:
        np.save(COMP / "artifacts" / f"oof_{name}_h.npy", H)
    print(f"{name}: MAP@3 {cv:.4f} ± {sigma:.4f} per fold {[round(float(rows[folds == k].mean()), 4) for k in range(5)]} -> artifacts/oof_{name}.npy")


if __name__ == "__main__":
    main()
