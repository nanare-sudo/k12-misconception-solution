"""Margin alignment of the predicted class distribution for the blend WITH the reasoning channel — transductive, as a set.

    python -I eda/c21_margin_align.py [--oof v1_zs27q_think18] [--strengths 0.25 0.5 0.75 1.0] [--splits 40]

The competition split is stratified by root_code: the class frequencies of the test set are (up to rounding) those of
the training set. Class weights w bring the mean predicted mass per class to these frequencies (iterative fit, only the
marginal distribution of the set — no labels of individual rows); applied with strength s (w to the power s). This uses
the test rows as a set: primary line and safety line with set alignment (not the strictly inductive line).

Four measurements on the cross-fitted predictions of the blend (artifacts/oof_<name>.npy):
  A  all 1,500 rows as ONE set, target = their own marginal distribution (too favourable: on the test set the target is
     only estimated);
  B  like the test set: random halves stratified by class; one half is aligned to the marginal distribution of the
     OTHER (honest; with 750 rows smaller than the test set, hence rather too unfavourable), mean over --splits splits;
  C  per fold as its own set of 300 rows, target = frequencies of the other folds;
  D  INDUCTIVE: weights learned on the other folds and only applied to the fold (no use of the set — also suitable for
     the inductive line).
Training rows only, aggregates only.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

COMP = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(COMP), str(COMP / "src")]

import labels  # noqa: E402
import metric  # noqa: E402
from vlm import common as C  # noqa: E402


def fit_weights(P: np.ndarray, target: np.ndarray, iters: int = 200) -> np.ndarray:
    """Class weights that bring the mean predicted mass per class to `target` (classes without a target: 1)."""
    ok = target > 0
    w = np.ones(P.shape[1])
    for _ in range(iters):
        Q = P * w
        Q /= Q.sum(1, keepdims=True)
        m = Q.mean(0)
        step = np.where(ok & (m > 0), target / np.maximum(m, 1e-12), 1.0)
        w *= step
        if np.abs(np.log(step[ok & (m > 0)])).max() < 1e-6:
            break
    return np.where(ok, w, 1.0)


def apply_weights(P: np.ndarray, w: np.ndarray, s: float) -> np.ndarray:
    Q = P * w ** s
    return Q / Q.sum(1, keepdims=True)


def align(P: np.ndarray, target: np.ndarray, s: float) -> np.ndarray:
    """Transductive: fit the weights on THIS set and apply them with strength s."""
    return apply_weights(P, fit_weights(P, target), s)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--oof", default="v1_zs27q_think18")
    ap.add_argument("--strengths", nargs="+", type=float, default=[0.25, 0.5, 0.75, 1.0])
    ap.add_argument("--splits", type=int, default=40)
    a = ap.parse_args()

    root = C.find_data_root()
    train = C.load_table(root, "train")
    folds = C.load_folds(len(train))
    classes = labels.load_classes(root / "competition_data" / "taxonomy_v1.csv")
    y = labels.encode(train["root_code"], classes)
    n, K = len(y), len(classes)
    P = np.clip(np.load(COMP / "artifacts" / f"oof_{a.oof}.npy"), 1e-9, 1)
    P /= P.sum(1, keepdims=True)
    r0 = metric.map3_rows_from_proba(y, P)
    freq = lambda idx: np.bincount(y[idx], minlength=K) / len(idx)  # noqa: E731
    q, code = train["question_id"].to_numpy(), train["root_code"].to_numpy()
    seen = np.array([np.any((q == q[i]) & (folds != folds[i])) for i in range(n)])
    cs = np.array([np.any((q == q[i]) & (folds != folds[i]) & (code == code[i])) for i in range(n)])
    capped = np.isin(code, ["ANS-01", "CAL-01", "WOP-01", "FRM-01", "OTH-00"])
    groups = {"NONE": y == 0, "seen": (y != 0) & cs, "never seen": (y != 0) & seen & ~cs, "new task": (y != 0) & ~seen,
              "generic codes": capped, "other error codes": (y != 0) & ~capped}
    print(f"blend {a.oof}: MAP@3 {r0.mean():.4f} ± {metric.bootstrap_sigma(r0):.4f} over {n} rows")

    for s in a.strengths:
        # A: one set, own target
        rA = metric.map3_rows_from_proba(y, align(P, freq(np.arange(n)), s))
        pA = metric.paired_bootstrap(rA, r0)
        # B: stratified halves, target = the other half
        dB = np.zeros(n)
        cnt = np.zeros(n)
        for rep in range(a.splits):
            rng = np.random.default_rng(1000 + rep)
            half = np.zeros(n, dtype=bool)
            for c in np.unique(y):
                idx = rng.permutation(np.where(y == c)[0])
                k = len(idx) // 2 + (rng.integers(2) if len(idx) % 2 else 0)
                half[idx[:k]] = True
            for me, other in ((half, ~half), (~half, half)):
                i_me, i_ot = np.where(me)[0], np.where(other)[0]
                r = metric.map3_rows_from_proba(y[i_me], align(P[i_me], freq(i_ot), s))
                dB[i_me] += r - r0[i_me]
                cnt[i_me] += 1
        dB /= cnt
        sB = metric.bootstrap_sigma(dB)
        # C: per fold
        rC = np.zeros(n)
        for k in range(5):
            va = np.where(folds == k)[0]
            rC[va] = metric.map3_rows_from_proba(y[va], align(P[va], freq(np.where(folds != k)[0]), s))
        pC = metric.paired_bootstrap(rC, r0)
        # D: inductive — weights learned on the other folds (there: mass = frequency), only applied to the fold
        rD = np.zeros(n)
        for k in range(5):
            tr, va = np.where(folds != k)[0], np.where(folds == k)[0]
            rD[va] = metric.map3_rows_from_proba(y[va], apply_weights(P[va], fit_weights(P[tr], freq(tr)), s))
        pD = metric.paired_bootstrap(rD, r0)
        print(f"          D inductive (weights from the other folds) {pD['delta']:+.4f} ± {pD['sigma']:.4f} (z {pD['z']:+.1f}; per fold "
              f"{[round(float((rD - r0)[folds == k].mean()), 4) for k in range(5)]}) | by group: " + " | ".join(f"{g} {(rD - r0)[m].mean():+.3f}" for g, m in groups.items()))
        print(f"s = {s:<4g} | A one set, own target {pA['delta']:+.4f} ± {pA['sigma']:.4f} (z {pA['z']:+.1f}) | "
              f"B halves, target of the other {dB.mean():+.4f} ± {sB:.4f} (z {dB.mean() / max(sB, 1e-9):+.1f}) | "
              f"C per fold {pC['delta']:+.4f} ± {pC['sigma']:.4f} (z {pC['z']:+.1f}; per fold {[round(float((rC - r0)[folds == k].mean()), 4) for k in range(5)]})")
        print("          B by group: " + " | ".join(f"{g} {dB[m].mean():+.3f}" for g, m in groups.items()))
    w = fit_weights(P, freq(np.arange(n)))
    top = np.bincount(P.argmax(1), minlength=K)
    true = np.bincount(y, minlength=K)
    order = np.argsort(w)
    print("weights on all rows — lowered most: " + ", ".join(f"{classes[c]} {w[c]:.2f} (top {top[c]}, true {true[c]})" for c in order[:8] if true[c] > 0))
    print("                     — raised most: " + ", ".join(f"{classes[c]} {w[c]:.2f} (top {top[c]}, true {true[c]})" for c in order[::-1][:8] if true[c] > 0))
    print(f"spread of the weights: median {np.median(w[true > 0]):.2f}, 10 % quantile {np.quantile(w[true > 0], 0.1):.2f}, 90 % quantile {np.quantile(w[true > 0], 0.9):.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
