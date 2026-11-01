"""MAP@3 — the competition metric.

Definition (evaluation page): every row has exactly one correct label. The row score is 1 if
the first prediction is correct, 1/2 for the second, 1/3 for the third, otherwise 0. At most
the first three predictions count. Repeating the correct label does not count twice. The
leaderboard score is the mean over rows.

Two forms:

* ``map3_rows`` / ``map3`` — on prediction lists (exactly the form of the submission file).
* ``map3_from_proba`` — on probability matrices (CV, blending); uses ``top3_indices``.

The behaviour for invalid codes is not reproduced: according to the evaluation page they are
"ignored, not penalised" — whether they still use up one of the three slots is not
documented. Our submissions never contain invalid codes (``submission_io`` checks this), so
an invalid code simply counts as a wrong prediction in its slot here.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

K = 3
_GAINS = np.array([1.0, 0.5, 1.0 / 3.0])


def map3_rows(y_true: Sequence[str], preds: Sequence[Sequence[str]]) -> np.ndarray:
    """Row scores in {0, 1/3, 1/2, 1} for prediction lists (up to 3 codes per row)."""
    if len(y_true) != len(preds):
        raise ValueError(f"Length of y_true {len(y_true)} != preds {len(preds)}.")
    out = np.zeros(len(y_true), dtype=np.float64)
    for i, (truth, row) in enumerate(zip(y_true, preds)):
        if isinstance(row, str):
            # Otherwise "ALG-06 CAL-01" would be iterated character by character and silently scored 0.
            raise TypeError(f"Row {i} is a string — split it into codes with .split() first.")
        for rank, code in enumerate(list(row)[:K]):
            if code == truth:
                out[i] = _GAINS[rank]
                break  # the first occurrence counts, repetitions are skipped
    return out


def map3(y_true: Sequence[str], preds: Sequence[Sequence[str]]) -> float:
    """MAP@3 as the mean over rows."""
    return float(map3_rows(y_true, preds).mean())


def top3_indices(P: np.ndarray, allowed: np.ndarray | None = None) -> np.ndarray:
    """Indices of the three most likely classes per row, sorted in descending order.

    Ties are broken deterministically in favour of the smaller class index (stable sort), so
    that two runs with identical probabilities produce identical submissions.

    Args:
        P: (n, C) scores or probabilities.
        allowed: optional boolean mask (C,); classes with False are never chosen
            (e.g. codes without a training example, see ``labels.seen_mask``).
    """
    P = np.asarray(P, dtype=np.float64)
    if P.ndim != 2 or P.shape[1] < K:
        raise ValueError(f"P must have shape (n, C>={K}), got {P.shape}.")
    if np.isnan(P).any():
        # Otherwise a NaN row would silently become classes 0, 1, 2 (NONE first).
        raise ValueError(f"P contains NaN in {int(np.isnan(P).any(axis=1).sum())} rows.")
    if allowed is not None:
        allowed = np.asarray(allowed, dtype=bool)
        if allowed.shape != (P.shape[1],) or allowed.sum() < K:
            raise ValueError("allowed must have shape (C,) and allow at least 3 classes.")
        P = np.where(allowed[None, :], P, -np.inf)
    return np.argsort(-P, axis=1, kind="stable")[:, :K]


def map3_rows_from_proba(y_idx: np.ndarray, P: np.ndarray, allowed: np.ndarray | None = None) -> np.ndarray:
    """Row scores for class indices ``y_idx`` against a probability matrix."""
    y_idx = np.asarray(y_idx)
    top = top3_indices(P, allowed)
    hits = top == y_idx[:, None]
    return (hits * _GAINS[None, :]).sum(axis=1)


def map3_from_proba(y_idx: np.ndarray, P: np.ndarray, allowed: np.ndarray | None = None) -> float:
    """MAP@3 for class indices against a probability matrix."""
    return float(map3_rows_from_proba(y_idx, P, allowed).mean())


def bootstrap_sigma(row_scores: np.ndarray, n: int | None = None, n_boot: int = 2000, seed: int = 0) -> float:
    """Standard deviation of the mean of ``n`` rows, estimated with a row bootstrap.

    With ``n`` = 1000 (public) or 2498 (private) this turns OOF row scores into the σ that an
    equally good model would have on the respective leaderboard split.
    """
    row_scores = np.asarray(row_scores, dtype=np.float64)
    n = int(n or len(row_scores))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(row_scores), size=(n_boot, n))
    return float(row_scores[idx].mean(axis=1).std(ddof=1))


def paired_bootstrap(rows_a: np.ndarray, rows_b: np.ndarray, n_boot: int = 5000, seed: int = 0) -> dict[str, float]:
    """Paired row bootstrap for the difference of two solutions on the same rows.

    Returns the difference (a − b), its σ and the share of bootstrap draws with a > b.
    Differences below about 2σ are treated as noise.
    """
    a = np.asarray(rows_a, dtype=np.float64)
    b = np.asarray(rows_b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("Both row-score arrays must have the same shape.")
    d = a - b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    means = d[idx].mean(axis=1)
    sigma = float(means.std(ddof=1))
    return {
        "delta": float(d.mean()),
        "sigma": sigma,
        "z": float(d.mean() / sigma) if sigma > 0 else float("inf") if d.mean() else 0.0,
        "p_a_better": float((means > 0).mean()),
    }


def _register() -> None:
    """Register ``map3`` in the kaggle_core metric registry (if importable)."""
    try:
        from kaggle_core.metrics import register_metric
    except ImportError:  # on Kaggle without the src dataset, or in isolated tests
        return
    register_metric("map3", lambda y_idx, P: map3_from_proba(y_idx, P), greater_is_better=True)


_register()
