"""Fast metrics, metric registry and class multipliers.

`fast_balanced_accuracy` is designed for repeated evaluation inside optimisation
loops (grid search, Nelder-Mead): `BalancedAccuracyScorer` precomputes the class
indices once and afterwards evaluates fully vectorised.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

MetricFn = Callable[[np.ndarray, np.ndarray], float]


def fast_balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Balanced accuracy = mean of the per-class recalls, vectorised via bincount.

    Much faster than sklearn for repeated calls; identical result (without the
    adjusted option).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    classes, y_idx = np.unique(y_true, return_inverse=True)
    totals = np.bincount(y_idx, minlength=len(classes)).astype(np.float64)
    hits = np.bincount(
        y_idx, weights=(y_pred == y_true).astype(np.float64), minlength=len(classes)
    )
    return float(np.mean(hits / totals))


class BalancedAccuracyScorer:
    """Precomputed class indices for many evaluations against the same y_true.

    Use in tuning loops: instantiate once, then call only ``scorer(y_pred)`` per
    candidate.
    """

    def __init__(self, y_true: np.ndarray) -> None:
        self.y_true = np.asarray(y_true)
        self.classes, self._y_idx = np.unique(self.y_true, return_inverse=True)
        self._totals = np.bincount(
            self._y_idx, minlength=len(self.classes)
        ).astype(np.float64)

    def __call__(self, y_pred: np.ndarray) -> float:
        hits = np.bincount(
            self._y_idx,
            weights=(np.asarray(y_pred) == self.y_true).astype(np.float64),
            minlength=len(self.classes),
        )
        return float(np.mean(hits / self._totals))


@dataclass(frozen=True)
class Metric:
    """Entry of the metric registry."""

    name: str
    fn: MetricFn
    greater_is_better: bool


_REGISTRY: dict[str, Metric] = {}


def register_metric(name: str, fn: MetricFn, greater_is_better: bool = True) -> Metric:
    """Register a metric under a name (overwrites an existing entry)."""
    metric = Metric(name=name, fn=fn, greater_is_better=greater_is_better)
    _REGISTRY[name] = metric
    return metric


def get_metric(name: str) -> Metric:
    """Return a registered metric; raises KeyError listing the available ones."""
    if name not in _REGISTRY:
        raise KeyError(
            f"Metric {name!r} is not registered. Available: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name]


def list_metrics() -> list[str]:
    """Names of all registered metrics."""
    return sorted(_REGISTRY)


def apply_class_multipliers(P: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Scale class probabilities by multipliers and take the argmax.

    Args:
        P: probabilities, shape (n, K).
        w: multipliers, shape (K,).

    Returns:
        hard class indices, shape (n,).
    """
    P = np.asarray(P, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    if P.ndim != 2 or w.shape != (P.shape[1],):
        raise ValueError(f"Shape mismatch: P {P.shape}, w {w.shape}.")
    return np.argmax(P * w[None, :], axis=1)


def multiclass_logloss(y_true: np.ndarray, P: np.ndarray, eps: float = 1e-15) -> float:
    """Log loss of class indices y_true against probabilities P (n, K)."""
    P = np.clip(np.asarray(P, dtype=np.float64), eps, 1.0)
    P = P / P.sum(axis=1, keepdims=True)
    y_true = np.asarray(y_true, dtype=np.int64)
    return float(-np.mean(np.log(P[np.arange(len(y_true)), y_true])))


# pre-register the standard metrics
register_metric("balanced_accuracy", fast_balanced_accuracy, greater_is_better=True)
register_metric(
    "accuracy",
    lambda y_true, y_pred: float(np.mean(np.asarray(y_true) == np.asarray(y_pred))),
    greater_is_better=True,
)
