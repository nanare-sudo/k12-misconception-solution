"""Fold creation and consistency.

All models of a blend MUST use identical folds; otherwise the OOF blend is
meaningless (leakage between training and validation rows of different
models). Therefore: create the folds once, save them, load them everywhere and
call `assert_same_folds` before every blend.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold


def make_folds(
    y: np.ndarray,
    n_splits: int = 5,
    seed: int = 42,
    strategy: str = "stratified",
    groups: np.ndarray | None = None,
) -> np.ndarray:
    """Create a fold array: ``folds[i]`` is the fold number of row ``i``.

    Args:
        y: target variable, length n.
        n_splits: number of folds.
        seed: seed for reproducible shuffling.
        strategy: ``"stratified"`` (default), ``"group"`` or ``"kfold"``.
        groups: group labels, only for ``strategy="group"`` (required there).

    Returns:
        int array of length n with values in ``[0, n_splits)``.
    """
    y = np.asarray(y)
    n = len(y)
    if strategy == "stratified":
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        split_iter = splitter.split(np.zeros(n), y)
    elif strategy == "group":
        if groups is None:
            raise ValueError("strategy='group' requires the groups argument.")
        try:
            # sklearn >= 1.6: GroupKFold can shuffle -> the seed takes effect.
            splitter = GroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        except TypeError:
            # Older sklearn (e.g. on Kaggle): deterministic, the seed has no
            # effect — say so explicitly instead of silently ignoring it.
            print(
                "   [folds] Note: the installed sklearn does not support "
                "GroupKFold(shuffle=...) — the seed has no effect for strategy='group'."
            )
            splitter = GroupKFold(n_splits=n_splits)
        split_iter = splitter.split(np.zeros(n), y, groups=np.asarray(groups))
    elif strategy == "kfold":
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        split_iter = splitter.split(np.zeros(n))
    else:
        raise ValueError(f"Unknown strategy: {strategy!r}")

    folds = np.full(n, -1, dtype=np.int64)
    for fold_id, (_, val_idx) in enumerate(split_iter):
        folds[val_idx] = fold_id
    assert (folds >= 0).all(), "Every row must be assigned to exactly one fold."
    return folds


def save_folds(path: str | Path, folds: np.ndarray) -> Path:
    """Save a fold array as .npy and return the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(folds, dtype=np.int64))
    # np.save appends .npy if it is missing
    return path if path.suffix == ".npy" else path.with_suffix(path.suffix + ".npy")


def load_folds(path: str | Path) -> np.ndarray:
    """Load a fold array saved with `save_folds`."""
    return np.load(Path(path)).astype(np.int64)


def folds_hash(folds: np.ndarray) -> str:
    """Short, stable hash of a fold array (for artifact metadata)."""
    arr = np.ascontiguousarray(np.asarray(folds, dtype=np.int64))
    return hashlib.sha256(arr.tobytes()).hexdigest()[:16]


def assert_same_folds(*fold_arrays: np.ndarray) -> None:
    """Raise AssertionError unless all fold arrays are identical.

    Call this before every blend: models with different folds must not be
    blended together.
    """
    if len(fold_arrays) < 2:
        return
    ref = np.asarray(fold_arrays[0])
    for i, other in enumerate(fold_arrays[1:], start=1):
        other = np.asarray(other)
        assert ref.shape == other.shape, (
            f"Fold array {i} has shape {other.shape}, expected {ref.shape}."
        )
        if not np.array_equal(ref, other):
            n_diff = int((ref != other).sum())
            raise AssertionError(
                f"Fold array {i} differs in {n_diff}/{len(ref)} rows — "
                "all blend candidates must use identical folds."
            )
