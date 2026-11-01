"""Submission validation and writing.

Any bug in a rerun means a score of 0. Every submission is therefore validated
strictly before it is written — row count, IDs, columns, allowed values, NaNs.
A violation raises SubmissionError; nothing is written silently.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


class SubmissionError(ValueError):
    """The submission violates the expected format."""


def validate_submission(
    df: pd.DataFrame,
    sample_submission: pd.DataFrame,
    required_cols: list[str] | None = None,
    id_col: str | None = None,
    allowed_labels: set | list | None = None,
    value_range: tuple[float, float] | None = None,
) -> None:
    """Check a submission against the sample submission. Raises on violation.

    Checks: row count, columns (names and order are free, presence is required),
    ID set and uniqueness, NaNs, optionally allowed labels or the value range of
    numeric predictions.

    Args:
        df: the submission to check.
        sample_submission: Kaggle's sample_submission.csv as a DataFrame.
        required_cols: required columns; default: all columns of the sample.
        id_col: ID column; default: first column of the sample.
        allowed_labels: allowed values for the prediction column(s).
        value_range: (min, max) for numeric prediction columns.
    """
    if required_cols is None:
        required_cols = list(sample_submission.columns)
    if id_col is None:
        id_col = str(sample_submission.columns[0])

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise SubmissionError(f"Missing columns: {missing}.")
    if id_col not in df.columns:
        raise SubmissionError(f"ID column {id_col!r} missing from the submission.")
    if id_col not in sample_submission.columns:
        raise SubmissionError(
            f"ID column {id_col!r} missing from sample_submission — wrong "
            "id_col parameter?"
        )

    if len(df) != len(sample_submission):
        raise SubmissionError(
            f"Row count {len(df)} != expected {len(sample_submission)}."
        )

    if df[id_col].duplicated().any():
        n_dup = int(df[id_col].duplicated().sum())
        raise SubmissionError(f"{n_dup} duplicate IDs in {id_col!r}.")

    expected_ids = set(sample_submission[id_col])
    got_ids = set(df[id_col])
    if got_ids != expected_ids:
        raise SubmissionError(
            f"ID set differs: {len(got_ids - expected_ids)} unknown, "
            f"{len(expected_ids - got_ids)} missing IDs."
        )

    pred_cols = [c for c in required_cols if c != id_col]
    for col in pred_cols:
        if df[col].isna().any():
            raise SubmissionError(f"{int(df[col].isna().sum())} NaNs in {col!r}.")
        if allowed_labels is not None:
            bad = set(df[col].unique()) - set(allowed_labels)
            if bad:
                raise SubmissionError(f"Disallowed labels in {col!r}: {sorted(bad)!r}.")
        if value_range is not None:
            lo, hi = value_range
            vals = pd.to_numeric(df[col], errors="coerce")
            if vals.isna().any():
                raise SubmissionError(f"Non-numeric values in {col!r}.")
            if (vals < lo).any() or (vals > hi).any():
                raise SubmissionError(
                    f"Values in {col!r} outside [{lo}, {hi}]: "
                    f"min={vals.min()}, max={vals.max()}."
                )


def write_submission(
    df: pd.DataFrame,
    path: str | Path,
    sample_submission: pd.DataFrame | None = None,
    **validate_kwargs,
) -> Path:
    """Validate (if a sample submission is given) and write the submission.

    On Kaggle the file must be named exactly `submission.csv` and live in the
    output directory — the caller is responsible for the path; this function only
    warns if the file name differs.
    """
    path = Path(path)
    if sample_submission is not None:
        validate_submission(df, sample_submission, **validate_kwargs)
    if path.name != "submission.csv":
        print(
            f"   [submission] Note: the file name is {path.name!r} — on Kaggle "
            "it must be 'submission.csv'."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path
