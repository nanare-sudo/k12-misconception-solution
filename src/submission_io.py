"""Build, strictly check and write a submission.

Format (evaluation page): header ``trace_id,root_code``; per ``trace_id`` up to three codes,
separated by spaces, ordered by confidence; file name ``submission.csv``. A missing
``trace_id`` scores 0. Codes outside the taxonomy (other than ``NONE``) are ignored.

The check is stricter than Kaggle's so that no slot is wasted:

* every test ``trace_id`` exactly once, no foreign IDs;
* 1–3 codes per row, no duplicates within a row;
* every code spelled exactly as in the taxonomy (``Dec-01``, not ``DEC-01``) or ``NONE``;
* optional: no code that never occurs in training (according to the host it cannot be correct
  on the test set).

The generic check (row count, ID set, NaNs) runs through ``kaggle_core.submission``.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from kaggle_core.submission import SubmissionError, validate_submission

ID_COL = "trace_id"
PRED_COL = "root_code"


def build_submission(trace_ids: Sequence[str], top_idx: np.ndarray, classes: list[str]) -> pd.DataFrame:
    """Build the submission frame from top-3 class indices (see ``metric.top3_indices``)."""
    top_idx = np.asarray(top_idx)
    if top_idx.ndim != 2 or len(trace_ids) != len(top_idx):
        raise SubmissionError(f"Shape {top_idx.shape} does not match {len(trace_ids)} trace_ids.")
    rows = [" ".join(classes[j] for j in row) for row in top_idx]
    return pd.DataFrame({ID_COL: list(trace_ids), PRED_COL: rows})


def check_submission(
    df: pd.DataFrame,
    sample_submission: pd.DataFrame,
    classes: list[str],
    never_seen: Iterable[str] = (),
    require_three: bool = False,
) -> dict:
    """Check a submission and return summary statistics. Raises ``SubmissionError`` on violation.

    Args:
        df: the submission.
        sample_submission: Kaggle's ``sample_submission.csv`` (provides the required IDs).
        classes: class list from ``labels.load_classes``.
        never_seen: codes without a training example — useless in a submission, hence an error.
        require_three: require exactly three codes per row (default: 1–3 allowed).
    """
    if list(df.columns) != [ID_COL, PRED_COL]:
        raise SubmissionError(f"Columns must be exactly {[ID_COL, PRED_COL]}, got {list(df.columns)}.")
    validate_submission(df, sample_submission, required_cols=[ID_COL, PRED_COL], id_col=ID_COL)

    valid = set(classes)
    useless = set(never_seen)
    n_codes = np.zeros(len(df), dtype=np.int64)
    first: list[str] = []
    for i, cell in enumerate(df[PRED_COL].astype(str)):
        codes = cell.split(" ")
        if cell != cell.strip() or "" in codes:
            raise SubmissionError(f"Row {i}: superfluous spaces in {cell!r}.")
        if not 1 <= len(codes) <= 3:
            raise SubmissionError(f"Row {i}: {len(codes)} codes — 1 to 3 are allowed.")
        if require_three and len(codes) != 3:
            raise SubmissionError(f"Row {i}: {len(codes)} codes — exactly 3 are required.")
        if len(set(codes)) != len(codes):
            raise SubmissionError(f"Row {i}: duplicate code in {cell!r} — wasted slot.")
        bad = [c for c in codes if c not in valid]
        if bad:
            raise SubmissionError(f"Row {i}: code(s) {bad} not in the taxonomy (exact spelling?).")
        dead = [c for c in codes if c in useless]
        if dead:
            raise SubmissionError(f"Row {i}: code(s) {dead} without a training example — cannot be correct on the test set.")
        n_codes[i] = len(codes)
        first.append(codes[0])

    first_s = pd.Series(first)
    return {
        "n_rows": int(len(df)),
        "share_none_rank1": float((first_s == "NONE").mean()),
        "n_distinct_rank1": int(first_s.nunique()),
        "rows_with_3_codes": int((n_codes == 3).sum()),
        "rows_with_fewer_codes": int((n_codes < 3).sum()),
    }


def write_checked(
    df: pd.DataFrame,
    path: str | Path,
    sample_submission: pd.DataFrame,
    classes: list[str],
    never_seen: Iterable[str] = (),
) -> dict:
    """Check and write the submission. Returns the summary statistics of the check."""
    stats = check_submission(df, sample_submission, classes, never_seen=never_seen)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return stats
