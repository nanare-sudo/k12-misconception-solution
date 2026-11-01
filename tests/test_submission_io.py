"""Submission check: every format trap of this competition must be caught as an error."""
import numpy as np
import pandas as pd
import pytest

import labels
import submission_io as sio
from kaggle_core.submission import SubmissionError

CLASSES = ["NONE", "ALG-06", "CAL-01", "Dec-01", "NOT-02"]
SAMPLE = pd.DataFrame({"trace_id": ["a", "b", "c"], "root_code": ["NONE"] * 3})


def frame(rows):
    return pd.DataFrame({"trace_id": ["a", "b", "c"], "root_code": rows})


def test_valid_submission_and_stats():
    stats = sio.check_submission(frame(["NONE ALG-06 CAL-01", "Dec-01", "CAL-01 NONE"]), SAMPLE, CLASSES)
    assert stats == {
        "n_rows": 3,
        "share_none_rank1": pytest.approx(1 / 3),
        "n_distinct_rank1": 3,
        "rows_with_3_codes": 1,
        "rows_with_fewer_codes": 2,
    }


@pytest.mark.parametrize(
    "rows, message",
    [
        (["NONE", "DEC-01", "NONE"], "not in the taxonomy"),  # wrong spelling
        (["NONE", "NONE NONE", "NONE"], "duplicate code"),
        (["NONE", "NONE ALG-06 CAL-01 Dec-01", "NONE"], "1 to 3 are allowed"),
        (["NONE", "NONE  ALG-06", "NONE"], "spaces"),
        (["NONE", " NONE", "NONE"], "spaces"),
        (["NONE", "XYZ-99", "NONE"], "not in the taxonomy"),
    ],
)
def test_format_violations(rows, message):
    with pytest.raises(SubmissionError, match=message):
        sio.check_submission(frame(rows), SAMPLE, CLASSES)


def test_never_seen_code_is_rejected():
    df = frame(["NONE", "NOT-02", "NONE"])
    sio.check_submission(df, SAMPLE, CLASSES)  # allowed without the argument
    with pytest.raises(SubmissionError, match="without a training example"):
        sio.check_submission(df, SAMPLE, CLASSES, never_seen=["NOT-02"])


def test_require_three():
    with pytest.raises(SubmissionError, match="exactly 3"):
        sio.check_submission(frame(["NONE", "NONE ALG-06", "NONE"]), SAMPLE, CLASSES, require_three=True)


def test_missing_duplicate_and_foreign_ids():
    with pytest.raises(SubmissionError):
        sio.check_submission(pd.DataFrame({"trace_id": ["a", "b"], "root_code": ["NONE"] * 2}), SAMPLE, CLASSES)
    with pytest.raises(SubmissionError):
        sio.check_submission(pd.DataFrame({"trace_id": ["a", "a", "c"], "root_code": ["NONE"] * 3}), SAMPLE, CLASSES)
    with pytest.raises(SubmissionError):
        sio.check_submission(pd.DataFrame({"trace_id": ["a", "b", "x"], "root_code": ["NONE"] * 3}), SAMPLE, CLASSES)


def test_wrong_columns_and_nan():
    with pytest.raises(SubmissionError, match="[Cc]olumns"):
        sio.check_submission(pd.DataFrame({"trace_id": ["a", "b", "c"], "pred": ["NONE"] * 3}), SAMPLE, CLASSES)
    with pytest.raises(SubmissionError):
        sio.check_submission(frame(["NONE", None, "NONE"]), SAMPLE, CLASSES)


def test_build_and_write_roundtrip(tmp_path):
    top = np.array([[0, 1, 2], [3, 0, 1], [2, 1, 0]])
    df = sio.build_submission(["a", "b", "c"], top, CLASSES)
    assert df.root_code.tolist() == ["NONE ALG-06 CAL-01", "Dec-01 NONE ALG-06", "CAL-01 ALG-06 NONE"]
    stats = sio.write_checked(df, tmp_path / "submission.csv", SAMPLE, CLASSES, never_seen=["NOT-02"])
    assert stats["rows_with_3_codes"] == 3
    back = pd.read_csv(tmp_path / "submission.csv")
    assert list(back.columns) == ["trace_id", "root_code"] and back.equals(df)


def test_labels_helpers(tmp_path):
    tax = tmp_path / "taxonomy.csv"
    pd.DataFrame({"id": ["ALG-06", "CAL-01", "Dec-01", "NOT-02"], "name": list("abcd")}).to_csv(tax, index=False)
    classes = labels.load_classes(tax)
    assert classes == ["NONE", "ALG-06", "CAL-01", "Dec-01", "NOT-02"]
    assert labels.encode(["Dec-01", "NONE"], classes).tolist() == [3, 0]
    with pytest.raises(ValueError, match="Unknown label"):
        labels.encode(["DEC-01"], classes)
    assert labels.seen_mask(["NONE", "CAL-01", "Dec-01"], classes).tolist() == [True, False, True, True, False]
