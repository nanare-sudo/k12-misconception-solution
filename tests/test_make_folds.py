"""tools/make_folds.py: artifacts/folds.npy from configs/folds.parquet (repository file, no competition data needed)."""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import numpy as np

COMP = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("make_folds_t", COMP / "tools" / "make_folds.py")
MF = importlib.util.module_from_spec(spec)
spec.loader.exec_module(MF)


def test_parquet_split_gives_the_recorded_file():
    folds = MF.folds_from_parquet()
    assert folds.dtype == np.int64 and len(folds) == 1500 and sorted(set(folds.tolist())) == [0, 1, 2, 3, 4]
    assert hashlib.sha256(MF.npy_bytes(folds)).hexdigest() == MF.FILE_SHA256


def test_existing_file_is_checked_not_overwritten(tmp_path, monkeypatch):
    out = tmp_path / "artifacts" / "folds.npy"
    monkeypatch.setattr(MF, "OUT", out)
    monkeypatch.setattr(MF, "ROOT", tmp_path)
    assert MF.main(["--check"]) == 3 and not out.exists()
    assert MF.main([]) == 0 and hashlib.sha256(out.read_bytes()).hexdigest() == MF.FILE_SHA256
    assert MF.main(["--check"]) == 0
    out.write_bytes(b"other")
    assert MF.main([]) == 4 and out.read_bytes() == b"other"


def test_recompute_from_train_labels():
    """Only with the competition data: the stored split is kaggle_core.folds.make_folds(5, seed 42, stratified by root_code)."""
    import pytest

    try:
        same = MF.recompute_equals_parquet()
    except FileNotFoundError:
        pytest.skip("competition data missing")
    assert same
