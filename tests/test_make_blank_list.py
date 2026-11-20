"""tools/make_blank_list.py: the blank-photo rule on synthetic photos; with the competition photos, the frozen test list.

The test with the competition photos is skipped if the photos or the frozen list are missing. It reads the photos by
program only and prints nothing about single rows.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

COMP = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("make_blank_list_t", COMP / "tools" / "make_blank_list.py")
BL = importlib.util.module_from_spec(spec)
spec.loader.exec_module(BL)

DATA = COMP / "data"
FROZEN_TEST = DATA / "derived" / "blank" / "test_rows.npy"


def _photo(path: Path, array: np.ndarray) -> Path:
    Image.fromarray(array.astype(np.uint8)).save(path, quality=95)
    return path


def test_rule_on_synthetic_photos(tmp_path):
    rng = np.random.default_rng(0)
    black = _photo(tmp_path / "black.jpeg", np.zeros((700, 500, 3)))
    dark_flat = _photo(tmp_path / "dark_flat.png", np.full((300, 400), 10))
    grey_flat = _photo(tmp_path / "grey_flat.png", np.full((300, 400), 14))
    dark_text = np.full((400, 400), 5)
    dark_text[::8, :] = 60  # dark photo with lines: mean below 12, but the spread is far above 8
    dark_text = _photo(tmp_path / "dark_text.png", dark_text)
    paper = _photo(tmp_path / "paper.jpeg", rng.integers(150, 256, size=(640, 480, 3)))
    assert BL.is_blank(*BL.grey_stats(black)) and BL.is_blank(*BL.grey_stats(dark_flat))
    assert not BL.is_blank(*BL.grey_stats(grey_flat))
    assert not BL.is_blank(*BL.grey_stats(dark_text))
    assert not BL.is_blank(*BL.grey_stats(paper))
    rows = BL.blank_rows([paper, black, grey_flat, dark_flat, dark_text])
    assert rows.dtype == np.int64 and rows.tolist() == [1, 3]


def test_list_bytes_are_those_of_np_save(tmp_path):
    rows = np.array([4, 17, 2001], dtype=np.int64)
    np.save(tmp_path / "x.npy", rows)
    assert BL.npy_bytes(rows) == (tmp_path / "x.npy").read_bytes()


def test_write_never_targets_the_frozen_folder(tmp_path):
    frozen = tmp_path / "derived" / "blank"
    frozen.mkdir(parents=True)
    assert BL.write(tmp_path, frozen, frozen) == 2
    assert list(frozen.iterdir()) == []


@pytest.mark.skipif(not (FROZEN_TEST.is_file() and (DATA / "test.csv").is_file() and (DATA / "competition_data" / "images" / "test").is_dir()),
                    reason="competition photos or frozen blank list missing")
def test_frozen_test_list_follows_the_rule():
    """Every row of the frozen test list is blank by the rule, and a fixed sample of the other rows is not."""
    from vlm import common as C

    df = C.load_table(DATA, "test")
    frozen = np.load(FROZEN_TEST)
    others = np.setdiff1d(np.arange(len(df)), frozen)
    sample = np.random.default_rng(0).choice(others, size=40, replace=False)
    path = lambda i: C.image_path(DATA, "test", df["image_file"].iloc[int(i)])  # noqa: E731
    assert all(BL.is_blank(*BL.grey_stats(path(i))) for i in frozen)
    assert not any(BL.is_blank(*BL.grey_stats(path(i))) for i in sample)


def test_check_reports_differences_and_missing_lists(tmp_path, monkeypatch, capsys):
    """check compares per table; a missing frozen list gives exit 3, a differing one exit 1, equal lists exit 0."""
    lists = {"train": np.array([3], dtype=np.int64), "test": np.array([1, 5], dtype=np.int64)}
    monkeypatch.setattr(BL, "table_rows", lambda root, table: (lists[table], 10))
    frozen = tmp_path / "blank"
    frozen.mkdir()
    (frozen / "test_rows.npy").write_bytes(BL.npy_bytes(lists["test"]))
    assert BL.check(tmp_path, frozen) == 3
    (frozen / "train_rows.npy").write_bytes(BL.npy_bytes(np.array([4], dtype=np.int64)))
    assert BL.check(tmp_path, frozen) == 1
    (frozen / "train_rows.npy").write_bytes(BL.npy_bytes(lists["train"]))
    assert BL.check(tmp_path, frozen) == 0
    out = capsys.readouterr().out
    assert "2 rows differ" in out and "check passed" in out
