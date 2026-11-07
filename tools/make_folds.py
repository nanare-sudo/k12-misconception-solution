"""Write artifacts/folds.npy from the fixed fold split configs/folds.parquet, or check an existing file.

The fold split is part of the repository (configs/folds.parquet: row position of train.csv -> fold, 5 folds stratified by
root_code, seed 42). Every model run reads it through vlm.common.load_folds, which checks its hash. Two CPU steps of the final
lines (eda/c118_stage2.py and eda/c120_s036_safe_stage2.py) additionally record the SHA-256 of artifacts/folds.npy among their
input checksums; this script writes that file in the same row order and dtype (int64, numpy .npy format).

    python -I tools/make_folds.py              # write artifacts/folds.npy (an existing different file is not overwritten)
    python -I tools/make_folds.py --check      # only compare an existing artifacts/folds.npy with the parquet split
    python -I tools/make_folds.py --recompute  # recompute the split from train.csv (kaggle_core.folds.make_folds) and compare

--recompute needs the competition data (train.csv, found as in vlm.common.find_data_root) and only prints whether the
recomputed split equals configs/folds.parquet; the parquet file itself is never rewritten.

Exit: 0 written or equal, 3 input missing, 4 the existing file differs.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PARQUET = ROOT / "configs" / "folds.parquet"
OUT = ROOT / "artifacts" / "folds.npy"
FOLDS_HASH = "07ec381cc58b2a80"  # first 16 hex digits of SHA-256 over the int64 fold array (as vlm.common.load_folds)
FILE_SHA256 = "fadf2771203c19d19e057d69a02413d9f01b26f85bbf76490046f2a6f8773b73"  # artifacts/folds.npy as recorded by the final lines


def folds_from_parquet(path: Path = PARQUET) -> np.ndarray:
    df = pd.read_parquet(path)
    folds = df.sort_values("row")["fold"].to_numpy().astype(np.int64)
    digest = hashlib.sha256(np.ascontiguousarray(folds).tobytes()).hexdigest()[:16]
    if len(folds) != 1500 or digest != FOLDS_HASH:
        raise SystemExit(f"{path}: {len(folds)} rows, hash {digest} (expected 1500 rows, {FOLDS_HASH})")
    return folds


def npy_bytes(folds: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, folds)
    return buf.getvalue()


def recompute_equals_parquet(data_root: Path | None = None) -> bool:
    """The split as created: kaggle_core.folds.make_folds over root_code of train.csv (5 folds, seed 42, stratified)."""
    sys.path[:0] = [str(ROOT), str(ROOT / "src")]
    import warnings

    from kaggle_core.folds import make_folds
    from vlm import common as C

    root = C.find_data_root(data_root)
    y = pd.read_csv(Path(root) / "train.csv", dtype=str, keep_default_na=False, usecols=["root_code"])["root_code"].to_numpy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # classes with a single row trigger an sklearn warning
        folds = make_folds(y, n_splits=5, seed=42, strategy="stratified")
    return bool(np.array_equal(np.asarray(folds, dtype=np.int64), folds_from_parquet()))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true", help="only compare an existing artifacts/folds.npy")
    ap.add_argument("--recompute", action="store_true", help="recompute the split from train.csv and compare with configs/folds.parquet")
    a = ap.parse_args(argv)
    if not PARQUET.is_file():
        print(f"ABORT: {PARQUET} missing", file=sys.stderr)
        return 3
    if a.recompute:
        try:
            same = recompute_equals_parquet()
        except FileNotFoundError as e:
            print(f"ABORT: {e}", file=sys.stderr)
            return 3
        print(f"make_folds(5 folds, seed 42, stratified by root_code) {'equals' if same else 'DIFFERS from'} configs/folds.parquet")
        return 0 if same else 4
    data = npy_bytes(folds_from_parquet())
    sha = hashlib.sha256(data).hexdigest()
    if sha != FILE_SHA256:
        print(f"ABORT: the fold array gives SHA-256 {sha[:16]}…, expected {FILE_SHA256[:16]}…", file=sys.stderr)
        return 4
    if OUT.is_file():
        same = OUT.read_bytes() == data
        print(f"{OUT.relative_to(ROOT)}: {'equal to' if same else 'DIFFERS from'} configs/folds.parquet (SHA-256 {sha[:16]}…)")
        return 0 if same else 4
    if a.check:
        print(f"ABORT: {OUT.relative_to(ROOT)} missing", file=sys.stderr)
        return 3
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_bytes(data)
    print(f"written: {OUT.relative_to(ROOT)} (SHA-256 {sha})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
