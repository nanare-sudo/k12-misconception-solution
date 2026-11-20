#!/usr/bin/env python3
"""Find the blank photos (black uploads) of the training and test tables.

    python -I tools/make_blank_list.py check                                  # compare with data/derived/blank/
    python -I tools/make_blank_list.py write --out data/derived/blank_repro   # write the lists into a new folder

THE RULE. A photo counts as blank if, converted to grey scale (values 0 to 255) and resized to 256 x 256 pixels with
bilinear filtering, its mean is below 12 and its standard deviation is below 8. The rule looks at one photo at a time.
It uses no label and no other row.

USE. Both final submissions put the code REL-01 at rank 1 for the test rows of the frozen list
data/derived/blank/test_rows.npy (tools/make_submission.py --blank-rows). A black photo shows no working, and the only
blank training photo carries REL-01, the code for an upload that does not match the task.

check  computes both lists from the photos and compares them with the frozen lists in data/derived/blank/ (or the
       folder given by --frozen). Exit 0 if both are equal, 1 if a list differs, 3 if a frozen list is missing.
write  writes train_rows.npy, test_rows.npy and blank.meta.json into the folder given by --out. It never writes into
       the folder of the frozen lists.

Each list holds table positions (int64, ascending) in the row order of train.csv and test.csv. The tool reads the photos
by program only and prints, per table, the number of blank photos, the SHA-256 of the list and the number of rows that
differ from the frozen list. It prints no row numbers. The photos are expected under
data/competition_data/images/{train,test}/ (as found by vlm.common.find_data_root). About one minute on one CPU core.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

COMP = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(COMP), str(COMP / "src")]

MEAN_BELOW = 12.0
STD_BELOW = 8.0
SIZE = (256, 256)
TABLES = ("train", "test")
RULE = (f"blank if the photo, converted to grey scale (0-255) and resized to {SIZE[0]} x {SIZE[1]} pixels (bilinear), "
        f"has a mean below {MEAN_BELOW:g} and a standard deviation below {STD_BELOW:g}")


def grey_stats(path: Path) -> tuple[float, float]:
    """Mean and standard deviation of the grey-scale photo at 256 x 256 pixels."""
    with Image.open(path) as im:
        g = np.asarray(im.convert("L").resize(SIZE, Image.Resampling.BILINEAR), dtype=np.float32)
    return float(g.mean()), float(g.std())


def is_blank(mean: float, std: float) -> bool:
    return mean < MEAN_BELOW and std < STD_BELOW


def blank_rows(paths: list[Path]) -> np.ndarray:
    """Table positions of the blank photos among `paths` (in table order)."""
    return np.array([k for k, p in enumerate(paths) if is_blank(*grey_stats(p))], dtype=np.int64)


def npy_bytes(rows: np.ndarray) -> bytes:
    """The bytes that np.save writes for this list."""
    buf = io.BytesIO()
    np.save(buf, np.asarray(rows, dtype=np.int64))
    return buf.getvalue()


def table_rows(data_root: Path, table: str) -> tuple[np.ndarray, int]:
    from vlm import common as C

    df = C.load_table(data_root, table)
    paths = [C.image_path(data_root, table, f) for f in df["image_file"]]
    missing = sum(not p.is_file() for p in paths)
    if missing:
        raise FileNotFoundError(f"{table}: {missing} of {len(paths)} photos missing under {C.image_path(data_root, table, '')}")
    return blank_rows(paths), len(df)


def check(data_root: Path, frozen: Path) -> int:
    worst = 0
    for table in TABLES:
        rows, n = table_rows(data_root, table)
        sha = hashlib.sha256(npy_bytes(rows)).hexdigest()
        f = frozen / f"{table}_rows.npy"
        if not f.is_file():
            print(f"{table}: {len(rows)} blank photos of {n}, SHA-256 {sha[:16]}…; frozen list missing ({f.name})")
            worst = max(worst, 3)
            continue
        old = np.load(f)
        differ = len(set(old.tolist()) ^ set(rows.tolist()))
        same = f.read_bytes() == npy_bytes(rows)
        print(f"{table}: {len(rows)} blank photos of {n}, SHA-256 {sha[:16]}…; frozen list {len(old)} rows, "
              f"{differ} rows differ, {'byte-identical' if same else 'NOT byte-identical'}")
        if differ or not same:
            worst = max(worst, 1)
    print("check passed" if worst == 0 else "check FAILED")
    return worst


def write(data_root: Path, out: Path, frozen: Path) -> int:
    if out.resolve() == frozen.resolve():
        print(f"ERROR: {out} is the folder of the frozen lists; choose another folder with --out", file=sys.stderr)
        return 2
    from vlm import common as C

    out.mkdir(parents=True, exist_ok=True)
    meta: dict = {"rule": RULE, "mean_below": MEAN_BELOW, "std_below": STD_BELOW, "size": list(SIZE), "tables": {}}
    for table in TABLES:
        rows, n = table_rows(data_root, table)
        f = out / f"{table}_rows.npy"
        f.write_bytes(npy_bytes(rows))
        info = {"rows": n, "blank": int(len(rows)), "sha256": hashlib.sha256(f.read_bytes()).hexdigest()}
        if table == "train":  # codes of the blank training rows (labels of training rows only)
            info["codes"] = C.load_table(data_root, "train")["root_code"].iloc[rows].value_counts().to_dict()
        meta["tables"][table] = info
        print(f"{table}: {len(rows)} blank photos of {n} -> {f} (SHA-256 {info['sha256'][:16]}…)")
    (out / "blank.meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("mode", choices=("check", "write"))
    ap.add_argument("--out", help="write: output folder, for example data/derived/blank_repro")
    ap.add_argument("--frozen", help="folder of the frozen lists (default: derived/blank/ in the data folder)")
    ap.add_argument("--data-root", help="data folder (default: as vlm.common.find_data_root, normally data/)")
    a = ap.parse_args(argv)
    from vlm import common as C

    data_root = Path(a.data_root) if a.data_root else C.find_data_root()
    frozen = Path(a.frozen) if a.frozen else data_root / "derived" / "blank"
    try:
        if a.mode == "check":
            return check(data_root, frozen)
        if not a.out:
            ap.error("write needs --out")
        return write(data_root, Path(a.out), frozen)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    sys.exit(main())
