#!/usr/bin/env python3
"""Write rotation lists for straightening rotated photos: which photos are rotated how often before reading.

    python tools/make_orient.py [--source cnn] [--min-conf 0.5] [--name c50]
    python tools/make_orient.py --source vlm --run orient_t4 [--min-conf 0.8] [--name v80]

Two sources of the orientation estimate:
  cnn  the image analysis (eda/i5_06_orientation.py cnn -> artifacts/eda/private/i5_rot_cnn_pred.parquet): a
       self-supervised network on 64x64 ink maps, trained only on train and pool images. Class r means "the upright
       sheet appears rotated by r quarter turns counter-clockwise"; it is straightened with (4 - r) quarter turns
       counter-clockwise (checked on two training images).
  vlm  the orientation probe `up` of a vision-language model without training (configs/vlm/orient_t4.yaml): jobs
       up0 … up3 measure for every photo in rotation k (k quarter turns counter-clockwise) the margin yes - no for the
       question "upright?". The rotation with the largest margin straightens the photo; the confidence is the softmax
       share of this rotation over the four margins.

Written per table (train, test) to data/derived/orient/ (gitignored; other folder with --out):
    <table>_<name>.json       {trace_id: quarter turns}   (input for `image.rotate_from`)
    <table>_<name>_rows.npy   table positions of these rows (input for `rows.only`)
    <table>_<name>.meta.json  count per rotation, threshold, source, SHA-256
Included are photos whose most probable orientation is not "upright" and reaches at least --min-conf. Only counts are
printed (for the training rows also the comparison of both sources); the lists for test rows are row-level derivatives
and are not inspected.

Writing the lists lives in src/orient.py (write_lists), so that the reproducible orientation network
(tools/orient_net.py) produces the same format. The source `cnn` of the first version (i5_rot_cnn_pred.parquet) is not
repeatable because its weights are missing; the reproducible version writes its predictions to
data/derived/orient_repro/cnn_pred.parquet (usable as --cnn).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

COMP = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(COMP / "src")]

import orient as O  # noqa: E402
from vlm import common as C  # noqa: E402


def from_cnn(source: str, table: str, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """-> (quarter turns counter-clockwise to straighten, confidence) per table row."""
    return O.from_pred_frame(pd.read_parquet(COMP / source), table, df)


def from_vlm(run: str, table: str, n: int) -> tuple[np.ndarray, np.ndarray]:
    tag = "bench" if table == "train" else "test"
    M = np.full((n, 4), np.nan)
    for k in range(4):
        d = COMP / "artifacts" / "runs" / run / run / f"up{k}"
        idx, lp = np.load(d / f"{tag}_idx.npy"), np.load(d / f"{tag}_up.npy")
        done = np.load(d / f"{tag}_up.done.npy") if (d / f"{tag}_up.done.npy").is_file() else np.ones(len(idx), bool)
        M[idx[done], k] = (lp[:, 0] - lp[:, 1])[done]
    bad = ~np.isfinite(M).all(1)
    if bad.any():
        print(f"  {table}: {int(bad.sum())} rows without a complete measurement stay unchanged")
    Mz = np.where(np.isfinite(M), M, -np.inf)
    Mz[bad] = np.array([0.0, -np.inf, -np.inf, -np.inf])
    e = np.exp(Mz - Mz.max(1, keepdims=True))
    p = e / e.sum(1, keepdims=True)
    return p.argmax(1).astype(int), p.max(1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--source", choices=["cnn", "vlm"], default="cnn")
    ap.add_argument("--run", default="orient_t4", help="run with the jobs up0 … up3 (source vlm)")
    ap.add_argument("--cnn", default="artifacts/eda/private/i5_rot_cnn_pred.parquet")
    ap.add_argument("--min-conf", type=float, default=0.5)
    ap.add_argument("--name", default="c50")
    ap.add_argument("--out", default="data/derived/orient", help="target folder, relative to the repository root")
    ap.add_argument("--force", action="store_true", help="overwrite existing lists of this name")
    a = ap.parse_args()

    root = C.find_data_root()
    out = COMP / a.out
    # Guard: the frozen lists in data/derived/orient are used by all submitted files; never overwrite them silently.
    existing = [p.name for t in ("train", "test") for p in (out / f"{t}_{a.name}.json", out / f"{t}_{a.name}_rows.npy") if p.exists()]
    if existing and not a.force:
        raise SystemExit(f"{a.out}: {', '.join(existing)} already exists; use another --out/--name or --force")
    out.mkdir(parents=True, exist_ok=True)
    for table in ("train", "test"):
        df = C.load_table(root, table)
        turns, conf = from_cnn(a.cnn, table, df) if a.source == "cnn" else from_vlm(a.run, table, len(df))
        source = a.cnn if a.source == "cnn" else f"{a.run}/up0..up3 (probe up)"
        meta = O.write_lists(out, table, df, turns, conf, a.min_conf, a.name, source)
        counts = meta["turns_ccw"]
        print(f"{table}: {meta['rows']} of {meta['of']} photos ({meta['rows'] / meta['of']:.1%}) are rotated; quarter turns counter-clockwise {counts}; "
              f"SHA-256 {meta['sha256'][:16]}…")
        if a.source == "vlm":  # comparison with the image analysis: shares only, for both tables
            t_c, c_c = from_cnn(a.cnn, table, df)
            print(f"   orientation not upright (all confidences): probe {float((turns != 0).mean()):.3f}, image analysis {float((t_c != 0).mean()):.3f}")
            for thr in (0.5, 0.8, 0.9):
                both = (conf >= thr) & (c_c >= thr)
                print(f"   both confidences from {thr:g}: {int(both.sum())} photos, same orientation {float((turns == t_c)[both].mean()) if both.any() else float('nan'):.3f} | "
                      f"image analysis confidently rotated -> probe same rotation {float((turns == t_c)[(t_c != 0) & (c_c >= thr)].mean()) if ((t_c != 0) & (c_c >= thr)).any() else float('nan'):.3f} | "
                      f"probe confidently rotated {int(((turns != 0) & (conf >= thr)).sum())}, of which image analysis upright with confidence from {thr:g}: "
                      f"{int(((turns != 0) & (conf >= thr) & (t_c == 0) & (c_c >= thr)).sum())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
