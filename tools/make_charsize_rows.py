"""Row lists by the script size of the photo: table positions of the rows whose median character height at long side 1,024 is at most
`--max-height` pixels (statistic cc_h_p50 of the image analysis eda/i5_01_image_stats.py — label-free, a pure function of the photo).

    python tools/make_charsize_rows.py --max-height 20 --name h20

Writes data/derived/charsize/{train,test}_<name>_rows.npy (ascending positions) and charsize_<name>.meta.json (threshold, counts,
checksums). Rows without the statistic (photo without detectable script) do NOT belong to the list. Of the test photos only the stored
statistics are read, no image; only counts are printed. Use: `rows.only` of the jobs that read the photo at long side 2,048 (the threshold
20 px is the two-thirds quantile of the TRAINING photos and is fixed). The lists themselves are private data (package), not part of the
repository.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

COMP = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(COMP), str(COMP / "src")]
from vlm import common as C  # noqa: E402

STATS = COMP / "artifacts" / "eda" / "private" / "i5_image_stats.parquet"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--max-height", type=float, required=True)
    ap.add_argument("--name", required=True)
    a = ap.parse_args()
    root = C.find_data_root()
    out = root / "derived" / "charsize"
    out.mkdir(parents=True, exist_ok=True)
    st = pd.read_parquet(STATS, columns=["folder", "file", "cc_h_p50"])
    # Recorded key and description of the frozen list meta file (data/derived/charsize/charsize_h20.meta.json, pinned by its checksum in
    # tools/builds/primary_m3x5h.sh and part of the digest TRAIN_H_SHA there): kept verbatim, so that a rerun reproduces the pinned file
    # byte for byte. Meaning: "statistic": "cc_h_p50 (median character height in pixels at long side 1,024)".
    meta = {"kennzahl": "cc_h_p50 (mittlere Zeichenhöhe in Pixeln bei langer Seite 1.024)", "max_height": a.max_height, "name": a.name,
            "stats_sha256": hashlib.sha256(STATS.read_bytes()).hexdigest(), "tables": {}}
    for split in ("train", "test"):
        t = C.load_table(root, split)
        v = pd.to_numeric(st[st["folder"] == split].set_index("file").loc[t["image_file"].to_numpy(), "cc_h_p50"], errors="coerce").to_numpy(np.float64)
        rows = np.flatnonzero(np.isfinite(v) & (v <= a.max_height)).astype(np.int64)
        f = out / f"{split}_{a.name}_rows.npy"
        np.save(f, rows)
        meta["tables"][split] = {"rows_total": int(len(t)), "rows_in_list": int(len(rows)), "rows_without_value": int((~np.isfinite(v)).sum()),
                                 "file": f"derived/charsize/{f.name}", "sha256": hashlib.sha256(f.read_bytes()).hexdigest()}
        print(f"{split}: {len(rows)} of {len(t)} rows with character height ≤ {a.max_height:g} px ({int((~np.isfinite(v)).sum())} without the statistic) -> {f.name}")
    (out / f"charsize_{a.name}.meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
