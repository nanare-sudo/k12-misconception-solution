#!/usr/bin/env python3
"""Reproducible orientation network: maps from images, training with fixed seeds, inference from stored weights.

    python -I tools/orient_net.py maps [--tables train,test,unlabelled] [--workers 4]            # images -> maps
    python -I tools/orient_net.py check-maps                                                      # against the statistics cache
    python -I tools/orient_net.py train [--seed 0] [--threads 2] [--weights artifacts/orient/orient_cnn.pt]
    python -I tools/orient_net.py infer [--from-images] [--tables train,test] [--out data/derived/orient_repro]
    python -I tools/orient_net.py compare [--v1r-run v1rot_t4] [--probe-run orient_t4] [--json …]

Rerun path (inference only, offline): `infer --from-images --tables test --weights <package>/orient_cnn.pt --out <folder>`
writes test_c50.json / test_c50_rows.npy / test_c80_rows.npy like tools/make_orient.py. Details: src/orient.py.
For test rows only counts and checksums are printed.
"""

from __future__ import annotations

import sys
from pathlib import Path

COMP = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(COMP), str(COMP / "src")]

import orient  # noqa: E402

if __name__ == "__main__":
    sys.exit(orient.main())
