#!/usr/bin/env bash
# Primary line, reranker MAIN form (second stage under the widened rule of the note guard, deployment file stage2_s034.json) with
# the RELEASED rotation list for v1r: derived/orient_repro/test_c80_rows.npy (344 rows, stored orientation network) instead of the
# first-version list (350 rows). No new GPU computation (all rows in the existing rot_test run, same rotation). Otherwise exactly
# the build of the main form (tools/builds/s034_stage2_main.sh best with ORIENT=repro, which builds the released-list fallback form
# and intermediate file): blend, β, second stage, guards, positive controls and file swap unchanged, referring to the released-list
# files. Needs the legacy-list files first (ORIENT=legacy bash tools/builds/s034_stage2_main.sh best).
#
#     bash tools/builds/s034n_stage2_main.sh   # -> artifacts/submissions/sub_s034n_stage2_main.csv (plus the two files above)
#
# Only aggregates and checksums are printed. Nothing is submitted from here.
set -euo pipefail
[ $# -eq 0 ] || { echo "usage: $0   (no arguments)" >&2; exit 2; }
cd "$(dirname "$0")/../.."
ORIENT=repro exec bash tools/builds/s034_stage2_main.sh best
