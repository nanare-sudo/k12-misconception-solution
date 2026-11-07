#!/usr/bin/env bash
# Primary line, reranker FALLBACK form (second stage with the deployment file stage2_s032.json) with the RELEASED rotation list for
# v1r: derived/orient_repro/test_c80_rows.npy (344 rows, stored orientation network) instead of the first-version list (350 rows).
# No new GPU computation (all rows in the existing rot_test run, same rotation). Otherwise exactly the build of the fallback form
# (tools/builds/s032_stage2.sh best with ORIENT=repro, which builds the released-list intermediate file through
# tools/builds/s029_q397m3.sh): blend, β, second stage, all other test channels and guards unchanged. Needs the legacy-list files
# first (ORIENT=legacy bash tools/builds/s034_stage2_main.sh best).
#
#     bash tools/builds/s033n_stage2.sh     # -> artifacts/submissions/sub_s033n_stage2.csv (intermediate file: sub_s032n_q397m3.csv)
#
# Only aggregates and checksums are printed. Nothing is submitted from here.
set -euo pipefail
[ $# -eq 0 ] || { echo "usage: $0   (no arguments)" >&2; exit 2; }
cd "$(dirname "$0")/../.."
ORIENT=repro exec bash tools/builds/s032_stage2.sh best
