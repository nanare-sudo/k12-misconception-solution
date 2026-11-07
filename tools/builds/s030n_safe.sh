#!/usr/bin/env bash
# Inductive line with the RELEASED rotation list for v1r: derived/orient_repro/test_c80_rows.npy (344 rows; stored orientation
# network tools/orient_net.py, artifacts/orient/orient_cnn.pt) instead of the first-version list (derived/orient/test_c80_rows.npy,
# 350 rows, network not stored). All 344 rows lie in the existing T4 rotation run rot_test with the same rotation: no new GPU
# computation, only the @ filter of the v1r source changes. Otherwise exactly the build of the inductive line
# (tools/builds/s029_m3x3.sh safe with ORIENT=repro): blend, θ, β, inductive class weights, all other test channels and guards
# unchanged. Needs the legacy-list file of the line first (ORIENT=legacy bash tools/builds/s029_m3x3.sh safe).
#
#     bash tools/builds/s030n_safe.sh       # -> artifacts/submissions/sub_s030n_safe.csv (the legacy-list file stays untouched)
#
# Only aggregates and checksums are printed. Nothing is submitted from here.
set -euo pipefail
[ $# -eq 0 ] || { echo "usage: $0   (no arguments)" >&2; exit 2; }
cd "$(dirname "$0")/../.."
ORIENT=repro exec bash tools/builds/s029_m3x3.sh safe
