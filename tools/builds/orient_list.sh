# shellcheck shell=bash
# Choice of the rotation list for the test side of channel v1r. Sourced by the build scripts of the final lines after the cd into
# the repository root (tools/builds/s029_m3x3.sh, s029_q397m3.sh, s032_stage2.sh, s034_stage2_main.sh, s035_stage2_guard.sh,
# s036_safe_stage2.sh):
#
#     . tools/builds/orient_list.sh
#     orient_select        # sets ORIENT (exported, applies to nested build scripts), ORIENT_ROWS, ORIENT_ROWS_SHA, V1R_TEST
#
#   ORIENT=repro (default)  derived/orient_repro/test_c80_rows.npy, 344 rows: list of the stored orientation network
#                 (tools/orient_net.py, artifacts/orient/orient_cnn.pt). The existing T4 rotation run rot_test (620 rows) computed
#                 all 344 rows with the same rotation — no new GPU computation, only the @ filter of the v1r source changes. The
#                 build scripts then write their own files (sub_s030n_safe.csv, sub_s032n_q397m3.csv, sub_s033n_stage2.csv,
#                 sub_s034n_stage2_main.csv) with their own suffixes of the test matrices; the legacy-list files stay untouched.
#   ORIENT=legacy  derived/orient/test_c80_rows.npy, 350 rows: list of the first version of the orientation network, from which the
#                 earlier submissions with v1r were built. Its weights were not stored, so the list cannot be recomputed. With
#                 legacy the build scripts rebuild those historical files byte-identically (the same make_submission arguments as
#                 before); the repro builds need them as references.
#
# Checks here (sha256sum only, no output): checksum of the chosen list; in repro first the rejection of the first-version list
# (path of the v1r source and checksum of the file), then the checksums of the released row and JSON lists. In repro the build
# scripts additionally call tools/orient_list.py check (coverage by rot_test, same rotation, provenance from orient_cnn.json)
# before make_submission and tools/orient_list.py verify-sub (source of v1r in the written sub_*.json) afterwards. The same
# constants are in tools/orient_list.py (tests/test_orient_list.py keeps both equal).
ORIENT_V1R_RUN="v1rot_t4/rot_test/test"
ORIENT_LEGACY_ROWS="derived/orient/test_c80_rows.npy"
ORIENT_LEGACY_ROWS_SHA="f0d10dd2179cc36d08fc6d2eaee734f3a9f3a5cc894274dceea5c198ce8bd91f"
ORIENT_REPRO_ROWS="derived/orient_repro/test_c80_rows.npy"
ORIENT_REPRO_ROWS_SHA="aa6515c8bf17bf593c5c90aa90d9240e9992d84f650c5445a37a5f015dbbff1d"
ORIENT_REPRO_JSON="derived/orient_repro/test_c80.json"
ORIENT_REPRO_JSON_SHA="2937d66d8c42c763ec5ba0b1c848585306ca72f8a63f0992a334d7117e4ae462"

orient_choose() {  # only the choice, no file access
  ORIENT="${ORIENT:-repro}"
  case "$ORIENT" in
    legacy) ORIENT_ROWS="$ORIENT_LEGACY_ROWS"; ORIENT_ROWS_SHA="$ORIENT_LEGACY_ROWS_SHA" ;;
    repro) ORIENT_ROWS="$ORIENT_REPRO_ROWS"; ORIENT_ROWS_SHA="$ORIENT_REPRO_ROWS_SHA" ;;
    *) echo "ABORT: ORIENT=$ORIENT — allowed are repro (default, released rotation list) and legacy (first-version rotation list)" >&2; exit 2 ;;
  esac
  export ORIENT
  V1R_TEST="v1r=${ORIENT_V1R_RUN}@${ORIENT_ROWS}"
}

orient_reject_legacy() {  # check of the repro variant: a v1r source on the first-version list is rejected (path or checksum)
  local spec="$1" rows sha
  case "$spec" in
    *@*) rows="${spec#*@}" ;;
    *) echo "ABORT: v1r source without a rotation list ($spec)" >&2; exit 4 ;;
  esac
  case "$rows" in
    "$ORIENT_LEGACY_ROWS" | derived/orient/*)
      echo "ABORT: ORIENT=repro, but v1r reads the first-version rotation list ($rows) — rejected in the repro variant" >&2; exit 4 ;;
  esac
  [ -f "data/$rows" ] || { echo "ABORT: rotation list data/$rows missing" >&2; exit 3; }
  sha="$(sha256sum "data/$rows" | cut -d' ' -f1)"
  if [ "$sha" = "$ORIENT_LEGACY_ROWS_SHA" ]; then
    echo "ABORT: ORIENT=repro, but data/$rows has the checksum of the first-version rotation list (${ORIENT_LEGACY_ROWS_SHA:0:16}…) — rejected" >&2
    exit 4
  fi
}

orient_verify() {  # checksums of the chosen list; in repro first the rejection of the first-version list, plus the JSON list
  if [ "$ORIENT" = repro ]; then
    orient_reject_legacy "$V1R_TEST"
  fi
  echo "$ORIENT_ROWS_SHA  data/$ORIENT_ROWS" | sha256sum -c --quiet \
    || { echo "ABORT: rotation list data/$ORIENT_ROWS differs (checksum; ORIENT=$ORIENT)" >&2; exit 4; }
  if [ "$ORIENT" = repro ]; then
    echo "$ORIENT_REPRO_JSON_SHA  data/$ORIENT_REPRO_JSON" | sha256sum -c --quiet \
      || { echo "ABORT: JSON list data/$ORIENT_REPRO_JSON differs (checksum)" >&2; exit 4; }
  fi
}

orient_select() {
  orient_choose
  orient_verify
}
