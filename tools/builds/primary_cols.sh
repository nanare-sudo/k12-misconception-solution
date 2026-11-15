#!/usr/bin/env bash
# INTERMEDIATE BUILD primary_cols (bundle test of the extra columns, expected-value rule; step 0 of the primary build primary_m3x5h):
# primary submission B (tools/builds/primary_m3x4.sh) with extra columns at the end of the second stage, variant qfp
# (src/stage2_sources.VARIANTS["qfp"]):
#   the six columns of the 397B group (c_vote, b_last, a_major, c_formed, d_minor per [pr27+q397], k_nov_single[q397]),
#   fac27_rule (pr27 final_answer_correct × {minor, ANS-01}) and psy_adm_sec (section of the candidate seen as gold within the topic);
#   without oth_novelty (its task count violates the test guard).
# Production code src/stage2_sources.py (frozen sources: pr27 training pr27_rp / test pr27te_rp; q397 training Tinker / test q397te_rp; blend
# channels and draws of the line; table, taxonomy), deployment file eda/c151_primary_cols.py fit (chain and deployment form of the base,
# feature set main_guard_i1_mm + columns), test side tools/make_submission.py --stage2 (field extra_columns, additive). Blend UNCHANGED (stack
# of the base). Order: blend -> margin alignment (one set) -> second stage (+ guard) -> REL-01. Rotation list ORIENT=repro.
#
#     bash tools/builds/primary_cols.sh                       # primary_cols -> artifacts/submissions/sub_primary_cols.csv
#     PRIMARY_COLS_REGRESSION=0 …                             # step 0 as checksums only (debugging; a deployment build requires step 0)
#
# STEPS: 0 regression check: build path of the base complete (primary_m3x4.sh) byte-identical with the additively extended make_submission;
#   1 inputs (checksum over all source files of the protocol columns, checksums of the base); 2 test sources of the columns (c151 test-sources:
#   completeness, parser cross-check, notes = frozen test notes); 3 second stage (c151 fit: regression of the main model of the base in the
#   process, MM + columns nested, null point (e)) and field comparison with the deployment file of the base; 4 copy null point of the
#   training side (c151 copy: columns := 0 -> rows and deployment file = base); 5 test guard: null model and file without second stage
#   (byte-identical), test guard per column and source (c151 test-guard, exit 5 if violated), positive controls (c151 controls, exit 6
#   without effect); 6 copy null point of the test side (copy deployment file through the new path -> CSV and test matrix byte-identical to
#   the base); 7 main file (guard (c), form main_guard_i1_mm + columns required); 8 check and comparison with the base (aggregates only);
#   9 checksums (the first build prints them -> pin -> second build byte-identical).
# MODE OF THE CHECK STEPS: fit runs with --save = writes what is missing, requires the same content for existing files (never overwrites a
#   differing file). Required are the recomputations in the process and the pinned checksums.
# EXIT: 2 usage; 3 input missing; 4 checksum/recomputation/null point; 5 test guard, form or guard (c) violated; 6 positive control without
#   effect.
#
# Only aggregates and checksums are printed. Nothing is submitted from here.
set -euo pipefail
[ $# -eq 0 ] || { echo "usage: $0   (no arguments; PRIMARY_COLS_REGRESSION=base|0)" >&2; exit 2; }
cd "$(dirname "$0")/../.."
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}" OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export ORIENT=repro
BASE_MODE="m3x4"  # the only base (eda/c151_primary_cols.py --base)
COLS="qfp"  # the only variant built here (eda/c151_primary_cols.py --cols)
RANK1="runs/m3_fullte_rp/m3_test_rank/test"
RANK2="runs/m3d2te_rp/m3d2_test_rank/test"
RANK3="runs/m3d3te_rp/m3d3_test_rank/test"
RANK4="runs/m3d4te_rp/m3d4_test_rank/test"
MID_TEST_RANK="runs/m3px27te_rp/m3px27_test_rank/test"
Q397M3_TEST="runs/q397m3te_rp/q397m3_test_rank/test"
LABEL="primary_cols (primary_m3x4 + columns $COLS)"; BASE_LABEL="primary_m3x4"
BASE_BUILD="tools/builds/primary_m3x4.sh"
STEM_BASE="primary_m3x4"; STEM="primary_cols"; TAG="cols"
BASE_SUB="artifacts/submissions/sub_primary_m3x4.csv"; BASE_SUB_SHA="76da786dfdbe6a6ddcbcc93070f385e18870cb5cdaea895d92104aa3565f9973"
BASE_MAT="artifacts/test_primary_m3x4_m3x4.npy"; BASE_MAT_SHA="30f74c8f72746f34cc423df7218bed60e7cf9b121b504a27e34606dbb3c97504"
# blend and deployment file of the base are written by eda/c135_primary_m3x4.py of this repository: empty = first build (printed, then pinned)
BASE_STACK="artifacts/stack_primary_m3x4.json"; BASE_STACK_SHA=""
BASE_STAGE2="artifacts/stage2_primary_m3x4.json"; BASE_STAGE2_SHA=""
MID="m3px27xm"; MID_PROTO="runs/pr27te_rp/proto_te/proto"
STAGE2="artifacts/stage2_${STEM}.json"
OUT="artifacts/submissions/sub_${STEM}.csv"
SUFFIX="_${TAG}"; TESTMAT="artifacts/test_${STEM_BASE}${SUFFIX}.npy"
# pins: deployment file (c151 fit; written by the code of this repository: empty = first build), file and test matrix (as submitted)
STAGE2_SHA=""; OUT_SHA="53ec296976834959caea500217aa8e01904fbdc8be40a0542f63a462e4251926"; TESTMAT_SHA="84cf58aac011166aa5c242f94d1b6c19a5e7f7d6892d60157185cdfb1148ec95"
SOURCES_SHA="ed4af2f667a02600aef494982b76860b1a4a91a214383505f3ae47a1a055e8d9"  # all source files of the protocol columns (src/stage2_sources.sources_digest: path + SHA-256 per file, training and test; 18 files)
PRIV="artifacts/eda/private/${STEM}"
COPY_STAGE2="$PRIV/copy/stage2_${TAG}k.json"
GUARDS="$PRIV/guards_${STEM}.json"
NX=""  # number of extra columns (from the deployment file, step 3)
FIRST=0
check() { echo "$1  $2" | sha256sum -c --quiet || { echo "ABORT: $2 differs (checksum)" >&2; exit 4; }; }
pin() {  # deployment checksum: pinned -> check; empty (first build) -> print it, then enter it here
  if [ -n "$1" ]; then check "$1" "$2"; else FIRST=1; echo "  FIRST BUILD — checksum $(sha256sum "$2" | cut -d' ' -f1)  $2 (pin it in tools/builds/primary_cols.sh)"; fi
}
c151() { python -I eda/c151_primary_cols.py "$@"; }
. tools/builds/orient_list.sh  # rotation list of the v1r test side: always repro here
orient_select
[ "$ORIENT" = repro ] || { echo "ABORT: $LABEL only with ORIENT=repro" >&2; exit 2; }
mkdir -p "$PRIV"
echo "== $LABEL: build path tools/builds/primary_cols.sh =="

# 0. regression check of the stored builds (make_submission is extended additively: extra columns only with the field extra_columns)
echo "== 0. regression check (PRIMARY_COLS_REGRESSION=${PRIMARY_COLS_REGRESSION:-base}: $BASE_BUILD) =="
case "${PRIMARY_COLS_REGRESSION:-base}" in
  base) bash "$BASE_BUILD" || { rc=$?; echo "ABORT: build path $BASE_LABEL (code $rc)" >&2; exit "$rc"; } ;;
  0) echo "  skipped (checksums only; no deployment build)" ;;
  *) echo "usage: PRIMARY_COLS_REGRESSION=base|0" >&2; exit 2 ;;
esac
for f in "$BASE_SUB" "$BASE_MAT" "$BASE_STACK" "$BASE_STAGE2"; do
  [ -f "$f" ] || { echo "BASE MISSING: $f — run bash $BASE_BUILD first (build path $BASE_LABEL)" >&2; exit 3; }
done
for pair in "$BASE_SUB_SHA $BASE_SUB" "$BASE_MAT_SHA $BASE_MAT"; do
  check ${pair% *} ${pair#* }
done
pin "$BASE_STACK_SHA" "$BASE_STACK"
pin "$BASE_STAGE2_SHA" "$BASE_STAGE2"
echo "  stored files unchanged: $BASE_LABEL (CSV, test matrix, blend, deployment file)"

TMP="$(mktemp -d "${TMPDIR:-/tmp}/primary_cols.XXXXXX")"
BAK=""
cleanup() {
  rm -rf -- "${TMP:?}"
  rm -f -- "artifacts/test_${STEM_BASE}_${TAG}k.npy" "artifacts/test_${STEM_BASE}_${TAG}base.npy" "artifacts/test_${STEM_BASE}_${TAG}null.npy"
  for c in q397_drop10 q397_unformed mech_zero q397_rotate fac_flip chan_rotate topic_unknown; do rm -f -- "artifacts/test_${STEM_BASE}_x3ctl_${c}.npy"; done
  [ -n "$BAK" ] && rm -rf -- "$BAK"
  return 0
}
trap cleanup EXIT

# 1. inputs
echo "== 1. inputs =="
got="$(python -I - <<'PY'
import sys
from pathlib import Path

sys.path[:0] = [str(Path.cwd()), str(Path.cwd() / "src")]
import stage2_sources as XS  # noqa: E402
from vlm import common as C  # noqa: E402

try:
    d, n = XS.sources_digest(Path.cwd(), C.find_data_root())
except XS.SourceError as e:
    print(f"MISSING {e}")
    raise SystemExit(0)
print(f"{n} {d}")
PY
)" || { echo "ABORT: checksum of the column sources not computed" >&2; exit 4; }
case "$got" in MISSING*) echo "INPUT MISSING: ${got#MISSING }" >&2; exit 3 ;; esac
n_src="${got%% *}"; got="${got#* }"
if [ -z "$SOURCES_SHA" ]; then
  FIRST=1; echo "  FIRST BUILD — checksum of the column sources ($n_src files) $got (pin it as SOURCES_SHA in tools/builds/primary_cols.sh)"
else
  [ "$got" = "$SOURCES_SHA" ] || { echo "ABORT: source files of the columns differ (checksum $got instead of $SOURCES_SHA)" >&2; exit 4; }
  echo "  source files of the columns ($n_src files: answers, cross-check files, notes, meta/summary; training and test) pinned"
fi

# 2. test sources of the columns (required)
echo "== 2. test sources of the columns (c151 test-sources) =="
c151 test-sources | cut -c1-420 || { rc=$?; echo "ABORT: test sources of the columns (code $rc)" >&2; exit "$rc"; }

# 3. second stage (regression of the base in the process, MM + columns nested, null point (e)) and field comparison with the base
echo "== 3. second stage (c151 fit) =="
c151 fit --save --base "$BASE_MODE" --cols "$COLS" | cut -c1-420 || { rc=$?; echo "ABORT: c151 fit (code $rc)" >&2; exit "$rc"; }
pin "$STAGE2_SHA" "$STAGE2"
python -I - "$BASE_STAGE2" "$STAGE2" "$BASE_STACK" "$COLS" <<'PY' || exit 4
import hashlib
import json
import sys

a, b = (json.load(open(p)) for p in sys.argv[1:3])
bad = []
if list(b) != [*a, "extra_columns", "cols_build"]:
    bad.append("order/set of the fields differs from the base (+ extra_columns, cols_build)")
for k in a:
    if k != "models" and b.get(k) != a[k]:
        bad.append(f"field {k} differs from the base")
if b["models"]["fallback"] != a["models"]["fallback"]:
    bad.append("fallback model differs")
if b["stack"] != sys.argv[3] or b["stack_sha256"] != hashlib.sha256(open(sys.argv[3], "rb").read()).hexdigest():
    bad.append(f"blend file {b['stack']} (the one of the base is required, unchanged)")
m, m0 = b["models"]["main"], a["models"]["main"]
x = b["extra_columns"]
sys.path[:0] = ["src"]
import stage2_sources as XS  # noqa: E402

want = list(XS.VARIANTS[sys.argv[4]])
feats = XS.feature_names(want)
if x["names"] != want or x["features"] != feats or x.get("variant") != sys.argv[4] or m.get("extra_columns") != feats or m["features"] != m0["features"] + feats:
    bad.append(f"sources {x['names']} / columns / features of the main model")
if m.get("feature_set") != "main_guard_i1_mm" or m.get("protect_none") is not True or len(m["w"]) != 64 + len(feats) or m.get("features_sha256_mm") != m0["features_sha256"]:
    bad.append(f"main model {m.get('feature_set')} / {len(m['w'])} weights / MM features")
if x["guard_rule"].get("rel") != 0.15 or x["guard_rule"].get("z_max") != 3.0 or not x["test"]["files_sha256"]:
    bad.append("rule of the test guard or test sources missing")
s = b["cols_build"]
if s.get("kind") != "deploy" or not s.get("null_point_e") or not (s.get("regression_base_main") or {}).get("equal"):
    bad.append("cols_build: regression or null point (e) not passed")
u = s["oof_units"]
print(f"  field comparison with the base: all fields equal (+ extra_columns, cols_build), fallback equal, blend of the base; main model "
      f"main_guard_i1_mm + {len(feats)} columns from {len(want)} sources ({64 + len(feats)}, guard on; OOF new top code {m['oof_changed_top_share']:.4f}); nested F {u['fold']['mm']} -> "
      f"{u['fold']['with_columns']}, M {u['set']['mm']} -> {u['set']['with_columns']} sixths; test sources {len(x['test']['files_sha256'])} files pinned"
      if not bad else "ABORT: " + "; ".join(bad))
raise SystemExit(4 if bad else 0)
PY

# 4. copy null point of the training side (columns := 0 -> rows, main model and fields = base)
echo "== 4. copy null point of the training side (c151 copy) =="
c151 copy --base "$BASE_MODE" --cols "$COLS" | cut -c1-420 || { rc=$?; echo "ABORT: c151 copy (code $rc)" >&2; exit "$rc"; }

# shared sources of the test side (as the build path of the base with ORIENT=repro)
python -I tools/orient_list.py check --mode repro --spec "$V1R_TEST" --json "$PRIV/check_orient_${STEM}.json" || exit $?
SRC=(--test "$V1R_TEST" --fill "v1r=v1" --test "v1=v1te_a_t4/v1_test_a/test+v1te_b_t4/v1_test_b/test"
     --test "zs27q=zs27q_test0_rp/zs27q_test0/test+zs27q_test1_rp/zs27q_test1/test" --fill "think18nb=zs27q"
     --test "think18nb=zs27v_te_rp/think_te/test/think@derived/think_rows/test.npy,zs27v_n3_rp/think_nte/test/think@derived/think_rows/test_tb_q.npy"
     --proto "$MID=$MID_PROTO" --rank "$MID=$MID_TEST_RANK"
     --rank-only "m3x4_full=$RANK1,$RANK2,$RANK3,$RANK4" --rank-only "q397m3=$Q397M3_TEST" --rank-none-band-for "q397m3=0.27,0.35"
     --align 0.75 --align-min 3 --blank-rows derived/blank/test_rows.npy)
ms() { PYTHONDONTWRITEBYTECODE=1 python -I tools/make_submission.py "$@" 2>&1 | grep -vi '^warning' | grep -E "SECOND STAGE|note guard|extra columns|written|ABORT|Traceback|Error|--stage2|--rank|--proto|MEAN|CONTROL" | cut -c1-320; return "${PIPESTATUS[0]}"; }

# 5. test guard per column: null model (computes the columns and the guard), file without second stage byte-identical, report, positive controls
echo "== 5. test guard of the extra columns (null model, c151 test-guard, c151 controls) =="
c151 null --base "$BASE_MODE" --cols "$COLS" --source "$STAGE2" --out "$TMP/null.json" >/dev/null
NX="$(python -I -c "import json,sys; print(len(json.load(open(sys.argv[1]))['extra_columns']['features']))" "$STAGE2")"
ms --name "$STEM_BASE" "${SRC[@]}" --out "$TMP/sub_base.csv" --suffix "_${TAG}base" || { echo "ABORT: file without second stage not computed" >&2; exit 4; }
rc=0; ms --name "$STEM_BASE" "${SRC[@]}" --out "$TMP/sub_null.csv" --suffix "_${TAG}null" --stage2 "$TMP/null.json" --stage2-report "$TMP/guards_null.json" || rc=$?
[ -f "$TMP/guards_null.json" ] || { echo "ABORT: null model not computed, no report (code $rc)" >&2; exit 4; }
cp -p "$TMP/guards_null.json" "$PRIV/guards_null_${STEM}.json"
rcg=0; c151 test-guard --report "$TMP/guards_null.json" || rcg=$?
if [ "$rcg" = 5 ]; then echo "ABORT: test guard of the extra columns violated (statistics above and in $PRIV/guards_null_${STEM}.json) — no file, no dropping of a column" >&2; exit 5; fi
[ "$rcg" = 0 ] || { echo "ABORT: c151 test-guard (code $rcg)" >&2; exit "$rcg"; }
[ "$rc" = 0 ] || { echo "ABORT: null model not computed (code $rc)" >&2; exit 4; }
cmp -s "$TMP/sub_base.csv" "$TMP/sub_null.csv" || { echo "ABORT: null model changes the file" >&2; exit 4; }
cmp -s "artifacts/test_${STEM_BASE}_${TAG}base.npy" "artifacts/test_${STEM_BASE}_${TAG}null.npy" || { echo "ABORT: null model changes the test matrix" >&2; exit 4; }
python -I - "$TMP/guards_null.json" "$NX" <<'PY' || exit 4
import json
import sys

g = json.load(open(sys.argv[1]))
nx = int(sys.argv[2])
x, mm = g.get("extra_columns") or {}, g.get("mech27") or {}
bad = []
if not g["null"] or g["feature_set"] != "main" or g.get("model_feature_set") != "main_guard_i1_mm" or g["n_features"] != 64 + nx or g.get("rows_protected") != 0:
    bad.append(f"null model {g['null']}, form {g['feature_set']}/{g.get('model_feature_set')}, {g['n_features']} features, guarded {g.get('rows_protected')}")
if g["guard_notes"].get("rule") != "widened" or not g["guard_notes"]["ok"]:
    bad.append("note guard")
if not x or not x["guard"]["ok"] or not x.get("used_by_model") or x.get("control") is not None:
    bad.append("extra columns: guard/use")
if not mm or mm.get("minor_codes") != 9:
    bad.append(f"mech27 {mm}")
print(f"  null model: file without second stage byte-identical (CSV and test matrix); {64 + nx} features (64 + {nx} extra columns from the frozen test sources) computed; "
      f"test guard per column held; note guard (widened rule) held" if not bad else "ABORT: " + "; ".join(bad))
raise SystemExit(4 if bad else 0)
PY
c151 controls --null "$TMP/null.json" --dir "$TMP/controls" --name "$STEM_BASE" -- "${SRC[@]}" | cut -c1-420 \
  || { rc=$?; echo "ABORT: positive controls of the test guard (code $rc)" >&2; exit "$rc"; }
cp -p "$TMP/controls/controls.json" "$PRIV/controls_${STEM}.json"

# 6. copy null point of the test side: copy deployment file (columns := 0 in the fit, real columns of the test side with weight 0) -> base byte-identical
echo "== 6. copy null point of the test side (columns := 0) =="
ms --name "$STEM_BASE" "${SRC[@]}" --out "$TMP/sub_copy.csv" --suffix "_${TAG}k" --stage2 "$COPY_STAGE2" --stage2-report "$TMP/guards_copy.json" \
  || { echo "ABORT: copy run of the test side not computed" >&2; exit 4; }
cmp -s "$TMP/sub_copy.csv" "$BASE_SUB" || { echo "ABORT: copy null point violated — CSV != $BASE_LABEL" >&2; exit 4; }
cmp -s "artifacts/test_${STEM_BASE}_${TAG}k.npy" "$BASE_MAT" || { echo "ABORT: copy null point violated — test matrix != $BASE_LABEL" >&2; exit 4; }
python -I - "$TMP/guards_copy.json" "$NX" <<'PY' || exit 4
import json
import sys

g = json.load(open(sys.argv[1]))
x = g.get("extra_columns") or {}
if g["n_features"] != 64 + int(sys.argv[2]) or not x.get("used_by_model") or not x["guard"]["ok"]:
    print(f"ABORT: copy run without the new path ({g['n_features']} features, extra columns {bool(x)})")
    raise SystemExit(4)
print(f"  copy run through the new path: {g['n_features']} features, extra columns computed from the test sources (weight 0), guard held")
PY
echo "  copy null point: CSV and test matrix byte-identical to $BASE_LABEL ($(sha256sum "$BASE_SUB" | cut -c1-16)… / $(sha256sum "$BASE_MAT" | cut -c1-16)…)"
rm -f -- "artifacts/test_${STEM_BASE}_${TAG}k.npy"

# 7. main file (form main with main_guard_i1_mm + columns and the guard required; guard (c) or test guard violated -> no file, exit 5)
echo "== 7. main file =="
if [ -f "$OUT" ] || [ -f "$TESTMAT" ]; then  # only replace byte-identically
  BAK="$(mktemp -d "${TMPDIR:-/tmp}/primary_cols_backup.XXXXXX")"
  for F in "$OUT" "${OUT%.csv}.json" "$TESTMAT"; do [ -f "$F" ] && cp -p "$F" "$BAK/"; done
fi
restore() {
  rm -f -- "$OUT" "${OUT%.csv}.json" "$TESTMAT"
  if [ -n "$BAK" ]; then
    for F in "$OUT" "${OUT%.csv}.json" "$TESTMAT"; do [ -f "$BAK/$(basename "$F")" ] && cp -p "$BAK/$(basename "$F")" "$F"; done
  fi
  return 0
}
rm -f "$GUARDS"
if ! ms --name "$STEM_BASE" "${SRC[@]}" --out "$OUT" --suffix "$SUFFIX" --stage2 "$STAGE2" --stage2-report "$GUARDS"; then
  restore
  if [ -f "$GUARDS" ] && python -I -c "import json,sys;g=json.load(open('$GUARDS'));sys.exit(1 if g['guard_changed_top']['ok'] and g['extra_columns']['guard']['ok'] else 0)"; then
    echo "ABORT: guard (c) or test guard of the extra columns violated (statistics in $GUARDS) — no file" >&2; exit 5
  fi
  echo "ABORT: make_submission with --stage2 failed" >&2; exit 4
fi
[ -f "$OUT" ] || { echo "ABORT: $OUT was not written" >&2; exit 4; }
python -I - "$GUARDS" "$PRIV/guards_null_${STEM}.json" "${OUT%.csv}.json" "$MID" "$NX" <<'PY' || { rc=$?; restore; exit "$rc"; }
import json
import sys

g, gnull, side = (json.load(open(p)) for p in sys.argv[1:4])
mid = sys.argv[4]
gn, x = g["guard_notes"], g.get("extra_columns") or {}
if g["feature_set"] != "main" or g.get("model_feature_set") != "main_guard_i1_mm" or g.get("protect_none") is not True or not x.get("used_by_model"):
    print(f"ABORT: form {g['feature_set']} / {g.get('model_feature_set')} / guard {g.get('protect_none')} / extra columns {x.get('used_by_model')} instead of "
          "main / main_guard_i1_mm / on / used — no file")
    raise SystemExit(5)
bad = []
if gn.get("rule") != "widened" or not gn["ok"] or g["null"] or g["n_features"] != 64 + int(sys.argv[5]) or not g["guard_changed_top"]["ok"] or not x["guard"]["ok"]:
    bad.append(f"rule {gn.get('rule')}, guard {gn['ok']}, null model {g['null']}, {g['n_features']} features, (c) {g['guard_changed_top']['ok']}, "
               f"columns {x['guard']['ok']}")
if gn["test"] != gnull["guard_notes"]["test"] or g.get("mech27") != gnull.get("mech27") or x["test"] != gnull["extra_columns"]["test"]:
    bad.append("test aggregates of the main file != those of step 5")
src = side.get("sources") or {}
notes = [z for z in src.get(mid) or [] if isinstance(z, dict) and "notes" in z]
if len(notes) != 1 or notes[0].get("rank_compare") != "sft":
    bad.append(f"sources of the channel {mid} without note guard/comparison form sft ({len(notes)})")
m4 = (src.get("m3x4_full") or [{}])[0]
if m4.get("draws") != 4 or len(str(m4.get("rank_only", "")).split(",")) != 4 or m4.get("share_all_draws") != 1.0:
    bad.append(f"channel m3x4_full: {m4.get('draws')} draws, share with all draws {m4.get('share_all_draws')}")
print(f"  main form main_guard_i1_mm + {int(sys.argv[5])} columns ({g['n_features']} features, guard on: {g['rows_protected']} test rows keep the base order); guard (c) new "
      f"top code {g['guard_changed_top']['test']:.4f} (OOF {g['guard_changed_top']['oof']:.4f}) ok; test guard of the columns held; mech27 "
      f"{g['mech27']['share_mech']:.4f}; note guard = step 5" if not bad else "ABORT: " + "; ".join(bad))
raise SystemExit(4 if bad else 0)
PY
if [ -n "$BAK" ] && [ -f "$BAK/$(basename "$OUT")" ]; then
  if ! cmp -s "$OUT" "$BAK/$(basename "$OUT")"; then
    restore
    echo "ABORT: rebuild differs from the existing $OUT — backup restored, investigate the cause" >&2; exit 4
  fi
  echo "  existing $OUT rebuilt byte-identically"
fi
python -I tools/orient_list.py verify-sub --mode repro "${OUT%.csv}.json" || { restore; exit 4; }

# 8. check of the file and comparison with the base (aggregates only)
echo "== 8. check and comparison with $BASE_LABEL (aggregates only) =="
check "$BASE_SUB_SHA" "$BASE_SUB"
python -I - "$OUT" "$BASE_SUB" "$GUARDS" "$PRIV/compare_${STEM}.json" <<'PY' || exit 4
import json
import sys
from pathlib import Path

import pandas as pd

sys.path[:0] = [str(Path.cwd()), str(Path.cwd() / "src")]
import labels  # noqa: E402
import submission_io  # noqa: E402
from vlm import common as C  # noqa: E402

root = C.find_data_root()
classes = labels.load_classes(root / "competition_data" / "taxonomy_v1.csv")
train = C.load_table(root, "train")
seen = labels.seen_mask(train["root_code"], classes)
sample = pd.read_csv(root / "competition_data" / "sample_submission.csv", dtype=str, keep_default_na=False)
sub = pd.read_csv(sys.argv[1], dtype=str, keep_default_na=False)
stats = submission_io.check_submission(sub, sample, classes, never_seen=[c for c, s in zip(classes, seen) if not s], require_three=True)
print(f"  check_submission: {stats}")


def read(p):
    return pd.read_csv(p, dtype=str, keep_default_na=False).set_index("trace_id")["root_code"]


new, ref = read(sys.argv[1]), read(sys.argv[2])
b = new.loc[ref.index]
t_new, t_ref = b.str.split().str[0], ref.str.split().str[0]
s_new, s_ref = b.str.split().apply(frozenset), ref.str.split().apply(frozenset)
e = {"rows": int(len(ref)), "rows_other_top1": int((t_ref != t_new).sum()), "share_other_top1": float((t_ref != t_new).mean()),
     "rows_other_top3_ordered": int((ref != b).sum()), "share_other_top3_ordered": float((ref != b).mean()),
     "rows_other_top3_as_set": int((s_new != s_ref).sum()), "share_top_none_new": float((t_new == "NONE").mean()),
     "share_top_none_base": float((t_ref == "NONE").mean())}
print(f"  vs. the base: other top code in {e['rows_other_top1']} of {e['rows']} rows ({e['share_other_top1']:.4f}), other first three (ordered) in "
      f"{e['rows_other_top3_ordered']} ({e['share_other_top3_ordered']:.4f}), as a set {e['rows_other_top3_as_set']}; NONE at the top "
      f"{e['share_top_none_base']:.4f} -> {e['share_top_none_new']:.4f}")
g = json.load(open(sys.argv[3]))
rep = {"check": stats, "vs_base": e, **{k: g[k] for k in ("guard_changed_top", "guard_novel_task_top5", "share_top3_changed", "share_top3_changed_oof",
                                                          "rows_protected", "mech27")}, "extra_columns_guard_ok": g["extra_columns"]["guard"]["ok"]}
gc = g["guard_changed_top"]
print(f"  guard (c): new top code {gc['test']:.4f} (OOF {gc['oof']:.4f}, band {gc['factor'][0] * gc['oof']:.4f}…{gc['factor'][1] * gc['oof']:.4f}) -> "
      f"{'ok' if gc['ok'] else 'VIOLATED'}; guarded {g['rows_protected']} test rows")
Path(sys.argv[4]).write_text(json.dumps(rep, indent=1, ensure_ascii=False) + "\n")
PY

# 9. checksums
echo "== 9. checksums =="
pin "$OUT_SHA" "$OUT"
pin "$TESTMAT_SHA" "$TESTMAT"
sha256sum "$STAGE2" "$OUT" "$TESTMAT" "${OUT%.csv}.json" "$GUARDS"
if [ "$FIRST" = 1 ]; then
  echo "primary_cols: FIRST BUILD — enter the printed checksums and run the build path again (second build byte-identical)"
else
  echo "primary_cols: $OUT built ($LABEL; nothing is submitted from here)"
fi
