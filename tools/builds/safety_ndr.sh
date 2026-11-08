#!/usr/bin/env bash
# Safety line WITH REGEX-EXACT NOTE COLUMNS (safety_ndr; safety slot): the safety line with note features (safety_nd,
# tools/builds/safety_nd.sh) with the feature set "fallback_i1_ndr" in the second stage = fallback_i1_nd (58, unchanged) + the six
# regex-exact note columns R (src/note_regex.py: mentions in the frozen M3 notes d1–d3 and in the frozen notes of the first pr27 draw).
# Deployment file artifacts/stage2_safety_ndr.json (eda/c130_safety_ndr.py fit). Order on the test side unchanged: blend -> margin alignment
# (one set) -> second stage -> NONE guard -> REL-01. No teacher channel, no 397B; notes gold-only (M3) or from the untrained 27B (pr27, the
# same run as the channel pr27xm). All guards ONLY as an abort (safety-ndr rule (6)).
#
#     bash tools/builds/safety_ndr.sh                  # deployment only if the safety-ndr rule is met, otherwise exit 5 (no deployment or submission file)
#     SAFETY_NDR_RECOMPUTE=1 …                         # recomputes the draw swap and the meta-splits (byte-identical result required)
#     SAFETY_NDR_NO_REGRESSION=1 …                     # skips step 0 (only for debugging; a deployment build requires step 0)
#
# STEPS: 0 regression check: build of safety_nd (tools/builds/safety_nd.sh; inside it safety_final with the override of (ii), the strictly
#   inductive line with second stage, s030n and primary_final) byte-identical; 1 inputs: M3 training notes (c129 export-notes --verify), pr27
#   notes = re-export from the runs (c130 export-notes --verify); 2 c130 oof (copy null point R 0 -> ND bit-identical, ND = safety_nd
#   bit-identical, 15 placebos); 3 draw swap d1–d4 (c130 swap); 4 meta-splits (c130 splits --n 20); 5 safety-ndr rule (1)–(4) (c130 rule;
#   exit 5 = no deployment); 6 deployment file (c130 fit); 7 copy null points of the test side: null model -> file without second stage
#   byte-identical; weights of safety_nd + 0 for R -> safety_nd byte-identical (CSV and test matrix); 8 positive controls (copy, permute,
#   truncate, pr27_shift, pr27_m3d1 -> a guard fires, no file); 9 main file (widened note guard, similarity guard, guard R, guard (c), class
#   masses L1 ≤ 0.035 = safety_nd); 10 comparison with safety_nd (aggregates only); 11 checksums.
# EXIT: 2 usage; 3 input missing; 4 checksum/recomputation/null point/control; 5 pre-registered rule not met or guard violated.
#
# Only aggregates and checksums are printed. Nothing is submitted from here.
set -euo pipefail
[ $# -eq 0 ] || { echo "usage: $0   (no arguments; SAFETY_NDR_RECOMPUTE=1, SAFETY_NDR_NO_REGRESSION=1)" >&2; exit 2; }
cd "$(dirname "$0")/../.."
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}" OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export ORIENT=repro
NAME="v1r_zs27t_think18sb_pr27xm_m3x3_full_ar"
STACK="artifacts/stack_${NAME}.json"; STACK_SHA="7fb543ae5b7cb460a869411c1a97759d5594af66140d6922485b1b1ff421490c"
SAFETY_ND_STAGE2="artifacts/stage2_safety_nd.json"; SAFETY_ND_STAGE2_SHA="50217a6c48f16bf35afb97fd755611eba85c43fa81187d0d00a91dbb31cddbee"
SAFETY_ND="artifacts/submissions/sub_safety_nd.csv"; SAFETY_ND_SHA="561a3c45ca3579763c44780b1f0cb009a6b75d5c730d4cbe42c4d11ce7e45025"
SAFETY_ND_MAT="artifacts/test_${NAME}_safety_nd.npy"; SAFETY_ND_MAT_SHA="32aac4537603992507321a65072e71cde1db95020322c0c7ae7183ec2d45da07"
PRIV="artifacts/eda/private/safety_ndr"
M3_NOTES_SHA=(389471a80e30d30cdc161cf487cfa1b0e47c0ef8c4496a6f2c06f9865b16f4ce 2a8f0813386e56a6c105039ef4639e1e909c5f8775051395a55d9dbfbb916455
              716085d463639ad4c2e030ce634dc71a3af5776d532980c57edfc9689bb05aba)
PR27_TRAIN="data/derived/proto_notes/train_pr27.jsonl"; PR27_TRAIN_SHA="d615b707778e1e7b685ff4a9b28c0b2b756b46aba7ce149c295f99527dea388d"
PR27_TEST="data/derived/proto_notes_test/test_pr27.jsonl"; PR27_TEST_SHA="1b459a40278b814922bfedbff9867028f9e0e56e39ca54b6c7b471c42a833f3a"
PR27_N2="data/derived/proto_notes/train_pr27d2.jsonl"; PR27_N2_SHA="4dfd8719e116fc5ebfff7b4223b6ebf5f790e1dc167174e98a1c99b70e85c4f9"
IOOF="$PRIV/c130_oof.json"; IOOF_SHA="ef4eb1944e5123da55f71d6f36fe6ed15c3fd28f4029591c1929fa76c040e35e"  # pinned after the first build
ISWAP="$PRIV/c130_swap.json"; ISWAP_SHA="fd47ced958c232047fa4516c7b15a00fb583bc0181e63da2ad82c8be17dcc6e2"
ISPL="$PRIV/c130_splits.json"; ISPL_SHA="db40c8b8294963f3303c5e700b220868b9338b157c91f5c97b09d2281f1fb9ec"
IRULE="$PRIV/c130_rule.json"
STAGE2="artifacts/stage2_safety_ndr.json"; STAGE2_SHA="bfd8f197fccb3ad9cc623fe2d45930151ac0debf40be3eb5cbafb953c79b2927"
OUT="artifacts/submissions/sub_safety_ndr.csv"; OUT_SHA="b8f17ce3e5fe1ed6dbf22febf9b153ec7c23870af123fa60d4342979d67270f6"  # safety_ndr candidate (pinned)
SUFFIX="_safety_ndr"
TESTMAT="artifacts/test_${NAME}${SUFFIX}.npy"; TESTMAT_SHA="abd63dfae32230ddd5ec5bba9e3ddcde91528c89d3fed4cf333da6f09b2c5101"
GUARDS="$PRIV/guards_safety_ndr.json"
RANK1="runs/m3_fullte_rp/m3_test_rank/test"
RANK2="runs/m3d2te_rp/m3d2_test_rank/test"
RANK3="runs/m3d3te_rp/m3d3_test_rank/test"
RANK_XM="runs/n1_rp/rank_test/test"
V1_T4="v1=v1te_a_t4/v1_test_a/test+v1te_b_t4/v1_test_b/test"
check() { echo "$1  $2" | sha256sum -c --quiet || { echo "ABORT: $2 differs (checksum)" >&2; exit 4; }; }
pin() {  # pinned checksum; empty = first build (the checksum is printed and then entered in this script)
  if [ -n "$1" ]; then check "$1" "$2"; else echo "  FIRST BUILD — checksum $(sha256sum "$2" | cut -c1-64)  $2 (enter it in this script)"; fi
}
. tools/builds/orient_list.sh
orient_select
[ "$ORIENT" = repro ] || { echo "ABORT: this build path only runs with ORIENT=repro" >&2; exit 2; }
mkdir -p "$PRIV"

# 0. regression check (code of the second stage and of the test side changed: safety_nd with its chain safety_final, the strictly inductive
#    line with second stage, s030n and primary_final must stay byte-identical)
if [ "${SAFETY_NDR_NO_REGRESSION:-0}" != 1 ]; then
  echo "== 0. build of safety_nd (tools/builds/safety_nd.sh; inside it safety_final with the override of (ii), the strictly inductive line with second stage, s030n and primary_final) =="
  bash tools/builds/safety_nd.sh > "$PRIV/regression_safety_nd.log" 2>&1 || { rc=$?; tail -5 "$PRIV/regression_safety_nd.log" >&2; echo "ABORT: build of safety_nd (code $rc)" >&2; exit "$rc"; }
  echo "  regression check safety_nd (with safety_final, the strictly inductive line with second stage, s030n, primary_final) byte-identical (log $PRIV/regression_safety_nd.log)"
else
  echo "== 0. regression check SKIPPED (SAFETY_NDR_NO_REGRESSION=1; no deployment build) =="
fi
check "$STACK_SHA" "$STACK"
check "$SAFETY_ND_STAGE2_SHA" "$SAFETY_ND_STAGE2"
check "$SAFETY_ND_SHA" "$SAFETY_ND"
check "$SAFETY_ND_MAT_SHA" "$SAFETY_ND_MAT"

TMP="$(mktemp -d "${TMPDIR:-/tmp}/safety_ndr.XXXXXX")"
BAK=""
cleanup() {
  rm -rf -- "${TMP:?}"
  rm -f -- "artifacts/test_${NAME}_ndrbase.npy" "artifacts/test_${NAME}_ndrnull.npy" "artifacts/test_${NAME}_ndrbzero.npy" \
           "artifacts/test_${NAME}_ndrctl.npy"
  [ -n "$BAK" ] && rm -rf -- "$BAK"  # the backup contains test identifiers: never leave it behind
  return 0
}
trap cleanup EXIT

# 1. inputs: frozen notes (recomputed from the runs)
echo "== 1. inputs (c129 export-notes --verify, c130 export-notes --verify) =="
python -I eda/c129_safety_nd.py export-notes --verify || exit $?
for d in 1 2 3; do check "${M3_NOTES_SHA[$((d-1))]}" "data/derived/sft/m3_d${d}_train_frozen_notes.jsonl"; done
python -I eda/c130_safety_ndr.py export-notes --verify || exit $?
check "$PR27_TRAIN_SHA" "$PR27_TRAIN"
check "$PR27_TEST_SHA" "$PR27_TEST"
check "$PR27_N2_SHA" "$PR27_N2"

# 2. nested reproduction
echo "== 2. c130 oof =="
python -I eda/c130_safety_ndr.py oof | grep -E "^\s+\[|ND\+R − ND|ABORT" | cut -c1-360 || { echo "ABORT: c130 oof" >&2; exit 4; }
python -I -c 'import json,sys; g=json.load(open(sys.argv[1])); n=g["null_point"]; sys.exit(0 if n["r_zero_equals_nd"]["ok"] and n["nd_prefix_equals_c129"]["ok"] and n["nd_equals_safety_nd_oof_rows"]["ok"] is True and g["placebo"]["n"]==15 else 4)' "$IOOF" \
  || { echo "ABORT: copy null point (R 0 -> ND), ND prefix or ND = safety_nd violated, or fewer than 15 placebos" >&2; exit 4; }
pin "$IOOF_SHA" "$IOOF"

# 3./4. draw swap and meta-splits (recomputed only without the file or with SAFETY_NDR_RECOMPUTE=1; then a byte-identical result is required)
recompute() {  # $1 file, $2 log, $3 pattern of the lines shown, $4… command
  local f="$1" log="$2" pat="$3"; shift 3
  if [ "${SAFETY_NDR_RECOMPUTE:-0}" = 1 ] || [ ! -f "$f" ]; then
    [ -f "$f" ] && cp -p "$f" "$TMP/before_$(basename "$f")"
    "$@" > "$log" 2>&1 || { rc=$?; tail -5 "$log" >&2; echo "ABORT: $* (code $rc; log $log)" >&2; exit 4; }
    grep -E "$pat" "$log" | cut -c1-330 || true
    if [ -f "$TMP/before_$(basename "$f")" ] && ! cmp -s "$f" "$TMP/before_$(basename "$f")"; then
      cp -p "$TMP/before_$(basename "$f")" "$f"
      echo "ABORT: recomputation of $f differs — backup restored" >&2; exit 4
    fi
    [ -f "$TMP/before_$(basename "$f")" ] && echo "  recomputation of $f byte-identical"
  else
    echo "  $f present (recomputation with SAFETY_NDR_RECOMPUTE=1)"
  fi
  return 0
}
echo "== 3. draw swap d1–d4 (c130 swap) =="
recompute "$ISWAP" "$PRIV/swap.log" "set |draw swap:" python -I eda/c130_safety_ndr.py swap
pin "$ISWAP_SHA" "$ISWAP"
echo "== 4. meta-splits (c130 splits --n 20) =="
recompute "$ISPL" "$PRIV/splits.log" "over" python -I eda/c130_safety_ndr.py splits --n 20
pin "$ISPL_SHA" "$ISPL"

# 5. pre-registered safety-ndr rule (1)–(4)
echo "== 5. pre-registered safety-ndr rule (1)–(4) (c130 rule) =="
python -I eda/c130_safety_ndr.py rule || { rc=$?; echo "ABORT: safety-ndr rule not met — no deployment, no deployment file and no submission file (code $rc)" >&2; exit "$rc"; }

# 6. deployment file
echo "== 6. deployment file (c130 fit) =="
if [ -f "$STAGE2" ]; then
  python -I eda/c130_safety_ndr.py fit --out "$STAGE2" --verify || { rc=$?; echo "ABORT: $STAGE2 is not the recomputation" >&2; exit "$rc"; }
else
  python -I eda/c130_safety_ndr.py fit --out "$STAGE2" --save || { rc=$?; echo "ABORT: $STAGE2 not built" >&2; exit "$rc"; }
fi
pin "$STAGE2_SHA" "$STAGE2"

python -I tools/orient_list.py check --mode repro --spec "$V1R_TEST" --json "$PRIV/check_orient_safety_ndr.json" || exit $?
BASE_SRC=(--test "$V1R_TEST" --fill "v1r=v1" --test "$V1_T4"
          --test "zs27t=px27_t4zt0/zs27t_test_t4/test+px27_t4zt1/zs27t_test_t4/test" --fill "think18sb=zs27t"
          --test "think18sb=zs27v_n3_rp/think_nte/test/think@derived/think_rows/test_tb_t.npy"
          --proto "pr27xm=runs/pr27te_rp/proto_te/proto" --rank "pr27xm=$RANK_XM"
          --rank-only "m3x3_full=$RANK1,$RANK2,$RANK3")
SRC=("${BASE_SRC[@]}" --align 0.75 --align-min 3 --blank-rows derived/blank/test_rows.npy)
ms() { PYTHONDONTWRITEBYTECODE=1 python -I tools/make_submission.py "$@" 2>&1 | grep -vi '^warning' | grep -E "SECOND STAGE|CONTROL|note guard|guard R|margin alignment as one set|written|ABORT|Traceback|Error|--stage2" | cut -c1-360; return "${PIPESTATUS[0]}"; }

# 7. copy null points of the test side
echo "== 7. copy null points of the test side =="
python -I eda/c130_safety_ndr.py null --source "$STAGE2" --out "$TMP/null.json" >/dev/null
python -I eda/c130_safety_ndr.py b-zero --source "$STAGE2" --out "$TMP/bzero.json" >/dev/null
ms --name "$NAME" "${SRC[@]}" --out "$TMP/sub_base.csv" --suffix "_ndrbase" || { echo "ABORT: file without second stage not computed" >&2; exit 4; }
ms --name "$NAME" "${SRC[@]}" --out "$TMP/sub_null.csv" --suffix "_ndrnull" --stage2 "$TMP/null.json" --stage2-report "$TMP/guards_null.json" \
  || { echo "ABORT: null model not computed" >&2; exit 4; }
cmp -s "$TMP/sub_base.csv" "$TMP/sub_null.csv" || { echo "ABORT: copy null point violated — weight 0 changes the file" >&2; exit 4; }
cmp -s "artifacts/test_${NAME}_ndrbase.npy" "artifacts/test_${NAME}_ndrnull.npy" || { echo "ABORT: copy null point violated — test matrix" >&2; exit 4; }
echo "  null model -> file without second stage byte-identical (CSV and test matrix; notes, reference, similarity, columns R and guards computed)"
ms --name "$NAME" "${SRC[@]}" --out "$TMP/sub_bzero.csv" --suffix "_ndrbzero" --stage2 "$TMP/bzero.json" --stage2-report "$TMP/guards_bzero.json" \
  || { echo "ABORT: copy null point (safety_nd + 0) not computed" >&2; exit 4; }
cmp -s "$TMP/sub_bzero.csv" "$SAFETY_ND" || { echo "ABORT: copy null point violated — weights of safety_nd + 0 do not give the file of safety_nd" >&2; exit 4; }
cmp -s "artifacts/test_${NAME}_ndrbzero.npy" "$SAFETY_ND_MAT" || { echo "ABORT: copy null point violated — test matrix != safety_nd" >&2; exit 4; }
echo "  weights of safety_nd + 0 for the six columns R -> file of safety_nd byte-identical (CSV and test matrix)"

# 8. positive controls: changed test notes -> a guard fires, no file (code 5)
echo "== 8. positive controls of the guards =="
for CTL in copy permute truncate pr27_shift pr27_m3d1; do
  rc=0
  ms --name "$NAME" "${SRC[@]}" --out "$TMP/sub_ctl.csv" --suffix "_ndrctl" --stage2 "$STAGE2" --stage2-control "$CTL" --stage2-report "$PRIV/control_${CTL}.json" || rc=$?
  [ "$rc" = 5 ] || { echo "ABORT: control $CTL — no guard fired (code $rc)" >&2; exit 4; }
  [ ! -f "$TMP/sub_ctl.csv" ] && [ ! -f "artifacts/test_${NAME}_ndrctl.npy" ] || { echo "ABORT: control $CTL wrote a file" >&2; exit 4; }
done
python -I - "$PRIV" <<'PY' || exit 4
import json
import sys
from pathlib import Path

d = Path(sys.argv[1])
for ctl in ("pr27_shift", "pr27_m3d1"):  # safety-ndr rule (7): the new guard of the columns R must detect these two (M3 notes unchanged)
    g = json.loads((d / f"control_{ctl}.json").read_text())
    if g["guard_r"]["ok"] or not (g["guard_notes"]["ok"] and g["guard_sim"]["ok"]):
        raise SystemExit(f"ABORT: control {ctl} — guard R {g['guard_r']['ok']}, note guard {g['guard_notes']['ok']}, similarity {g['guard_sim']['ok']} "
                         "(expected: only the guard R fires)")
print("  controls pr27_shift and pr27_m3d1 are detected by the guard of the columns R (note and similarity guards hold there, M3 unchanged)")
PY
echo "  controls copy, permute, truncate, pr27_shift, pr27_m3d1: each time a guard fires, no file"

# 9. main file
echo "== 9. main file =="
if [ -f "$OUT" ] || [ -f "$TESTMAT" ]; then  # replace only byte-identically
  BAK="$(mktemp -d "${TMPDIR:-/tmp}/safety_ndr_backup.XXXXXX")"
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
if ! ms --name "$NAME" "${SRC[@]}" --stage2 "$STAGE2" --stage2-report "$GUARDS" --out "$OUT" --suffix "$SUFFIX"; then
  restore
  echo "ABORT: make_submission with --stage2 failed (guard violated or error; statistics in $GUARDS) — no file" >&2; exit 5
fi
[ -f "$OUT" ] || { echo "ABORT: $OUT was not written" >&2; exit 4; }
python -I - "$GUARDS" "$TMP/guards_bzero.json" "$STAGE2" <<'PY' || { rc=$?; restore; exit "$rc"; }
import json
import sys

g, gb, s2 = (json.load(open(f)) for f in sys.argv[1:4])
cm, cmb = g.get("class_mass") or {}, gb.get("class_mass") or {}
bad = []
if g.get("align_mode") != "set" or g["feature_set"] != "fallback_i1_ndr" or g.get("protect_none") is not True or g["null"] \
        or g["n_features"] != len(s2["model"]["features"]) or g["n_features"] != 64:
    bad.append(f"form {g.get('align_mode')} / {g['feature_set']} / NONE guard {g.get('protect_none')} / null model {g['null']} / {g['n_features']} features")
if not (g["guard_changed_top"]["ok"] and g["guard_notes"]["ok"] and g["guard_sim"]["ok"] and g["guard_r"]["ok"]):
    bad.append(f"guards: (c) {g['guard_changed_top']['ok']}, notes {g['guard_notes']['ok']}, similarity {g['guard_sim']['ok']}, R {g['guard_r']['ok']}")
if abs(float(cm.get("l1_after", -1)) - float(cmb.get("l1_after", -2))) > 1e-12 or not float(cm["l1_after"]) <= 0.035:
    bad.append(f"class masses L1 {cm.get('l1_after')} != path of safety_nd ({cmb.get('l1_after')}) or above 0.035")
gs, gr = g["guard_sim"], g["guard_r"]
print(f"  form set / {g['feature_set']} ({g['n_features']}) / NONE guard on ({g['rows_protected']} rows); widened note guard held; similarity guard test "
      f"{gs['test']['mean']:.4f} against {gs['reference']['mean']:.4f} (z {gs['z']:+.2f}, {gs['passed_by']}); guard R held ("
      + ", ".join(f"{c['stat']} {c['test']:.4f}/{c['ref']:.4f} {c['passed_by']}" for c in gr["checks"]) + f"); guard (c) new top code "
      f"{g['guard_changed_top']['test']:.4f} (OOF {g['guard_changed_top']['oof']:.4f}); class masses L1 {cm['l1_after']:.4f} = safety_nd"
      if not bad else "ABORT: " + "; ".join(bad))
raise SystemExit(5 if bad else 0)
PY
if [ -n "$BAK" ] && [ -f "$BAK/$(basename "$OUT")" ]; then
  if ! cmp -s "$OUT" "$BAK/$(basename "$OUT")" || { [ -f "$BAK/$(basename "$TESTMAT")" ] && ! cmp -s "$TESTMAT" "$BAK/$(basename "$TESTMAT")"; }; then
    restore
    echo "ABORT: the rebuild differs from the existing $OUT or $TESTMAT — backup restored, find the cause" >&2; exit 4
  fi
  echo "  existing $OUT and $TESTMAT rebuilt byte-identically"
fi
python -I tools/orient_list.py verify-sub --mode repro "${OUT%.csv}.json" || { restore; exit 4; }

# 10. check and comparison with safety_nd (aggregates only)
echo "== 10. check and comparison (aggregates only) =="
python -I - "$OUT" "$SAFETY_ND" "$PRIV/compare_safety_ndr.json" <<'PY' || exit 4
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
gen = {"ANS-01", "CAL-01", "WOP-01", "FRM-01", "OTH-00"}
e = {"share_other_top1": float((t_ref != t_new).mean()), "rows_other_top1": int((t_ref != t_new).sum()), "share_other_top3_ordered": float((ref != b).mean()),
     "share_top_none_new": float((t_new == "NONE").mean()), "share_top_none_ref": float((t_ref == "NONE").mean()),
     "share_top_general_new": float(t_new.isin(gen).mean()), "share_top_general_ref": float(t_ref.isin(gen).mean())}
print(f"  vs. safety_nd: other top code in {e['share_other_top1']:.4f} of the rows ({e['rows_other_top1']}), other first three (ordered) in "
      f"{e['share_other_top3_ordered']:.4f}; NONE at the top {e['share_top_none_ref']:.4f} -> {e['share_top_none_new']:.4f}, general codes "
      f"{e['share_top_general_ref']:.4f} -> {e['share_top_general_new']:.4f}")
Path(sys.argv[3]).write_text(json.dumps({"check": stats, "vs_safety_nd": e}, indent=1, ensure_ascii=False) + "\n")
PY

# 11. checksums
echo "== 11. checksums =="
pin "$OUT_SHA" "$OUT"
pin "$TESTMAT_SHA" "$TESTMAT"
sha256sum "$STACK" "$SAFETY_ND_STAGE2" "$STAGE2" "$SAFETY_ND" "$OUT" "${OUT%.csv}.json" "$TESTMAT" "$GUARDS" "$IOOF" "$ISWAP" "$ISPL" "$IRULE"
echo "safety_ndr: $OUT built (safety_ndr candidate of the safety slot; nothing is submitted from here)"
