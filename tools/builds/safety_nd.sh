#!/usr/bin/env bash
# Safety line WITH NOTE FEATURES (safety_nd): the safety line with set alignment (tools/builds/safety_final.sh, variant b) with the feature
# set "fallback_i1_nd" in the second stage = fallback_i1 (55, unchanged) + note mentions m3_men, m3_last (frozen M3 notes d1–d3) + example
# similarity demo_sim (src/note_sim.py; reference: frozen training notes without the same question). Deployment file
# artifacts/stage2_safety_nd.json (eda/c129_safety_nd.py fit). Order on the test side unchanged: blend -> margin alignment (one set) -> second
# stage -> NONE guard -> REL-01. No teacher channel, no 397B; notes gold-only from m3x3_full. Widened note guard and similarity guard ONLY as
# an abort (safety-nd rule (4)).
#
#     bash tools/builds/safety_nd.sh                   # deployment only if the pre-registered safety-nd rule is met, otherwise exit 5
#     SAFETY_ND_SPLITS=1 …                             # recomputes the 20 meta-splits (byte-identical result required)
#     SAFETY_ND_NO_REGRESSION=1 …                      # skips step 0 (only for debugging; a deployment build requires step 0)
#
# STEPS: 0 regression checks: build of safety_final (tools/builds/safety_final.sh with the override of (ii), inside it the strictly inductive
#   line with and without second stage) and build of the final primary form (tools/builds/primary_final.sh) byte-identical; 1 training notes
#   (c129 export-notes --verify, pinned); 2 c129 oof (copy null point ND 0 -> b bit-identical, result file pinned); 3 meta-splits (pinned);
#   4 safety-nd rule (6) (c129 rule; exit 5 = no deployment); 5 deployment file (c129 fit, pinned); 6 copy null points of the test side: null
#   model -> file without second stage byte-identical; weights of safety_final + 0 for the ND columns -> safety_final byte-identical (CSV and
#   test matrix); 7 positive controls (copied, permuted, truncated test notes -> a guard fires, no file); 8 main file (widened note guard,
#   similarity guard, guard (c), class masses L1 ≤ 0.035 = safety_final); 9 comparison with safety_final (aggregates only); 10 checksums.
# EXIT: 2 usage; 3 input missing; 4 checksum/recomputation/null point/control; 5 pre-registered rule not met or guard violated.
#
# Only aggregates and checksums are printed. Nothing is submitted from here.
set -euo pipefail
[ $# -eq 0 ] || { echo "usage: $0   (no arguments; SAFETY_ND_SPLITS=1, SAFETY_ND_NO_REGRESSION=1)" >&2; exit 2; }
cd "$(dirname "$0")/../.."
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}" OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export ORIENT=repro
NAME="v1r_zs27t_think18sb_pr27xm_m3x3_full_ar"
STACK="artifacts/stack_${NAME}.json"; STACK_SHA="7fb543ae5b7cb460a869411c1a97759d5594af66140d6922485b1b1ff421490c"
SAFETY_FINAL_STAGE2="artifacts/stage2_safety_final.json"; SAFETY_FINAL_STAGE2_SHA="9b6dfbed3ac97a6083e45bab68a52b55118c3b0e3a35dd58371bad47c6e8c81a"
SAFETY_FINAL="artifacts/submissions/sub_safety_final.csv"; SAFETY_FINAL_SHA="736d87b3cb61494aeb6dcac29bc248a909138c6e85e3939f37a01180984e78f7"
SAFETY_FINAL_MAT="artifacts/test_${NAME}_safety_final.npy"
PRIMARY_FINAL="artifacts/submissions/sub_primary_final.csv"; PRIMARY_FINAL_SHA="d9c68c460c6da06d4975513911e11c4541d70d07b78f132566a3e43e7ad60b24"
PRIV="artifacts/eda/private/safety_nd"
NOTES_SHA=(389471a80e30d30cdc161cf487cfa1b0e47c0ef8c4496a6f2c06f9865b16f4ce 2a8f0813386e56a6c105039ef4639e1e909c5f8775051395a55d9dbfbb916455
           716085d463639ad4c2e030ce634dc71a3af5776d532980c57edfc9689bb05aba)
IOOF="$PRIV/c129_oof.json"; IOOF_SHA="53b50aa590538c82b6b98f88dd0e318da5d18d02ebce5ad1d442aed38b079979"  # pinned after the first build
ISPL="$PRIV/c129_splits.json"; ISPL_SHA="703703d46c913b6d5607c47183b734980cfb919d14283c2d83ac9e6c634e891e"
STAGE2="artifacts/stage2_safety_nd.json"; STAGE2_SHA="50217a6c48f16bf35afb97fd755611eba85c43fa81187d0d00a91dbb31cddbee"
OUT="artifacts/submissions/sub_safety_nd.csv"; OUT_SHA="561a3c45ca3579763c44780b1f0cb009a6b75d5c730d4cbe42c4d11ce7e45025"  # safety_nd candidate (pinned)
SUFFIX="_safety_nd"
TESTMAT="artifacts/test_${NAME}${SUFFIX}.npy"
GUARDS="$PRIV/guards_safety_nd.json"
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

# 0. regression checks (code of the second stage and of the test side changed: safety_final, the strictly inductive line with and without
#    second stage, and primary_final must stay byte-identical)
if [ "${SAFETY_ND_NO_REGRESSION:-0}" != 1 ]; then
  echo "== 0a. build of safety_final (tools/builds/safety_final.sh, with the override of (ii)) =="
  OVERRIDE_CV_II=override_cv_condition_ii bash tools/builds/safety_final.sh > "$PRIV/regression_safety_final.log" 2>&1 || { rc=$?; tail -5 "$PRIV/regression_safety_final.log" >&2; echo "ABORT: build of safety_final (code $rc)" >&2; exit "$rc"; }
  echo "== 0b. build of the final primary form (tools/builds/primary_final.sh) =="
  bash tools/builds/primary_final.sh > "$PRIV/regression_primary_final.log" 2>&1 || { rc=$?; tail -5 "$PRIV/regression_primary_final.log" >&2; echo "ABORT: build of primary_final (code $rc)" >&2; exit "$rc"; }
  echo "  regression checks safety_final (with the strictly inductive line with and without second stage) and primary_final byte-identical (logs $PRIV/regression_safety_final.log, $PRIV/regression_primary_final.log)"
else
  echo "== 0. regression checks SKIPPED (SAFETY_ND_NO_REGRESSION=1; no deployment build) =="
fi
check "$STACK_SHA" "$STACK"
check "$SAFETY_FINAL_STAGE2_SHA" "$SAFETY_FINAL_STAGE2"
check "$SAFETY_FINAL_SHA" "$SAFETY_FINAL"
check "$PRIMARY_FINAL_SHA" "$PRIMARY_FINAL"

TMP="$(mktemp -d "${TMPDIR:-/tmp}/safety_nd.XXXXXX")"
BAK=""
cleanup() {
  rm -rf -- "${TMP:?}"
  rm -f -- "artifacts/test_${NAME}_ndbase.npy" "artifacts/test_${NAME}_ndnull.npy" "artifacts/test_${NAME}_ndbzero.npy" \
           "artifacts/test_${NAME}_ndctl.npy"
  [ -n "$BAK" ] && rm -rf -- "$BAK"  # the backup contains test identifiers: never leave it behind
  return 0
}
trap cleanup EXIT

# 1. frozen training notes (recomputation from the runs)
echo "== 1. training notes (c129 export-notes --verify) =="
python -I eda/c129_safety_nd.py export-notes --verify || exit $?
for d in 1 2 3; do check "${NOTES_SHA[$((d-1))]}" "data/derived/sft/m3_d${d}_train_frozen_notes.jsonl"; done

# 2. nested reproduction
echo "== 2. c129 oof =="
python -I eda/c129_safety_nd.py oof | grep -E "^\s+\[|ND − b|ABORT" | cut -c1-330 || { echo "ABORT: c129 oof" >&2; exit 4; }
python -I -c 'import json,sys; g=json.load(open(sys.argv[1])); sys.exit(0 if g["null_point"]["nd_zero_equals_b"]["ok"] and g["placebo"]["n"]==15 else 4)' "$IOOF" \
  || { echo "ABORT: copy null point (ND 0 -> b) violated or fewer than 15 placebos" >&2; exit 4; }
pin "$IOOF_SHA" "$IOOF"

# 3. meta-splits
echo "== 3. meta-splits (c129 splits) =="
if [ "${SAFETY_ND_SPLITS:-0}" = 1 ] || [ ! -f "$ISPL" ]; then
  [ -f "$ISPL" ] && cp -p "$ISPL" "$TMP/splits_before.json"
  python -I eda/c129_safety_nd.py splits --n 20 | grep -E "over|ABORT" | cut -c1-300 || { echo "ABORT: c129 splits" >&2; exit 4; }
  if [ -f "$TMP/splits_before.json" ] && ! cmp -s "$ISPL" "$TMP/splits_before.json"; then
    cp -p "$TMP/splits_before.json" "$ISPL"
    echo "ABORT: recomputation of the meta-splits differs — backup restored" >&2; exit 4
  fi
fi
pin "$ISPL_SHA" "$ISPL"

# 4. pre-registered safety-nd rule (6)
echo "== 4. safety-nd rule (6) (c129 rule) =="
python -I eda/c129_safety_nd.py rule || { rc=$?; echo "ABORT: pre-registered safety-nd rule not met — no deployment (code $rc)" >&2; exit "$rc"; }

# 5. deployment file
echo "== 5. deployment file (c129 fit) =="
if [ -f "$STAGE2" ]; then
  python -I eda/c129_safety_nd.py fit --out "$STAGE2" --verify || { rc=$?; echo "ABORT: $STAGE2 is not the recomputation" >&2; exit "$rc"; }
else
  python -I eda/c129_safety_nd.py fit --out "$STAGE2" --save || { rc=$?; echo "ABORT: $STAGE2 not built" >&2; exit "$rc"; }
fi
pin "$STAGE2_SHA" "$STAGE2"

python -I tools/orient_list.py check --mode repro --spec "$V1R_TEST" --json "$PRIV/check_orient_safety_nd.json" || exit $?
BASE_SRC=(--test "$V1R_TEST" --fill "v1r=v1" --test "$V1_T4"
          --test "zs27t=px27_t4zt0/zs27t_test_t4/test+px27_t4zt1/zs27t_test_t4/test" --fill "think18sb=zs27t"
          --test "think18sb=zs27v_n3_rp/think_nte/test/think@derived/think_rows/test_tb_t.npy"
          --proto "pr27xm=runs/pr27te_rp/proto_te/proto" --rank "pr27xm=$RANK_XM"
          --rank-only "m3x3_full=$RANK1,$RANK2,$RANK3")
SRC=("${BASE_SRC[@]}" --align 0.75 --align-min 3 --blank-rows derived/blank/test_rows.npy)
ms() { PYTHONDONTWRITEBYTECODE=1 python -I tools/make_submission.py "$@" 2>&1 | grep -vi '^warning' | grep -E "SECOND STAGE|CONTROL|note guard|margin alignment as one set|written|ABORT|Traceback|Error|--stage2" | cut -c1-320; return "${PIPESTATUS[0]}"; }

# 6. copy null points of the test side
echo "== 6. copy null points of the test side =="
python -I eda/c129_safety_nd.py null --source "$STAGE2" --out "$TMP/null.json" >/dev/null
python -I eda/c129_safety_nd.py b-zero --source "$STAGE2" --out "$TMP/bzero.json" >/dev/null
ms --name "$NAME" "${SRC[@]}" --out "$TMP/sub_base.csv" --suffix "_ndbase" || { echo "ABORT: file without second stage not computed" >&2; exit 4; }
ms --name "$NAME" "${SRC[@]}" --out "$TMP/sub_null.csv" --suffix "_ndnull" --stage2 "$TMP/null.json" --stage2-report "$TMP/guards_null.json" \
  || { echo "ABORT: null model not computed" >&2; exit 4; }
cmp -s "$TMP/sub_base.csv" "$TMP/sub_null.csv" || { echo "ABORT: copy null point violated — weight 0 changes the file" >&2; exit 4; }
cmp -s "artifacts/test_${NAME}_ndbase.npy" "artifacts/test_${NAME}_ndnull.npy" || { echo "ABORT: copy null point violated — test matrix" >&2; exit 4; }
echo "  null model -> file without second stage byte-identical (CSV and test matrix; notes, reference and similarity computed)"
ms --name "$NAME" "${SRC[@]}" --out "$TMP/sub_bzero.csv" --suffix "_ndbzero" --stage2 "$TMP/bzero.json" --stage2-report "$TMP/guards_bzero.json" \
  || { echo "ABORT: copy null point (safety_final + 0) not computed" >&2; exit 4; }
cmp -s "$TMP/sub_bzero.csv" "$SAFETY_FINAL" || { echo "ABORT: copy null point violated — weights of safety_final + 0 do not give the file of safety_final" >&2; exit 4; }
cmp -s "artifacts/test_${NAME}_ndbzero.npy" "$SAFETY_FINAL_MAT" || { echo "ABORT: copy null point violated — test matrix != safety_final" >&2; exit 4; }
echo "  weights of safety_final + 0 for m3_men, m3_last, demo_sim -> file of safety_final byte-identical (CSV and test matrix)"

# 7. positive controls: changed test notes -> a guard fires, no file (code 5)
echo "== 7. positive controls of the guards =="
for CTL in copy permute truncate; do
  rc=0
  ms --name "$NAME" "${SRC[@]}" --out "$TMP/sub_ctl.csv" --suffix "_ndctl" --stage2 "$STAGE2" --stage2-control "$CTL" --stage2-report "$PRIV/control_${CTL}.json" || rc=$?
  [ "$rc" = 5 ] || { echo "ABORT: control $CTL — no guard fired (code $rc)" >&2; exit 4; }
  [ ! -f "$TMP/sub_ctl.csv" ] && [ ! -f "artifacts/test_${NAME}_ndctl.npy" ] || { echo "ABORT: control $CTL wrote a file" >&2; exit 4; }
done
echo "  controls copy, permute, truncate: a guard fires each time, no file"

# 8. main file
echo "== 8. main file =="
if [ -f "$OUT" ] || [ -f "$TESTMAT" ]; then  # only replace byte-identically
  BAK="$(mktemp -d "${TMPDIR:-/tmp}/safety_nd_backup.XXXXXX")"
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
if g.get("align_mode") != "set" or g["feature_set"] != "fallback_i1_nd" or g.get("protect_none") is not True or g["null"] \
        or g["n_features"] != len(s2["model"]["features"]):
    bad.append(f"form {g.get('align_mode')} / {g['feature_set']} / guard {g.get('protect_none')} / null model {g['null']} / {g['n_features']} features")
if not (g["guard_changed_top"]["ok"] and g["guard_notes"]["ok"] and g["guard_sim"]["ok"]):
    bad.append(f"guards: (c) {g['guard_changed_top']['ok']}, notes {g['guard_notes']['ok']}, similarity {g['guard_sim']['ok']}")
if abs(float(cm.get("l1_after", -1)) - float(cmb.get("l1_after", -2))) > 1e-12 or not float(cm["l1_after"]) <= 0.035:
    bad.append(f"class masses L1 {cm.get('l1_after')} != path of safety_final ({cmb.get('l1_after')}) or above 0.035")
gs = g["guard_sim"]
print(f"  form set / {g['feature_set']} ({g['n_features']}) / guard on ({g['rows_protected']} rows); widened note guard held; similarity guard test "
      f"{gs['test']['mean']:.4f} vs. {gs['reference']['mean']:.4f} (z {gs['z']:+.2f}, {gs['passed_by']}); guard (c) new top code "
      f"{g['guard_changed_top']['test']:.4f} (OOF {g['guard_changed_top']['oof']:.4f}); class masses L1 {cm['l1_after']:.4f} = safety_final"
      if not bad else "ABORT: " + "; ".join(bad))
raise SystemExit(5 if bad else 0)
PY
if [ -n "$BAK" ] && [ -f "$BAK/$(basename "$OUT")" ]; then
  if ! cmp -s "$OUT" "$BAK/$(basename "$OUT")"; then
    restore
    echo "ABORT: rebuild differs from the existing $OUT — backup restored, investigate the cause" >&2; exit 4
  fi
  echo "  existing $OUT rebuilt byte-identically"
fi
python -I tools/orient_list.py verify-sub --mode repro "${OUT%.csv}.json" || { restore; exit 4; }

# 9. check and comparison with safety_final (aggregates only)
echo "== 9. check and comparison (aggregates only) =="
python -I - "$OUT" "$SAFETY_FINAL" "$PRIV/compare_safety_nd.json" <<'PY' || exit 4
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
print(f"  against safety_final: other top code in {e['share_other_top1']:.4f} of the rows ({e['rows_other_top1']}), other first three (ordered) in "
      f"{e['share_other_top3_ordered']:.4f}; NONE at the top {e['share_top_none_ref']:.4f} -> {e['share_top_none_new']:.4f}, general codes "
      f"{e['share_top_general_ref']:.4f} -> {e['share_top_general_new']:.4f}")
Path(sys.argv[3]).write_text(json.dumps({"check": stats, "vs_safety_final": e}, indent=1, ensure_ascii=False) + "\n")
PY

# 10. checksums
echo "== 10. checksums =="
pin "$OUT_SHA" "$OUT"
sha256sum "$STACK" "$SAFETY_FINAL_STAGE2" "$STAGE2" "$SAFETY_FINAL" "$OUT" "${OUT%.csv}.json" "$TESTMAT" "$GUARDS" "$IOOF" "$ISPL"
echo "safety_nd: $OUT built (fallback form of the safety line; nothing is submitted from here)"
