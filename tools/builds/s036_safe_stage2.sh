#!/usr/bin/env bash
# STRICTLY INDUCTIVE LINE with the SECOND STAGE without notes and with the NONE guard, released rotation list: the inductive line
# with the released rotation list (tools/builds/s030n_safe.sh, ORIENT=repro) + SECOND STAGE without notes + NONE guard:
#   conditional logit over the first 5 candidates AFTER the inductive alignment (src/stage2.py, feature set "fallback", 40 features,
#   K 5, λ 0.1; knowledge of the training rows only from the other folds), fitted on all 1,500 training rows in the per-fold form
#   (eda/c120_s036_safe_stage2.py fit -> artifacts/stage2_s036_safe.json); then the NONE guard (base top NONE -> the base order of
#   the row stays); then REL-01 for blank photos. No teacher output, no 397B channel, no margin alignment over the test set, no
#   notes (hence no note guard), no new GPU computation: all test inputs are those of the inductive line with the released list.
# Order on the test side: blend -> inductive alignment (class weights from the training rows) -> second stage -> NONE guard -> REL-01.
#
#     bash tools/builds/s036_safe_stage2.sh     # -> artifacts/submissions/sub_s036_safe_stage2.csv (the files without second stage stay untouched)
#
# STEPS: 0 build of the inductive line with the released list (tools/builds/s030n_safe.sh: all guards of the test side, file
#   byte-identical 8cca7bbf…); 1 checksums (blend file, file and test matrix without second stage); 2 nested OOF reproduction (c120
#   oof: anchor 7,366/9,000, deployment form 7,410/9,000, copy null point on the OOF rows); 3 deployment file (c120 fit --verify, on
#   the first build --save; null point (e): features of the deployment form = evaluated rows of the OOF reproduction), pinned
#   checksum; 4 copy null point on the test side (weight 0 -> file without second stage byte-identical, CSV and test matrix); 5 main
#   file with --stage2 (guard (c): share of new top codes after the guard ½…2× OOF, otherwise no file; (b) reported only;
#   statistics to artifacts/eda/private/stage2_inductive/guards_s036.json); 6 check (check_submission, source of v1r = released
#   list) and comparison with the file without second stage (aggregates only). An existing output file is only replaced by a
#   byte-identical rebuild; the checksum of the file is pinned (OUT_SHA).
# EXIT: 2 usage; 3 input missing; 4 checksum/recomputation/null point; 5 guard (c) of the second stage violated; 1 a check
#   inside a called Python script that stops with a message (for example the checksum of a base channel).
#
# Only aggregates and checksums are printed. Nothing is submitted from here.
set -euo pipefail
[ $# -eq 0 ] || { echo "usage: $0   (no arguments)" >&2; exit 2; }
cd "$(dirname "$0")/../.."
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}" OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export ORIENT=repro
NAME="v1r_zs27t_think18sb_pr27xm_m3x3_full_ar"
CH="m3x3_full"
STACK="artifacts/stack_${NAME}.json"
STACK_SHA="7fb543ae5b7cb460a869411c1a97759d5594af66140d6922485b1b1ff421490c"          # blend file of the inductive line (both rotation lists)
BASE_FILE="artifacts/submissions/sub_s030n_safe.csv"
BASE_FILE_SHA="8cca7bbfe9b8c9dce47d6153d59712575bb7b8ad6dea8a286f7ef530df448b67"          # inductive line, released list (tools/builds/s030n_safe.sh)
BASE_MAT="artifacts/test_${NAME}_ia_t4_s030n.npy"
BASE_MAT_SHA="64647c1b3624b4861589a9d881bf653a5f909403e91afac656105cbdb68965ee"      # its test matrix
STAGE2="artifacts/stage2_s036_safe.json"
STAGE2_SHA="${STAGE2_SHA:-7acd67bc7602369b75838870f1e362a2b4065a7deb3dca515dc81a9dedaa145f}"  # c120 fit
OUT="artifacts/submissions/sub_s036_safe_stage2.csv"
OUT_SHA="ff45c2f72d21fd9d6b227fc47bad10b4b4092cd79008734da917e2ba3bcefd13"  # sub_s036_safe_stage2.csv (pinned after the first build)
SUFFIX="_ia_t4_s036"
GUARDS="artifacts/eda/private/stage2_inductive/guards_s036.json"
RANK1="runs/m3_fullte_rp/m3_test_rank/test"
RANK2="runs/m3d2te_rp/m3d2_test_rank/test"
RANK3="runs/m3d3te_rp/m3d3_test_rank/test"
RANK_XM="runs/n1_rp/rank_test/test"
V1_T4="v1=v1te_a_t4/v1_test_a/test+v1te_b_t4/v1_test_b/test"
check() { echo "$1  $2" | sha256sum -c --quiet || { echo "ABORT: $2 differs (checksum)" >&2; exit 4; }; }
. tools/builds/orient_list.sh  # rotation list of the v1r test side: always repro here
orient_select
PRIV="artifacts/eda/private/stage2_inductive"
mkdir -p "$PRIV"

# 0. build of the inductive line with the released list: test side complete, all guards, blend file and file byte-identical
echo "== 0. inductive line with the released rotation list (tools/builds/s030n_safe.sh) =="
bash tools/builds/s030n_safe.sh || { rc=$?; echo "ABORT: build of the inductive line with the released list (code $rc)" >&2; exit "$rc"; }

# 1. checksums
echo "== 1. checksums =="
check "$STACK_SHA" "$STACK"
check "$BASE_FILE_SHA" "$BASE_FILE"
check "$BASE_MAT_SHA" "$BASE_MAT"

# 2. nested OOF reproduction (anchor, deployment form, copy null point on the OOF rows)
echo "== 2. OOF reproduction (c120 oof) =="
python -I eda/c120_s036_safe_stage2.py oof || { rc=$?; echo "ABORT: OOF reproduction (code $rc)" >&2; exit "$rc"; }

# 3. deployment file (null point (e) inside c120 fit)
echo "== 3. deployment file (c120 fit) =="
if [ -f "$STAGE2" ]; then
  python -I eda/c120_s036_safe_stage2.py fit --verify || { rc=$?; echo "ABORT: $STAGE2 is not the recomputation" >&2; exit "$rc"; }
else
  python -I eda/c120_s036_safe_stage2.py fit --save || { rc=$?; echo "ABORT: $STAGE2 not built" >&2; exit "$rc"; }
fi
check "$STAGE2_SHA" "$STAGE2"

# shared sources of the test side (as in tools/builds/s029_m3x3.sh safe with ORIENT=repro)
python -I tools/orient_list.py check --mode repro --spec "$V1R_TEST" || exit $?
SRC=(--test "$V1R_TEST" --fill "v1r=v1" --test "$V1_T4"
     --test "zs27t=px27_t4zt0/zs27t_test_t4/test+px27_t4zt1/zs27t_test_t4/test" --fill "think18sb=zs27t"
     --test "think18sb=zs27v_n3_rp/think_nte/test/think@derived/think_rows/test_tb_t.npy"
     --proto "pr27xm=runs/pr27te_rp/proto_te/proto" --rank "pr27xm=$RANK_XM"
     --rank-only "$CH=$RANK1,$RANK2,$RANK3"
     --align-inductive 0.75 --align-min 3 --blank-rows derived/blank/test_rows.npy)
ms() { PYTHONDONTWRITEBYTECODE=1 python -I tools/make_submission.py "$@" 2>&1 | grep -vi '^warning' | grep -E "SECOND STAGE|written|ABORT|Traceback|Error|--stage2" | cut -c1-320; return "${PIPESTATUS[0]}"; }

TMP="$(mktemp -d "${TMPDIR:-/tmp}/s036_safe_stage2.XXXXXX")"
BAK=""
cleanup() {
  rm -rf -- "${TMP:?}"
  rm -f -- "artifacts/test_${NAME}${SUFFIX}null.npy"
  [ -n "$BAK" ] && rm -rf -- "$BAK"  # the backup holds test identifiers: never leave it behind
  return 0
}
trap cleanup EXIT

# 4. copy null point on the test side (weight 0): the features are computed, the order stays -> file without second stage byte-identical
echo "== 4. copy null point =="
python -I eda/c120_s036_safe_stage2.py null --out "$TMP/null.json" >/dev/null
ms --name "$NAME" "${SRC[@]}" --out "$TMP/sub_null.csv" --suffix "${SUFFIX}null" --stage2 "$TMP/null.json" --stage2-report "$TMP/guards_null.json" \
  || { echo "ABORT: copy null point not computed" >&2; exit 4; }
check "$BASE_FILE_SHA" "$TMP/sub_null.csv"
check "$BASE_MAT_SHA" "artifacts/test_${NAME}${SUFFIX}null.npy"
echo "  copy null point: weight 0 -> $(basename "$BASE_FILE") byte-identical (${BASE_FILE_SHA:0:16}…), test matrix byte-identical (${BASE_MAT_SHA:0:16}…)"

# 5. main file (guard (c) violated -> no file, exit 5)
echo "== 5. main file with second stage =="
if [ -f "$OUT" ]; then  # only replace byte-identically
  BAK="$(mktemp -d "${TMPDIR:-/tmp}/s036_backup.XXXXXX")"
  for F in "$OUT" "${OUT%.csv}.json" "artifacts/test_${NAME}${SUFFIX}.npy"; do [ -f "$F" ] && cp -p "$F" "$BAK/"; done
fi
rm -f "$GUARDS"
if ! ms --name "$NAME" "${SRC[@]}" --stage2 "$STAGE2" --stage2-report "$GUARDS" --out "$OUT" --suffix "$SUFFIX"; then
  if [ -f "$GUARDS" ] && python -I -c "import json,sys;sys.exit(1 if json.load(open('$GUARDS'))['guard_changed_top']['ok'] else 0)"; then
    echo "ABORT: guard (c) of the second stage violated (statistics in $GUARDS) — no file" >&2; exit 5
  fi
  echo "ABORT: make_submission with --stage2 failed" >&2; exit 4
fi
[ -f "$OUT" ] || { echo "ABORT: $OUT was not written" >&2; exit 4; }
if [ -n "$BAK" ] && [ -f "$BAK/$(basename "$OUT")" ]; then
  if ! cmp -s "$OUT" "$BAK/$(basename "$OUT")"; then
    for F in "$OUT" "${OUT%.csv}.json"; do [ -f "$BAK/$(basename "$F")" ] && cp -p "$BAK/$(basename "$F")" "$F"; done
    [ -f "$BAK/test_${NAME}${SUFFIX}.npy" ] && cp -p "$BAK/test_${NAME}${SUFFIX}.npy" "artifacts/test_${NAME}${SUFFIX}.npy"
    echo "ABORT: rebuild differs from the existing $OUT — backup restored, investigate the cause" >&2; exit 4
  fi
  echo "  existing $OUT rebuilt byte-identically"
fi
python -I tools/orient_list.py verify-sub --mode repro "${OUT%.csv}.json" || exit $?

# 6. check of the file and comparison with the file without second stage (aggregates only)
echo "== 6. check and comparison =="
check "$BASE_FILE_SHA" "$BASE_FILE"
python -I - "$OUT" "$BASE_FILE" "$GUARDS" <<'PY'
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
print(f"  vs. the file without second stage: other top code in {(t_ref != t_new).mean():.4f} of the rows, other first three (ordered) in {(ref != b).mean():.4f}; "
      f"NONE at rank 1: without second stage {(t_ref == 'NONE').mean():.4f}, with second stage {(t_new == 'NONE').mean():.4f}")
g = json.load(open(sys.argv[3]))
gc, gn = g["guard_changed_top"], g["guard_novel_task_top5"]
print(f"  form: feature set {g['feature_set']} ({g['n_features']}), NONE guard {'on' if g['protect_none'] else 'off'} ({g['rows_protected']} test rows guarded); "
      f"guard (c): new top code {gc['test']:.4f} (OOF {gc['oof']:.4f}, band {gc['factor'][0] * gc['oof']:.4f}…{gc['factor'][1] * gc['oof']:.4f}) -> "
      f"{'ok' if gc['ok'] else 'VIOLATED'}; (b) \"novel for the task\" among the first 5: test {gn['test']:.4f}, OOF {gn['oof']:.4f}; "
      f"first three changed test {g['share_top3_changed']:.4f}, OOF {float(g['share_top3_changed_oof'] or 0.0):.4f}")
PY
python -I tools/orient_list.py compare "$BASE_FILE" "$OUT" --label "inductive line without second stage, released rotation list" --json "$PRIV/compare_s036_vs_s030n.json"
if [ -n "$OUT_SHA" ]; then
  check "$OUT_SHA" "$OUT"
else
  echo "  NOTE: OUT_SHA is not pinned yet — enter the checksum printed below in this build script"
fi
sha256sum "$STACK" "$STAGE2" "$OUT" "${OUT%.csv}.json" "artifacts/test_${NAME}${SUFFIX}.npy" "$GUARDS"
echo "s036 safe: $OUT built (strictly inductive line with the second stage and the NONE guard, released rotation list)"
