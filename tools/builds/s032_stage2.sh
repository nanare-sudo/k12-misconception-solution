#!/usr/bin/env bash
# Primary line with the SECOND STAGE, as submitted (reranker fallback form): the primary-line file of tools/builds/s029_q397m3.sh
# (v1r, zs27q, think18nb × 1.4, pr27xm, M3 from three draws, q397m3; floors 0.01/0.05/0.05/0.05; margin alignment 0.75 for classes
# with at least 3 rows, applied to the test rows as ONE set; REL-01 for blank photos; note guard and NONE band 0.27–0.35 for
# q397m3) with a SECOND STAGE:
#   (1) arithmetic share β chosen by the rule argmin log loss before the last step on all 1,500 OOF rows (grid 0.1…0.8) instead
#       of 0.2 -> artifacts/stack_s032_stage2.json (eda/c110_q397_deploy.py stack --replace-q397 q397m3 --arith-beta-select --stem s032_stage2);
#   (2) conditional logit over the first 5 candidates AFTER the alignment (src/stage2.py; K 5, λ 0.1 fixed, feature set "main" =
#       without 397B notes and without 397B channel features; "fallback" = "main" without notes) -> artifacts/stage2_s032.json
#       (eda/c118_stage2.py fit; knowledge of a training row only from the other folds).
#   On the test side the note guard of this deployment file fails one statistic (identical[d1-d2]), so the reranker runs in the
#   fallback form; tools/builds/s034_stage2_main.sh builds the main form under the widened rule of the note guard.
# Order on the test side: blend -> margin alignment -> second stage -> REL-01.
#
#     bash tools/builds/s032_stage2.sh best        # -> artifacts/submissions/sub_s032_stage2.csv
#
# STEPS: 0 build of the primary-line file without second stage complete (tools/builds/s029_q397m3.sh best: all guards of the test
#   side, file byte-identical 768b7778…); 1 checksums (blend file and output file of that build, frozen M3 test notes d1–d3);
#   2 blend file with β (cross-check inside c110: β 0.2 reproduces the blend file of that build character by character);
#   3 nested OOF reproduction (c118 oof; deviation from the target values of the independent cross-check ≤ 0.0005, prototype
#   bit-identical, label placebo abstains); 4 deployment file (c118 fit; null point (e): training features = features of the
#   OOF recomputation, bit-identical); 5 copy null points with null models (weight 0): (a) blend file of that build ->
#   sub_s029_best_q397m3.csv byte-identical; (b) new blend -> file without second stage byte-identical (CSV and test matrix);
#   6 main file with --stage2 (guards (a)–(c), statistics to artifacts/eda/private/stage2/guards.json); 7 comparison with the file
#   without second stage at β 0.2 and at the new β (aggregates only). An existing output file is only replaced by a
#   byte-identical rebuild.
# EXIT: 2 usage; 3 input missing; 4 checksum/recomputation/null point; 5 guard of the test side violated (including guard (c) of
#   the second stage); 6 c108 verdict missing (from s029_q397m3.sh); 1 a check inside a called Python script that stops
#   with a message.
# ROTATION LIST (tools/builds/orient_list.sh): ORIENT=repro (default; entry point tools/builds/s033n_stage2.sh) reads for v1r the
#   rotation list of the released orientation network, derived/orient_repro/test_c80_rows.npy: step 0 builds the intermediate
#   file sub_s032n_q397m3.csv (s029_q397m3.sh with ORIENT=repro), copy null point (a) requires that file byte-identically, and the
#   main file is the fallback form with the released list -> artifacts/submissions/sub_s033n_stage2.csv (test matrix
#   test_s032_stage2_s033n.npy, statistics stage2/guards_s033n.json, throw-away suffixes with _s033n). Blend file, deployment
#   file stage2_s032.json and all other inputs and guards as with ORIENT=legacy; in addition tools/orient_list.py check/verify-sub
#   and the comparison with the legacy-list file as aggregates. ORIENT=legacy rebuilds the historical file sub_s032_stage2.csv
#   with the first-version rotation list byte-identically; the released-list build needs it (and sub_s029_best_q397m3.csv) first.
#
# Only aggregates and checksums are printed. Nothing is submitted from here.
set -euo pipefail
cd "$(dirname "$0")/../.."
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}" OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
usage() { echo "usage: $0 best" >&2; exit 2; }
[ $# -eq 1 ] && [ "$1" = best ] || usage
STEM="s032_stage2"
STACK="artifacts/stack_${STEM}.json"
STAGE2="artifacts/stage2_s032.json"
GUARDS="artifacts/eda/private/stage2/guards.json"
REF_BASE="artifacts/submissions/sub_s029_best_q397m3.csv"; REF_BASE_SHA="768b7778ea049caeb23455ecad26b9bc90ebff52bee3cb97e356aac7cadf07e8"
STACK_BASE="artifacts/stack_s029_best_q397m3.json"; STACK_BASE_SHA="4910a2ac4be346da19fbfd0776711b51e7bbaea5e125d7f2423854d37b26d1f3"
STACK_SHA="d278bbedf7fb4d7564e584a55d45959b826748dfe937d8becf1e0818007281eb"   # artifacts/stack_s032_stage2.json (c110, β 0.6)
STAGE2_SHA="${STAGE2_SHA:-c33174e3ab203b5c2559586eb6b02f67f1588aca4da6bde25078cfb318fed73e}"  # artifacts/stage2_s032.json (c118 fit)
NOTES_SHA=("5e2921752b8e20c7058649989ba5d23b76ab009916477a614644976d9e3a8a9b" "7bad2517add60acf7248e796149b5dbef735fb3115c63bf23c18ebb1262568c4"
           "03df5b4caf68f22e2c0e6e6e8910e20cbf4385b49b1914907dbfc26948e39734")  # data/derived/sft/m3_d{1,2,3}_test_frozen_notes.jsonl
RANK1="runs/m3_fullte_rp/m3_test_rank/test"
RANK2="runs/m3d2te_rp/m3d2_test_rank/test"
RANK3="runs/m3d3te_rp/m3d3_test_rank/test"
Q397M3_TEST="runs/q397m3te_rp/q397m3_test_rank/test"
check() { echo "$1  $2" | sha256sum -c --quiet || { echo "ABORT: $2 differs (checksum)" >&2; exit 4; }; }
. tools/builds/orient_list.sh  # rotation list of the v1r test side: ORIENT=repro (default, released list) | legacy (first-version list)
orient_select
OUT="artifacts/submissions/sub_${STEM}.csv"
RSUF=""; MAINARGS=()  # empty = legacy-list file (without --out/--suffix)
REF_BASE_LABEL="file without second stage (β 0.2)"
if [ "$ORIENT" = repro ]; then  # fallback form with the released rotation list on the intermediate file: own file, test matrix, statistics and throw-away suffixes
  REF_BASE_OLD="$REF_BASE"; REF_BASE_OLD_SHA="$REF_BASE_SHA"
  REF_BASE="artifacts/submissions/sub_s032n_q397m3.csv"; REF_BASE_SHA="d1667f6d2fb118844b72ecd8abd85220754a709d468519b7a047a893bbea12f1"  # intermediate file (pinned after the first build)
  REF_BASE_LABEL="file without second stage (β 0.2, released rotation list)"
  LEGACY_OUT="artifacts/submissions/sub_s032_stage2.csv"; LEGACY_OUT_SHA="96bafc6fa4c0fb89a32e45fe24816454685b137d0c8264171bfcea93ea371809"  # legacy-list file of this build
  RSUF="_s033n"
  OUT="artifacts/submissions/sub_s033n_stage2.csv"
  MAINARGS=(--out "$OUT" --suffix "$RSUF")
  GUARDS="artifacts/eda/private/stage2/guards_s033n.json"
  PRIVF="artifacts/eda/private/orient_check"
  mkdir -p "$PRIVF"
fi

# 0. build of the primary-line file without second stage: test side complete, all guards (q397m3, 397B, M3 draws, note guard), blend file and output file byte-identical
echo "== 0. primary-line file without second stage (tools/builds/s029_q397m3.sh best) =="
bash tools/builds/s029_q397m3.sh best || { rc=$?; echo "ABORT: build of the primary-line file without second stage (code $rc)" >&2; exit "$rc"; }

# 1. checksums
echo "== 1. checksums =="
check "$REF_BASE_SHA" "$REF_BASE"
if [ "$ORIENT" = repro ]; then check "$REF_BASE_OLD_SHA" "$REF_BASE_OLD"; check "$LEGACY_OUT_SHA" "$LEGACY_OUT"; fi
check "$STACK_BASE_SHA" "$STACK_BASE"
for d in 1 2 3; do check "${NOTES_SHA[$((d - 1))]}" "data/derived/sft/m3_d${d}_test_frozen_notes.jsonl"; done

# 2. blend file with β by the rule (the cross-check β 0.2 = blend file of step 0, character by character, is inside c110)
echo "== 2. blend file with β (c110) =="
if [ -f "$STACK" ]; then
  python -I eda/c110_q397_deploy.py stack --line best --m3 m3x3_full --replace-q397 q397m3 --arith-beta-select --stem "$STEM" --verify \
    || { rc=$?; echo "ABORT: $STACK is not the recomputation" >&2; exit "$rc"; }
else
  python -I eda/c110_q397_deploy.py stack --line best --m3 m3x3_full --replace-q397 q397m3 --arith-beta-select --stem "$STEM" --save \
    || { rc=$?; echo "ABORT: $STACK not built" >&2; exit "$rc"; }
fi
check "$STACK_SHA" "$STACK"

# 3. nested OOF reproduction (target values of the independent cross-check ±0.0005; label placebo; reference of guard (c))
echo "== 3. OOF reproduction (c118 oof) =="
python -I eda/c118_stage2.py oof --stack "$STEM" || { rc=$?; echo "ABORT: OOF reproduction (code $rc)" >&2; exit "$rc"; }
python -I - <<'PY' || exit 4
import json
r = json.load(open("artifacts/eda/private/stage2/c118_oof.json"))
bad = []
if not r["reproduction"]["ok"]:
    bad.append("reproduction missed")
for f, e in r["sets"]["main"].items():
    if e.get("prototype_rows_equal") is False:
        bad.append(f"prototype not bit-identical [{f}]")
for f, e in r["sets"].get("main+placebo", {}).items():
    if e["abstain_folds"] != 5 or e["abstain"]["vs_recipe_beta"]["units"] != 0:
        bad.append(f"label placebo does not abstain completely [{f}]")
if "main+placebo" not in r["sets"]:
    bad.append("label placebo missing")
print("  OOF reproduction: " + ("PASSED (prototype bit-identical, label placebo abstains 5/5, Δ 0)" if not bad else "; ".join(bad)))
raise SystemExit(4 if bad else 0)
PY

# 4. deployment file (null point (e) inside c118 fit: training features = features of the OOF recomputation, bit-identical)
echo "== 4. deployment file (c118 fit) =="
if [ -f "$STAGE2" ]; then
  python -I eda/c118_stage2.py fit --stack "$STEM" --verify || { rc=$?; echo "ABORT: $STAGE2 is not the recomputation" >&2; exit "$rc"; }
else
  python -I eda/c118_stage2.py fit --stack "$STEM" --save || { rc=$?; echo "ABORT: $STAGE2 not built" >&2; exit "$rc"; }
fi
check "$STAGE2_SHA" "$STAGE2"

# shared sources of the test side (as in tools/builds/s029_q397m3.sh)
[ "$ORIENT" = repro ] && { python -I tools/orient_list.py check --mode repro --spec "$V1R_TEST" || exit $?; }
SRC=(--test "$V1R_TEST" --fill "v1r=v1" --test "v1=v1te_a_t4/v1_test_a/test+v1te_b_t4/v1_test_b/test"
     --test "zs27q=zs27q_test0_rp/zs27q_test0/test+zs27q_test1_rp/zs27q_test1/test" --fill "think18nb=zs27q"
     --test "think18nb=zs27v_te_rp/think_te/test/think@derived/think_rows/test.npy,zs27v_n3_rp/think_nte/test/think@derived/think_rows/test_tb_q.npy"
     --proto "pr27xm=runs/pr27te_rp/proto_te/proto" --rank "pr27xm=runs/n1_rp/rank_test/test"
     --rank-only "m3x3_full=$RANK1,$RANK2,$RANK3"
     --rank-only "q397m3=$Q397M3_TEST" --rank-none-band-for "q397m3=0.27,0.35"
     --align 0.75 --align-min 3 --blank-rows derived/blank/test_rows.npy)
ms() { PYTHONDONTWRITEBYTECODE=1 python -I tools/make_submission.py "$@" 2>&1 | grep -vi '^warning' | grep -E "SECOND STAGE|note guard|written|ABORT|Traceback|Error|--stage2" | cut -c1-320; return "${PIPESTATUS[0]}"; }

TMP="$(mktemp -d "${TMPDIR:-/tmp}/s032_stage2.XXXXXX")"
BAK=""
cleanup() {
  rm -rf -- "${TMP:?}"
  rm -f -- "artifacts/test_s029_best_q397m3_null_a${RSUF}.npy" "artifacts/test_${STEM}_base${RSUF}.npy" "artifacts/test_${STEM}_null_b${RSUF}.npy"
  [ -n "$BAK" ] && rm -rf -- "$BAK"
  return 0
}
trap cleanup EXIT

# 5. copy null points (weight 0): the features are computed, the order stays
echo "== 5. copy null points =="
python -I eda/c118_stage2.py null --stack s029_best_q397m3 --out "$TMP/null_a.json" >/dev/null
python -I eda/c118_stage2.py null --stack "$STEM" --out "$TMP/null_b.json" >/dev/null
ms --name s029_best_q397m3 "${SRC[@]}" --out "$TMP/sub_null_a.csv" --suffix "_null_a${RSUF}" --stage2 "$TMP/null_a.json" --stage2-report "$TMP/guards_null_a.json" \
  || { echo "ABORT: null point (a) not computed" >&2; exit 4; }
check "$REF_BASE_SHA" "$TMP/sub_null_a.csv"
echo "  null point (a): weight 0 + β 0.2 -> $(basename "$REF_BASE") byte-identical (${REF_BASE_SHA:0:16}…)"
ms --name "$STEM" "${SRC[@]}" --out "$TMP/sub_base.csv" --suffix "_base${RSUF}" || { echo "ABORT: file without second stage not computed" >&2; exit 4; }
ms --name "$STEM" "${SRC[@]}" --out "$TMP/sub_null_b.csv" --suffix "_null_b${RSUF}" --stage2 "$TMP/null_b.json" --stage2-report "$TMP/guards_null_b.json" \
  || { echo "ABORT: null point (b) not computed" >&2; exit 4; }
cmp -s "$TMP/sub_base.csv" "$TMP/sub_null_b.csv" || { echo "ABORT: null point (b) violated — weight 0 changes the file" >&2; exit 4; }
cmp -s "artifacts/test_${STEM}_base${RSUF}.npy" "artifacts/test_${STEM}_null_b${RSUF}.npy" || { echo "ABORT: null point (b) violated — test matrix" >&2; exit 4; }
echo "  null point (b): weight 0 + β $(python -I -c "import json;print(json.load(open('$STACK'))['arith_beta'])") -> file without second stage byte-identical (CSV and test matrix; $(sha256sum "$TMP/sub_base.csv" | cut -c1-16)…)"

# 6. main file (guards of the second stage; guard (c) violated -> no file, exit 5)
echo "== 6. main file with second stage =="
if [ -f "$OUT" ]; then  # replace only by a byte-identical rebuild
  BAK="$(mktemp -d "${TMPDIR:-/tmp}/s032_stage2_backup.XXXXXX")"
  for F in "$OUT" "${OUT%.csv}.json" "artifacts/test_${STEM}${RSUF}.npy"; do [ -f "$F" ] && cp -p "$F" "$BAK/"; done
fi
rm -f "$GUARDS"
if ! ms --name "$STEM" "${SRC[@]}" --stage2 "$STAGE2" --stage2-report "$GUARDS" "${MAINARGS[@]}"; then
  if [ -f "$GUARDS" ] && python -I -c "import json,sys;sys.exit(1 if json.load(open('$GUARDS'))['guard_changed_top']['ok'] else 0)"; then
    echo "ABORT: guard (c) of the second stage violated (statistics in $GUARDS) — no file" >&2; exit 5
  fi
  echo "ABORT: make_submission with --stage2 failed" >&2; exit 4
fi
[ -f "$OUT" ] || { echo "ABORT: $OUT was not written" >&2; exit 4; }
if [ -n "$BAK" ] && [ -f "$BAK/$(basename "$OUT")" ]; then
  if ! cmp -s "$OUT" "$BAK/$(basename "$OUT")"; then
    for F in "$OUT" "${OUT%.csv}.json"; do [ -f "$BAK/$(basename "$F")" ] && cp -p "$BAK/$(basename "$F")" "$F"; done
    [ -f "$BAK/test_${STEM}${RSUF}.npy" ] && cp -p "$BAK/test_${STEM}${RSUF}.npy" "artifacts/test_${STEM}${RSUF}.npy"
    echo "ABORT: rebuild differs from the existing $OUT — backup restored, investigate the cause" >&2; exit 4
  fi
  echo "  existing $OUT rebuilt byte-identically"
fi
[ "$ORIENT" = repro ] && { python -I tools/orient_list.py verify-sub --mode repro "${OUT%.csv}.json" || exit $?; }

# 7. check of the file and comparison (aggregates only)
echo "== 7. check and comparison =="
check "$REF_BASE_SHA" "$REF_BASE"
python -I - "$OUT" "$REF_BASE" "$TMP/sub_base.csv" "$GUARDS" "$REF_BASE_LABEL" <<'PY'
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


new = read(sys.argv[1])
for path, label in ((sys.argv[2], sys.argv[5]), (sys.argv[3], "file without reranker (new β)")):
    ref = read(path)
    b = new.loc[ref.index]
    print(f"  against {label}: different top code in {(ref.str.split().str[0] != b.str.split().str[0]).mean():.4f} of the rows, "
          f"different first three (with order) in {(ref != b).mean():.4f}")
g = json.load(open(sys.argv[4]))
print(f"  form: {'FALLBACK (feature set without notes)' if g['fallback'] else 'MAIN FORM (main)'} — note guard {'held' if g['guard_notes']['ok'] else 'failed'}"
      + "".join(f"; failed {c['stat']} {c['test']:.4f} outside [{c['lo']:.4f}, {c['hi']:.4f}]" for c in g["guard_notes"]["checks"] if not c["ok"]))
gc, gn = g["guard_changed_top"], g["guard_novel_task_top5"]
print(f"  guard (c): new top code from the reranker {gc['test']:.4f} (OOF {gc['oof']:.4f}, band {gc['factor'][0] * gc['oof']:.4f}…{gc['factor'][1] * gc['oof']:.4f}) -> "
      f"{'ok' if gc['ok'] else 'VIOLATED'}; (b) \"new for the task\" among the first 5: test {gn['test']:.4f}, OOF {gn['oof']:.4f}; "
      f"first three changed test {g['share_top3_changed']:.4f}, OOF {float(g['share_top3_changed_oof'] or 0.0):.4f}")
PY
if [ "$ORIENT" = repro ]; then  # legacy-list file unchanged; difference from it only as aggregates (split by the rows whose list membership differs)
  check "$LEGACY_OUT_SHA" "$LEGACY_OUT"
  python -I tools/orient_list.py compare "$LEGACY_OUT" "$OUT" --label "reranker fallback form, first-version rotation list" --json "$PRIVF/compare_s033n_vs_s033.json"
fi
sha256sum "$STACK" "$STAGE2" "$OUT"
if [ "$ORIENT" = repro ]; then
  sha256sum "${OUT%.csv}.json" "artifacts/test_${STEM}${RSUF}.npy" "$GUARDS"
  echo "s032_stage2 best (ORIENT=repro): $OUT built (reranker fallback form, released rotation list)"
else
  echo "s032_stage2 best: $OUT built"
fi
