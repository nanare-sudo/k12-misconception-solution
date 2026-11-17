#!/usr/bin/env bash
# PRIMARY SUBMISSION A primary_m3x5h (five-draw rule, expected-value rule): the intermediate build primary_cols (tools/builds/primary_cols.sh:
# primary_m3x4 with the extra columns qfp) with ONE change — the M3 channel m3x5h is per row the mean of the draws that cover the row: d1, d2,
# d3, d4 on all rows, the fifth draw h (M3 with the photo at long side 2,048 px, seed 5; OOF m3ha_oof_rp/m3hb_oof_rp, test m3hte_a_rp/
# m3hte_b_rp) ONLY on the rows of the list h20 (character height ≤ 20 px; data/derived/charsize/{train,test}_h20_rows.npy). Blend (θ on log
# loss from all 1,500 rows, β rule) and second stage (chain s032_stage2 -> … -> primary_final in the recipe of primary_m3px27, then the
# columns qfp) refitted — as primary_m3x4 from primary_m3px27 and primary_cols from primary_m3x4. Production code eda/c166_primary_m3x5h.py,
# test side tools/make_submission.py (path "draw with row list", field rank.row_list_draws, additive). Order: blend -> margin alignment (one
# set) -> second stage (+ guard) -> REL-01. Rotation list ORIENT=repro.
#
#     bash tools/builds/primary_m3x5h.sh                  # complete; if the test side h is missing: exit 7 after all other steps (final build after fetching)
#     PRIMARY_M3X5H_REGRESSION=0 bash …                   # step 0 as checksums only (debugging; a deployment build requires step 0)
#     PRIMARY_M3X5H_MEASURE=short|full bash …             # step 6 on orig,11000,11001 (short) or on all 20 untouched meta-splits 11000–11019 (full)
#
# STEPS: 0 regression check: tools/builds/primary_cols.sh complete (inside it primary_m3x4.sh and the whole chain) byte-identical with the
#   additively extended make_submission; 1 inputs (checksums: stored builds primary_cols/primary_m3x4, row lists, all files of the OOF draw h);
#   2 channel m3x5h (c166 oof: = add5_sm of the evaluation bit-identical); 3 blend (c166 stack); 4 second stage, base (c166 fit-base: MM rows =
#   bank variant, sixths = evaluation) and field comparison with stage2_primary_m3x4.json; 5 columns qfp (c166 fit: sixths = evaluation add5_sm)
#   and field comparison with stage2_primary_cols.json; 6 comparison over meta-splits (c166 measure: self-contained recomputation of the CV
#   evidence; only with PRIMARY_M3X5H_MEASURE=short|full); 7 copy null point of the training side "empty list" (c166 copy: blend, chain, qfp =
#   primary_m3x4/primary_cols); 8 copy null point of the test side (empty list through the new path -> CSV and test matrix byte-identical to
#   primary_cols); 9 stand-in (test draw d4 on the test list as fifth draw, placeholder -> formally valid file, only counts against
#   primary_cols, deleted afterwards); 10 test side h present? (otherwise exit 7); 11 checksum of the test shards; 12 large-photo draw guard
#   (c166 h-guard, exit 5) and its positive controls (c166 h-controls, exit 6); 13 null model (file without second stage byte-identical), test
#   guard of the columns (c151 test-guard), positive controls of the columns (c151 controls); 14 main file (guard (c), form main_guard_i1_mm + 9
#   columns, M3 channel with five draws and h exactly on the test list); 15 check and comparison with primary_cols (aggregates only);
#   16 checksums (the first build prints them -> pin -> second build byte-identical).
# MODE OF THE CHECK STEPS: oof/stack/fit-base/fit run with --save = write missing files, require the same content for existing ones (never
#   overwrite a differing file). The sixths of the internal evaluation on the original split are pinned in eda/c166_primary_m3x5h.py (UNITS).
# EXIT: 2 usage; 3 input missing; 4 checksum/recomputation/null point; 5 test guard, form or guard (c) violated; 6 positive control without
#   effect; 7 test side h missing or incomplete (all steps before passed).
#
# Only aggregates and checksums are printed. Nothing is submitted from here.
set -euo pipefail
[ $# -eq 0 ] || { echo "usage: $0   (no arguments; PRIMARY_M3X5H_REGRESSION=base|0, PRIMARY_M3X5H_MEASURE=short|full)" >&2; exit 2; }
cd "$(dirname "$0")/../.."
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}" OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export ORIENT=repro
STEM="primary_m3x5h"; CH="m3x5h"; TAG="m3x5h"
LABEL="primary_m3x5h (primary_cols with the M3 channel m3x5h)"
# stored builds primary_cols / primary_m3x4 (the intermediate build of step 0 and its base)
COLS_FILE="artifacts/submissions/sub_primary_cols.csv"; COLS_FILE_SHA="53ec296976834959caea500217aa8e01904fbdc8be40a0542f63a462e4251926"
COLS_MAT="artifacts/test_primary_m3x4_cols.npy"; COLS_MAT_SHA="84cf58aac011166aa5c242f94d1b6c19a5e7f7d6892d60157185cdfb1148ec95"
COLS_STAGE2="artifacts/stage2_primary_cols.json"; COLS_STAGE2_SHA="102f98ce9c50c955753305a1795b808136d098c82c5fa1d25a0fde6f3e988a7f"  # written by eda/c151_primary_cols.py fit: pinned after the first build
M3X4_FILE="artifacts/submissions/sub_primary_m3x4.csv"; M3X4_FILE_SHA="76da786dfdbe6a6ddcbcc93070f385e18870cb5cdaea895d92104aa3565f9973"
M3X4_STACK="artifacts/stack_primary_m3x4.json"; M3X4_STACK_SHA="1e90852a68f53efd6dfd0a51730d6035e78345dafc5c0730b59a976de369420f"  # written by eda/c135_primary_m3x4.py stack: pinned after the first build
M3X4_STAGE2="artifacts/stage2_primary_m3x4.json"; M3X4_STAGE2_SHA="7259ce1446a85aa700448f6c283170e119c414b24b0aea7bd243d67b6090c58a"  # written by eda/c135_primary_m3x4.py fit: pinned after the first build
M3X4_OOF="artifacts/oof_m3x4_full.npy"; M3X4_OOF_SHA="96ad8725bbaa9d274376a0bb68549abc4051878171ebb572f7707962600399a6"
# row lists (tools/make_charsize_rows.py --max-height 20 --name h20) and all files of the OOF draw h (c166 digest --side train)
LIST_TRAIN="data/derived/charsize/train_h20_rows.npy"; LIST_TRAIN_SHA="11afce9af96fca09032153b566ee81725f466ab927e782fcdad4324b55d18e77"
LIST_TEST_REL="derived/charsize/test_h20_rows.npy"; LIST_TEST="data/$LIST_TEST_REL"; LIST_TEST_SHA="9b1a1837bc2cf399e2e221c49def4b4ae11d666b5346ecb1eddc076622c8340a"
LIST_META="data/derived/charsize/charsize_h20.meta.json"; LIST_META_SHA="4a48a02b00a53aaf022a48d3031ef2ad7cd8f1fc10052139cbb875c3431007df"
TRAIN_H_SHA="cffb5d0e8f69ad725d6ed1f6e9a331c395635818ac59e3a6734b646546baa004"  # 42 files: shards, generation, list, list meta
TEST_H_SHA="5023ddc9b3fe1679dc4e20bc3d2f5350206d3586963d6d8312d5396978f7832f"   # test shards h (c166 digest --side test): printed by the first complete build, then pinned here
# results (pinned after the first build)
CH_OOF="artifacts/oof_${CH}.npy"; CH_OOF_SHA="388546e508a603fc993bc52faf887e3831895cd57513887c3d21aed7bf78cc2f"
CH_META="artifacts/meta_${CH}.json"; CH_META_SHA="a2979b02d58572252575c202cd759d0f80b003824ed421d09945ce98ab9188f3"
STACK="artifacts/stack_${STEM}.json"; STACK_SHA="50b46374e9703131301bb771bf46c967a8f9b3b02fdf994b8e7f9bbb1545ed66"
STAGE2_BASE="artifacts/stage2_${STEM}_base.json"; STAGE2_BASE_SHA="4e3ad9d8c6310701ac214793c5cdd172e00f9828607567bbe376843d2a32030d"
STAGE2="artifacts/stage2_${STEM}.json"; STAGE2_SHA="572ff297dc04b7d0e4a9a28b477af74f532bcd742371717d10e172daad894d05"
OUT="artifacts/submissions/sub_${STEM}.csv"; OUT_SHA="4505c15950167650a66cdc60eb9c153eebfb9ed6192277bd283045f3797354e7"        # pinned after the first complete build (the submitted file)
SUFFIX="_${TAG}"; TESTMAT="artifacts/test_${STEM}${SUFFIX}.npy"; TESTMAT_SHA="12c72dccc5995a90f6f406355ec7526a87a8c742dd5402945f003a69c073424a"
PRIV="artifacts/eda/private/primary_m3x5h"
COPY_STAGE2="$PRIV/copy/stage2_m3x5hk_cols.json"
COPY_EMPTY_TEST="$PWD/$PRIV/copy/test_empty_rows.npy"
GUARDS="$PRIV/guards_${STEM}.json"
RANK1="runs/m3_fullte_rp/m3_test_rank/test"
RANK2="runs/m3d2te_rp/m3d2_test_rank/test"
RANK3="runs/m3d3te_rp/m3d3_test_rank/test"
RANK4="runs/m3d4te_rp/m3d4_test_rank/test"
H_TEST="runs/m3hte_a_rp/m3h_test_rank/test+runs/m3hte_b_rp/m3h_test_rank/test"
MID="m3px27xm"; MID_PROTO="runs/pr27te_rp/proto_te/proto"; MID_TEST_RANK="runs/m3px27te_rp/m3px27_test_rank/test"
Q397M3_TEST="runs/q397m3te_rp/q397m3_test_rank/test"
FIRST=0
check() { echo "$1  $2" | sha256sum -c --quiet || { echo "ABORT: $2 differs (checksum)" >&2; exit 4; }; }
pin() {  # deployment checksum: pinned -> check; empty (first build) -> print it, then enter it here
  if [ -n "$1" ]; then check "$1" "$2"; else FIRST=1; echo "  FIRST BUILD — checksum $(sha256sum "$2" | cut -d' ' -f1)  $2 (pin it in tools/builds/primary_m3x5h.sh)"; fi
}
c166() { python -I eda/c166_primary_m3x5h.py "$@"; }
c151() { python -I eda/c151_primary_cols.py "$@"; }
. tools/builds/orient_list.sh  # rotation list of the v1r test side: always repro here
orient_select
[ "$ORIENT" = repro ] || { echo "ABORT: $LABEL only with ORIENT=repro" >&2; exit 2; }
mkdir -p "$PRIV"
echo "== $LABEL: build script tools/builds/primary_m3x5h.sh =="

# 0. regression check of the stored builds (make_submission is extended additively: draw with row list only with the field rank.row_list_draws)
echo "== 0. regression check (PRIMARY_M3X5H_REGRESSION=${PRIMARY_M3X5H_REGRESSION:-base}: tools/builds/primary_cols.sh) =="
case "${PRIMARY_M3X5H_REGRESSION:-base}" in
  base) bash tools/builds/primary_cols.sh || { rc=$?; echo "ABORT: build path of primary_cols (code $rc)" >&2; exit "$rc"; } ;;
  0) echo "  skipped (checksums only; no deployment build)" ;;
  *) echo "usage: PRIMARY_M3X5H_REGRESSION=base|0" >&2; exit 2 ;;
esac
for pair in "$COLS_FILE_SHA $COLS_FILE" "$COLS_MAT_SHA $COLS_MAT" "$M3X4_FILE_SHA $M3X4_FILE" "$M3X4_OOF_SHA $M3X4_OOF"; do
  check ${pair% *} ${pair#* }
done
pin "$COLS_STAGE2_SHA" "$COLS_STAGE2"
pin "$M3X4_STACK_SHA" "$M3X4_STACK"
pin "$M3X4_STAGE2_SHA" "$M3X4_STAGE2"
echo "  stored builds unchanged: primary_cols (CSV, test matrix, deployment file), primary_m3x4 (CSV, blend, deployment file), m3x4_full"

TMP="$(mktemp -d "${TMPDIR:-/tmp}/primary_m3x5h.XXXXXX")"
BAK=""
cleanup() {
  python -I eda/c166_primary_m3x5h.py clean-copy >/dev/null 2>&1 || true
  rm -rf -- "${TMP:?}"
  rm -f -- "artifacts/test_${STEM}_copy_m3x5hk.npy" "artifacts/test_${STEM}_${TAG}base.npy" "artifacts/test_${STEM}_${TAG}null.npy" \
           "artifacts/test_${STEM}_${TAG}stv.npy"
  for c in q397_drop10 q397_unformed mech_zero q397_rotate fac_flip chan_rotate topic_unknown; do rm -f -- "artifacts/test_${STEM}_x3ctl_${c}.npy"; done
  [ -n "$BAK" ] && rm -rf -- "$BAK"
  return 0
}
trap cleanup EXIT

# 1. inputs
echo "== 1. inputs =="
check "$LIST_TRAIN_SHA" "$LIST_TRAIN"
check "$LIST_TEST_SHA" "$LIST_TEST"
check "$LIST_META_SHA" "$LIST_META"
got="$(c166 digest --side train)" || { echo "INPUT MISSING: ${got}" >&2; exit 3; }
[ "${got#* }" = "$TRAIN_H_SHA" ] || { echo "ABORT: files of the OOF draw h differ (checksum ${got#* } instead of $TRAIN_H_SHA)" >&2; exit 4; }
echo "  row lists (training 1,001, test 2,459 rows; count and checksum only) and all ${got%% *} files of the OOF draw h pinned"

# 2. channel m3x5h (mean of the draws that cover the row; = add5_sm of the evaluation bit-identical; outside the list = m3x4_full bit-identical)
echo "== 2. channel m3x5h (c166 oof) =="
c166 oof --save | cut -c1-320 || { rc=$?; echo "ABORT: c166 oof (code $rc)" >&2; exit "$rc"; }
pin "$CH_OOF_SHA" "$CH_OOF"
pin "$CH_META_SHA" "$CH_META"

# 3. blend (θ on log loss from all 1,500 rows, β rule; recipe of primary_m3px27 with m3x5h in place of m3x3_full, as primary_m3x4)
echo "== 3. blend (c166 stack) =="
c166 stack --save | cut -c1-320 || { rc=$?; echo "ABORT: c166 stack (code $rc)" >&2; exit "$rc"; }
pin "$STACK_SHA" "$STACK"

# 4. second stage, base (chain as primary_m3x4; MM rows = bank variant bit-identical, sixths = evaluation add5_sm) and field comparison with stage2_primary_m3x4.json
echo "== 4. second stage, base (c166 fit-base) =="
c166 fit-base --save | grep -vE "^\s+\[(main|fallback)\]" | cut -c1-320 || { rc=$?; echo "ABORT: c166 fit-base (code $rc)" >&2; exit "$rc"; }
pin "$STAGE2_BASE_SHA" "$STAGE2_BASE"
python -I - "$M3X4_STAGE2" "$STAGE2_BASE" "$STACK" <<'PY' || exit 4
import hashlib
import json
import sys

a, b = (json.load(open(p)) for p in sys.argv[1:3])
ren = lambda f: f.replace("m3x4_full_", "m3x5h_", 1) if f.startswith("m3x4_full_") else ("m3x5h" if f == "m3x4_full" else f)  # noqa: E731
bad = []
if list(b) != list(a):
    bad.append("order/set of the fields differs from stage2_primary_m3x4.json")
if b["stack"] != sys.argv[3] or b["stack_sha256"] != hashlib.sha256(open(sys.argv[3], "rb").read()).hexdigest():
    bad.append(f"blend file {b['stack']}")
if b["channels"] != [ren(c) for c in a["channels"]] or b["draws_channel"] != "m3x5h" or b["draws_first"] != 3:
    bad.append(f"channels {b['channels']} / draws_channel {b['draws_channel']} / draws_first {b['draws_first']}")
for k in ("align", "align_min", "K", "lambda", "null", "draw_sources", "draw_rank_source", "test_notes", "guard_rule", "order", "kind", "mech27"):
    if b[k] != a[k]:
        bad.append(f"field {k} differs")
for fs in ("main", "fallback"):
    ma, mb = a["models"][fs], b["models"][fs]
    if mb["features"] != [ren(f) for f in ma["features"]] or mb.get("feature_set") != ma.get("feature_set") or mb.get("protect_none") != ma.get("protect_none"):
        bad.append(f"model {fs}: features/form/guard")
m = b["models"]["main"]
if m.get("feature_set") != "main_guard_i1_mm" or m.get("protect_none") is not True or len(m["w"]) != 64 or len(b["models"]["fallback"]["w"]) != 40:
    bad.append(f"main model {m.get('feature_set')} / {len(m['w'])} weights; fallback {len(b['models']['fallback']['w'])}")
s = b["m3x4_build"]
if s.get("kind") != "deploy" or not (s.get("oof_rows") or {}).get("bit_identical") or s["oof_rows"].get("units") != [7584, 7542] \
        or not all(v["ok"] for v in s["null_point_i"].values()):
    bad.append("m3x4_build: MM rows not = evaluation add5_sm (7584/7542) or copy null point (i) violated")
print(f"  field comparison with stage2_primary_m3x4.json: same fields, channel name m3x4_full -> m3x5h in channels and features, draws_channel m3x5h "
      f"with draws_first 3, test notes, note guard, K, λ, mech27 and recipe fields equal; main model main_guard_i1_mm (64, guard on; OOF new top code "
      f"{m['oof_changed_top_share']:.4f}), fallback 40; MM sixths {s['oof_rows']['units']} = evaluation add5_sm" if not bad else "ABORT: " + "; ".join(bad))
raise SystemExit(4 if bad else 0)
PY

# 5. columns qfp (as primary_cols; sixths = evaluation add5_sm 7583/7564) and field comparison with stage2_primary_cols.json
echo "== 5. columns qfp (c166 fit) =="
c166 fit --save | cut -c1-320 || { rc=$?; echo "ABORT: c166 fit (code $rc)" >&2; exit "$rc"; }
pin "$STAGE2_SHA" "$STAGE2"
python -I - "$COLS_STAGE2" "$STAGE2" "$STAGE2_BASE" <<'PY' || exit 4
import hashlib
import json
import sys

a, b, e0 = (json.load(open(p)) for p in sys.argv[1:4])
bad = []
want = [k for k in a if k != "cols_build"] + ["m3x5h_build"]
if list(b) != want:
    bad.append("order/set of the fields differs from stage2_primary_cols.json (cols_build -> m3x5h_build)")
for k in e0:
    if k != "models" and b.get(k) != e0[k]:
        bad.append(f"field {k} differs from the base stage2_primary_m3x5h_base.json")
if b["models"]["fallback"] != e0["models"]["fallback"]:
    bad.append("fallback model differs from the base")
x, xa = b["extra_columns"], a["extra_columns"]
for k in ("decision", "module", "version", "variant", "names", "features", "definitions", "knowledge", "minor_codes", "guard_rule", "controls", "test"):
    if x.get(k) != xa.get(k):
        bad.append(f"extra_columns.{k} differs from primary_cols")
if x["train"]["files_sha256"] != xa["train"]["files_sha256"] or x["train"]["matrices_sha256"] != xa["train"]["matrices_sha256"]:
    bad.append("extra_columns.train (sources/matrices) differs from primary_cols")
m = b["models"]["main"]
if m.get("feature_set") != "main_guard_i1_mm" or m.get("protect_none") is not True or len(m["w"]) != 73 or m.get("features_sha256_mm") != e0["models"]["main"]["features_sha256"]:
    bad.append(f"main model {m.get('feature_set')} / {len(m['w'])} weights / MM features")
s = b["m3x5h_build"]
if s["measurement"]["units_line"] != [7584, 7542] or s["measurement"]["units_with"] != [7583, 7564] or not s.get("null_point_e") \
        or s["base"]["sha256"] != hashlib.sha256(open(sys.argv[3], "rb").read()).hexdigest():
    bad.append("m3x5h_build: sixths != evaluation add5_sm or null point (e) / base")
print(f"  field comparison: fields = base (+ extra_columns, m3x5h_build), fallback equal; columns as primary_cols (sources, definitions, test files, "
      f"training matrices, rule of the test guard); main model main_guard_i1_mm + 9 (73, guard on; OOF new top code {m['oof_changed_top_share']:.4f}); "
      f"nested MM {s['measurement']['units_line']}, with columns {s['measurement']['units_with']} = evaluation add5_sm" if not bad else "ABORT: " + "; ".join(bad))
raise SystemExit(4 if bad else 0)
PY

# 6. comparison over meta-splits (c166 measure: self-contained recomputation of the comparison of the internal evaluation, arms base and add5_sm)
echo "== 6. comparison over meta-splits (c166 measure) =="
if [ "${PRIMARY_M3X5H_MEASURE:-0}" != 0 ]; then
  if [ "${PRIMARY_M3X5H_MEASURE}" = full ]; then SEEDS="orig,11000-11019"; else SEEDS="orig,11000,11001"; fi
  c166 measure --seeds "$SEEDS" | cut -c1-320 || { rc=$?; echo "ABORT: comparison over meta-splits (code $rc)" >&2; exit "$rc"; }
else
  echo "  skipped (PRIMARY_M3X5H_MEASURE=short|full runs it; the chain checks the pinned sixths of the original split)"
fi

# shared sources of the test side (as primary_cols with ORIENT=repro), without the M3 channel
python -I tools/orient_list.py check --mode repro --spec "$V1R_TEST" --json "$PRIV/check_orient_${STEM}.json" || exit $?
BASE=(--test "$V1R_TEST" --fill "v1r=v1" --test "v1=v1te_a_t4/v1_test_a/test+v1te_b_t4/v1_test_b/test"
      --test "zs27q=zs27q_test0_rp/zs27q_test0/test+zs27q_test1_rp/zs27q_test1/test" --fill "think18nb=zs27q"
      --test "think18nb=zs27v_te_rp/think_te/test/think@derived/think_rows/test.npy,zs27v_n3_rp/think_nte/test/think@derived/think_rows/test_tb_q.npy"
      --proto "$MID=$MID_PROTO" --rank "$MID=$MID_TEST_RANK")
TAIL=(--rank-only "q397m3=$Q397M3_TEST" --rank-none-band-for "q397m3=0.27,0.35" --align 0.75 --align-min 3 --blank-rows derived/blank/test_rows.npy)
ms() { PYTHONDONTWRITEBYTECODE=1 python -I tools/make_submission.py "$@" 2>&1 | grep -vi '^warning' | grep -E "SECOND STAGE|note guard|extra columns|ROW LIST|MEAN of|written|ABORT|Traceback|Error|--stage2|--rank|--proto|CONTROL|channel m3x5h" | cut -c1-320; return "${PIPESTATUS[0]}"; }
compare() {  # file against primary_cols: format check and counts (aggregates only) -> JSON
  python -I - "$1" "$COLS_FILE" "$2" <<'PY'
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


def read(p):
    return pd.read_csv(p, dtype=str, keep_default_na=False).set_index("trace_id")["root_code"]


new, ref = read(sys.argv[1]), read(sys.argv[2])
b = new.loc[ref.index]
t_new, t_ref = b.str.split().str[0], ref.str.split().str[0]
s_new, s_ref = b.str.split().apply(frozenset), ref.str.split().apply(frozenset)
e = {"rows": int(len(ref)), "rows_other_top1": int((t_ref != t_new).sum()), "share_other_top1": float((t_ref != t_new).mean()),
     "rows_other_top3_ordered": int((ref != b).sum()), "share_other_top3_ordered": float((ref != b).mean()),
     "rows_other_top3_as_set": int((s_new != s_ref).sum()), "share_top_none_new": float((t_new == "NONE").mean()),
     "share_top_none_primary_cols": float((t_ref == "NONE").mean())}
print(f"  check_submission: {stats}")
print(f"  vs. primary_cols: other top code in {e['rows_other_top1']} of {e['rows']} rows ({e['share_other_top1']:.4f}), other first three (ordered) in "
      f"{e['rows_other_top3_ordered']} ({e['share_other_top3_ordered']:.4f}), as a set {e['rows_other_top3_as_set']}; NONE at the top "
      f"{e['share_top_none_primary_cols']:.4f} -> {e['share_top_none_new']:.4f}")
Path(sys.argv[3]).write_text(json.dumps({"check": stats, "vs_primary_cols": e}, indent=1, ensure_ascii=False) + "\n")
ok = stats.get("n_rows") == 3498 and stats.get("rows_with_3_codes") == 3498 and stats.get("rows_with_fewer_codes") == 0
raise SystemExit(0 if ok else 4)
PY
}

# 7. copy null point of the training side "empty list" (channel m3x5hk = m3x4_full bit-identical -> blend, chain and qfp = primary_m3x4/primary_cols)
echo "== 7. copy null point of the training side (c166 copy, empty list) =="
c166 copy --keep | grep -vE "^\s+\[(main|fallback)\]" | cut -c1-320 || { rc=$?; echo "ABORT: c166 copy (code $rc)" >&2; exit "$rc"; }

# 8. copy null point of the test side: empty test list through the new path (placeholder, stand-in d4 with an empty list) -> primary_cols byte-identical
echo "== 8. copy null point of the test side (empty list) =="
ms --name "${STEM}_copy" "${BASE[@]}" --rank-only "m3x5hk=$RANK1,$RANK2,$RANK3,$RANK4,$RANK4@$COPY_EMPTY_TEST" --rank-only-placeholder m3x5hk "${TAIL[@]}" \
  --out "$TMP/sub_copy.csv" --suffix "_m3x5hk" --stage2 "$COPY_STAGE2" --stage2-report "$TMP/guards_copy.json" \
  || { echo "ABORT: copy run of the test side not computed" >&2; exit 4; }
cmp -s "$TMP/sub_copy.csv" "$COLS_FILE" || { echo "ABORT: copy null point violated — CSV != primary_cols" >&2; exit 4; }
cmp -s "artifacts/test_${STEM}_copy_m3x5hk.npy" "$COLS_MAT" || { echo "ABORT: copy null point violated — test matrix != primary_cols" >&2; exit 4; }
python -I - "$TMP/sub_copy.json" <<'PY' || exit 4
import json
import sys

side = json.load(open(sys.argv[1]))
m = (side.get("sources") or {}).get("m3x5hk") or [{}]
m = m[0]
pd5 = (m.get("per_draw") or [{}] * 5)[-1]
ok = m.get("draws") == 5 and m.get("rows_by_draws") == {"4": 3498} and (m.get("row_list_draws") or [{}])[0].get("test_rows") == 0 and pd5.get("rows_list") == 0
print("  copy run through the new path: channel m3x5hk with 5 draws, draw 5 with an EMPTY test list (0 rows), test rows by draws {4: 3498}"
      if ok else f"ABORT: copy run not through the path \"draw with row list\" ({m.get('rows_by_draws')}, {pd5.get('rows_list')})")
raise SystemExit(0 if ok else 4)
PY
echo "  copy null point: CSV and test matrix byte-identical to primary_cols ($(sha256sum "$COLS_FILE" | cut -c1-16)… / $(sha256sum "$COLS_MAT" | cut -c1-16)…)"
rm -f -- "artifacts/test_${STEM}_copy_m3x5hk.npy"
python -I eda/c166_primary_m3x5h.py clean-copy >/dev/null

# 9. stand-in: test draw d4 on the test list as fifth draw (placeholder; never submit) -> formally valid, counts against primary_cols
echo "== 9. stand-in (d4 on the test list as fifth draw; placeholder) =="
ms --name "$STEM" "${BASE[@]}" --rank-only "$CH=$RANK1,$RANK2,$RANK3,$RANK4,$RANK4@$LIST_TEST_REL" --rank-only-placeholder "$CH" "${TAIL[@]}" \
  --out "$TMP/sub_stv.csv" --suffix "_${TAG}stv" --stage2 "$STAGE2" --stage2-report "$TMP/guards_stv.json" \
  || { echo "ABORT: stand-in run not computed" >&2; exit 4; }
compare "$TMP/sub_stv.csv" "$PRIV/compare_standin_d4.json" || { echo "ABORT: stand-in file formally invalid" >&2; exit 4; }
rm -f -- "$TMP/sub_stv.csv" "$TMP/sub_stv.json" "artifacts/test_${STEM}_${TAG}stv.npy"
echo "  stand-in file deleted (only counts in $PRIV/compare_standin_d4.json)"

# 10. test side h present and complete? (ranking files of both shards; the run's status.json is not in the offline package)
echo "== 10. test side h =="
rc=0; c166 h-guard --spec "$H_TEST" > "$TMP/h_guard_probe.log" 2>&1 || rc=$?
if [ "$rc" = 3 ]; then
  grep -E "MISSING|INCOMPLETE" "$TMP/h_guard_probe.log" | cut -c1-240 || true
  echo "TEST SIDE h MISSING OR INCOMPLETE: $H_TEST (runs m3hte_a_rp / m3hte_b_rp not finished or not completely fetched)."
  echo "  Everything else is built and checked (steps 0–9). Final build after fetching: bash tools/builds/primary_m3x5h.sh"
  echo "  (the first complete build prints the checksums of the test shards, of $OUT and $TESTMAT -> pin TEST_H_SHA/OUT_SHA/TESTMAT_SHA -> second build byte-identical)"
  exit 7
fi

# 11. checksum of the test shards h (the first complete build prints it -> pin)
echo "== 11. test shards h (checksum) =="
got="$(c166 digest --side test)" || { echo "ABORT: test shards h: ${got}" >&2; exit 4; }
if [ -z "$TEST_H_SHA" ]; then
  FIRST=1; echo "  FIRST BUILD — checksum of the test shards h (${got%% *} files) ${got#* } (pin it as TEST_H_SHA in tools/builds/primary_m3x5h.sh)"
else
  [ "${got#* }" = "$TEST_H_SHA" ] || { echo "ABORT: test shards h differ (checksum ${got#* } instead of $TEST_H_SHA)" >&2; exit 4; }
  echo "  test shards h (${got%% *} files: rankings, generation, merge jobs, test list) pinned"
fi

# 12. large-photo draw guard of the test draw h (fixed in advance) and its positive controls
echo "== 12. large-photo draw guard (c166 h-guard, h-controls) =="
rc=0; c166 h-guard --spec "$H_TEST" | cut -c1-320 || rc=$?
if [ "$rc" = 5 ]; then echo "ABORT: large-photo draw guard of the draw h violated (statistics above and in $PRIV/c166_h_guard.json) — no file, the draw is not dropped" >&2; exit 5; fi
[ "$rc" = 0 ] || { echo "ABORT: c166 h-guard (code $rc)" >&2; exit "$rc"; }
c166 h-controls --spec "$H_TEST" | cut -c1-320 || { rc=$?; echo "ABORT: positive controls of the large-photo draw guard (code $rc)" >&2; exit "$rc"; }

SRC=("${BASE[@]}" --rank-only "$CH=$RANK1,$RANK2,$RANK3,$RANK4,$H_TEST@$LIST_TEST_REL" "${TAIL[@]}")

# 13. null model (file without second stage byte-identical), test guard of the columns (c151 test-guard), positive controls of the columns (c151 controls)
echo "== 13. null model and test guard of the extra columns =="
c166 null --source "$STAGE2" --out "$TMP/null.json" >/dev/null
ms --name "$STEM" "${SRC[@]}" --out "$TMP/sub_base.csv" --suffix "_${TAG}base" || { echo "ABORT: file without second stage not computed" >&2; exit 4; }
rc=0; ms --name "$STEM" "${SRC[@]}" --out "$TMP/sub_null.csv" --suffix "_${TAG}null" --stage2 "$TMP/null.json" --stage2-report "$TMP/guards_null.json" || rc=$?
[ -f "$TMP/guards_null.json" ] || { echo "ABORT: null model not computed, no report (code $rc)" >&2; exit 4; }
cp -p "$TMP/guards_null.json" "$PRIV/guards_null_${STEM}.json"
rcg=0; c151 test-guard --report "$TMP/guards_null.json" | tail -3 || rcg=$?
if [ "$rcg" = 5 ]; then echo "ABORT: test guard of the extra columns violated (statistics in $PRIV/guards_null_${STEM}.json) — no file" >&2; exit 5; fi
[ "$rcg" = 0 ] || { echo "ABORT: c151 test-guard (code $rcg)" >&2; exit "$rcg"; }
[ "$rc" = 0 ] || { echo "ABORT: null model not computed (code $rc)" >&2; exit 4; }
cmp -s "$TMP/sub_base.csv" "$TMP/sub_null.csv" || { echo "ABORT: null model changes the file" >&2; exit 4; }
cmp -s "artifacts/test_${STEM}_${TAG}base.npy" "artifacts/test_${STEM}_${TAG}null.npy" || { echo "ABORT: null model changes the test matrix" >&2; exit 4; }
python -I - "$TMP/guards_null.json" <<'PY' || exit 4
import json
import sys

g = json.load(open(sys.argv[1]))
x, mm = g.get("extra_columns") or {}, g.get("mech27") or {}
bad = []
if not g["null"] or g["feature_set"] != "main" or g.get("model_feature_set") != "main_guard_i1_mm" or g["n_features"] != 73 or g.get("rows_protected") != 0:
    bad.append(f"null model {g['null']}, form {g['feature_set']}/{g.get('model_feature_set')}, {g['n_features']} features, guarded {g.get('rows_protected')}")
if g["guard_notes"].get("rule") != "widened" or not g["guard_notes"]["ok"]:
    bad.append("note guard")
if not x or not x["guard"]["ok"] or not x.get("used_by_model") or x.get("control") is not None:
    bad.append("extra columns: guard/use")
if not mm or mm.get("minor_codes") != 9:
    bad.append(f"mech27 {mm}")
print("  null model: file without second stage byte-identical (CSV and test matrix); 73 features (64 + 9 extra columns) computed; test guard per column held; "
      "note guard (widened rule) held" if not bad else "ABORT: " + "; ".join(bad))
raise SystemExit(4 if bad else 0)
PY
c151 controls --null "$TMP/null.json" --dir "$TMP/controls" --name "$STEM" -- "${SRC[@]}" | cut -c1-320 \
  || { rc=$?; echo "ABORT: positive controls of the test guard of the columns (code $rc)" >&2; exit "$rc"; }
cp -p "$TMP/controls/controls.json" "$PRIV/controls_${STEM}.json"

# 14. main file (form main with main_guard_i1_mm + 9 columns and the guard required; M3 channel with h exactly on the test list)
echo "== 14. main file =="
if [ -f "$OUT" ] || [ -f "$TESTMAT" ]; then  # only replace byte-identically
  BAK="$(mktemp -d "${TMPDIR:-/tmp}/primary_m3x5h_backup.XXXXXX")"
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
if ! ms --name "$STEM" "${SRC[@]}" --out "$OUT" --suffix "$SUFFIX" --stage2 "$STAGE2" --stage2-report "$GUARDS"; then
  restore
  if [ -f "$GUARDS" ] && python -I -c "import json,sys;g=json.load(open('$GUARDS'));sys.exit(1 if g['guard_changed_top']['ok'] and g['extra_columns']['guard']['ok'] else 0)"; then
    echo "ABORT: guard (c) or test guard of the extra columns violated (statistics in $GUARDS) — no file" >&2; exit 5
  fi
  echo "ABORT: make_submission with --stage2 failed" >&2; exit 4
fi
[ -f "$OUT" ] || { echo "ABORT: $OUT was not written" >&2; exit 4; }
python -I - "$GUARDS" "$PRIV/guards_null_${STEM}.json" "${OUT%.csv}.json" "$MID" "$LIST_TEST_REL" <<'PY' || { rc=$?; restore; exit "$rc"; }
import json
import sys

g, gnull, side = (json.load(open(p)) for p in sys.argv[1:4])
mid, lst = sys.argv[4], sys.argv[5]
gn, x = g["guard_notes"], g.get("extra_columns") or {}
if g["feature_set"] != "main" or g.get("model_feature_set") != "main_guard_i1_mm" or g.get("protect_none") is not True or not x.get("used_by_model"):
    print(f"ABORT: form {g['feature_set']} / {g.get('model_feature_set')} / guard {g.get('protect_none')} / extra columns {x.get('used_by_model')} — no file")
    raise SystemExit(5)
bad = []
if gn.get("rule") != "widened" or not gn["ok"] or g["null"] or g["n_features"] != 73 or not g["guard_changed_top"]["ok"] or not x["guard"]["ok"]:
    bad.append(f"rule {gn.get('rule')}, guard {gn['ok']}, null model {g['null']}, {g['n_features']} features, (c) {g['guard_changed_top']['ok']}, columns {x['guard']['ok']}")
if gn["test"] != gnull["guard_notes"]["test"] or g.get("mech27") != gnull.get("mech27") or x["test"] != gnull["extra_columns"]["test"]:
    bad.append("test aggregates of the main file != those of step 13")
src = side.get("sources") or {}
notes = [z for z in src.get(mid) or [] if isinstance(z, dict) and "notes" in z]
if len(notes) != 1 or notes[0].get("rank_compare") != "sft":
    bad.append(f"sources of the channel {mid} without note guard/comparison form sft ({len(notes)})")
m5 = (src.get("m3x5h") or [{}])[0]
rl = (m5.get("row_list_draws") or [{}])[0]
d5 = (m5.get("per_draw") or [{}] * 5)[-1]
if m5.get("draws") != 5 or len(str(m5.get("rank_only", "")).split(",")) != 5 or m5.get("rows_by_draws") != {"4": 1039, "5": 2459}:
    bad.append(f"channel m3x5h: {m5.get('draws')} draws, test rows by draws {m5.get('rows_by_draws')} (required {{4: 1039, 5: 2459}})")
if rl.get("draw") != 5 or rl.get("test_list") != lst or rl.get("test_rows") != 2459 or rl.get("placeholder") is not False:
    bad.append(f"draw with row list: {rl}")
if d5.get("rows_list") != 2459 or d5.get("rows_covered") != 2459 or d5.get("rows_covered_outside_list") != 0 or d5.get("settings_checked") is not True or len(d5.get("shards") or []) != 2:
    bad.append(f"test draw h: coverage {d5.get('rows_covered')} (outside {d5.get('rows_covered_outside_list')}), settings checked {d5.get('settings_checked')}")
if side.get("placeholder"):
    bad.append("file marked as a placeholder")
print(f"  main form main_guard_i1_mm + 9 columns (73 features, guard on: {g['rows_protected']} test rows keep the base order); guard (c) new top code "
      f"{g['guard_changed_top']['test']:.4f} (OOF {g['guard_changed_top']['oof']:.4f}) ok; test guard of the columns held; note guard = step 13; channel m3x5h: "
      f"5 draws, test rows by draws {{4: 1039, 5: 2459}}, h exactly on the test list (2 shards, settings checked)" if not bad else "ABORT: " + "; ".join(bad))
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

# 15. check of the file and comparison with primary_cols (aggregates only)
echo "== 15. check and comparison with primary_cols (aggregates only) =="
check "$COLS_FILE_SHA" "$COLS_FILE"
compare "$OUT" "$PRIV/compare_${STEM}.json" || { echo "ABORT: $OUT formally invalid" >&2; exit 4; }
python -I - "$GUARDS" <<'PY'
import json
import sys

g = json.load(open(sys.argv[1]))
gc = g["guard_changed_top"]
print(f"  guard (c): new top code {gc['test']:.4f} (OOF {gc['oof']:.4f}, band {gc['factor'][0] * gc['oof']:.4f}…{gc['factor'][1] * gc['oof']:.4f}) -> "
      f"{'ok' if gc['ok'] else 'VIOLATED'}; guarded {g['rows_protected']} test rows")
PY

# 16. checksums
echo "== 16. checksums =="
pin "$OUT_SHA" "$OUT"
pin "$TESTMAT_SHA" "$TESTMAT"
sha256sum "$CH_OOF" "$CH_META" "$STACK" "$STAGE2_BASE" "$STAGE2" "$OUT" "$TESTMAT" "${OUT%.csv}.json" "$GUARDS"
if [ "$FIRST" = 1 ]; then
  echo "primary_m3x5h: FIRST BUILD — enter the printed checksums and run the build path again (second build byte-identical)"
else
  echo "primary_m3x5h: $OUT built ($LABEL; nothing is submitted from here)"
fi
