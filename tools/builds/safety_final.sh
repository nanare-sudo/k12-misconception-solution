#!/usr/bin/env bash
# Safety line, FINAL CANDIDATE: the channels of the strictly inductive line with second stage (tools/builds/s036_safe_stage2.sh) with the
# MARGIN ALIGNMENT OVER THE TEST SET (0.75 for classes with at least 3 training rows, as one set as in the primary line) instead of the
# inductive one; second stage fitted anew in this form (eda/c123_safety_set.py: conditional logit K 5, λ 0.1, feature set fallback,
# 40 features, no notes, NONE guard; knowledge of the training rows only from the other folds; variant b adds the interactions I1); then
# REL-01. Channels, floors, β 0.4 and all test inputs as in the strictly inductive line (rotation list repro); no teacher channel, no
# 397B, no notes. Order on the test side: blend -> margin alignment (one set over the test rows) -> second stage -> NONE guard -> REL-01
# (tools/make_submission.py --align 0.75 --align-min 3 --stage2 artifacts/stage2_safety_final.json; align_mode "set" -> apply_stage2_safe).
#
#     bash tools/builds/safety_final.sh                       # deployment only if the safety-set rule is met, otherwise exit 5 after step 4
#     SAFETY_FINAL_PROBE=1 bash tools/builds/safety_final.sh  # if the rule is missed: PROBE of the same steps 5–8, only under
#                                                             #   artifacts/eda/private/safety_set/probe/ (never deployed, no file under submissions/)
#     SAFETY_FINAL_PROBE=1 SAFETY_FINAL_VARIANT=b bash …      # the same probe with variant b (fallback_i1 = fallback + interactions I1)
#     SAFETY_FINAL_SPLITS=1 …                                 # recomputes the 20 meta-splits (byte-identical result required)
#     OVERRIDE_CV_II=override_cv_condition_ii bash tools/builds/safety_final.sh   # DEPLOYMENT although the CV condition (ii) is missed:
#                                                             #   override only with the decision file decisions/override_cv_condition_ii.json
#                                                             #   (checksum pinned in eda/c123_safety_set.py); variant as chosen by (iii)
#
# STEPS: 0 build of the strictly inductive line with second stage complete (tools/builds/s036_safe_stage2.sh, inside it the line without
#   second stage) = regression check (fixed checksums); 1 c123 oof (copy null points a: weight 0 -> base, b: I1 columns 0 -> a; sixths
#   fixed; result file byte-identical); 2 meta-splits (c123 splits, fixed checksum); 3 class masses of the test side (c123 test-mass: as one
#   set and — control against an independent check — inductive); 4 safety-set rule (c123 rule with the test report; exit 5 = no
#   deployment); 5 deployment file (c123 fit; probe: fit --probe, private); 6 copy null point of the test side (null model -> file without
#   second stage byte-identical, CSV and test matrix); 7 main file (guard (c), class masses of the report = step 3, L1 ≤ 0.035);
#   8 comparison with the strictly inductive line with and without second stage (aggregates only); 9 checksums (deployment: pinned after
#   the first build). With variant b additionally 6b: I1 weights 0 with the 40 weights of a -> file of variant a byte-identical (CSV and
#   test matrix; = probe a); in 9 also deployment model = model of probe b (same file), if probe b is present. OVERRIDE: only the CV
#   condition of (ii); test masses (L1), null points, guard (c) and all checksums stay in force; the result file
#   safety_set/c123_rule_override_cv_condition_ii.json and the deployment file carry override_ii.
# EXIT: 2 usage; 3 input missing; 4 checksum/recomputation/null point/control; 5 pre-registered rule not met or guard (c) violated;
#   1 a check inside a called Python script that stops with a message.
#
# Only aggregates and checksums are printed. Nothing is submitted from here.
set -euo pipefail
[ $# -eq 0 ] || { echo "usage: $0   (no arguments; SAFETY_FINAL_PROBE=1, SAFETY_FINAL_SPLITS=1, OVERRIDE_CV_II=override_cv_condition_ii)" >&2; exit 2; }
cd "$(dirname "$0")/../.."
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}" OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export ORIENT=repro
NAME="v1r_zs27t_think18sb_pr27xm_m3x3_full_ar"
STACK="artifacts/stack_${NAME}.json"; STACK_SHA="7fb543ae5b7cb460a869411c1a97759d5594af66140d6922485b1b1ff421490c"
STAGE2_INDUCTIVE="artifacts/stage2_s036_safe.json"; STAGE2_INDUCTIVE_SHA="7acd67bc7602369b75838870f1e362a2b4065a7deb3dca515dc81a9dedaa145f"
INDUCTIVE_FILE="artifacts/submissions/sub_s036_safe_stage2.csv"; INDUCTIVE_FILE_SHA="ff45c2f72d21fd9d6b227fc47bad10b4b4092cd79008734da917e2ba3bcefd13"
INDUCTIVE_MAT="artifacts/test_${NAME}_ia_t4_s036.npy"; INDUCTIVE_MAT_SHA="7c348940c2e9b8a5d1e463165d5341e1a117494dbc3c739e558fcd20a1fe5e47"
BASE_FILE="artifacts/submissions/sub_s030n_safe.csv"; BASE_FILE_SHA="8cca7bbfe9b8c9dce47d6153d59712575bb7b8ad6dea8a286f7ef530df448b67"
PRIV="artifacts/eda/private/safety_set"
IOOF="$PRIV/c123_oof.json"; IOOF_SHA="ee8a2462438670d689db717a92f609962eaceba7aa5c08942348b5945b8d1ff1"
ISPL="$PRIV/c123_splits.json"; ISPL_SHA="7ed397f4dcd9ef57312f6f65a07beeb012b426f0b1fba830adf1ce1d0c8648e5"
MASS_SET="$PRIV/c123_test_mass_set.json"; MASS_SET_SHA="72e5b6ec0b075cc35079870dd55715b5a4f61a4ec93f5e50cf9dd8f2dff16766"
MASS_IND="$PRIV/c123_test_mass_inductive.json"; MASS_IND_SHA="27a8f44bf5f0fe6e39cce42e1c1a9297129078118b2672b08fc6e8f1a6fc9a6a"
A_STAGE2="$PRIV/probe/stage2_safety_a_probe.json"; A_STAGE2_SHA="49c37d1e5cb903e26adc5f7af4894ea3bb9ea32aa6ca1ae9f68a45c3b057b546"  # probe a (deployment file of variant a)
A_PROBE_CSV_SHA="94d11062ae7270d7708aea2cff2d25a1a920b98f9dd75e57244e3db1e9aa08ab"  # file of variant a (probe a)
B_STAGE2="$PRIV/probe/stage2_safety_b_probe.json"; B_PROBE_CSV="$PRIV/probe/sub_safety_b_probe.csv"  # probe b (compared with the deployment if present)
OOF_UNITS='{"a": {"fold": 7430, "pre": 7404, "set": 7391}, "b": {"fold": 7450, "pre": 7417, "set": 7395}}'  # c123 oof (gate + guard)
STAGE2="artifacts/stage2_safety_final.json"; STAGE2_SHA="9b6dfbed3ac97a6083e45bab68a52b55118c3b0e3a35dd58371bad47c6e8c81a"  # deployment build (variant b, with the override of (ii))
OUT="artifacts/submissions/sub_safety_final.csv"; OUT_SHA="736d87b3cb61494aeb6dcac29bc248a909138c6e85e3939f37a01180984e78f7"  # final safety candidate (= probe b)
SUFFIX="_safety_final"
RANK1="runs/m3_fullte_rp/m3_test_rank/test"
RANK2="runs/m3d2te_rp/m3d2_test_rank/test"
RANK3="runs/m3d3te_rp/m3d3_test_rank/test"
RANK_XM="runs/n1_rp/rank_test/test"
V1_T4="v1=v1te_a_t4/v1_test_a/test+v1te_b_t4/v1_test_b/test"
check() { echo "$1  $2" | sha256sum -c --quiet || { echo "ABORT: $2 differs (checksum)" >&2; exit 4; }; }
pinned() { check "$1" "$2"; }  # deployment checksums: pinned after the first deployment build, confirmed in a second build
. tools/builds/orient_list.sh  # rotation list of the v1r test side: always repro here
orient_select
[ "$ORIENT" = repro ] || { echo "ABORT: this build path only runs with ORIENT=repro" >&2; exit 2; }
OVR=()
if [ -n "${OVERRIDE_CV_II:-}" ]; then  # override ONLY of the CV condition of (ii)
  [ "$OVERRIDE_CV_II" = override_cv_condition_ii ] || { echo "ABORT: OVERRIDE_CV_II=$OVERRIDE_CV_II — only override_cv_condition_ii is admitted (decision file)" >&2; exit 2; }
  [ "${SAFETY_FINAL_PROBE:-0}" != 1 ] || { echo "ABORT: OVERRIDE_CV_II and SAFETY_FINAL_PROBE are mutually exclusive" >&2; exit 2; }
  OVR=(--override-ii "$OVERRIDE_CV_II")
fi
mkdir -p "$PRIV"

# 0. build of the strictly inductive line with second stage complete (inside it the line without second stage) — regression check
echo "== 0. strictly inductive line with second stage (tools/builds/s036_safe_stage2.sh) =="
bash tools/builds/s036_safe_stage2.sh || { rc=$?; echo "ABORT: build of the strictly inductive line with second stage (code $rc)" >&2; exit "$rc"; }
check "$STACK_SHA" "$STACK"
check "$STAGE2_INDUCTIVE_SHA" "$STAGE2_INDUCTIVE"
check "$INDUCTIVE_FILE_SHA" "$INDUCTIVE_FILE"
check "$INDUCTIVE_MAT_SHA" "$INDUCTIVE_MAT"
check "$BASE_FILE_SHA" "$BASE_FILE"

TMP="$(mktemp -d "${TMPDIR:-/tmp}/safety_final.XXXXXX")"
BAK=""
cleanup() {
  rm -rf -- "${TMP:?}"
  rm -f -- "artifacts/test_${NAME}_safebase.npy" "artifacts/test_${NAME}_safenull.npy" "artifacts/test_${NAME}_safeprobe.npy" \
           "artifacts/test_${NAME}_safemass_set.npy" "artifacts/test_${NAME}_safemass_inductive.npy" \
           "artifacts/test_${NAME}_safevara.npy" "artifacts/test_${NAME}_safei1zero.npy"
  [ -n "$BAK" ] && rm -rf -- "$BAK"  # the backup contains test identifiers: never leave it behind
  return 0
}
trap cleanup EXIT

# 1. nested reproduction (copy null points, sixths fixed)
echo "== 1. c123 oof =="
python -I eda/c123_safety_set.py oof | grep -E "^\s+\[|safety-set a|ABORT" | cut -c1-330 || { echo "ABORT: c123 oof" >&2; exit 4; }
python -I - "$IOOF" "$OOF_UNITS" <<'PY' || exit 4
import json
import sys

g, want = json.load(open(sys.argv[1])), json.loads(sys.argv[2])
bad = []
if not (g["null_point"]["a_zero_model"]["ok"] and g["null_point"]["b_i1_zero"]["ok"]):
    bad.append(f"copy null points violated {g['null_point']}")
got = {v: {f: g["sets"][v][f]["units_gated"] for f in ("fold", "pre", "set")} for v in ("a", "b")}
if got != want or g["placebo"]["n"] != 15 or g["inductive_units"] != 7410:
    bad.append(f"sixths {got} instead of {want} / placebos {g['placebo']['n']} / strictly inductive line {g['inductive_units']}")
print(f"  c123 oof: copy null points a (weight 0 -> base) and b (I1 0 -> a) bit-identical; sixths (gate + guard) {got}" if not bad else "ABORT: " + "; ".join(bad))
raise SystemExit(4 if bad else 0)
PY
check "$IOOF_SHA" "$IOOF"

# 2. meta-splits (b − a; fixed checksum, SAFETY_FINAL_SPLITS=1 recomputes and requires byte identity)
echo "== 2. meta-splits (c123 splits) =="
if [ "${SAFETY_FINAL_SPLITS:-0}" = 1 ] || [ ! -f "$ISPL" ]; then
  [ -f "$ISPL" ] && cp -p "$ISPL" "$TMP/splits_before.json"
  python -I eda/c123_safety_set.py splits --n 20 | grep -E "over|ABORT" | cut -c1-300 || { echo "ABORT: c123 splits" >&2; exit 4; }
  if [ -f "$TMP/splits_before.json" ] && ! cmp -s "$ISPL" "$TMP/splits_before.json"; then
    cp -p "$TMP/splits_before.json" "$ISPL"
    echo "ABORT: recomputation of the meta-splits differs from the existing $ISPL — backup restored" >&2; exit 4
  fi
fi
check "$ISPL_SHA" "$ISPL"

# shared sources of the test side (as in tools/builds/s036_safe_stage2.sh, without the alignment; ORIENT=repro)
python -I tools/orient_list.py check --mode repro --spec "$V1R_TEST" --json "$PRIV/check_orient_safety_final.json" || exit $?
BASE_SRC=(--test "$V1R_TEST" --fill "v1r=v1" --test "$V1_T4"
          --test "zs27t=px27_t4zt0/zs27t_test_t4/test+px27_t4zt1/zs27t_test_t4/test" --fill "think18sb=zs27t"
          --test "think18sb=zs27v_n3_rp/think_nte/test/think@derived/think_rows/test_tb_t.npy"
          --proto "pr27xm=runs/pr27te_rp/proto_te/proto" --rank "pr27xm=$RANK_XM"
          --rank-only "m3x3_full=$RANK1,$RANK2,$RANK3")
SRC=("${BASE_SRC[@]}" --align 0.75 --align-min 3 --blank-rows derived/blank/test_rows.npy)
ms() { PYTHONDONTWRITEBYTECODE=1 python -I tools/make_submission.py "$@" 2>&1 | grep -vi '^warning' | grep -E "SECOND STAGE|margin alignment as one set|written|ABORT|Traceback|Error|--stage2" | cut -c1-320; return "${PIPESTATUS[0]}"; }

# 3. class masses of the test side after the alignment (aggregates only; as one set and — control against an independent check — inductive)
echo "== 3. class masses of the test side (c123 test-mass) =="
for MODE_A in set inductive; do
  python -I eda/c123_safety_set.py test-mass --name "$NAME" --align-mode "$MODE_A" -- "${BASE_SRC[@]}" | grep -E "class masses|ABORT" | cut -c1-300 \
    || { echo "ABORT: test-mass $MODE_A" >&2; exit 4; }
done
check "$MASS_SET_SHA" "$MASS_SET"
check "$MASS_IND_SHA" "$MASS_IND"

# 4. safety-set rule (exit 5 = no deployment; with SAFETY_FINAL_PROBE=1 continue as a probe into the private folder only)
echo "== 4. safety-set rule (c123 rule) =="
MODE=deploy
rc=0
python -I eda/c123_safety_set.py rule --test-report "$MASS_SET" "${OVR[@]}" || rc=$?
if [ "$rc" != 0 ]; then
  if [ "$rc" = 5 ] && [ "${SAFETY_FINAL_PROBE:-0}" = 1 ]; then
    MODE=probe
    echo "  safety-set rule NOT met -> PROBE (SAFETY_FINAL_PROBE=1): steps 5–8 only under $PRIV/probe/, no deployment file, no file under submissions/"
  else
    echo "ABORT: safety-set rule not met — no deployment (code $rc)" >&2; exit "$rc"
  fi
fi
VAR="${SAFETY_FINAL_VARIANT:-a}"
case "$VAR" in a|b) ;; *) echo "ABORT: SAFETY_FINAL_VARIANT=$VAR (a or b)" >&2; exit 2 ;; esac
if [ "$MODE" = probe ]; then
  mkdir -p "$PRIV/probe"
  STAGE2="$PRIV/probe/stage2_safety_${VAR}_probe.json"
  OUT="$PRIV/probe/sub_safety_${VAR}_probe.csv"
  SUFFIX="_safeprobe"
  FIT=(--probe --variant "$VAR")
  TAG="probe$VAR"
else
  [ -z "${SAFETY_FINAL_VARIANT:-}" ] || { echo "ABORT: SAFETY_FINAL_VARIANT only with SAFETY_FINAL_PROBE (in a deployment the rule chooses the variant)" >&2; exit 2; }
  FIT=("${OVR[@]}")
  TAG=""
fi
TESTMAT="artifacts/test_${NAME}${SUFFIX}.npy"
GUARDS="$PRIV/guards_safety_final${TAG}.json"

# 5. deployment file (c123 fit; null point (e) against c123 oof)
echo "== 5. deployment file (c123 fit ${FIT[*]}) =="
if [ -f "$STAGE2" ]; then
  python -I eda/c123_safety_set.py fit "${FIT[@]}" --out "$STAGE2" --verify || { rc=$?; echo "ABORT: $STAGE2 is not the recomputation" >&2; exit "$rc"; }
else
  python -I eda/c123_safety_set.py fit "${FIT[@]}" --out "$STAGE2" --save || { rc=$?; echo "ABORT: $STAGE2 not built" >&2; exit "$rc"; }
fi
[ "$MODE" = deploy ] && pinned "$STAGE2_SHA" "$STAGE2"

# 6. copy null point of the test side: weight 0 -> file without second stage byte-identical (CSV and test matrix)
echo "== 6. copy null point (null model) =="
python -I eda/c123_safety_set.py null --source "$STAGE2" --out "$TMP/null.json" >/dev/null
ms --name "$NAME" "${SRC[@]}" --out "$TMP/sub_base.csv" --suffix "_safebase" || { echo "ABORT: file without second stage not computed" >&2; exit 4; }
ms --name "$NAME" "${SRC[@]}" --out "$TMP/sub_null.csv" --suffix "_safenull" --stage2 "$TMP/null.json" --stage2-report "$TMP/guards_null.json" \
  || { echo "ABORT: copy null point not computed" >&2; exit 4; }
cmp -s "$TMP/sub_base.csv" "$TMP/sub_null.csv" || { echo "ABORT: copy null point violated — weight 0 changes the file" >&2; exit 4; }
cmp -s "artifacts/test_${NAME}_safebase.npy" "artifacts/test_${NAME}_safenull.npy" || { echo "ABORT: copy null point violated — test matrix" >&2; exit 4; }
echo "  copy null point: weight 0 -> file without second stage byte-identical (CSV and test matrix; features of the deployment file computed)"

# 6b. variant b only: I1 weights 0 with the 40 weights of a -> file of variant a byte-identical (CSV and test matrix; = probe a)
VARIANT="$(python -I -c 'import json,sys; print(json.load(open(sys.argv[1])).get("variant", "a"))' "$STAGE2")"
if [ "$VARIANT" = b ]; then
  echo "== 6b. copy null point: I1 weights 0 with the weights of a -> variant a =="
  mkdir -p "$PRIV/probe"
  if [ -f "$A_STAGE2" ]; then
    python -I eda/c123_safety_set.py fit --probe --variant a --out "$A_STAGE2" --verify >/dev/null \
      || { rc=$?; echo "ABORT: $A_STAGE2 is not the recomputation" >&2; exit "$rc"; }
  else
    python -I eda/c123_safety_set.py fit --probe --variant a --out "$A_STAGE2" --save >/dev/null || { rc=$?; echo "ABORT: $A_STAGE2 not built" >&2; exit "$rc"; }
  fi
  check "$A_STAGE2_SHA" "$A_STAGE2"
  python -I - "$STAGE2" "$A_STAGE2" "$TMP/stage2_i1zero.json" <<'PY' || exit 4
import json
import sys

b, a = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
mb, ma = b["model"], a["model"]
if b.get("feature_set") != "fallback_i1" or a.get("feature_set") != "fallback" or mb["features"][:40] != ma["features"] or len(mb["features"]) != 55:
    raise SystemExit("ABORT: features of the variants a/b do not match")
s = json.loads(json.dumps(b))
s["model"].update({"w": ma["w"] + [0.0] * 15, "mu": ma["mu"] + [0.0] * 15, "sd": ma["sd"] + [1.0] * 15,
                   "oof_changed_top_share": ma["oof_changed_top_share"], "oof_changed_top3_share": ma["oof_changed_top3_share"]})
open(sys.argv[3], "w").write(json.dumps(s, indent=1, ensure_ascii=False) + "\n")
PY
  ms --name "$NAME" "${SRC[@]}" --out "$TMP/sub_vara.csv" --suffix "_safevara" --stage2 "$A_STAGE2" --stage2-report "$TMP/guards_vara.json" \
    || { echo "ABORT: file of variant a not computed" >&2; exit 4; }
  ms --name "$NAME" "${SRC[@]}" --out "$TMP/sub_i1zero.csv" --suffix "_safei1zero" --stage2 "$TMP/stage2_i1zero.json" --stage2-report "$TMP/guards_i1zero.json" \
    || { echo "ABORT: null point (I1 0) not computed" >&2; exit 4; }
  cmp -s "$TMP/sub_vara.csv" "$TMP/sub_i1zero.csv" || { echo "ABORT: null point violated — I1 weights 0 do not give the file of variant a" >&2; exit 4; }
  cmp -s "artifacts/test_${NAME}_safevara.npy" "artifacts/test_${NAME}_safei1zero.npy" || { echo "ABORT: null point violated — test matrix (I1 0) != a" >&2; exit 4; }
  check "$A_PROBE_CSV_SHA" "$TMP/sub_vara.csv"
  echo "  copy null point: I1 weights 0 (55 features with the 40 weights of a) -> file of variant a byte-identical (CSV and test matrix; = probe a ${A_PROBE_CSV_SHA:0:16}…)"
fi

# 7. main file (deployment) or probe (private)
echo "== 7. main file ($MODE) =="
if [ "$MODE" = deploy ] && { [ -f "$OUT" ] || [ -f "$TESTMAT" ]; }; then  # only replace byte-identically
  BAK="$(mktemp -d "${TMPDIR:-/tmp}/safety_final_backup.XXXXXX")"
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
  if [ -f "$GUARDS" ] && python -I -c "import json,sys;sys.exit(1 if json.load(open('$GUARDS'))['guard_changed_top']['ok'] else 0)"; then
    echo "ABORT: guard (c) of the second stage violated (statistics in $GUARDS) — no file" >&2; exit 5
  fi
  echo "ABORT: make_submission with --stage2 failed" >&2; exit 4
fi
[ -f "$OUT" ] || { echo "ABORT: $OUT was not written" >&2; exit 4; }
python -I - "$GUARDS" "$MASS_SET" "$STAGE2" <<'PY' || { rc=$?; restore; exit "$rc"; }
import json
import sys

g, m, s2 = (json.load(open(f)) for f in sys.argv[1:4])
cm = g.get("class_mass") or {}
bad = []
if g.get("align_mode") != "set" or g["feature_set"] != s2["feature_set"] or g["feature_set"] not in ("fallback", "fallback_i1") \
        or g.get("protect_none") is not True or g["null"] or g["n_features"] != len(s2["model"]["features"]):
    bad.append(f"form {g.get('align_mode')} / {g['feature_set']} / guard {g.get('protect_none')} / null model {g['null']} / {g['n_features']} features")
if not g["guard_changed_top"]["ok"]:
    bad.append("guard (c) violated")
if abs(float(cm.get("l1_after", -1)) - float(m["l1_after"])) > 1e-12 or not float(cm["l1_after"]) <= 0.035:
    bad.append(f"class masses of the report L1 {cm.get('l1_after')} != step 3 ({m['l1_after']}) or above 0.035")
print(f"  form set / {g['feature_set']} ({g['n_features']}) / guard on: {g['rows_protected']} test rows keep the base order; guard (c) new top code "
      f"{g['guard_changed_top']['test']:.4f} (OOF {g['guard_changed_top']['oof']:.4f}) ok; class masses L1 {cm['l1_after']:.4f} = step 3"
      if not bad else "ABORT: " + "; ".join(bad))
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

# 8. check and comparison with the strictly inductive line with and without second stage (aggregates only)
echo "== 8. check and comparison (aggregates only) =="
check "$INDUCTIVE_FILE_SHA" "$INDUCTIVE_FILE"
check "$BASE_FILE_SHA" "$BASE_FILE"
python -I - "$OUT" "$INDUCTIVE_FILE" "$BASE_FILE" "$TMP/sub_base.csv" "$PRIV/compare_safety_final${TAG}.json" <<'PY' || exit 4
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
rep = {"check": stats, "vs": {}}
gen = {"ANS-01", "CAL-01", "WOP-01", "FRM-01", "OTH-00"}
for path, label in ((sys.argv[2], "strictly inductive line with second stage"), (sys.argv[3], "strictly inductive line without second stage"),
                    (sys.argv[4], "set alignment without second stage")):
    ref = read(path)
    b = new.loc[ref.index]
    t_new, t_ref = b.str.split().str[0], ref.str.split().str[0]
    s_new, s_ref = b.str.split().apply(frozenset), ref.str.split().apply(frozenset)
    e = {"share_other_top1": float((t_ref != t_new).mean()), "rows_other_top1": int((t_ref != t_new).sum()),
         "share_other_top3_ordered": float((ref != b).mean()), "share_other_top3_as_set": float((s_new != s_ref).mean()),
         "share_top_none_new": float((t_new == "NONE").mean()), "share_top_none_ref": float((t_ref == "NONE").mean()),
         "share_top_general_new": float(t_new.isin(gen).mean()), "share_top_general_ref": float(t_ref.isin(gen).mean())}
    rep["vs"][label] = e
    print(f"  vs. {label}: other top code in {e['share_other_top1']:.4f} of the rows ({e['rows_other_top1']}), other first three (ordered) in "
          f"{e['share_other_top3_ordered']:.4f}, as a set {e['share_other_top3_as_set']:.4f}; NONE at the top {e['share_top_none_ref']:.4f} -> "
          f"{e['share_top_none_new']:.4f}, general codes {e['share_top_general_ref']:.4f} -> {e['share_top_general_new']:.4f}")
Path(sys.argv[5]).write_text(json.dumps(rep, indent=1, ensure_ascii=False) + "\n")
PY

# 9. checksums
echo "== 9. checksums ($MODE) =="
if [ "$MODE" = deploy ]; then
  pinned "$OUT_SHA" "$OUT"
  if [ "$VARIANT" = b ] && [ -f "$B_STAGE2" ] && [ -f "$B_PROBE_CSV" ]; then  # deployment = probe b (same model, same file)
    python -I -c 'import json,sys; a,b=(json.load(open(f)) for f in sys.argv[1:3]); sys.exit(0 if a["model"]==b["model"] and a["feature_set"]==b["feature_set"] else 1)' \
      "$STAGE2" "$B_STAGE2" || { echo "ABORT: deployment model != model of probe b" >&2; exit 4; }
    cmp -s "$OUT" "$B_PROBE_CSV" || { echo "ABORT: deployment file != file of probe b" >&2; exit 4; }
    echo "  deployment model and file = probe b (byte-identical)"
  fi
  sha256sum "$STACK" "$STAGE2_INDUCTIVE" "$STAGE2" "$INDUCTIVE_FILE" "$BASE_FILE" "$OUT" "${OUT%.csv}.json" "$TESTMAT" "$GUARDS" "$IOOF" "$ISPL" "$MASS_SET"
  echo "safety_final: $OUT built (final safety candidate; nothing is submitted from here)"
else
  sha256sum "$STAGE2" "$OUT" "$TESTMAT" "$GUARDS"
  rm -f -- "$TESTMAT"  # probe: do not leave the test matrix behind
  echo "safety_final: PROBE under $PRIV/probe/ (safety-set rule not met — no candidate, never submit)"
fi
