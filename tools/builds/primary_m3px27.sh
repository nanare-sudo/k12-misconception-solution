#!/usr/bin/env bash
# PRIMARY CANDIDATE primary_m3px27 (m3px27 rule: expected-value rule DEPLOY): the final primary form (tools/builds/primary_final.sh: blend
# s032_stage2 + second stage main_guard_i1_mm with the NONE guard under the widened rule) with the mean channel m3px27xm = ½(pr27 + m3px27) in
# place of pr27xm = ½(pr27 + px27): named pr27 codes (the same draw as before) and the exact ranking of the fine-tuned M3 behind the same
# frozen notes of the first pr27 pass behind which px27 ranked (OOF: five fold adapters, run m3px27_oof_rp; test: full fit
# m3_fullte_rp/m3_train_full, run m3px27te_rp). Blend refitted (eda/c133_primary_m3px27.py stack: θ on log loss from all 1,500 rows,
# β rule as in s032_stage2), second stage refitted (c133 fit: chain main form -> widened rule -> guard form -> I1 form -> final form with the
# swapped channel; the deployment form of the training rows is the standard cross-fit of the reproduction, because fold 1 chooses β 0.7
# instead of 0.6 there — for the final primary form it is bit-identical to the old form). Test side through tools/make_submission.py (field
# rank_mean.rank_compare "sft", note guard rank_mean.notes). Order unchanged: blend -> margin alignment (one set) -> second stage (+ guard)
# -> REL-01. Rotation list ORIENT=repro.
#
#     bash tools/builds/primary_m3px27.sh                      # complete; if the test side is missing: exit 3 after the copy null point, with the command for the final build
#     PRIMARY_M3PX27_REGRESSION=full …                         # step 0 with the build of the safety_ndr form (inside it safety_nd, safety_final with the override decision file, primary_final) instead of primary_final alone
#     PRIMARY_M3PX27_REGRESSION=0 …                            # step 0 as checksums only (debugging; a deployment build requires step 0)
#
# STEPS: 0 regression check: build of the final primary form complete (primary_final.sh, inside it primary_i1, the guard form, s034n, s033n,
#   s032n) byte-identical, checksums of safety_nd and safety_ndr; 1 inputs (checksums; oof_m3px27 recomputed from the five shards: c132
#   --oof-verify); 2 mean channel (c133 oof); 3 blend (c133 stack); 4 second stage (c133 fit: chain, MM rows = rows of the expected-value
#   rule computation of c132 bit-identical, sensitivity with fixed β) and field comparison with stage2_primary_final.json; 5 copy null point
#   of the training side (c133 copy: m3px27 := px27 -> blend, chain rows and stage2_s032 … primary_final bit-identical); 6 copy null point of
#   the test side (m3px27 test := px27 test -> CSV and test matrix of the final primary form byte-identical); 7 test guard (c133 test-guard:
#   3,498/3,498 exact, "of which without a finding 3498", NONE band, settings, note guard, training regime; same top code against the px27
#   test ranking as an aggregate); 8 null model -> file without second stage byte-identical; 9 main file (guard (c), form main_guard_i1_mm
#   required); 10 check and comparison with the final primary form (aggregates only: number of changed top codes); 11 sensitivity with
#   fixed β (aggregates only, private); 12 checksums (the first build prints them -> pin OUT_SHA/TESTMAT_SHA here -> second build
#   byte-identical).
# EXIT: 2 usage; 3 input missing (also: test side not fetched yet); 4 checksum/recomputation/null point; 5 test guard, form or guard (c)
#   violated.
#
# Only aggregates and checksums are printed. Nothing is submitted from here.
set -euo pipefail
[ $# -eq 0 ] || { echo "usage: $0   (no arguments; PRIMARY_M3PX27_REGRESSION=primary_final|full|0)" >&2; exit 2; }
cd "$(dirname "$0")/../.."
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}" OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export ORIENT=repro
STEM="primary_m3px27"
STACK="artifacts/stack_${STEM}.json"; STACK_SHA="ebdd585e4650922d59972b3a84eb85c44af4f2dc469dd723ef2100ab7d797a3c"  # c133 stack
REF_STEM="s032_stage2"
REF_STACK="artifacts/stack_${REF_STEM}.json"; REF_STACK_SHA="d278bbedf7fb4d7564e584a55d45959b826748dfe937d8becf1e0818007281eb"
FINAL_FILE="artifacts/submissions/sub_primary_final.csv"; FINAL_FILE_SHA="d9c68c460c6da06d4975513911e11c4541d70d07b78f132566a3e43e7ad60b24"
FINAL_MAT="artifacts/test_${REF_STEM}_primary_final.npy"; FINAL_MAT_SHA="ba3d9785ef502fc8688d8760873a7313adf5adfead9b3c9b630477c6cae05e47"
STAGE2_FINAL="artifacts/stage2_primary_final.json"; STAGE2_FINAL_SHA="40481bc4c83d0ce41b69ebfd10b0d1aaea0a8156c693f1ee7fc89ceb99674604"
I1_FILE="artifacts/submissions/sub_primary_i1.csv"; I1_FILE_SHA="60b63d69bcdd5970cafaf4d5645440d8d375953871d6dd688ae5091b25256bee"
GUARD_FILE="artifacts/submissions/sub_s035_stage2_guard.csv"; GUARD_FILE_SHA="d4dce1aaa5cfbd1fd5668a9e7073a47734c0c5aacda7b7c2020255005a81f360"
FB_FILE="artifacts/submissions/sub_s033n_stage2.csv"; FB_FILE_SHA="76aba64fbc431b9ad403a5dec9fb2cefdb7fe2e76260149c96176459c4cd8dfe"
SAFE="v1r_zs27t_think18sb_pr27xm_m3x3_full_ar"
ND_FILE="artifacts/submissions/sub_safety_nd.csv"; ND_FILE_SHA="561a3c45ca3579763c44780b1f0cb009a6b75d5c730d4cbe42c4d11ce7e45025"
ND_MAT="artifacts/test_${SAFE}_safety_nd.npy"; ND_MAT_SHA="32aac4537603992507321a65072e71cde1db95020322c0c7ae7183ec2d45da07"
STAGE2_ND="artifacts/stage2_safety_nd.json"; STAGE2_ND_SHA="50217a6c48f16bf35afb97fd755611eba85c43fa81187d0d00a91dbb31cddbee"
NDR_FILE="artifacts/submissions/sub_safety_ndr.csv"; NDR_FILE_SHA="b8f17ce3e5fe1ed6dbf22febf9b153ec7c23870af123fa60d4342979d67270f6"
NDR_MAT="artifacts/test_${SAFE}_safety_ndr.npy"; NDR_MAT_SHA="abd63dfae32230ddd5ec5bba9e3ddcde91528c89d3fed4cf333da6f09b2c5101"
STAGE2_NDR="artifacts/stage2_safety_ndr.json"; STAGE2_NDR_SHA="bfd8f197fccb3ad9cc623fe2d45930151ac0debf40be3eb5cbafb953c79b2927"
# inputs (notes tools/make_m3px27_notes.py, channel c132 --oof-save, evaluation c132 = verdict DEPLOY)
RANK_OOF="artifacts/oof_m3px27.npy"; RANK_OOF_SHA="a0a9660f6f20099ff1019702a11d881eb4c85a3ce089a645ccbc0ac94428986f"
RANK_META="artifacts/meta_m3px27.json"; RANK_META_SHA="1f872ace4f3950622deadb90403551aa8275b14b30d3bd25acf003feeb79445e"
C132="artifacts/eda/private/m3px27_eval/c132_m3px27_eval.json"; C132_SHA="b3b9c7b87de3d7d25f017369c19e68aec219516a1dd75f9dfcf9455a7bc30ce0"
C132_ROWS="artifacts/eda/private/m3px27_eval/c132_m3px27_eval_rows.npz"; C132_ROWS_SHA="cf4af62a74cd3d315d7a0e0c3eefe8f3b8151f9db3d36cd4d2855ee40b5f0fc5"
NOTES_TR="data/derived/sft_m3px27/m3px27_train_notes.jsonl"; NOTES_TR_SHA="907bdf35fa10ac06fb07eebef5f441d5a3eeae186193ae561ea451a457457816"
NOTES_TE="data/derived/sft_m3px27te/m3px27_test_notes.jsonl"; NOTES_TE_SHA="dddc7ce63816a60459ef7610484e141574ae47411f2ed1a5de932fc89ae02ab7"
PR27_TE_NOTES="data/derived/proto_notes_test/test_pr27.jsonl"; PR27_TE_NOTES_SHA="1b459a40278b814922bfedbff9867028f9e0e56e39ca54b6c7b471c42a833f3a"
OOF_PR27="artifacts/oof_pr27.npy"; OOF_PR27_SHA="c3a5cd851579a4168989658b198c211fecb080bad544f2ae7e204719b1527426"
OOF_PR27X="artifacts/oof_pr27x.npy"; OOF_PR27X_SHA="9f84a8ea566f4c6ddbe77f2601cb60f0d29cfb9eea965e561147c883da9abe1f"
OOF_PR27XM="artifacts/oof_pr27xm.npy"; OOF_PR27XM_SHA="bf3d1b58abfc57eb64139418ed7646f45ba9f879791661658035171b9a2f0d03"
TEST_ANSWERS="artifacts/runs/pr27te_rp/pr27te_rp/proto_te/proto_raw.jsonl"; TEST_ANSWERS_SHA="3e9b05623a0dbb7eb7bb9b211b662a93af3b6c85a096c70523a62dc1e35eae62"
# results (pinned after the first build)
CH_OOF="artifacts/oof_m3px27xm.npy"; CH_OOF_SHA="47e128eed7865deec8a5818cdbd514ed29246dec79a56f7da59d86ed3897cfe8"
CH_META="artifacts/meta_m3px27xm.json"; CH_META_SHA="35cd59551ee04c05567dc2a52412553730d5b74e51a5268f05685de345237f77"
STAGE2="artifacts/stage2_primary_m3px27.json"; STAGE2_SHA="9321d597b175d672fe5b31bf284ce984fbce149cb2e2cd8941a31aa8070d2500"
PRIV="artifacts/eda/private/primary_m3px27"
SENS="$PRIV/sens/stage2_primary_m3px27_fixbeta.json"; SENS_SHA="ccb4447dd5775c3443fa8f196637fafc3ee02094ca2a92a9f85c0881be0e125c"
COPY_STAGE2="$PRIV/copy/stage2_primary_m3px27_copy_e.json"
TEST_RUN="m3px27te_rp"
TEST_STATUS="artifacts/runs/${TEST_RUN}/${TEST_RUN}/m3px27_test_rank/test_proto_rank.done.npy"  # ranking of the test side (in the package; the run's status.json is not)
TEST_RANK="runs/${TEST_RUN}/m3px27_test_rank/test"
OUT="artifacts/submissions/sub_${STEM}.csv"; OUT_SHA="23101e74aa530b9c280087442b15f72bdf25a0c4627f0207e92c648e83a2176a"  # candidate primary_m3px27 (first build with the real test side; pinned)
SUFFIX="_primary_m3px27"
TESTMAT="artifacts/test_${STEM}${SUFFIX}.npy"; TESTMAT_SHA="f3988d9c18c0d2485d085f7b4ab160f368f8da13b381a6bab48115e7b11446ff"  # likewise
GUARDS="$PRIV/guards_primary_m3px27.json"
RANK1="runs/m3_fullte_rp/m3_test_rank/test"
RANK2="runs/m3d2te_rp/m3d2_test_rank/test"
RANK3="runs/m3d3te_rp/m3d3_test_rank/test"
Q397M3_TEST="runs/q397m3te_rp/q397m3_test_rank/test"
FIRST=0
check() { echo "$1  $2" | sha256sum -c --quiet || { echo "ABORT: $2 differs (checksum)" >&2; exit 4; }; }
pin() {  # deployment checksum: pinned -> check; empty (first build) -> print it, then enter it here
  if [ -n "$1" ]; then check "$1" "$2"; else FIRST=1; echo "  FIRST BUILD — checksum $(sha256sum "$2" | cut -d' ' -f1)  $2 (pin it in tools/builds/primary_m3px27.sh)"; fi
}
. tools/builds/orient_list.sh  # rotation list of the v1r test side: always repro here
orient_select
[ "$ORIENT" = repro ] || { echo "ABORT: this build path only runs with ORIENT=repro" >&2; exit 2; }
mkdir -p "$PRIV"

# 0. regression check of the stored builds (the second stage and make_submission are extended additively — existing build paths stay byte-identical)
echo "== 0. regression check (PRIMARY_M3PX27_REGRESSION=${PRIMARY_M3PX27_REGRESSION:-primary_final}) =="
case "${PRIMARY_M3PX27_REGRESSION:-primary_final}" in
  primary_final) bash tools/builds/primary_final.sh || { rc=$?; echo "ABORT: build of the final primary form (code $rc)" >&2; exit "$rc"; } ;;
  full) bash tools/builds/safety_ndr.sh || { rc=$?; echo "ABORT: build of the safety_ndr form (code $rc)" >&2; exit "$rc"; } ;;
  0) echo "  skipped (checksums only; no deployment build)" ;;
  *) echo "usage: PRIMARY_M3PX27_REGRESSION=primary_final|full|0" >&2; exit 2 ;;
esac
for pair in "$FINAL_FILE_SHA $FINAL_FILE" "$FINAL_MAT_SHA $FINAL_MAT" "$STAGE2_FINAL_SHA $STAGE2_FINAL" "$I1_FILE_SHA $I1_FILE" "$GUARD_FILE_SHA $GUARD_FILE" "$FB_FILE_SHA $FB_FILE" \
            "$REF_STACK_SHA $REF_STACK" "$ND_FILE_SHA $ND_FILE" "$ND_MAT_SHA $ND_MAT" "$STAGE2_ND_SHA $STAGE2_ND" "$NDR_FILE_SHA $NDR_FILE" \
            "$NDR_MAT_SHA $NDR_MAT" "$STAGE2_NDR_SHA $STAGE2_NDR"; do
  check ${pair% *} ${pair#* }
done
echo "  stored files unchanged: final primary form (CSV, test matrix, stage2_primary_final.json), primary_i1, guard form, s033n, stack_s032_stage2.json, safety_nd and safety_ndr (CSV, test matrix, deployment file)"

TMP="$(mktemp -d "${TMPDIR:-/tmp}/primary_m3px27.XXXXXX")"
BAK=""
cleanup() {
  python -I eda/c133_primary_m3px27.py clean-copy >/dev/null 2>&1 || true
  rm -rf -- "${TMP:?}"
  rm -f -- "artifacts/test_primary_m3px27_copy_copy.npy" "artifacts/test_${STEM}_m3pxbase.npy" "artifacts/test_${STEM}_m3pxnull.npy" \
           "artifacts/test_${STEM}_m3pxsens.npy"
  [ -n "$BAK" ] && rm -rf -- "$BAK"
  return 0
}
trap cleanup EXIT

# 1. inputs
echo "== 1. inputs =="
for pair in "$RANK_OOF_SHA $RANK_OOF" "$RANK_META_SHA $RANK_META" "$C132_SHA $C132" "$C132_ROWS_SHA $C132_ROWS" "$NOTES_TR_SHA $NOTES_TR" \
            "$NOTES_TE_SHA $NOTES_TE" "$PR27_TE_NOTES_SHA $PR27_TE_NOTES" "$OOF_PR27_SHA $OOF_PR27" "$OOF_PR27X_SHA $OOF_PR27X" \
            "$OOF_PR27XM_SHA $OOF_PR27XM" "$TEST_ANSWERS_SHA $TEST_ANSWERS"; do
  check ${pair% *} ${pair#* }
done
python -I - "$C132" <<'PY' || exit 4
import json
import sys

e = json.load(open(sys.argv[1]))
d = e.get("decision") or {}
if e.get("dry") is not False or d.get("verdict") != "DEPLOY":
    print(f"ABORT: c132 gave {d.get('verdict')!r} (dry {e.get('dry')}) — no deployment under the m3px27 rule")
    raise SystemExit(4)
print(f"  c132 (m3px27 rule, expected-value rule): verdict {d['verdict']} (real evaluation)")
PY
python -I eda/c132_m3px27_eval.py --oof-verify | grep -E "VERIFIED|ABORT" | cut -c1-200 || { echo "ABORT: oof_m3px27 is not the recomputation of the five shards" >&2; exit 4; }

# 2. mean channel m3px27xm = ½(pr27 + m3px27) (training rows; = matrix of the expected-value rule computation)
echo "== 2. mean channel (c133 oof) =="
# always --save: writes missing files and requires the same content for existing ones (never overwrites a differing file); a
# package may lack the report files and the OOF matrix, the pins below apply unchanged
OOFMODE=--save
python -I eda/c133_primary_m3px27.py oof "$OOFMODE" | cut -c1-300 || { rc=$?; echo "ABORT: c133 oof (code $rc)" >&2; exit "$rc"; }
pin "$CH_OOF_SHA" "$CH_OOF"
pin "$CH_META_SHA" "$CH_META"

# 3. blend (θ on log loss from all 1,500 rows, β rule; rank_only, rows, folds, last step as in the final primary form)
echo "== 3. blend (c133 stack) =="
STMODE=--save  # as above
python -I eda/c133_primary_m3px27.py stack "$STMODE" | cut -c1-320 || { rc=$?; echo "ABORT: c133 stack (code $rc)" >&2; exit "$rc"; }
pin "$STACK_SHA" "$STACK"

# 4. second stage (chain with the swapped channel; MM rows = rows of the expected-value rule computation of c132 bit-identical; sensitivity with fixed β)
echo "== 4. second stage (c133 fit) =="
FITMODE=--save  # as above: chain a–e, the report c133_oof_primary_m3px27.json and the sensitivity are written or required byte-identical
python -I eda/c133_primary_m3px27.py fit "$FITMODE" | grep -vE "^\s+\[(main|fallback)\]" | cut -c1-320 || { rc=$?; echo "ABORT: c133 fit (code $rc)" >&2; exit "$rc"; }
pin "$STAGE2_SHA" "$STAGE2"
pin "$SENS_SHA" "$SENS"
python -I - "$STAGE2_FINAL" "$STAGE2" "$STACK" "$TEST_ANSWERS" "$TEST_ANSWERS_SHA" <<'PY' || exit 4
import hashlib
import json
import sys

a, b = (json.load(open(p)) for p in sys.argv[1:3])
ren = lambda f: f.replace("pr27xm_", "m3px27xm_", 1) if f.startswith("pr27xm_") else ("m3px27xm" if f == "pr27xm" else f)  # noqa: E731
bad = []
if list(b) != [*a, "m3px27_build"]:
    bad.append("order/set of the fields differs from stage2_primary_final.json (+ m3px27_build)")
if b["stack"] != sys.argv[3] or b["stack_sha256"] != hashlib.sha256(open(sys.argv[3], "rb").read()).hexdigest():
    bad.append(f"blend file {b['stack']}")
if b["channels"] != [ren(c) for c in a["channels"]]:
    bad.append(f"channels {b['channels']}")
for k in ("arith_beta", "align", "align_min", "K", "lambda", "null", "draws_channel", "draw_sources", "draw_rank_source", "test_notes", "guard_rule",
          "order", "kind"):
    if b[k] != a[k]:
        bad.append(f"field {k} differs")
for fs in ("main", "fallback"):
    ma, mb = a["models"][fs], b["models"][fs]
    if mb["features"] != [ren(f) for f in ma["features"]] or mb.get("feature_set") != ma.get("feature_set") or mb.get("protect_none") != ma.get("protect_none"):
        bad.append(f"model {fs}: features/form/guard")
m = b["models"]["main"]
if m.get("feature_set") != "main_guard_i1_mm" or m.get("protect_none") is not True or len(m["w"]) != 64 or len(b["models"]["fallback"]["w"]) != 40:
    bad.append(f"main model {m.get('feature_set')} / {len(m['w'])} weights; fallback {len(b['models']['fallback']['w'])}")
t = (b.get("mech27") or {}).get("test") or {}
if b.get("mech27") != a.get("mech27") or t.get("file") != sys.argv[4] or t.get("sha256") != sys.argv[5]:
    bad.append("mech27 differs from stage2_primary_final.json")
for k in ("guard_build", "i1_build", "mm_build"):
    keep = ("oof", "splits", "rule_oof", "fallback")
    if {x: y for x, y in b[k].items() if x not in keep} != {x: y for x, y in a[k].items() if x not in keep}:
        bad.append(f"{k} differs (except provenance)")
s = b["m3px27_build"]
df = s.get("deploy_forms") or {}
if s.get("kind") != "deploy" or not (s.get("rule_rows") or {}).get("bit_identical") or not all(v["ok"] for v in s["null_point_i"].values()):
    bad.append("m3px27_build: rows of the expected-value rule computation not bit-identical or copy null point (i) violated")
print(f"  field comparison with stage2_primary_final.json: same fields (+ m3px27_build), channel name pr27xm -> m3px27xm in channels and features, note "
      f"guard/β/K/λ/mech27/recipe fields equal; main model main_guard_i1_mm (64, guard on; OOF new top code {m['oof_changed_top_share']:.4f}), fallback 40; "
      f"deployment form standard cross-fit (β per fold {df.get('beta_per_fold')})" if not bad else "ABORT: " + "; ".join(bad))
raise SystemExit(4 if bad else 0)
PY

# 5. copy null point of the training side (m3px27 := px27: blend = stack_s032_stage2.json, chain rows = stage2/stage2_guard/stage2_i1/stage2_mm, files = stage2_s032 … primary_final)
echo "== 5. copy null point of the training side (c133 copy) =="
python -I eda/c133_primary_m3px27.py copy --keep | grep -vE "^\s+\[(main|fallback)\]" | cut -c1-320 || { rc=$?; echo "ABORT: c133 copy (code $rc)" >&2; exit "$rc"; }

# shared sources of the test side (as in tools/builds/primary_final.sh with ORIENT=repro), without the mean channel
python -I tools/orient_list.py check --mode repro --spec "$V1R_TEST" --json "$PRIV/check_orient_primary_m3px27.json" || exit $?
BASE=(--test "$V1R_TEST" --fill "v1r=v1" --test "v1=v1te_a_t4/v1_test_a/test+v1te_b_t4/v1_test_b/test"
      --test "zs27q=zs27q_test0_rp/zs27q_test0/test+zs27q_test1_rp/zs27q_test1/test" --fill "think18nb=zs27q"
      --test "think18nb=zs27v_te_rp/think_te/test/think@derived/think_rows/test.npy,zs27v_n3_rp/think_nte/test/think@derived/think_rows/test_tb_q.npy")
TAIL=(--rank-only "m3x3_full=$RANK1,$RANK2,$RANK3"
      --rank-only "q397m3=$Q397M3_TEST" --rank-none-band-for "q397m3=0.27,0.35"
      --align 0.75 --align-min 3 --blank-rows derived/blank/test_rows.npy)
SRC_COPY=("${BASE[@]}" --proto "m3px27xmc=runs/pr27te_rp/proto_te/proto" --rank "m3px27xmc=runs/n1_rp/rank_test/test" "${TAIL[@]}")
SRC=("${BASE[@]}" --proto "m3px27xm=runs/pr27te_rp/proto_te/proto" --rank "m3px27xm=$TEST_RANK" "${TAIL[@]}")
ms() { PYTHONDONTWRITEBYTECODE=1 python -I tools/make_submission.py "$@" 2>&1 | grep -vi '^warning' | grep -E "SECOND STAGE|note guard|written|ABORT|Traceback|Error|--stage2|--rank" | cut -c1-320; return "${PIPESTATUS[0]}"; }

# 6. copy null point of the test side: the same path with m3px27 test := px27 test (channel m3px27xmc, copy blend and copy deployment file) -> final primary form
echo "== 6. copy null point of the test side (m3px27 test := px27 test) =="
ms --name primary_m3px27_copy "${SRC_COPY[@]}" --out "$TMP/sub_copy.csv" --suffix "_copy" --stage2 "$COPY_STAGE2" --stage2-report "$TMP/guards_copy.json" \
  || { echo "ABORT: copy run of the test side not computed" >&2; exit 4; }
cmp -s "$TMP/sub_copy.csv" "$FINAL_FILE" || { echo "ABORT: copy null point violated — CSV != final primary form" >&2; exit 4; }
cmp -s "artifacts/test_primary_m3px27_copy_copy.npy" "$FINAL_MAT" || { echo "ABORT: copy null point violated — test matrix != final primary form" >&2; exit 4; }
echo "  copy null point: CSV and test matrix byte-identical to the final primary form ($(sha256sum "$FINAL_FILE" | cut -c1-16)… / $(sha256sum "$FINAL_MAT" | cut -c1-16)…)"
python -I eda/c133_primary_m3px27.py clean-copy >/dev/null
rm -f -- "artifacts/test_primary_m3px27_copy_copy.npy"

# 7. test side: present? technical checks (aggregates only)
echo "== 7. test guard (c133 test-guard) =="
if [ ! -f "$TEST_STATUS" ]; then
  echo "TEST SIDE MISSING: $TEST_STATUS (run $TEST_RUN not finished or not completely fetched)."
  echo "  Everything else is built and checked (steps 0–6). Final build after fetching: bash tools/builds/primary_m3px27.sh"
  echo "  (the first build prints the checksums of $OUT and $TESTMAT -> pin OUT_SHA/TESTMAT_SHA here -> second build byte-identical)"
  exit 3
fi
python -I eda/c133_primary_m3px27.py test-guard || { rc=$?; echo "ABORT: test guard (code $rc)" >&2; exit "$rc"; }

# 8. null model (weight 0) -> file without second stage byte-identical (computes the 64 features with the new channel and mech27 of the test side)
echo "== 8. null model =="
python -I eda/c133_primary_m3px27.py null --source "$STAGE2" --out "$TMP/null.json" >/dev/null
ms --name "$STEM" "${SRC[@]}" --out "$TMP/sub_base.csv" --suffix "_m3pxbase" || { echo "ABORT: file without second stage not computed" >&2; exit 4; }
ms --name "$STEM" "${SRC[@]}" --out "$TMP/sub_null.csv" --suffix "_m3pxnull" --stage2 "$TMP/null.json" --stage2-report "$TMP/guards_null.json" \
  || { echo "ABORT: null model not computed" >&2; exit 4; }
cmp -s "$TMP/sub_base.csv" "$TMP/sub_null.csv" || { echo "ABORT: null model changes the file" >&2; exit 4; }
cmp -s "artifacts/test_${STEM}_m3pxbase.npy" "artifacts/test_${STEM}_m3pxnull.npy" || { echo "ABORT: null model changes the test matrix" >&2; exit 4; }
python -I - "$TMP/guards_null.json" <<'PY' || exit 4
import json
import sys

g = json.load(open(sys.argv[1]))
gn, mm = g["guard_notes"], g.get("mech27") or {}
bad = []
if not g["null"] or g["feature_set"] != "main" or g.get("model_feature_set") != "main_guard_i1_mm" or g["n_features"] != 64 or g.get("rows_protected") != 0:
    bad.append(f"null model {g['null']}, form {g['feature_set']}/{g.get('model_feature_set')}, {g['n_features']} features, guarded {g.get('rows_protected')}")
if gn.get("rule") != "widened" or not gn["ok"]:
    bad.append(f"note guard {gn.get('rule')} / {gn['ok']}")
if not mm or mm.get("minor_codes") != 9:
    bad.append(f"mech27 {mm}")
print(f"  null model: file without second stage byte-identical (CSV and test matrix); 64 features computed with m3px27xm, mech27 {mm.get('share_mech'):.4f}; "
      "note guard (widened rule) held" if not bad else "ABORT: " + "; ".join(bad))
raise SystemExit(4 if bad else 0)
PY

# 9. main file (form main with main_guard_i1_mm and the guard required; guard (c) violated -> no file, exit 5)
echo "== 9. main file =="
if [ -f "$OUT" ] || [ -f "$TESTMAT" ]; then  # only replace byte-identically
  BAK="$(mktemp -d "${TMPDIR:-/tmp}/primary_m3px27_backup.XXXXXX")"
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
  if [ -f "$GUARDS" ] && python -I -c "import json,sys;sys.exit(1 if json.load(open('$GUARDS'))['guard_changed_top']['ok'] else 0)"; then
    echo "ABORT: guard (c) of the second stage violated (statistics in $GUARDS) — no file" >&2; exit 5
  fi
  echo "ABORT: make_submission with --stage2 failed" >&2; exit 4
fi
[ -f "$OUT" ] || { echo "ABORT: $OUT was not written" >&2; exit 4; }
python -I - "$GUARDS" "$TMP/guards_null.json" "${OUT%.csv}.json" <<'PY' || { rc=$?; restore; exit "$rc"; }
import json
import sys

g, gnull, side = (json.load(open(p)) for p in sys.argv[1:4])
gn = g["guard_notes"]
if g["feature_set"] != "main" or g.get("model_feature_set") != "main_guard_i1_mm" or g.get("protect_none") is not True:
    print(f"ABORT: form {g['feature_set']} / {g.get('model_feature_set')} / guard {g.get('protect_none')} instead of main / main_guard_i1_mm / on — no file")
    raise SystemExit(5)
bad = []
if gn.get("rule") != "widened" or not gn["ok"] or g["null"] or g["n_features"] != 64 or not g["guard_changed_top"]["ok"]:
    bad.append(f"rule {gn.get('rule')}, guard {gn['ok']}, null model {g['null']}, {g['n_features']} features, (c) {g['guard_changed_top']['ok']}")
if gn["test"] != gnull["guard_notes"]["test"] or gn["checks"] != gnull["guard_notes"]["checks"] or g.get("mech27") != gnull.get("mech27"):
    bad.append("test aggregates of the main file != those of step 8")
src = (side.get("sources") or {}).get("m3px27xm") or []
notes = [x for x in src if isinstance(x, dict) and "notes" in x]
if len(notes) != 1 or notes[0].get("rank_compare") != "sft":
    bad.append(f"sources of the channel m3px27xm without note guard/comparison form sft ({len(notes)})")
print(f"  main form main_guard_i1_mm (64 features, guard on: {g['rows_protected']} test rows keep the base order); guard (c) new top code "
      f"{g['guard_changed_top']['test']:.4f} (OOF {g['guard_changed_top']['oof']:.4f}) ok; mech27 {g['mech27']['share_mech']:.4f}; note guard = step 8; "
      "sources with note guard (sft)" if not bad else "ABORT: " + "; ".join(bad))
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

# 10. check of the file and comparison with the final primary form (aggregates only)
echo "== 10. check and comparison with the final primary form (aggregates only) =="
check "$FINAL_FILE_SHA" "$FINAL_FILE"
python -I - "$OUT" "$FINAL_FILE" "$GUARDS" "$PRIV/compare_primary_m3px27.json" <<'PY' || exit 4
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
     "share_top_none_primary_final": float((t_ref == "NONE").mean())}
print(f"  vs. the final primary form: other top code in {e['rows_other_top1']} of {e['rows']} rows ({e['share_other_top1']:.4f}), other first three (ordered) in "
      f"{e['rows_other_top3_ordered']} ({e['share_other_top3_ordered']:.4f}), as a set {e['rows_other_top3_as_set']}; NONE at the top "
      f"{e['share_top_none_primary_final']:.4f} -> {e['share_top_none_new']:.4f}")
g = json.load(open(sys.argv[3]))
rep = {"check": stats, "vs_primary_final": e, **{k: g[k] for k in ("guard_changed_top", "guard_novel_task_top5", "share_top3_changed", "share_top3_changed_oof",
                                                                    "rows_protected", "mech27")}}
gc = g["guard_changed_top"]
print(f"  guard (c): new top code {gc['test']:.4f} (OOF {gc['oof']:.4f}, band {gc['factor'][0] * gc['oof']:.4f}…{gc['factor'][1] * gc['oof']:.4f}) -> "
      f"{'ok' if gc['ok'] else 'VIOLATED'}; guarded {g['rows_protected']} test rows")
Path(sys.argv[4]).write_text(json.dumps(rep, indent=1, ensure_ascii=False) + "\n")
PY

# 11. sensitivity (report only): main model on the deployment form with the fixed β of the blend file instead of the standard cross-fit
echo "== 11. sensitivity with fixed β (aggregates only, private) =="
ms --name "$STEM" "${SRC[@]}" --out "$TMP/sub_sens.csv" --suffix "_m3pxsens" --stage2 "$SENS" --stage2-report "$TMP/guards_sens.json" \
  || echo "  sensitivity run without a file (guard) — report only"
if [ -f "$TMP/sub_sens.csv" ]; then
  python -I - "$OUT" "$TMP/sub_sens.csv" "$PRIV/sens/compare_fixbeta.json" <<'PY' || exit 4
import json
import sys

import pandas as pd


def read(p):
    return pd.read_csv(p, dtype=str, keep_default_na=False).set_index("trace_id")["root_code"]


a, b = read(sys.argv[1]), read(sys.argv[2])
b = b.loc[a.index]
r = {"rows_other_top1": int((a.str.split().str[0] != b.str.split().str[0]).sum()), "rows_other_top3_ordered": int((a != b).sum()), "rows": int(len(a))}
print(f"  fixed β instead of the standard cross-fit: other top code in {r['rows_other_top1']} of {r['rows']} test rows, other first three in {r['rows_other_top3_ordered']}")
open(sys.argv[3], "w").write(json.dumps(r, indent=1) + "\n")
PY
fi
rm -f -- "artifacts/test_${STEM}_m3pxsens.npy"

# 12. checksums
echo "== 12. checksums =="
pin "$OUT_SHA" "$OUT"
pin "$TESTMAT_SHA" "$TESTMAT"
sha256sum "$CH_OOF" "$CH_META" "$STACK" "$STAGE2" "$OUT" "$TESTMAT" "${OUT%.csv}.json" "$GUARDS"
if [ "$FIRST" = 1 ]; then
  echo "primary_m3px27: FIRST BUILD — enter the printed checksums and run the build path again (second build byte-identical)"
else
  echo "primary_m3px27: $OUT built (candidate primary_m3px27, primary line with m3px27xm; nothing is submitted from here)"
fi
