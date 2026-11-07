#!/usr/bin/env bash
# Primary line, FINAL CANDIDATE: the second stage with interactions I1 (tools/builds/primary_i1.sh) with one more column — feature set
# "main_guard_i1_mm" = main_guard_i1 (63, unchanged) + minor_x_mech27 = 1[the candidate has the default severity "minor"] ·
# 1[the final answer read by the pr27 channel matches the answer key mechanically] (src/answer_match.py, without eval). Deployment file
# artifacts/stage2_primary_final.json (eda/c124_stage2_mm.py fit): stage2_primary_i1.json recomputed character for character, then ONLY
# models.main replaced; fallback model, note guard under the widened rule, β, K, λ unchanged; new the field mech27 (frozen pr27 test
# answers with SHA-256, minor codes). Test side directly through tools/make_submission.py (the set main_guard_i1_mm is admitted for the
# main model). Order unchanged: blend -> margin alignment (one set) -> second stage (+ guard) -> REL-01. Rotation list ORIENT=repro.
#
#     bash tools/builds/primary_final.sh                         # deployment only if the expected-value rule is met, otherwise exit 5 after step 4
#     PRIMARY_FINAL_PROBE=1 bash tools/builds/primary_final.sh   # if the rule is missed: PROBE only under artifacts/eda/private/stage2_mm/probe/
#     PRIMARY_FINAL_SPLITS=1 …                                   # recomputes the 20 meta-splits (byte-identical result required)
#
# STEPS: 0 build of the I1 form complete (tools/builds/primary_i1.sh, inside it the guard form and its chain) = regression check; 1 c124
#   parser (mech27 training/test, shares; result file pinned); 2 c124 oof (I1 form bit-identical, copy null point, sixths fixed, result
#   file pinned); 3 meta-splits (pinned); 4 expected-value rule (c124 rule; exit 5 = no deployment); 5 deployment file (c124 fit; probe:
#   --probe, private); 6 copy null point (ii): null model -> file without second stage byte-identical (computes the 64 features and
#   mech27 of the test side); 7 copy null point (iii) = rule (i), file part: column weight 0 with the 63 weights of the I1 form -> I1-form
#   CSV and test matrix byte-identical; 8 file swap (notes file d2 as d1) -> guard (a) -> fallback = fallback-form file byte-identical;
#   9 main file (guard (c), form main_guard_i1_mm required); 10 comparison with the I1 form and the guard form (aggregates only);
#   11 checksums (deployment: pinned after the first build).
# EXIT: 2 usage; 3 input missing; 4 checksum/recomputation/null point/control; 5 pre-registered rule not met, wrong form or guard (c)
#   violated; 1 a check inside a called Python script that stops with a message.
#
# Only aggregates and checksums are printed. Nothing is submitted from here.
set -euo pipefail
[ $# -eq 0 ] || { echo "usage: $0   (no arguments; PRIMARY_FINAL_PROBE=1, PRIMARY_FINAL_SPLITS=1)" >&2; exit 2; }
cd "$(dirname "$0")/../.."
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}" OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export ORIENT=repro
STEM="s032_stage2"
STACK="artifacts/stack_${STEM}.json"; STACK_SHA="d278bbedf7fb4d7564e584a55d45959b826748dfe937d8becf1e0818007281eb"
STAGE2_I1="artifacts/stage2_primary_i1.json"; STAGE2_I1_SHA="f5e24adfd382d24b36bd88b57deb8904a9d5288863250d5d314ceb09e54b181d"
I1_FILE="artifacts/submissions/sub_primary_i1.csv"; I1_FILE_SHA="60b63d69bcdd5970cafaf4d5645440d8d375953871d6dd688ae5091b25256bee"
I1_MAT="artifacts/test_${STEM}_primary_i1.npy"; I1_MAT_SHA="7ce36020bb4f5bf2e61291f07817a783acefec677856a15bab780464a59902a7"
GUARD_FILE="artifacts/submissions/sub_s035_stage2_guard.csv"; GUARD_FILE_SHA="d4dce1aaa5cfbd1fd5668a9e7073a47734c0c5aacda7b7c2020255005a81f360"
FB_FILE="artifacts/submissions/sub_s033n_stage2.csv"; FB_FILE_SHA="76aba64fbc431b9ad403a5dec9fb2cefdb7fe2e76260149c96176459c4cd8dfe"
TEST_ANSWERS="artifacts/runs/pr27te_rp/pr27te_rp/proto_te/proto_raw.jsonl"; TEST_ANSWERS_SHA="3e9b05623a0dbb7eb7bb9b211b662a93af3b6c85a096c70523a62dc1e35eae62"
PRIV="artifacts/eda/private/stage2_mm"
PARSER="$PRIV/c124_parser.json"; PARSER_SHA="a5b8093112948cd55c8a8c69b6193bb5ec4c0afb0a55083e1eea85fa4af26788"
IOOF="$PRIV/c124_mm_oof.json"; IOOF_SHA="654cf628bd62e2ba7384416f3c5c00e97a11f053812127018481a8ff4ee8833d"
ISPL="$PRIV/c124_mm_splits.json"; ISPL_SHA="511a42a9fdfbd55574d315d4921d4f38440a65ef0abc4afd1340af0dfd31b3b6"
OOF_UNITS_VS_I1='{"fold": 1, "pre": 10, "set": 10}'  # main_guard_i1_mm − I1 form in sixths (c124 oof; = the independent cross-check)
STAGE2="artifacts/stage2_primary_final.json"; STAGE2_SHA="40481bc4c83d0ce41b69ebfd10b0d1aaea0a8156c693f1ee7fc89ceb99674604"  # c124 fit (pinned after the first build)
OUT="artifacts/submissions/sub_primary_final.csv"; OUT_SHA="d9c68c460c6da06d4975513911e11c4541d70d07b78f132566a3e43e7ad60b24"  # final primary candidate (pinned)
SUFFIX="_primary_final"
NOTES1="derived/sft/m3_d1_test_frozen_notes.jsonl"
NOTES2="derived/sft/m3_d2_test_frozen_notes.jsonl"
RANK1="runs/m3_fullte_rp/m3_test_rank/test"
RANK2="runs/m3d2te_rp/m3d2_test_rank/test"
RANK3="runs/m3d3te_rp/m3d3_test_rank/test"
Q397M3_TEST="runs/q397m3te_rp/q397m3_test_rank/test"
check() { echo "$1  $2" | sha256sum -c --quiet || { echo "ABORT: $2 differs (checksum)" >&2; exit 4; }; }
pinned() { check "$1" "$2"; }  # deployment checksums (pinned after the first build)
. tools/builds/orient_list.sh  # rotation list of the v1r test side: always repro here
orient_select
[ "$ORIENT" = repro ] || { echo "ABORT: this build path only runs with ORIENT=repro" >&2; exit 2; }
mkdir -p "$PRIV"

# 0. build of the I1 form complete (inside it the guard form and its chain) — regression check of the earlier files
echo "== 0. I1 form (tools/builds/primary_i1.sh) =="
bash tools/builds/primary_i1.sh || { rc=$?; echo "ABORT: build of the I1 form (code $rc)" >&2; exit "$rc"; }
check "$I1_FILE_SHA" "$I1_FILE"
check "$I1_MAT_SHA" "$I1_MAT"
check "$STAGE2_I1_SHA" "$STAGE2_I1"
check "$GUARD_FILE_SHA" "$GUARD_FILE"
check "$FB_FILE_SHA" "$FB_FILE"
check "$STACK_SHA" "$STACK"
check "$TEST_ANSWERS_SHA" "$TEST_ANSWERS"

TMP="$(mktemp -d "${TMPDIR:-/tmp}/primary_final.XXXXXX")"
BAK=""
cleanup() {
  rm -rf -- "${TMP:?}"
  rm -f -- "artifacts/test_${STEM}_mmbase.npy" "artifacts/test_${STEM}_mmnull.npy" "artifacts/test_${STEM}_mmzero.npy" \
           "artifacts/test_${STEM}_mmswap.npy" "artifacts/test_${STEM}_mmprobe.npy"
  [ -n "$BAK" ] && rm -rf -- "$BAK"
  return 0
}
trap cleanup EXIT

# 1. mech27 of the training and test side (shares only)
echo "== 1. c124 parser =="
python -I eda/c124_stage2_mm.py parser | grep -E "mech27|ABORT" | cut -c1-300 || { echo "ABORT: c124 parser" >&2; exit 4; }
check "$PARSER_SHA" "$PARSER"

# 2. nested reproduction (I1 form bit-identical, copy null point, sixths fixed, 15 placebos)
echo "== 2. c124 oof =="
python -I eda/c124_stage2_mm.py oof | grep -E "^\s+\[MM|MM vs\.|ABORT" | cut -c1-320 || { echo "ABORT: c124 oof" >&2; exit 4; }
python -I - "$IOOF" "$OOF_UNITS_VS_I1" <<'PY' || exit 4
import json
import sys

g, want = json.load(open(sys.argv[1])), json.loads(sys.argv[2])
bad = []
if not g["null_point_i"]["ok"]:
    bad.append(f"copy null point (i) violated {g['null_point_i']}")
got = {f: g["sets"]["mm"][f]["vs_i1"]["units"] for f in ("fold", "pre", "set")}
if got != want or g["placebo"]["n"] != 15 or not all(g["sets"]["i1"][f]["rows_equal_reference"] for f in ("fold", "pre", "set")):
    bad.append(f"sixths {got} instead of {want} / placebos {g['placebo']['n']} / I1 form not bit-identical")
print(f"  c124 oof: I1 form bit-identical, copy null point bit-identical, Δ against the I1 form {got} sixths, ½(F+M) {g['summary']['A']:+.5f}, placebo mean "
      f"{g['placebo']['A_mean']:+.5f} (n {g['placebo']['n']})" if not bad else "ABORT: " + "; ".join(bad))
raise SystemExit(4 if bad else 0)
PY
check "$IOOF_SHA" "$IOOF"

# 3. meta-splits (fixed checksum; PRIMARY_FINAL_SPLITS=1 recomputes and requires byte identity)
echo "== 3. meta-splits (c124 splits) =="
if [ "${PRIMARY_FINAL_SPLITS:-0}" = 1 ] || [ ! -f "$ISPL" ]; then
  [ -f "$ISPL" ] && cp -p "$ISPL" "$TMP/splits_before.json"
  python -I eda/c124_stage2_mm.py splits --n 20 | grep -E "over|ABORT" | cut -c1-300 || { echo "ABORT: c124 splits" >&2; exit 4; }
  if [ -f "$TMP/splits_before.json" ] && ! cmp -s "$ISPL" "$TMP/splits_before.json"; then
    cp -p "$TMP/splits_before.json" "$ISPL"
    echo "ABORT: recomputation of the meta-splits differs from the existing $ISPL — backup restored" >&2; exit 4
  fi
fi
check "$ISPL_SHA" "$ISPL"

# 4. expected-value rule (exit 5 = no deployment; with PRIMARY_FINAL_PROBE=1 continue as a probe into the private folder only)
echo "== 4. expected-value rule (c124 rule) =="
MODE=deploy
rc=0
python -I eda/c124_stage2_mm.py rule || rc=$?
if [ "$rc" != 0 ]; then
  if [ "$rc" = 5 ] && [ "${PRIMARY_FINAL_PROBE:-0}" = 1 ]; then
    MODE=probe
    echo "  expected-value rule NOT met -> PROBE (PRIMARY_FINAL_PROBE=1): steps 5–10 only under $PRIV/probe/, no deployment file, no file under submissions/"
  else
    echo "ABORT: expected-value rule not met — no deployment (code $rc)" >&2; exit "$rc"
  fi
fi
if [ "$MODE" = probe ]; then
  mkdir -p "$PRIV/probe"
  STAGE2="$PRIV/probe/stage2_primary_final_probe.json"
  OUT="$PRIV/probe/sub_primary_final_probe.csv"
  SUFFIX="_mmprobe"
  FIT=(--probe)
  TAG="probe"
else
  FIT=()
  TAG=""
fi
TESTMAT="artifacts/test_${STEM}${SUFFIX}.npy"
GUARDS="$PRIV/guards_primary_final${TAG}.json"

# 5. deployment file (c124 fit: stage2_primary_i1.json recomputed character for character, null point (e) against c124 oof)
echo "== 5. deployment file (c124 fit ${FIT[*]}) =="
if [ -f "$STAGE2" ]; then
  python -I eda/c124_stage2_mm.py fit "${FIT[@]}" --out "$STAGE2" --verify || { rc=$?; echo "ABORT: $STAGE2 is not the recomputation" >&2; exit "$rc"; }
else
  python -I eda/c124_stage2_mm.py fit "${FIT[@]}" --out "$STAGE2" --save || { rc=$?; echo "ABORT: $STAGE2 not built" >&2; exit "$rc"; }
fi
[ "$MODE" = deploy ] && pinned "$STAGE2_SHA" "$STAGE2"
python -I - "$STAGE2_I1" "$STAGE2" "$STAGE2_I1_SHA" "$TEST_ANSWERS" "$TEST_ANSWERS_SHA" <<'PY' || exit 4
import json
import sys

a, b = (json.load(open(p)) for p in sys.argv[1:3])
bad = []
extra = ("mm_build", "mech27", "derived_from", "probe", "probe_note")
if [k for k in b if k not in extra] != [k for k in a if k != "derived_from"]:
    bad.append("order/set of the fields differs")
diff = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
if set(diff) - {"models", *extra}:
    bad.append(f"fields other than models/mm_build/mech27/derived_from differ: {diff}")
if list(b["models"]) != ["main", "fallback"] or b["models"]["fallback"] != a["models"]["fallback"]:
    bad.append("fallback model not bit-identical to stage2_primary_i1.json")
m, am = b["models"]["main"], a["models"]["main"]
if m.get("feature_set") != "main_guard_i1_mm" or m.get("protect_none") is not True or m["features"][:63] != am["features"] \
        or m["features"][63:] != ["minor_x_mech27"] or len(m["w"]) != 64:
    bad.append(f"main model: {m.get('feature_set')} / guard {m.get('protect_none')} / {len(m['features'])} features")
t = (b.get("mech27") or {}).get("test") or {}
if t.get("file") != sys.argv[4] or t.get("sha256") != sys.argv[5] or len(b["mech27"].get("minor_codes") or []) != 9:
    bad.append(f"mech27 {b.get('mech27')}")
d = b.get("derived_from") or {}
if d.get("file") != sys.argv[1] or d.get("sha256") != sys.argv[3] or d.get("previous") != a.get("derived_from"):
    bad.append(f"derived_from {d}")
print(f"  field comparison with stage2_primary_i1.json: only models.main replaced (main_guard_i1_mm, 64 features, guard on; weight minor_x_mech27 "
      f"{m['w'][-1]:+.4f}; OOF share of new top codes {m['oof_changed_top_share']:.4f}), fallback/note guard/β/K/λ bit-identical" if not bad
      else "ABORT: " + "; ".join(bad))
raise SystemExit(4 if bad else 0)
PY

# shared sources of the test side (as in tools/builds/primary_i1.sh with ORIENT=repro)
python -I tools/orient_list.py check --mode repro --spec "$V1R_TEST" --json "$PRIV/check_orient_primary_final.json" || exit $?
SRC=(--test "$V1R_TEST" --fill "v1r=v1" --test "v1=v1te_a_t4/v1_test_a/test+v1te_b_t4/v1_test_b/test"
     --test "zs27q=zs27q_test0_rp/zs27q_test0/test+zs27q_test1_rp/zs27q_test1/test" --fill "think18nb=zs27q"
     --test "think18nb=zs27v_te_rp/think_te/test/think@derived/think_rows/test.npy,zs27v_n3_rp/think_nte/test/think@derived/think_rows/test_tb_q.npy"
     --proto "pr27xm=runs/pr27te_rp/proto_te/proto" --rank "pr27xm=runs/n1_rp/rank_test/test"
     --rank-only "m3x3_full=$RANK1,$RANK2,$RANK3"
     --rank-only "q397m3=$Q397M3_TEST" --rank-none-band-for "q397m3=0.27,0.35"
     --align 0.75 --align-min 3 --blank-rows derived/blank/test_rows.npy)
ms() { PYTHONDONTWRITEBYTECODE=1 python -I tools/make_submission.py "$@" 2>&1 | grep -vi '^warning' | grep -E "SECOND STAGE|note guard|written|ABORT|Traceback|Error|--stage2" | cut -c1-320; return "${PIPESTATUS[0]}"; }

# 6. copy null point (ii): weight 0 -> file without second stage byte-identical (computes the 64 features and mech27 of the test side)
echo "== 6. copy null point (ii) with the null model =="
python -I eda/c124_stage2_mm.py null --source "$STAGE2" --out "$TMP/null.json" >/dev/null
ms --name "$STEM" "${SRC[@]}" --out "$TMP/sub_base.csv" --suffix "_mmbase" || { echo "ABORT: file without second stage not computed" >&2; exit 4; }
ms --name "$STEM" "${SRC[@]}" --out "$TMP/sub_null.csv" --suffix "_mmnull" --stage2 "$TMP/null.json" --stage2-report "$TMP/guards_null.json" \
  || { echo "ABORT: null point (ii) not computed" >&2; exit 4; }
cmp -s "$TMP/sub_base.csv" "$TMP/sub_null.csv" || { echo "ABORT: null point (ii) violated — weight 0 changes the file" >&2; exit 4; }
cmp -s "artifacts/test_${STEM}_mmbase.npy" "artifacts/test_${STEM}_mmnull.npy" || { echo "ABORT: null point (ii) violated — test matrix" >&2; exit 4; }
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
if not mm or mm.get("minor_codes") != 9 or not 0.2 < float(mm.get("share_mech", 0)) < 0.35:
    bad.append(f"mech27 {mm}")
print(f"  null point (ii): weight 0 -> file without second stage byte-identical (CSV and test matrix); 64 features computed, mech27 of the test side "
      f"{mm.get('share_mech'):.4f} (answer {mm.get('share_with_answer'):.4f}); note guard (widened rule) held" if not bad else "ABORT: " + "; ".join(bad))
raise SystemExit(4 if bad else 0)
PY

# 7. copy null point (iii) = rule (i), file part: the 63 weights of the I1 form, column weight 0 -> I1-form file byte-identical
echo "== 7. copy null point (iii): I1-form weights, column weight 0 -> I1-form file =="
python -I - "$STAGE2" "$STAGE2_I1" "$TMP/stage2_zero.json" <<'PY' || exit 4
import json
import sys

b, a = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
am, m = a["models"]["main"], b["models"]["main"]
if m["features"][:63] != am["features"] or am.get("feature_set") != "main_guard_i1" or am.get("protect_none") is not True:
    raise SystemExit("ABORT: features/form of the I1 form do not match")
s = json.loads(json.dumps(b))
s["models"]["main"].update({"w": am["w"] + [0.0], "mu": am["mu"] + [0.0], "sd": am["sd"] + [1.0],
                            "oof_changed_top_share": am["oof_changed_top_share"], "oof_changed_top3_share": am["oof_changed_top3_share"]})
open(sys.argv[3], "w").write(json.dumps(s, indent=1, ensure_ascii=False) + "\n")
PY
ms --name "$STEM" "${SRC[@]}" --out "$TMP/sub_zero.csv" --suffix "_mmzero" --stage2 "$TMP/stage2_zero.json" --stage2-report "$TMP/guards_zero.json" \
  || { echo "ABORT: null point (iii) not computed" >&2; exit 4; }
cmp -s "$TMP/sub_zero.csv" "$I1_FILE" || { echo "ABORT: null point (iii) violated — column weight 0 does not give the I1-form file (CSV)" >&2; exit 4; }
cmp -s "artifacts/test_${STEM}_mmzero.npy" "$I1_MAT" || { echo "ABORT: null point (iii) violated — test matrix != I1 form" >&2; exit 4; }
python -I - "$TMP/guards_zero.json" "artifacts/eda/private/stage2_i1/guards_primary_i1.json" "$PRIV/nullpoint_iii_primary_final.json" <<'PY' || exit 4
import json
import sys

g, g37 = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
bad = []
if g.get("model_feature_set") != "main_guard_i1_mm" or g["n_features"] != 64 or g.get("protect_none") is not True or g["null"]:
    bad.append(f"form {g.get('model_feature_set')} / {g['n_features']} / guard {g.get('protect_none')} / null model {g['null']}")
for k in ("rows_protected", "share_top3_changed"):
    if g.get(k) != g37.get(k):
        bad.append(f"{k} {g.get(k)} != I1 form {g37.get(k)}")
if g["guard_changed_top"]["test"] != g37["guard_changed_top"]["test"] or g["guard_notes"]["checks"] != g37["guard_notes"]["checks"]:
    bad.append("guards (a)/(c) != I1 form")
json.dump({"iii_equals_i1_form_csv": True, "iii_equals_i1_form_testmatrix": True, "rows_protected": g.get("rows_protected"),
           "changed_top_test": g["guard_changed_top"]["test"]}, open(sys.argv[3], "w"), indent=1)
print(f"  (iii) column weight 0: CSV and test matrix byte-identical to the I1 form; guarded {g.get('rows_protected')}, new top code "
      f"{g['guard_changed_top']['test']:.4f} as in the I1 form (rule (i), file part, passed)" if not bad else "ABORT: " + "; ".join(bad))
raise SystemExit(4 if bad else 0)
PY

# 8. file swap: notes file of draw d2 in place of d1 (consistent provenance) -> guard (a) fires -> fallback = fallback-form file
echo "== 8. file swap (notes file d2 as d1) =="
python -I - "$STAGE2" "$TMP" "$NOTES1" "$NOTES2" "$RANK2" "_mmswap" "${SRC[@]}" <<'PY' > "$TMP/swap.log" 2>&1 \
  || { echo "ABORT: file swap not computed" >&2; grep -E "ABORT|Error|SystemExit|--stage2" "$TMP/swap.log" >&2; exit 4; }
import importlib.util
import sys
from pathlib import Path

stage2, tmp, n1, n2, r2, swap_suffix = sys.argv[1:7]
src = sys.argv[7:]
spec = importlib.util.spec_from_file_location("make_submission_mmswap", Path.cwd() / "tools" / "make_submission.py")
MS = importlib.util.module_from_spec(spec)
spec.loader.exec_module(MS)
orig, calls = MS._test_notes_texts, []


def swapped(root, rel_file, rank_spec, n_test):
    """File swap: the frozen notes of draw d2 (with their own, consistent provenance) in place of d1."""
    if rel_file == n1:
        calls.append(rel_file)
        texts, info = orig(root, n2, r2, n_test)
        return texts, {**info, "swapped_in_for": rel_file}
    return orig(root, rel_file, rank_spec, n_test)


MS._test_notes_texts = swapped
sys.argv = ["make_submission.py", "--name", "s032_stage2", *src, "--out", f"{tmp}/sub_swap.csv", "--suffix", swap_suffix, "--stage2", stage2,
            "--stage2-report", f"{tmp}/guards_swap.json"]
rc = MS.main()
if calls != [n1]:
    raise SystemExit(f"swap not executed ({len(calls)} calls)")
raise SystemExit(rc)
PY
python -I - "$TMP/guards_swap.json" "$TMP/sub_swap.csv" "$FB_FILE_SHA" "$PRIV/swap_control_primary_final${TAG}.json" <<'PY' || exit 4
import hashlib
import json
import sys

g = json.load(open(sys.argv[1]))
gn = g["guard_notes"]
failed = sorted(c["stat"] for c in gn["checks"] if not c["ok"])
sha = hashlib.sha256(open(sys.argv[2], "rb").read()).hexdigest()
bad = []
if gn["ok"] or g["feature_set"] != "fallback" or "identical[d1-d2]" not in failed or gn.get("rule") != "widened" or "model_feature_set" in g \
        or "mech27" in g:
    bad.append(f"guard (a) did not fire on the swap, or the fallback used a foreign feature set (form {g['feature_set']}, failed {failed})")
if sha != sys.argv[3]:
    bad.append(f"fallback file {sha[:16]}… != fallback-form file")
json.dump({"feature_set": g["feature_set"], "failed": failed, "sub_sha256": sha, "equals_fallback_form_file": sha == sys.argv[3]}, open(sys.argv[4], "w"), indent=1)
print(f"  guard (a) fires under the widened rule (failed {', '.join(failed)}) -> fallback (40 features, without the guard, without mech27); file = "
      f"fallback-form file byte-identical ({sha[:16]}…)" if not bad else "ABORT: " + "; ".join(bad))
raise SystemExit(4 if bad else 0)
PY

# 9. main file (deployment) or probe (private); form main with main_guard_i1_mm and the guard required; guard (c) violated -> no file, exit 5
echo "== 9. main file ($MODE) =="
if [ "$MODE" = deploy ] && { [ -f "$OUT" ] || [ -f "$TESTMAT" ]; }; then  # only replace byte-identically
  BAK="$(mktemp -d "${TMPDIR:-/tmp}/primary_final_backup.XXXXXX")"
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
python -I - "$GUARDS" "$TMP/guards_null.json" <<'PY' || { rc=$?; restore; exit "$rc"; }
import json
import sys

g, gnull = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
gn = g["guard_notes"]
if g["feature_set"] != "main" or g.get("model_feature_set") != "main_guard_i1_mm" or g.get("protect_none") is not True:
    print(f"ABORT: form {g['feature_set']} / {g.get('model_feature_set')} / guard {g.get('protect_none')} instead of main / main_guard_i1_mm / on — no file")
    raise SystemExit(5)
bad = []
if gn.get("rule") != "widened" or not gn["ok"] or g["null"] or g["n_features"] != 64 or not g["guard_changed_top"]["ok"]:
    bad.append(f"rule {gn.get('rule')}, guard {gn['ok']}, null model {g['null']}, {g['n_features']} features, (c) {g['guard_changed_top']['ok']}")
if gn["test"] != gnull["guard_notes"]["test"] or gn["checks"] != gnull["guard_notes"]["checks"] or g.get("mech27") != gnull.get("mech27"):
    bad.append("test aggregates of the main file != those of step 6")
print(f"  main form main_guard_i1_mm (64 features, guard on: {g['rows_protected']} test rows keep the base order); guard (c) new top code "
      f"{g['guard_changed_top']['test']:.4f} (OOF {g['guard_changed_top']['oof']:.4f}) ok; mech27 {g['mech27']['share_mech']:.4f}; note guard = step 6"
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

# 10. check of the file and comparison (aggregates only)
echo "== 10. check and comparison (aggregates only) =="
check "$I1_FILE_SHA" "$I1_FILE"
check "$GUARD_FILE_SHA" "$GUARD_FILE"
python -I - "$OUT" "$I1_FILE" "$GUARD_FILE" "$GUARDS" "$PRIV/compare_primary_final${TAG}.json" <<'PY' || exit 4
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
tax = pd.read_csv(root / "competition_data" / "taxonomy_v1.csv", dtype=str, keep_default_na=False)
minor = {c for c, s in zip(tax["id"], tax["default_severity"]) if str(s).strip().lower() == "minor"}  # codes of default severity minor
for path, label in ((sys.argv[2], "I1 form (main_guard_i1)"), (sys.argv[3], "guard form (main_guard)")):
    ref = read(path)
    b = new.loc[ref.index]
    t_new, t_ref = b.str.split().str[0], ref.str.split().str[0]
    s_new, s_ref = b.str.split().apply(frozenset), ref.str.split().apply(frozenset)
    in3_new = b.str.split().apply(lambda xs: any(x in minor for x in xs))
    in3_ref = ref.str.split().apply(lambda xs: any(x in minor for x in xs))
    e = {"share_other_top1": float((t_ref != t_new).mean()), "rows_other_top1": int((t_ref != t_new).sum()),
         "share_other_top3_ordered": float((ref != b).mean()), "rows_other_top3_ordered": int((ref != b).sum()),
         "share_other_top3_as_set": float((s_new != s_ref).mean()), "share_minor_in_top3_new": float(in3_new.mean()),
         "share_minor_in_top3_ref": float(in3_ref.mean()), "share_top_none_new": float((t_new == "NONE").mean()),
         "share_top_none_ref": float((t_ref == "NONE").mean())}
    rep["vs"][label] = e
    print(f"  vs. {label}: other top code in {e['share_other_top1']:.4f} of the rows ({e['rows_other_top1']}), other first three (ordered) in "
          f"{e['share_other_top3_ordered']:.4f} ({e['rows_other_top3_ordered']}), as a set {e['share_other_top3_as_set']:.4f}; minor code among the first three "
          f"{e['share_minor_in_top3_ref']:.4f} -> {e['share_minor_in_top3_new']:.4f}; NONE at the top {e['share_top_none_ref']:.4f} -> {e['share_top_none_new']:.4f}")
g = json.load(open(sys.argv[4]))
rep.update({k: g[k] for k in ("guard_changed_top", "guard_novel_task_top5", "share_top3_changed", "share_top3_changed_oof", "rows_protected", "mech27")})
gc = g["guard_changed_top"]
print(f"  guard (c): new top code {gc['test']:.4f} (OOF {gc['oof']:.4f}, band {gc['factor'][0] * gc['oof']:.4f}…{gc['factor'][1] * gc['oof']:.4f}) -> "
      f"{'ok' if gc['ok'] else 'VIOLATED'}; guarded {g['rows_protected']} test rows; mech27 of the test side {g['mech27']['share_mech']:.4f}")
Path(sys.argv[5]).write_text(json.dumps(rep, indent=1, ensure_ascii=False) + "\n")
PY

# 11. checksums
echo "== 11. checksums ($MODE) =="
if [ "$MODE" = deploy ]; then
  pinned "$OUT_SHA" "$OUT"
  sha256sum "$STACK" "$STAGE2_I1" "$STAGE2" "$I1_FILE" "$GUARD_FILE" "$OUT" "$TESTMAT" "${OUT%.csv}.json" "$PARSER" "$IOOF" "$ISPL" "$GUARDS"
  echo "primary_final: $OUT built (final primary candidate: main_guard_i1_mm + NONE guard under the widened rule, rotation list repro; nothing is submitted from here)"
else
  sha256sum "$STAGE2" "$OUT" "$TESTMAT" "$GUARDS"
  rm -f -- "$TESTMAT"
  echo "primary_final: PROBE under $PRIV/probe/ (expected-value rule not met — no candidate, never submit)"
fi
