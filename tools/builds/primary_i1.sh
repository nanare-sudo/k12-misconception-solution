#!/usr/bin/env bash
# Primary line, second stage with the INTERACTIONS I1 (predecessor and fallback of the final primary candidate): the guard form of the
# second stage (main form + NONE guard + general-code identifiers, tools/builds/s035_stage2_guard.sh) with the feature set
# "main_guard_i1" = main_guard (48 features, unchanged) + 15 interactions {row_ent, row_max, row_d_ntop, row_ch_ntop, task_rows} ×
# {f_logp, is_general, is_none} (fixed before the evaluation) — with the NONE guard of the guard form. Deployment file
# artifacts/stage2_primary_i1.json (eda/c121_stage2_i1.py fit): stage2_s035.json recomputed character for character, then ONLY
# models.main replaced by the main_guard_i1 model; fallback model, note guard under the widened rule (reference and rule), β, K, λ and
# all input checksums stay those of the guard form. Rotation list ORIENT=repro (released list). Order on the test side unchanged: blend
# -> margin alignment (one set) -> second stage (+ guard) -> REL-01. The test side with main_guard_i1 is computed by
# eda/c121_stage2_i1.py submit: tools/make_submission.py (unchanged) as a module, main_guard_i1 admitted for the main model only; without
# the second stage and for the fallback model the path is exactly that of make_submission.py.
#
#     bash tools/builds/primary_i1.sh                        # -> artifacts/submissions/sub_primary_i1.csv
#     PRIMARY_I1_SPLITS=1 bash tools/builds/primary_i1.sh    # also recomputes the 20 meta-splits (≈ 8 min; byte-identical result required)
#
# STEPS: 0 build of the guard form complete (tools/builds/s035_stage2_guard.sh; inside it the main form, the fallback form, the
#   intermediate file and all their checks) — also the regression check of the earlier files (fixed checksums of the guard, main and
#   fallback form); 1 nested reproduction (c121 oof: guard form = row values in stage2_guard/ bit for bit, copy null point (i) I1 columns
#   0 -> guard form bit-identical, Δ against the guard form in sixths fixed, 15 placebos; result file byte-identical, fixed checksum);
#   2 expected-value rule (c121 rule; the meta-splits from c121 splits with a fixed checksum, PRIMARY_I1_SPLITS=1 recomputes them);
#   3 deployment file (c121 fit --save or --verify; fixed checksum; field comparison with stage2_s035.json); 4 copy null point (ii):
#   null model from stage2_primary_i1.json -> file without second stage byte-identical (CSV and test matrix; computes the 63 features);
#   5 copy null point (iii) = rule (i), file part: main_guard_i1 model with the 48 weights of the guard form and I1 weights 0 -> guard-form
#   file byte-identical (CSV and test matrix); 6 file swap (notes file d2 in place of d1, consistent provenance) -> guard (a) fires under
#   the widened rule -> fallback, file = fallback-form file byte-identical; 7 main file (form main with feature set main_guard_i1 and the
#   guard required; guard (c) with the OOF share of main_guard_i1, otherwise exit 5 without a file); 8 comparison with the guard, main
#   and fallback form (aggregates only); 9 checksums. An existing output file is only replaced by a byte-identical rebuild. Statistics
#   and logs under artifacts/eda/private/stage2_i1/.
# EXIT: 2 usage; 3 input missing; 4 checksum/recomputation/null point/control; 5 pre-registered rule not met, form not main_guard_i1 or
#   guard (c) violated (codes from step 0 are passed on); 1 a check inside a called Python script that stops with a message.
#
# Only aggregates and checksums are printed. Nothing is submitted from here.
set -euo pipefail
[ $# -eq 0 ] || { echo "usage: $0   (no arguments; PRIMARY_I1_SPLITS=1 recomputes the meta-splits)" >&2; exit 2; }
cd "$(dirname "$0")/../.."
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}" OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export ORIENT=repro
STEM="s032_stage2"
STACK="artifacts/stack_${STEM}.json"; STACK_SHA="d278bbedf7fb4d7564e584a55d45959b826748dfe937d8becf1e0818007281eb"
STAGE2_GUARD="artifacts/stage2_s035.json"; STAGE2_GUARD_SHA="12f15243872681aeddad83c407e3e36955db18426e62331ee729869d474f4b86"
STAGE2="artifacts/stage2_primary_i1.json"
STAGE2_SHA="f5e24adfd382d24b36bd88b57deb8904a9d5288863250d5d314ceb09e54b181d"  # artifacts/stage2_primary_i1.json (c121 fit; pinned after the first build)
GUARD_FILE="artifacts/submissions/sub_s035_stage2_guard.csv"; GUARD_FILE_SHA="d4dce1aaa5cfbd1fd5668a9e7073a47734c0c5aacda7b7c2020255005a81f360"
GUARD_MAT="artifacts/test_${STEM}_s035guard.npy"; GUARD_MAT_SHA="79de6ca05cd297f3d069627a5b698cda6e7d053653bc59a8bc18a046db6bd9f9"
MAIN_FILE="artifacts/submissions/sub_s034n_stage2_main.csv"; MAIN_FILE_SHA="8b8892e474ace44048d6c02eba7622288599990f5f5f0814cbefaeb6bd9ce242"
FB_FILE="artifacts/submissions/sub_s033n_stage2.csv"; FB_FILE_SHA="76aba64fbc431b9ad403a5dec9fb2cefdb7fe2e76260149c96176459c4cd8dfe"
PRIV="artifacts/eda/private/stage2_i1"
IOOF="$PRIV/c121_i1_oof.json"
IOOF_SHA="7d28f6b0e0d437e3373c0e7e171d5ed5a37d255ce7ff0bf731d4559b2a70c0ac"  # c121 oof (without run time, repeatable; pinned after the first build)
ISPL="$PRIV/c121_i1_splits.json"
SPLITS_SHA="7e28c8f8920a57f951c464ba0416ed6ad9bb387e5a5686b9086d0619b139a606"  # c121 splits --n 20 (pinned after the first build)
GUARDS="$PRIV/guards_primary_i1.json"
SUFFIX="_primary_i1"
OUT="artifacts/submissions/sub_primary_i1.csv"
OUT_SHA="60b63d69bcdd5970cafaf4d5645440d8d375953871d6dd688ae5091b25256bee"  # sub_primary_i1.csv (pinned after the first build)
TESTMAT="artifacts/test_${STEM}${SUFFIX}.npy"
TESTMAT_SHA="7ce36020bb4f5bf2e61291f07817a783acefec677856a15bab780464a59902a7"  # its test matrix (pinned after the first build)
NOTES1="derived/sft/m3_d1_test_frozen_notes.jsonl"
NOTES2="derived/sft/m3_d2_test_frozen_notes.jsonl"
RANK1="runs/m3_fullte_rp/m3_test_rank/test"
RANK2="runs/m3d2te_rp/m3d2_test_rank/test"
RANK3="runs/m3d3te_rp/m3d3_test_rank/test"
Q397M3_TEST="runs/q397m3te_rp/q397m3_test_rank/test"
OOF_UNITS_VS_GUARD='{"fold": 17, "pre": -18, "set": 9}'  # main_guard_i1 − guard form in sixths (c121 oof; = the independent cross-check)
check() { echo "$1  $2" | sha256sum -c --quiet || { echo "ABORT: $2 differs (checksum)" >&2; exit 4; }; }
. tools/builds/orient_list.sh  # rotation list of the v1r test side: always repro here
orient_select
[ "$ORIENT" = repro ] || { echo "ABORT: this build path only runs with ORIENT=repro" >&2; exit 2; }
mkdir -p "$PRIV"

# 0. build of the guard form complete (contains the main form, the fallback form, the intermediate file, the reproductions c118 oof/guard-oof,
#    the deployment files stage2_s034/s035, positive controls, file swap and the fixed checksums) — also the regression check of the earlier files
echo "== 0. guard form (tools/builds/s035_stage2_guard.sh) =="
bash tools/builds/s035_stage2_guard.sh || { rc=$?; echo "ABORT: build of the guard form (code $rc)" >&2; exit "$rc"; }
check "$GUARD_FILE_SHA" "$GUARD_FILE"
check "$GUARD_MAT_SHA" "$GUARD_MAT"
check "$MAIN_FILE_SHA" "$MAIN_FILE"
check "$FB_FILE_SHA" "$FB_FILE"
check "$STACK_SHA" "$STACK"
check "$STAGE2_GUARD_SHA" "$STAGE2_GUARD"

TMP="$(mktemp -d "${TMPDIR:-/tmp}/primary_i1.XXXXXX")"
BAK=""
cleanup() {
  rm -rf -- "${TMP:?}"
  rm -f -- "artifacts/test_${STEM}_i1base.npy" "artifacts/test_${STEM}_i1null.npy" "artifacts/test_${STEM}_i1zero.npy" \
           "artifacts/test_${STEM}_i1swap.npy"
  [ -n "$BAK" ] && rm -rf -- "$BAK"
  return 0
}
trap cleanup EXIT

# 1. nested reproduction (guard form bit-identical to the row values in stage2_guard/, copy null point (i), Δ in sixths fixed, 15 placebos)
echo "== 1. c121 oof =="
python -I eda/c121_stage2_i1.py oof --stack "$STEM" | grep -E "I1|ABORT" | cut -c1-300 || { echo "ABORT: c121 oof" >&2; exit 4; }
python -I - "$IOOF" "$OOF_UNITS_VS_GUARD" <<'PY' || exit 4
import json
import sys

g, want = json.load(open(sys.argv[1])), json.loads(sys.argv[2])
bad = []
if not g["null_point_i"]["ok"]:
    bad.append(f"copy null point (i) violated {g['null_point_i']}")
got = {f: g["sets"]["i1"][f]["vs_guard"]["units"] for f in ("fold", "pre", "set")}
if got != want:
    bad.append(f"Δ against the guard form in sixths {got} instead of {want}")
if not all(g["sets"]["guard"][f]["rows_equal_reference"] for f in ("fold", "pre", "set")) or g["placebo"]["n"] != 15:
    bad.append(f"guard form not bit-identical to the row values in stage2_guard/ or {g['placebo']['n']} placebos instead of 15")
print(f"  c121 oof: guard form bit-identical, copy null point (i) bit-identical, Δ against the guard form {got} sixths, ½(F+M) {g['summary']['A']:+.5f}, "
      f"placebo mean {g['placebo']['A_mean']:+.5f} (n {g['placebo']['n']})" if not bad else "ABORT: " + "; ".join(bad))
raise SystemExit(4 if bad else 0)
PY
check "$IOOF_SHA" "$IOOF"

# 2. expected-value rule: (i) OOF part, (ii) against the placebos, (iii) over the meta-splits (file part of (i): step 5)
echo "== 2. expected-value rule (c121 rule) =="
if [ "${PRIMARY_I1_SPLITS:-0}" = 1 ] || [ ! -f "$ISPL" ]; then
  [ -f "$ISPL" ] && cp -p "$ISPL" "$TMP/splits_before.json"
  python -I eda/c121_stage2_i1.py splits --stack "$STEM" --n 20 | grep -E "over|ABORT" | cut -c1-300 \
    || { echo "ABORT: c121 splits" >&2; exit 4; }
  if [ -f "$TMP/splits_before.json" ] && ! cmp -s "$ISPL" "$TMP/splits_before.json"; then
    cp -p "$TMP/splits_before.json" "$ISPL"
    echo "ABORT: recomputation of the meta-splits differs from the existing $ISPL — backup restored" >&2; exit 4
  fi
fi
check "$SPLITS_SHA" "$ISPL"
python -I eda/c121_stage2_i1.py rule || { rc=$?; echo "ABORT: expected-value rule not met — no deployment (code $rc)" >&2; exit "$rc"; }

# 3. deployment file (c121 fit: stage2_s035.json recomputed character for character, null point (e) against c121 oof, rule required)
echo "== 3. deployment file stage2_primary_i1.json (c121 fit) =="
if [ -f "$STAGE2" ]; then
  python -I eda/c121_stage2_i1.py fit --stack "$STEM" --verify || { rc=$?; echo "ABORT: $STAGE2 is not the recomputation" >&2; exit "$rc"; }
else
  python -I eda/c121_stage2_i1.py fit --stack "$STEM" --save || { rc=$?; echo "ABORT: $STAGE2 not built" >&2; exit "$rc"; }
fi
check "$STAGE2_SHA" "$STAGE2"
python -I - "$STAGE2_GUARD" "$STAGE2" "$STAGE2_GUARD_SHA" <<'PY' || exit 4
import json
import sys

a, b = (json.load(open(p)) for p in sys.argv[1:3])
bad = []
if [k for k in b if k not in ("i1_build", "derived_from")] != [k for k in a if k != "derived_from"]:
    bad.append("order/set of the fields differs")
diff = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
if set(diff) - {"models", "i1_build", "derived_from"}:
    bad.append(f"fields other than models/i1_build/derived_from differ: {diff}")
if list(b["models"]) != ["main", "fallback"] or b["models"]["fallback"] != a["models"]["fallback"]:
    bad.append("fallback model not bit-identical to stage2_s035.json")
m, am = b["models"]["main"], a["models"]["main"]
i1 = ["row_ent", "row_max", "row_d_ntop", "row_ch_ntop", "task_rows"]
want_i1 = [f"{r}_x_{c}" for r in i1 for c in ("f_logp", "is_general", "is_none")]
if m.get("feature_set") != "main_guard_i1" or m.get("protect_none") is not True or m["features"][:48] != am["features"] \
        or m["features"][48:] != want_i1 or len(m["w"]) != 63 or m["abstain"]:
    bad.append(f"main model: {m.get('feature_set')} / guard {m.get('protect_none')} / {len(m['features'])} features / abstention {m['abstain']}")
d = b.get("derived_from") or {}
if d.get("file") != sys.argv[1] or d.get("sha256") != sys.argv[3] or d.get("previous") != a.get("derived_from"):
    bad.append(f"derived_from {d}")
if (b.get("i1_build") or {}).get("rule_oof", {}).get("verdict_oof") != "deploy":
    bad.append("expected-value rule (OOF part) in the deployment file is not \"deploy\"")
print((f"  field comparison with stage2_s035.json: only models.main replaced (main_guard_i1, 63 features, guard on; OOF share of new top codes "
       f"{m['oof_changed_top_share']:.4f}), fallback/note guard/β/K/λ bit-identical") if not bad else "ABORT: " + "; ".join(bad))
raise SystemExit(4 if bad else 0)
PY

# shared sources of the test side (as in tools/builds/s035_stage2_guard.sh with ORIENT=repro)
python -I tools/orient_list.py check --mode repro --spec "$V1R_TEST" --json "$PRIV/check_orient_primary_i1.json" || exit $?
SRC=(--test "$V1R_TEST" --fill "v1r=v1" --test "v1=v1te_a_t4/v1_test_a/test+v1te_b_t4/v1_test_b/test"
     --test "zs27q=zs27q_test0_rp/zs27q_test0/test+zs27q_test1_rp/zs27q_test1/test" --fill "think18nb=zs27q"
     --test "think18nb=zs27v_te_rp/think_te/test/think@derived/think_rows/test.npy,zs27v_n3_rp/think_nte/test/think@derived/think_rows/test_tb_q.npy"
     --proto "pr27xm=runs/pr27te_rp/proto_te/proto" --rank "pr27xm=runs/n1_rp/rank_test/test"
     --rank-only "m3x3_full=$RANK1,$RANK2,$RANK3"
     --rank-only "q397m3=$Q397M3_TEST" --rank-none-band-for "q397m3=0.27,0.35"
     --align 0.75 --align-min 3 --blank-rows derived/blank/test_rows.npy)
FILTER="SECOND STAGE|note guard|written|ABORT|Traceback|Error|--stage2"
# file without second stage: the unchanged path tools/make_submission.py
ms() { PYTHONDONTWRITEBYTECODE=1 python -I tools/make_submission.py "$@" 2>&1 | grep -vi '^warning' | grep -E "$FILTER" | cut -c1-320; return "${PIPESTATUS[0]}"; }
# with the second stage main_guard_i1: eda/c121_stage2_i1.py submit (make_submission.py as a module, the set admitted for the main model only)
ms_i1() { PYTHONDONTWRITEBYTECODE=1 python -I eda/c121_stage2_i1.py submit "$@" 2>&1 | grep -vi '^warning' | grep -E "$FILTER" | cut -c1-320; return "${PIPESTATUS[0]}"; }

# 4. copy null point (ii): weight 0 from stage2_primary_i1.json -> file without second stage byte-identical (computes the 63 features of main_guard_i1)
echo "== 4. copy null point (ii) with the null model from stage2_primary_i1.json =="
python -I eda/c118_stage2.py null --stack "$STEM" --source "$STAGE2" --out "$TMP/null_primary_i1.json" >/dev/null
ms --name "$STEM" "${SRC[@]}" --out "$TMP/sub_base.csv" --suffix "_i1base" || { echo "ABORT: file without second stage not computed" >&2; exit 4; }
ms_i1 --name "$STEM" "${SRC[@]}" --out "$TMP/sub_null.csv" --suffix "_i1null" --stage2 "$TMP/null_primary_i1.json" --stage2-report "$TMP/guards_null.json" \
  || { echo "ABORT: null point (ii) not computed" >&2; exit 4; }
cmp -s "$TMP/sub_base.csv" "$TMP/sub_null.csv" || { echo "ABORT: null point (ii) violated — weight 0 changes the file" >&2; exit 4; }
cmp -s "artifacts/test_${STEM}_i1base.npy" "artifacts/test_${STEM}_i1null.npy" || { echo "ABORT: null point (ii) violated — test matrix" >&2; exit 4; }
python -I - "$TMP/guards_null.json" <<'PY' || exit 4
import json
import sys

g = json.load(open(sys.argv[1]))
gn = g["guard_notes"]
bad = []
if not g["null"] or g["feature_set"] != "main" or g.get("model_feature_set") != "main_guard_i1" or g["n_features"] != 63 or g.get("rows_protected") != 0:
    bad.append(f"null model {g['null']}, form {g['feature_set']}/{g.get('model_feature_set')}, {g['n_features']} features, guarded {g.get('rows_protected')}")
if gn.get("rule") != "widened" or gn.get("share_z_max") != 3.0 or gn.get("n_ref") != 1500 or not gn["ok"]:
    bad.append(f"rule {gn.get('rule')} / {gn.get('share_z_max')} / {gn.get('n_ref')}, guard {gn['ok']}")
print(f"  null point (ii): weight 0 from stage2_primary_i1.json -> file without second stage byte-identical (CSV and test matrix); 63 features of "
      f"main_guard_i1 computed, guard (a) held under the widened rule (n_test {gn.get('n_test')})" if not bad else "ABORT: " + "; ".join(bad))
raise SystemExit(4 if bad else 0)
PY

# 5. copy null point (iii) = rule (i), file part: main_guard_i1 model with the 48 weights of the guard form, I1 weights 0 -> guard-form file
echo "== 5. copy null point (iii): guard-form weights in the main_guard_i1 model, I1 weights 0 -> guard-form file =="
python -I - "$STAGE2" "$STAGE2_GUARD" "$TMP/stage2_zero.json" <<'PY' || exit 4
import json
import sys

b, a = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
am, m = a["models"]["main"], b["models"]["main"]
if m["features"][:48] != am["features"] or am.get("feature_set") != "main_guard" or am.get("protect_none") is not True:
    raise SystemExit("ABORT: features/form of the guard form do not match")
s = json.loads(json.dumps(b))
s["models"]["main"].update({"w": am["w"] + [0.0] * 15, "mu": am["mu"] + [0.0] * 15, "sd": am["sd"] + [1.0] * 15,
                            "oof_changed_top_share": am["oof_changed_top_share"], "oof_changed_top3_share": am["oof_changed_top3_share"]})
open(sys.argv[3], "w").write(json.dumps(s, indent=1, ensure_ascii=False) + "\n")
PY
ms_i1 --name "$STEM" "${SRC[@]}" --out "$TMP/sub_zero.csv" --suffix "_i1zero" --stage2 "$TMP/stage2_zero.json" --stage2-report "$TMP/guards_zero.json" \
  || { echo "ABORT: null point (iii) not computed" >&2; exit 4; }
cmp -s "$TMP/sub_zero.csv" "$GUARD_FILE" || { echo "ABORT: null point (iii) violated — I1 weights 0 do not give the guard-form file (CSV)" >&2; exit 4; }
cmp -s "artifacts/test_${STEM}_i1zero.npy" "$GUARD_MAT" || { echo "ABORT: null point (iii) violated — test matrix != guard form" >&2; exit 4; }
python -I - "$TMP/guards_zero.json" "artifacts/eda/private/stage2_guard/guards_s035.json" "$PRIV/nullpoint_iii_primary_i1.json" <<'PY' || exit 4
import json
import sys

g, g35 = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
bad = []
if g.get("model_feature_set") != "main_guard_i1" or g["n_features"] != 63 or g.get("protect_none") is not True or g["null"]:
    bad.append(f"form {g.get('model_feature_set')} / {g['n_features']} / guard {g.get('protect_none')} / null model {g['null']}")
for k in ("rows_protected", "share_top3_changed"):
    if g.get(k) != g35.get(k):
        bad.append(f"{k} {g.get(k)} != guard form {g35.get(k)}")
if g["guard_changed_top"]["test"] != g35["guard_changed_top"]["test"] or g["guard_notes"]["checks"] != g35["guard_notes"]["checks"]:
    bad.append("guards (a)/(c) != guard form")
json.dump({"iii_equals_guard_form_csv": True, "iii_equals_guard_form_testmatrix": True, "rows_protected": g.get("rows_protected"),
           "changed_top_test": g["guard_changed_top"]["test"]}, open(sys.argv[3], "w"), indent=1)
print(f"  (iii) I1 weights 0: CSV and test matrix byte-identical to the guard form; guarded {g.get('rows_protected')}, new top code "
      f"{g['guard_changed_top']['test']:.4f} as in the guard form (rule (i), file part, passed)" if not bad else "ABORT: " + "; ".join(bad))
raise SystemExit(4 if bad else 0)
PY

# 6. file swap: notes file of draw d2 in place of d1 (consistent provenance) -> guard (a) fires -> fallback = fallback-form file
echo "== 6. file swap (notes file d2 as d1) =="
python -I - "$STAGE2" "$TMP" "$NOTES1" "$NOTES2" "$RANK2" "_i1swap" "${SRC[@]}" <<'PY' > "$TMP/swap.log" 2>&1 \
  || { echo "ABORT: file swap not computed" >&2; grep -E "ABORT|Error|SystemExit|--stage2" "$TMP/swap.log" >&2; exit 4; }
import importlib.util
import sys
from pathlib import Path

stage2, tmp, n1, n2, r2, swap_suffix = sys.argv[1:7]
src = sys.argv[7:]
spec = importlib.util.spec_from_file_location("c121_i1swap", Path.cwd() / "eda" / "c121_stage2_i1.py")
C121 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(C121)
MS = C121.load_make_submission("make_submission_i1swap")  # as ms_i1: main_guard_i1 for the main model only
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
python -I - "$TMP/guards_swap.json" "$TMP/sub_swap.csv" "$FB_FILE_SHA" "$PRIV/swap_control_primary_i1.json" <<'PY' || exit 4
import hashlib
import json
import sys

g = json.load(open(sys.argv[1]))
gn = g["guard_notes"]
failed = sorted(c["stat"] for c in gn["checks"] if not c["ok"])
sha = hashlib.sha256(open(sys.argv[2], "rb").read()).hexdigest()
bad = []
if gn["ok"] or g["feature_set"] != "fallback" or "identical[d1-d2]" not in failed or gn.get("rule") != "widened" or "model_feature_set" in g:
    bad.append(f"guard (a) did not fire on the swap, or the fallback used a foreign feature set (form {g['feature_set']}, failed {failed})")
if sha != sys.argv[3]:
    bad.append(f"fallback file {sha[:16]}… != fallback-form file")
json.dump({"feature_set": g["feature_set"], "failed": failed, "sub_sha256": sha, "equals_fallback_form_file": sha == sys.argv[3]}, open(sys.argv[4], "w"), indent=1)
print(f"  guard (a) fires under the widened rule (failed {', '.join(failed)}) -> fallback (40 features, without the guard); file = fallback-form file "
      f"byte-identical ({sha[:16]}…)" if not bad else "ABORT: " + "; ".join(bad))
raise SystemExit(4 if bad else 0)
PY

# 7. main file (form main with main_guard_i1 and the guard required; guard (c) violated -> no file, exit 5)
echo "== 7. main file with stage2_primary_i1.json =="
if [ -f "$OUT" ] || [ -f "$TESTMAT" ]; then  # only replace byte-identically
  BAK="$(mktemp -d "${TMPDIR:-/tmp}/primary_i1_backup.XXXXXX")"
  for F in "$OUT" "${OUT%.csv}.json" "$TESTMAT"; do [ -f "$F" ] && cp -p "$F" "$BAK/"; done
fi
restore() {  # remove the new outputs, restore an existing backup
  rm -f -- "$OUT" "${OUT%.csv}.json" "$TESTMAT"
  if [ -n "$BAK" ]; then
    for F in "$OUT" "${OUT%.csv}.json" "$TESTMAT"; do [ -f "$BAK/$(basename "$F")" ] && cp -p "$BAK/$(basename "$F")" "$F"; done
  fi
  return 0
}
rm -f "$GUARDS"
if ! ms_i1 --name "$STEM" "${SRC[@]}" --out "$OUT" --suffix "$SUFFIX" --stage2 "$STAGE2" --stage2-report "$GUARDS"; then
  restore
  if [ -f "$GUARDS" ] && python -I -c "import json,sys;sys.exit(1 if json.load(open('$GUARDS'))['guard_changed_top']['ok'] else 0)"; then
    echo "ABORT: guard (c) of the second stage violated (statistics in $GUARDS) — no file" >&2; exit 5
  fi
  echo "ABORT: make_submission (c121 submit) with --stage2 failed" >&2; exit 4
fi
[ -f "$OUT" ] || { echo "ABORT: $OUT was not written" >&2; exit 4; }
python -I - "$GUARDS" "$TMP/guards_null.json" <<'PY' || { rc=$?; restore; exit "$rc"; }
import json
import sys

g, gnull = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
gn = g["guard_notes"]
if g["feature_set"] != "main" or g.get("model_feature_set") != "main_guard_i1" or g.get("protect_none") is not True:
    print(f"ABORT: form {g['feature_set']} / {g.get('model_feature_set')} / guard {g.get('protect_none')} instead of main / main_guard_i1 / on — no file")
    raise SystemExit(5)
bad = []
if gn.get("rule") != "widened" or not gn["ok"] or g["null"] or g["n_features"] != 63 or not g["guard_changed_top"]["ok"]:
    bad.append(f"rule {gn.get('rule')}, guard {gn['ok']}, null model {g['null']}, {g['n_features']} features, (c) {g['guard_changed_top']['ok']}")
if gn["test"] != gnull["guard_notes"]["test"] or gn["checks"] != gnull["guard_notes"]["checks"]:
    bad.append("test aggregates of the main file != those of step 4")
print(f"  main form main_guard_i1 (63 features, guard on: {g['rows_protected']} test rows keep the base order); guard (c) new top code "
      f"{g['guard_changed_top']['test']:.4f} (OOF {g['guard_changed_top']['oof']:.4f}) ok; note guard = step 4" if not bad else "ABORT: " + "; ".join(bad))
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

# 8. check of the file and comparison (aggregates only)
echo "== 8. check and comparison (aggregates only) =="
check "$GUARD_FILE_SHA" "$GUARD_FILE"
check "$MAIN_FILE_SHA" "$MAIN_FILE"
check "$FB_FILE_SHA" "$FB_FILE"
python -I - "$OUT" "$GUARD_FILE" "$MAIN_FILE" "$FB_FILE" "$GUARDS" "$PRIV/compare_primary_i1.json" <<'PY' || exit 4
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
for path, label in ((sys.argv[2], "guard form (main_guard)"), (sys.argv[3], "main form"), (sys.argv[4], "fallback form")):
    ref = read(path)
    b = new.loc[ref.index]
    t_new, t_ref = b.str.split().str[0], ref.str.split().str[0]
    top = float((t_ref != t_new).mean())
    top3 = float((ref != b).mean())
    s_new, s_ref = b.str.split().apply(frozenset), ref.str.split().apply(frozenset)
    e = {"share_other_top1": top, "share_other_top3_ordered": top3, "share_other_top3_as_set": float((s_new != s_ref).mean()),
         "rows_other_top1": int((t_ref != t_new).sum()), "rows_other_top3_ordered": int((ref != b).sum()),
         "share_top_general_new": float(t_new.isin(gen).mean()), "share_top_general_ref": float(t_ref.isin(gen).mean()),
         "share_top_none_new": float((t_new == "NONE").mean()), "share_top_none_ref": float((t_ref == "NONE").mean())}
    rep["vs"][Path(path).name] = e
    print(f"  vs. {label}: other top code in {top:.4f} of the rows ({e['rows_other_top1']}), other first three (ordered) in {top3:.4f} "
          f"({e['rows_other_top3_ordered']}), as a set {e['share_other_top3_as_set']:.4f}; share of general codes at the top "
          f"{e['share_top_general_ref']:.4f} -> {e['share_top_general_new']:.4f}, NONE {e['share_top_none_ref']:.4f} -> {e['share_top_none_new']:.4f}")
g = json.load(open(sys.argv[5]))
rep.update({k: g[k] for k in ("guard_changed_top", "guard_novel_task_top5", "share_top3_changed", "share_top3_changed_oof", "rows_protected")})
gc, gn = g["guard_changed_top"], g["guard_novel_task_top5"]
print(f"  guard (c): new top code {gc['test']:.4f} (OOF main_guard_i1 {gc['oof']:.4f}, band {gc['factor'][0] * gc['oof']:.4f}…{gc['factor'][1] * gc['oof']:.4f}) -> "
      f"{'ok' if gc['ok'] else 'VIOLATED'}; (b) \"novel for the task\" among the first 5: test {gn['test']:.4f}, OOF {gn['oof']:.4f}; guarded "
      f"{g['rows_protected']} test rows")
Path(sys.argv[6]).write_text(json.dumps(rep, indent=1, ensure_ascii=False) + "\n")
PY
python -I tools/orient_list.py compare "$GUARD_FILE" "$OUT" --label "guard form (main_guard)" --json "$PRIV/compare_primary_i1_vs_guard_orient.json"

# 9. checksums
echo "== 9. checksums =="
check "$OUT_SHA" "$OUT"
check "$TESTMAT_SHA" "$TESTMAT"
sha256sum "$STACK" "$STAGE2_GUARD" "$STAGE2" "$GUARD_FILE" "$GUARD_MAT" "$MAIN_FILE" "$FB_FILE" "$OUT" "$TESTMAT" "${OUT%.csv}.json" "$IOOF" "$ISPL" "$GUARDS"
echo "primary_i1: $OUT built (main_guard_i1 + NONE guard under the widened rule, rotation list repro; nothing is submitted from here)"
