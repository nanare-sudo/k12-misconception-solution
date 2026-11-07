#!/usr/bin/env bash
# Primary line, reranker MAIN form with the NONE guard and general-code identifiers, released rotation list: the main form with the
# feature set "main_guard" of the second stage — the 43 features of the main form + identifiers of the five general codes (48) —
# and the NONE guard (if the base form has NONE at the top, the base order of the row stays). Deployment file
# artifacts/stage2_s035.json (eda/c118_stage2.py guard-fit): stage2_s034.json recomputed character for character, then ONLY
# models.main replaced by the main_guard model; fallback model (40 features), note guard under the widened rule (reference and
# rule), β, K, λ and all input checksums stay those of the main form. Rotation list ORIENT=repro (released list); base = the main
# form with the released list (tools/builds/s034n_stage2_main.sh).
# Order on the test side unchanged: blend -> margin alignment (one set) -> second stage (+ guard) -> REL-01.
#
#     bash tools/builds/s035_stage2_guard.sh      # -> artifacts/submissions/sub_s035_stage2_guard.csv
#
# STEPS: 0 build of the main form with the released list complete (tools/builds/s034n_stage2_main.sh, including the fallback form
#   and the intermediate file, positive controls of the note guard, file swap, fixed checksums); 1 nested reproduction (c118
#   guard-oof: main form = row values of stage2/c118_oof_rows.npz bit for bit, copy null point (i), Δ against the main form in
#   sixths fixed); 2 deployment file (c118 guard-fit --save or --verify; fixed checksum; field comparison with stage2_s034.json);
#   3 copy null point (ii): null model from stage2_s035.json -> file without second stage byte-identical (CSV and test matrix;
#   computes the 48 features of main_guard); 4 copy null point (iii): main_guard model with the 43 weights of the main form,
#   identifier weights 0, guard off -> main-form file byte-identical; plus (iv) the same with the guard on -> differences from the
#   main-form file only in the guarded rows (count); 5 file swap (notes file d2 in place of d1, consistent provenance) -> guard
#   (a) fires under the widened rule -> fallback, file = fallback-form file byte-identical; 6 main file (form main with feature
#   set main_guard and the guard required; guard (c) with the OOF share of main_guard, otherwise exit 5 without a file);
#   7 comparison with the main-form and fallback-form files (aggregates only); 8 checksums. An existing output file is only
#   replaced by a byte-identical rebuild. Statistics and logs under artifacts/eda/private/stage2_guard/.
# EXIT: 2 usage; 3 input missing; 4 checksum/recomputation/null point/control; 5 form not main_guard or guard (c) violated
#   (codes from step 0 are passed on); 1 a check inside a called Python script that stops with a message.
#
# Only aggregates and checksums are printed. Nothing is submitted from here.
set -euo pipefail
[ $# -eq 0 ] || { echo "usage: $0   (no arguments)" >&2; exit 2; }
cd "$(dirname "$0")/../.."
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}" OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export ORIENT=repro
STEM="s032_stage2"
STACK="artifacts/stack_${STEM}.json"; STACK_SHA="d278bbedf7fb4d7564e584a55d45959b826748dfe937d8becf1e0818007281eb"
STAGE2_MAIN="artifacts/stage2_s034.json"; STAGE2_MAIN_SHA="377d863bf08b07dda20bb37fab1af4a0c929e7fbbb8c3f221f1aed6e14f4397a"
STAGE2="artifacts/stage2_s035.json"
STAGE2_SHA="12f15243872681aeddad83c407e3e36955db18426e62331ee729869d474f4b86"  # artifacts/stage2_s035.json (c118 guard-fit; pinned after the first build)
MAIN_FILE="artifacts/submissions/sub_s034n_stage2_main.csv"; MAIN_FILE_SHA="8b8892e474ace44048d6c02eba7622288599990f5f5f0814cbefaeb6bd9ce242"
FB_FILE="artifacts/submissions/sub_s033n_stage2.csv"; FB_FILE_SHA="76aba64fbc431b9ad403a5dec9fb2cefdb7fe2e76260149c96176459c4cd8dfe"
GOOF="artifacts/eda/private/stage2_guard/c118_guard_oof.json"
PRIV="artifacts/eda/private/stage2_guard"
GUARDS="$PRIV/guards_s035.json"
SUFFIX="_s035guard"
OUT="artifacts/submissions/sub_s035_stage2_guard.csv"
OUT_SHA="d4dce1aaa5cfbd1fd5668a9e7073a47734c0c5aacda7b7c2020255005a81f360"  # sub_s035_stage2_guard.csv (pinned after the first build)
TESTMAT="artifacts/test_${STEM}${SUFFIX}.npy"
NOTES1="derived/sft/m3_d1_test_frozen_notes.jsonl"
NOTES2="derived/sft/m3_d2_test_frozen_notes.jsonl"
RANK1="runs/m3_fullte_rp/m3_test_rank/test"
RANK2="runs/m3d2te_rp/m3d2_test_rank/test"
RANK3="runs/m3d3te_rp/m3d3_test_rank/test"
Q397M3_TEST="runs/q397m3te_rp/q397m3_test_rank/test"
OOF_UNITS_VS_MAIN='{"fold": 12, "pre": 46, "set": 28}'  # main_guard − main form in sixths (c118 guard-oof; fixed before the build)
check() { echo "$1  $2" | sha256sum -c --quiet || { echo "ABORT: $2 differs (checksum)" >&2; exit 4; }; }
. tools/builds/orient_list.sh  # rotation list of the v1r test side: always repro here
orient_select
[ "$ORIENT" = repro ] || { echo "ABORT: this build path only runs with ORIENT=repro" >&2; exit 2; }
mkdir -p "$PRIV"

# 0. build of the main form with the released list complete (includes the fallback form, the intermediate file, the OOF reproduction
#    c118 oof, the deployment file stage2_s034.json, positive controls, file swap and the fixed checksums)
echo "== 0. main form with the released rotation list (tools/builds/s034n_stage2_main.sh) =="
bash tools/builds/s034n_stage2_main.sh || { rc=$?; echo "ABORT: build of the main form with the released list (code $rc)" >&2; exit "$rc"; }
check "$MAIN_FILE_SHA" "$MAIN_FILE"
check "$FB_FILE_SHA" "$FB_FILE"
check "$STACK_SHA" "$STACK"
check "$STAGE2_MAIN_SHA" "$STAGE2_MAIN"

# 1. nested reproduction of main_guard (main form bit-identical to the row values in stage2/, copy null point (i))
echo "== 1. c118 guard-oof =="
python -I eda/c118_stage2.py guard-oof --stack "$STEM" | grep -E "main_guard|null point|ABORT" | cut -c1-260 \
  || { echo "ABORT: guard-oof" >&2; exit 4; }
python -I - "$GOOF" "$OOF_UNITS_VS_MAIN" <<'PY' || exit 4
import json
import sys

g, want = json.load(open(sys.argv[1])), json.loads(sys.argv[2])
bad = []
if not g["null_point_i"]["ok"]:
    bad.append(f"copy null point (i) violated {g['null_point_i']}")
got = {f: g["sets"]["main_guard"][f]["vs_main"]["units"] for f in ("fold", "pre", "set")}
if got != want:
    bad.append(f"Δ against the main form in sixths {got} instead of {want}")
print(f"  guard-oof: copy null point (i) bit-identical, Δ against the main form {got} sixths (= the check fixed before the build)" if not bad else "ABORT: " + "; ".join(bad))
raise SystemExit(4 if bad else 0)
PY

# 2. deployment file (guard-fit: recomputation under the widened rule = stage2_s034.json character for character; null point (e) against guard-oof)
echo "== 2. deployment file stage2_s035.json (c118 guard-fit) =="
if [ -f "$STAGE2" ]; then
  python -I eda/c118_stage2.py guard-fit --stack "$STEM" --out "$STAGE2" --verify || { rc=$?; echo "ABORT: $STAGE2 is not the recomputation" >&2; exit "$rc"; }
else
  python -I eda/c118_stage2.py guard-fit --stack "$STEM" --out "$STAGE2" --save || { rc=$?; echo "ABORT: $STAGE2 not built" >&2; exit "$rc"; }
fi
check "$STAGE2_SHA" "$STAGE2"
python -I - "$STAGE2_MAIN" "$STAGE2" "$STAGE2_MAIN_SHA" <<'PY' || exit 4
import json
import sys

a, b = (json.load(open(p)) for p in sys.argv[1:3])
bad = []
if [k for k in b if k not in ("guard_build", "derived_from")] != [k for k in a if k != "derived_from"]:
    bad.append("order/set of the fields differs")
diff = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
if set(diff) - {"models", "guard_build", "derived_from"}:
    bad.append(f"fields other than models/guard_build/derived_from differ: {diff}")
if list(b["models"]) != ["main", "fallback"] or b["models"]["fallback"] != a["models"]["fallback"]:
    bad.append("fallback model not bit-identical to stage2_s034.json")
m = b["models"]["main"]
if m.get("feature_set") != "main_guard" or m.get("protect_none") is not True or m["features"][:43] != a["models"]["main"]["features"] \
        or len(m["features"]) != 48 or m["abstain"]:
    bad.append(f"main model: {m.get('feature_set')} / guard {m.get('protect_none')} / {len(m['features'])} features / abstention {m['abstain']}")
d = b.get("derived_from") or {}
if d.get("file") != sys.argv[1] or d.get("sha256") != sys.argv[3] or d.get("previous") != a.get("derived_from"):
    bad.append(f"derived_from {d}")
print((f"  field comparison with stage2_s034.json: only models.main replaced (main_guard, 48 features, guard on; OOF share of new top codes "
       f"{m['oof_changed_top_share']:.4f}), fallback/note guard/β/K/λ bit-identical") if not bad else "ABORT: " + "; ".join(bad))
raise SystemExit(4 if bad else 0)
PY

# shared sources of the test side (as in tools/builds/s034_stage2_main.sh with ORIENT=repro)
python -I tools/orient_list.py check --mode repro --spec "$V1R_TEST" --json "$PRIV/check_orient_s035.json" || exit $?
SRC=(--test "$V1R_TEST" --fill "v1r=v1" --test "v1=v1te_a_t4/v1_test_a/test+v1te_b_t4/v1_test_b/test"
     --test "zs27q=zs27q_test0_rp/zs27q_test0/test+zs27q_test1_rp/zs27q_test1/test" --fill "think18nb=zs27q"
     --test "think18nb=zs27v_te_rp/think_te/test/think@derived/think_rows/test.npy,zs27v_n3_rp/think_nte/test/think@derived/think_rows/test_tb_q.npy"
     --proto "pr27xm=runs/pr27te_rp/proto_te/proto" --rank "pr27xm=runs/n1_rp/rank_test/test"
     --rank-only "m3x3_full=$RANK1,$RANK2,$RANK3"
     --rank-only "q397m3=$Q397M3_TEST" --rank-none-band-for "q397m3=0.27,0.35"
     --align 0.75 --align-min 3 --blank-rows derived/blank/test_rows.npy)
ms() { PYTHONDONTWRITEBYTECODE=1 python -I tools/make_submission.py "$@" 2>&1 | grep -vi '^warning' | grep -E "SECOND STAGE|note guard|written|ABORT|Traceback|Error|--stage2" | cut -c1-320; return "${PIPESTATUS[0]}"; }

TMP="$(mktemp -d "${TMPDIR:-/tmp}/s035_guard.XXXXXX")"
BAK=""
cleanup() {
  rm -rf -- "${TMP:?}"
  rm -f -- "artifacts/test_${STEM}_guard_base.npy" "artifacts/test_${STEM}_guard_null.npy" "artifacts/test_${STEM}_guard_iii.npy" \
           "artifacts/test_${STEM}_guard_iv.npy" "artifacts/test_${STEM}_guard_swap.npy"
  [ -n "$BAK" ] && rm -rf -- "$BAK"
  return 0
}
trap cleanup EXIT

# 3. copy null point (ii): weight 0 from stage2_s035.json -> file without second stage byte-identical (computes the 48 features of main_guard)
echo "== 3. copy null point (ii) with the null model from stage2_s035.json =="
python -I eda/c118_stage2.py null --stack "$STEM" --source "$STAGE2" --out "$TMP/null_s035.json" >/dev/null
ms --name "$STEM" "${SRC[@]}" --out "$TMP/sub_base.csv" --suffix "_guard_base" || { echo "ABORT: file without second stage not computed" >&2; exit 4; }
ms --name "$STEM" "${SRC[@]}" --out "$TMP/sub_null.csv" --suffix "_guard_null" --stage2 "$TMP/null_s035.json" --stage2-report "$TMP/guards_null.json" \
  || { echo "ABORT: null point (ii) not computed" >&2; exit 4; }
cmp -s "$TMP/sub_base.csv" "$TMP/sub_null.csv" || { echo "ABORT: null point (ii) violated — weight 0 changes the file" >&2; exit 4; }
cmp -s "artifacts/test_${STEM}_guard_base.npy" "artifacts/test_${STEM}_guard_null.npy" || { echo "ABORT: null point (ii) violated — test matrix" >&2; exit 4; }
python -I - "$TMP/guards_null.json" <<'PY' || exit 4
import json
import sys

g = json.load(open(sys.argv[1]))
gn = g["guard_notes"]
bad = []
if not g["null"] or g["feature_set"] != "main" or g.get("model_feature_set") != "main_guard" or g["n_features"] != 48 or g.get("rows_protected") != 0:
    bad.append(f"null model {g['null']}, form {g['feature_set']}/{g.get('model_feature_set')}, {g['n_features']} features, guarded {g.get('rows_protected')}")
if gn.get("rule") != "widened" or gn.get("share_z_max") != 3.0 or gn.get("n_ref") != 1500 or not gn["ok"]:
    bad.append(f"rule {gn.get('rule')} / {gn.get('share_z_max')} / {gn.get('n_ref')}, guard {gn['ok']}")
print(f"  null point (ii): weight 0 from stage2_s035.json -> file without second stage byte-identical (CSV and test matrix); 48 features of main_guard "
      f"computed, guard (a) held under the widened rule (n_test {gn.get('n_test')})" if not bad else "ABORT: " + "; ".join(bad))
raise SystemExit(4 if bad else 0)
PY

# 4. copy null points (iii)/(iv): main_guard model with the weights of the main form and identifier weights 0
echo "== 4. copy null point (iii): main-form weights in the main_guard model, identifiers 0, guard off -> main-form file =="
python -I - "$STAGE2" "$STAGE2_MAIN" "$TMP/stage2_iii.json" "$TMP/stage2_iv.json" <<'PY' || exit 4
import json
import sys

b, a = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
am = a["models"]["main"]
for out, protect in ((sys.argv[3], False), (sys.argv[4], True)):
    s = json.loads(json.dumps(b))
    m = s["models"]["main"]
    m.update({"w": am["w"] + [0.0] * 5, "mu": am["mu"] + [0.0] * 5, "sd": am["sd"] + [1.0] * 5, "protect_none": protect,
              "oof_changed_top_share": am["oof_changed_top_share"]})
    open(out, "w").write(json.dumps(s, indent=1, ensure_ascii=False) + "\n")
PY
ms --name "$STEM" "${SRC[@]}" --out "$TMP/sub_iii.csv" --suffix "_guard_iii" --stage2 "$TMP/stage2_iii.json" --stage2-report "$TMP/guards_iii.json" \
  || { echo "ABORT: null point (iii) not computed" >&2; exit 4; }
cmp -s "$TMP/sub_iii.csv" "$MAIN_FILE" || { echo "ABORT: null point (iii) violated — the main-form weights in the main_guard model do not give the main-form file" >&2; exit 4; }
ms --name "$STEM" "${SRC[@]}" --out "$TMP/sub_iv.csv" --suffix "_guard_iv" --stage2 "$TMP/stage2_iv.json" --stage2-report "$TMP/guards_iv.json" \
  || { echo "ABORT: control (iv) not computed" >&2; exit 4; }
python -I - "$TMP/sub_iv.csv" "$MAIN_FILE" "$TMP/guards_iii.json" "$TMP/guards_iv.json" "$PRIV/nullpoint_iii_iv_s035.json" <<'PY' || exit 4
import json
import sys

import pandas as pd

iv = pd.read_csv(sys.argv[1], dtype=str, keep_default_na=False).set_index("trace_id")["root_code"]
ref = pd.read_csv(sys.argv[2], dtype=str, keep_default_na=False).set_index("trace_id")["root_code"].loc[iv.index]
g3, g4 = json.load(open(sys.argv[3])), json.load(open(sys.argv[4]))
diff = int((iv != ref).sum())
top_none_iv = int((iv.str.split().str[0] == "NONE").sum())
top_none_ref = int((ref.str.split().str[0] == "NONE").sum())
bad = []
if g3.get("rows_protected") != 0 or g3.get("protect_none") is not False or g3.get("model_feature_set") != "main_guard":
    bad.append(f"(iii) fields {g3.get('model_feature_set')} / {g3.get('protect_none')} / {g3.get('rows_protected')}")
if diff != g4.get("rows_protected") or not 0 <= top_none_iv - top_none_ref <= diff:
    bad.append(f"(iv) {diff} rows differ, guarded {g4.get('rows_protected')}, NONE tops {top_none_iv - top_none_ref:+d}")
json.dump({"iii_equals_main_form_file": True, "iv_rows_differ": diff, "iv_rows_protected": g4.get("rows_protected"),
           "iv_extra_none_tops": top_none_iv - top_none_ref}, open(sys.argv[5], "w"), indent=1)
print(f"  (iii) byte-identical to the main-form file; (iv) with the guard: {diff} test rows differ from the main-form file = guarded rows ({top_none_iv - top_none_ref:+d} "
      "NONE tops; counts only)" if not bad else "ABORT: " + "; ".join(bad))
raise SystemExit(4 if bad else 0)
PY

# 5. file swap: notes file of draw d2 in place of d1 (consistent provenance) -> guard (a) fires -> fallback = fallback-form file
echo "== 5. file swap (notes file d2 as d1) =="
python -I - "$STAGE2" "$TMP" "$NOTES1" "$NOTES2" "$RANK2" "_guard_swap" "${SRC[@]}" <<'PY' > "$TMP/swap.log" 2>&1 \
  || { echo "ABORT: file swap not computed" >&2; grep -E "ABORT|Error|SystemExit|--stage2" "$TMP/swap.log" >&2; exit 4; }
import importlib.util
import sys
from pathlib import Path

stage2, tmp, n1, n2, r2, swap_suffix = sys.argv[1:7]
src = sys.argv[7:]
spec = importlib.util.spec_from_file_location("make_submission_swap", Path.cwd() / "tools" / "make_submission.py")
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
python -I - "$TMP/guards_swap.json" "$TMP/sub_swap.csv" "$FB_FILE_SHA" "$PRIV/swap_control_s035.json" <<'PY' || exit 4
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
print(f"  guard (a) fires under the widened rule (failed {', '.join(failed)}) -> fallback (40 features, without the guard); file = fallback-form file byte-identical "
      f"({sha[:16]}…)" if not bad else "ABORT: " + "; ".join(bad))
raise SystemExit(4 if bad else 0)
PY

# 6. main file (form main with main_guard and the guard required; guard (c) violated -> no file, exit 5)
echo "== 6. main file with stage2_s035.json =="
if [ -f "$OUT" ] || [ -f "$TESTMAT" ]; then  # only replace byte-identically
  BAK="$(mktemp -d "${TMPDIR:-/tmp}/s035_guard_backup.XXXXXX")"
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
if g["feature_set"] != "main" or g.get("model_feature_set") != "main_guard" or g.get("protect_none") is not True:
    print(f"ABORT: form {g['feature_set']} / {g.get('model_feature_set')} / guard {g.get('protect_none')} instead of main / main_guard / on — no file")
    raise SystemExit(5)
bad = []
if gn.get("rule") != "widened" or not gn["ok"] or g["null"] or g["n_features"] != 48 or not g["guard_changed_top"]["ok"]:
    bad.append(f"rule {gn.get('rule')}, guard {gn['ok']}, null model {g['null']}, {g['n_features']} features, (c) {g['guard_changed_top']['ok']}")
if gn["test"] != gnull["guard_notes"]["test"] or gn["checks"] != gnull["guard_notes"]["checks"]:
    bad.append("test aggregates of the main file != those of step 3")
print(f"  main form main_guard (48 features, guard on: {g['rows_protected']} test rows keep the base order); guard (c) new top code "
      f"{g['guard_changed_top']['test']:.4f} (OOF {g['guard_changed_top']['oof']:.4f}) ok; note guard = step 3" if not bad else "ABORT: " + "; ".join(bad))
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

# 7. check of the file and comparison (aggregates only)
echo "== 7. check and comparison (aggregates only) =="
check "$MAIN_FILE_SHA" "$MAIN_FILE"
check "$FB_FILE_SHA" "$FB_FILE"
python -I - "$OUT" "$MAIN_FILE" "$FB_FILE" "$GUARDS" "$PRIV/compare_s035.json" <<'PY' || exit 4
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
for path, label in ((sys.argv[2], "main-form file (released rotation list)"), (sys.argv[3], "fallback-form file (released rotation list)")):
    ref = read(path)
    b = new.loc[ref.index]
    t_new, t_ref = b.str.split().str[0], ref.str.split().str[0]
    top = float((t_ref != t_new).mean())
    top3 = float((ref != b).mean())
    e = {"share_other_top1": top, "share_other_top3_ordered": top3, "rows_other_top1": int((t_ref != t_new).sum()),
         "share_top_general_new": float(t_new.isin(gen).mean()), "share_top_general_ref": float(t_ref.isin(gen).mean()),
         "share_top_none_new": float((t_new == "NONE").mean()), "share_top_none_ref": float((t_ref == "NONE").mean())}
    rep["vs"][Path(path).name] = e
    print(f"  vs. {label}: other top code in {top:.4f} of the rows ({e['rows_other_top1']}), other first three (ordered) in {top3:.4f}; "
          f"share of general codes at the top {e['share_top_general_ref']:.4f} -> {e['share_top_general_new']:.4f}, NONE {e['share_top_none_ref']:.4f} "
          f"-> {e['share_top_none_new']:.4f}")
g = json.load(open(sys.argv[4]))
rep.update({k: g[k] for k in ("guard_changed_top", "guard_novel_task_top5", "share_top3_changed", "share_top3_changed_oof", "rows_protected")})
gc, gn = g["guard_changed_top"], g["guard_novel_task_top5"]
print(f"  guard (c): new top code {gc['test']:.4f} (OOF main_guard {gc['oof']:.4f}, band {gc['factor'][0] * gc['oof']:.4f}…{gc['factor'][1] * gc['oof']:.4f}) -> "
      f"{'ok' if gc['ok'] else 'VIOLATED'}; (b) \"novel for the task\" among the first 5: test {gn['test']:.4f}, OOF {gn['oof']:.4f}; guarded "
      f"{g['rows_protected']} test rows")
Path(sys.argv[5]).write_text(json.dumps(rep, indent=1, ensure_ascii=False) + "\n")
PY
python -I tools/orient_list.py compare "$MAIN_FILE" "$OUT" --label "main-form file (released rotation list)" --json "$PRIV/compare_s035_vs_s034n_orient.json"

# 8. checksums
echo "== 8. checksums =="
check "$OUT_SHA" "$OUT"
sha256sum "$STACK" "$STAGE2_MAIN" "$STAGE2" "$MAIN_FILE" "$FB_FILE" "$OUT" "$TESTMAT" "${OUT%.csv}.json" "$GUARDS"
echo "s035_stage2_guard: $OUT built (main form with the NONE guard and general-code identifiers under the widened rule, released rotation list)"
