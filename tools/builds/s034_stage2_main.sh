#!/usr/bin/env bash
# Primary line with the SECOND STAGE in the MAIN FORM (43 features, including the code mentions in the M3 notes), under the
# widened rule of the note guard. Same second stage as tools/builds/s032_stage2.sh (β 0.6 + conditional logit over the first 5
# candidates after the margin alignment): weights, standardisation, feature lists, β, K, λ, gate, OOF shares, the reference of
# the note guard and the input checksums are those of artifacts/stage2_s032.json (recomputed bit-identically); ONLY the rule of
# note guard (a) is new:
#   Widened rule: a statistic passes within the old band (range of the three training draws ± 10 % relative) OR — shares only
#   (top-code mention per draw, mention distribution per level, word-identical notes per pair, each against its own training
#   value) — at |z| ≤ 3 in the two-sided two-sample binomial test (pooled share; n = 1,500 training rows and all test rows,
#   respectively). codes_per_row only within the old band. The rule only widens, it never narrows.
#   -> artifacts/stage2_s034.json (eda/c118_stage2.py fit --guard-rule widened; all fields except guard_rule/derived_from = stage2_s032.json).
# Order on the test side unchanged: blend -> margin alignment (one set) -> second stage -> REL-01.
#
#     bash tools/builds/s034_stage2_main.sh best        # -> artifacts/submissions/sub_s034_stage2_main.csv
#
# STEPS: 0 build of the fallback-form file complete (tools/builds/s032_stage2.sh best): afterwards sub_s032_stage2.csv
#   byte-identical 96bafc6f…, blend file, stage2_s032.json and the file without second stage with unchanged checksums, the guard
#   statistics of the fallback-form build byte-identical (path without the widened rule unchanged); 1 deployment file
#   stage2_s034.json (c118 fit --guard-rule widened, --save or --verify; checksum; field comparison with stage2_s032.json); 2 copy
#   null point (b′): null model from stage2_s034.json -> file without second stage byte-identical (CSV and test matrix; computes
#   the 43 features of the main form); 3 positive controls on the REAL test aggregates (planted statistics; a planted |z| = 3.5
#   outside the old band fires, |z| = 2.5 passes); 4 real file swap (notes file d2 as d1): via the command line -> abort by the
#   provenance check; with consistent provenance (texts of d2 in place of d1) -> guard (a) fires under the widened rule,
#   fallback, file = fallback-form file byte-identical; 5 main file (form "main" and guard (c) required, otherwise exit 5
#   without a file; test aggregates = those of step 2); 6 comparison with the fallback-form file and the file without second
#   stage (aggregates only); 7 CV of the main form against the deployed fallback form (training rows, nested row vectors from
#   c118 oof); 8 checksums. An existing output file is only replaced by a byte-identical rebuild. Statistics and logs under
#   artifacts/eda/private/stage2_main/.
# EXIT: 2 usage; 3 input missing; 4 checksum/recomputation/null point/positive control; 5 form not "main" or guard (c) violated
#   (codes from step 0 are passed on, see s032_stage2.sh).
#
# Only aggregates and checksums are printed. Nothing is submitted from here.
set -euo pipefail
cd "$(dirname "$0")/../.."
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}" OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
usage() { echo "usage: $0 best" >&2; exit 2; }
[ $# -eq 1 ] && [ "$1" = best ] || usage
STEM="s032_stage2"
STACK="artifacts/stack_${STEM}.json"; STACK_SHA="d278bbedf7fb4d7564e584a55d45959b826748dfe937d8becf1e0818007281eb"
STAGE2_32="artifacts/stage2_s032.json"; STAGE2_32_SHA="c33174e3ab203b5c2559586eb6b02f67f1588aca4da6bde25078cfb318fed73e"
STAGE2="artifacts/stage2_s034.json"
STAGE2_SHA="377d863bf08b07dda20bb37fab1af4a0c929e7fbbb8c3f221f1aed6e14f4397a"  # artifacts/stage2_s034.json (c118 fit --guard-rule widened)
FB_FILE="artifacts/submissions/sub_s032_stage2.csv"; FB_FILE_SHA="96bafc6fa4c0fb89a32e45fe24816454685b137d0c8264171bfcea93ea371809"
REF_BASE="artifacts/submissions/sub_s029_best_q397m3.csv"; REF_BASE_SHA="768b7778ea049caeb23455ecad26b9bc90ebff52bee3cb97e356aac7cadf07e8"
GUARDS_FB="artifacts/eda/private/stage2/guards.json"
PRIV="artifacts/eda/private/stage2_main"
GUARDS="$PRIV/guards_s034.json"
SUFFIX="_s034main"
OUT="artifacts/submissions/sub_s034_stage2_main.csv"
OUT_SHA="d9dc8b480b3c87c968426c2c16b264b69ca1b1bf533049105875dc666ec8f7ef"   # main-form file (pinned after the checks)
TESTMAT="artifacts/test_${STEM}${SUFFIX}.npy"
NOTES1="derived/sft/m3_d1_test_frozen_notes.jsonl"
NOTES2="derived/sft/m3_d2_test_frozen_notes.jsonl"
RANK1="runs/m3_fullte_rp/m3_test_rank/test"
RANK2="runs/m3d2te_rp/m3d2_test_rank/test"
RANK3="runs/m3d3te_rp/m3d3_test_rank/test"
Q397M3_TEST="runs/q397m3te_rp/q397m3_test_rank/test"
check() { echo "$1  $2" | sha256sum -c --quiet || { echo "ABORT: $2 differs (checksum)" >&2; exit 4; }; }
mkdir -p "$PRIV"

# 0. build of the fallback-form file complete (includes the file without second stage, the OOF reproduction, the copy null points (a)/(b) and the guards of the second stage)
echo "== 0. fallback-form file (tools/builds/s032_stage2.sh best) =="
GUARDS_FB_BEFORE=""
[ -f "$GUARDS_FB" ] && GUARDS_FB_BEFORE="$(sha256sum "$GUARDS_FB" | cut -d' ' -f1)"
bash tools/builds/s032_stage2.sh best || { rc=$?; echo "ABORT: build of the fallback-form file (code $rc)" >&2; exit "$rc"; }
check "$FB_FILE_SHA" "$FB_FILE"
check "$STACK_SHA" "$STACK"
check "$STAGE2_32_SHA" "$STAGE2_32"
check "$REF_BASE_SHA" "$REF_BASE"
if [ -n "$GUARDS_FB_BEFORE" ]; then check "$GUARDS_FB_BEFORE" "$GUARDS_FB"; fi
python -I - "$GUARDS_FB" <<'PY' || exit 4
import json
import sys

g = json.load(open(sys.argv[1]))
gn = g["guard_notes"]
failed = [c["stat"] for c in gn["checks"] if not c["ok"]]
bad = []
if g["feature_set"] != "fallback" or failed != ["identical[d1-d2]"]:
    bad.append(f"the fallback-form file no longer falls back because of identical[d1-d2] (form {g['feature_set']}, failed {failed})")
if "rule" in gn or any("z" in c for c in gn["checks"]):
    bad.append("guard statistics of the fallback-form build carry fields of the widened rule (path without the widened rule changed)")
print("  fallback-form file byte-identical (96bafc6f…); path without the widened rule unchanged: the note guard fails only identical[d1-d2] -> fallback"
      if not bad else "ABORT: " + "; ".join(bad))
raise SystemExit(4 if bad else 0)
PY

# 1. deployment file under the widened rule (the recomputation must be stage2_s032.json character by character; afterwards only guard_rule + derived_from are new)
echo "== 1. deployment file stage2_s034.json (c118 fit --guard-rule widened) =="
if [ -f "$STAGE2" ]; then
  python -I eda/c118_stage2.py fit --stack "$STEM" --guard-rule widened --out "$STAGE2" --verify \
    || { rc=$?; echo "ABORT: $STAGE2 is not the recomputation" >&2; exit "$rc"; }
else
  python -I eda/c118_stage2.py fit --stack "$STEM" --guard-rule widened --out "$STAGE2" --save \
    || { rc=$?; echo "ABORT: $STAGE2 not built" >&2; exit "$rc"; }
fi
check "$STAGE2_SHA" "$STAGE2"
python -I - "$STAGE2_32" "$STAGE2" "$STAGE2_32_SHA" <<'PY' || exit 4
import json
import sys

a, b = (json.load(open(p)) for p in sys.argv[1:3])
bad = [f"field {k} differs" for k in sorted(set(a) | set(b)) if k not in ("guard_rule", "derived_from") and a.get(k) != b.get(k)]
if [k for k in b if k != "derived_from"] != list(a):
    bad.append("order of the fields differs")
ga, gb = a["guard_rule"], b["guard_rule"]
if any(gb.get(k) != v for k, v in ga.items()):
    bad.append("old fields of guard_rule changed")
new = {k: v for k, v in gb.items() if k not in ga}
# "widened" is the rule label recorded in stage2_s034.json
if set(new) != {"rule", "share_z_max", "n_ref", "rule_text"} or new.get("rule") != "widened" or new.get("share_z_max") != 3.0 \
        or new.get("n_ref") != 1500:
    bad.append(f"new fields of guard_rule {sorted(new)} / {new.get('rule')} / {new.get('share_z_max')} / {new.get('n_ref')}")
d = b.get("derived_from") or {}
if d.get("file") != sys.argv[1] or d.get("sha256") != sys.argv[3]:
    bad.append(f"derived_from {d}")
mm = [fs for fs in ("main", "fallback") if a["models"][fs] != b["models"][fs]]
print(("  field comparison with stage2_s032.json: all fields except guard_rule/derived_from equal (weights main "
       f"{len(b['models']['main']['w'])}/fallback {len(b['models']['fallback']['w'])} bit-identical); guard_rule + widened rule, "
       f"share_z_max {new['share_z_max']:g}, n_ref {new['n_ref']}") if not bad and not mm else "ABORT: " + "; ".join(bad + mm))
raise SystemExit(4 if bad or mm else 0)
PY

# shared sources of the test side (as in tools/builds/s032_stage2.sh)
SRC=(--test "v1r=v1rot_t4/rot_test/test@derived/orient/test_c80_rows.npy" --fill "v1r=v1" --test "v1=v1te_a_t4/v1_test_a/test+v1te_b_t4/v1_test_b/test"
     --test "zs27q=zs27q_test0_rp/zs27q_test0/test+zs27q_test1_rp/zs27q_test1/test" --fill "think18nb=zs27q"
     --test "think18nb=zs27v_te_rp/think_te/test/think@derived/think_rows/test.npy,zs27v_n3_rp/think_nte/test/think@derived/think_rows/test_tb_q.npy"
     --proto "pr27xm=runs/pr27te_rp/proto_te/proto" --rank "pr27xm=runs/n1_rp/rank_test/test"
     --rank-only "m3x3_full=$RANK1,$RANK2,$RANK3"
     --rank-only "q397m3=$Q397M3_TEST" --rank-none-band-for "q397m3=0.27,0.35"
     --align 0.75 --align-min 3 --blank-rows derived/blank/test_rows.npy)
ms() { PYTHONDONTWRITEBYTECODE=1 python -I tools/make_submission.py "$@" 2>&1 | grep -vi '^warning' | grep -E "SECOND STAGE|note guard|written|ABORT|Traceback|Error|--stage2" | cut -c1-320; return "${PIPESTATUS[0]}"; }

TMP="$(mktemp -d "${TMPDIR:-/tmp}/s034_stage2.XXXXXX")"
BAK=""
cleanup() {
  rm -rf -- "${TMP:?}"
  rm -f -- "artifacts/test_${STEM}_main_base.npy" "artifacts/test_${STEM}_main_null.npy" "artifacts/test_${STEM}_swap.npy" \
           "artifacts/test_${STEM}_swapcli.npy"
  [ -n "$BAK" ] && rm -rf -- "$BAK"
  return 0
}
trap cleanup EXIT

# 2. copy null point (b′): weight 0 from the deployment file of the widened rule -> file without second stage byte-identical; the guard uses the widened rule
echo "== 2. copy null point (b′) with the null model from stage2_s034.json =="
python -I eda/c118_stage2.py null --stack "$STEM" --source "$STAGE2" --out "$TMP/null_s034.json" >/dev/null
ms --name "$STEM" "${SRC[@]}" --out "$TMP/sub_base.csv" --suffix _main_base || { echo "ABORT: file without second stage not computed" >&2; exit 4; }
ms --name "$STEM" "${SRC[@]}" --out "$TMP/sub_null.csv" --suffix _main_null --stage2 "$TMP/null_s034.json" --stage2-report "$TMP/guards_null.json" \
  || { echo "ABORT: null point (b′) not computed" >&2; exit 4; }
cmp -s "$TMP/sub_base.csv" "$TMP/sub_null.csv" || { echo "ABORT: null point (b′) violated — weight 0 changes the file" >&2; exit 4; }
cmp -s "artifacts/test_${STEM}_main_base.npy" "artifacts/test_${STEM}_main_null.npy" || { echo "ABORT: null point (b′) violated — test matrix" >&2; exit 4; }
python -I - "$TMP/guards_null.json" <<'PY' || exit 4
import json
import sys

g = json.load(open(sys.argv[1]))
gn = g["guard_notes"]
bad = []
if not g["null"] or g["feature_set"] != "main" or g["n_features"] != 43:
    bad.append(f"null model {g['null']}, form {g['feature_set']}, {g['n_features']} features")
# "widened" is the rule label recorded in stage2_s034.json
if gn.get("rule") != "widened" or gn.get("share_z_max") != 3.0 or gn.get("n_ref") != 1500 or not gn["ok"]:
    bad.append(f"rule {gn.get('rule')} / {gn.get('share_z_max')} / {gn.get('n_ref')}, guard {gn['ok']}")
print(f"  null point (b′): weight 0 from stage2_s034.json -> file without second stage byte-identical (CSV and test matrix); guard (a) under the widened rule "
      f"held, 43 features of the main form computed (n_test {gn.get('n_test')})" if not bad else "ABORT: " + "; ".join(bad))
raise SystemExit(4 if bad else 0)
PY

# 3. positive controls on the REAL test aggregates (statistics from step 2; only aggregates are changed, no row)
echo "== 3. positive controls of the note guard under the widened rule (real test aggregates) =="
python -I - "$TMP/guards_null.json" "$STAGE2" "$PRIV/positive_controls.json" <<'PY' || { echo "ABORT: positive control failed" >&2; exit 4; }
import copy
import json
import re
import sys
from pathlib import Path

sys.path[:0] = [str(Path.cwd() / "src")]
import stage2 as S2  # noqa: E402

g, s2 = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
rule, ref, gn = s2["guard_rule"], s2["guard_reference"], g["guard_notes"]
test = gn["test"]
kw = {"share_z_max": float(rule["share_z_max"]), "n_ref": int(rule["n_ref"]), "n_test": int(gn["n_test"])}
rel = float(rule["notes_rel_tolerance"])
bad = []
if gn["reference"] != ref:
    bad.append("reference of the statistics file != deployment file")


def run(t):
    return S2.guard_notes(ref, t, rel, **kw)


def slot(t, stat):
    m = re.fullmatch(r"(\w+)\[([\w-]+)\]", stat)
    key, arg = m.group(1), m.group(2)
    if key in ("codes_per_row", "share_top_mentioned"):
        return t["per_draw"][arg], key
    if key == "mention_hist_top5":
        return t["mention_hist_top5"], int(arg)
    return t["identical"], arg


def planted(stat, value):
    t = copy.deepcopy(test)
    obj, k = slot(t, stat)
    obj[k] = value
    r = run(t)
    return r, next(c for c in r["checks"] if c["stat"] == stat)


real = run(test)
old = S2.guard_notes(ref, test, rel)
out = {"rule": rule.get("rule"), "kw": kw, "real": {"ok": real["ok"], "passed_by_z": [c["stat"] for c in real["checks"] if c["passed_by"] == "z"],
                                                    "max_abs_z": max(abs(c["z"]) for c in real["checks"] if c["z"] is not None)},
       "band_rule_on_same_aggregates": {"ok": old["ok"], "failed": [c["stat"] for c in old["checks"] if not c["ok"]]}, "fixed": [], "planted_z": []}
if not real["ok"]:
    bad.append("real statistics do not pass the widened rule")
for stat, v in (("identical[d1-d2]", 1.0), ("identical[d1-d3]", 1.0), ("codes_per_row[d1]", 1.7), ("share_top_mentioned[d1]", 0.40)):
    r, c = planted(stat, v)
    hit = (not r["ok"]) and (not c["ok"])
    out["fixed"].append({"stat": stat, "value": v, "z": c["z"], "alarm": hit})
    if not hit:
        bad.append(f"{stat} = {v} does not fire")


def solve(r0, target):
    """Share t with share_z(r0, n_ref, t, n_test) = target (bisection); None if not reachable."""
    lo, hi = (r0, 1.0) if target > 0 else (0.0, r0)
    f = lambda t: S2.share_z(r0, kw["n_ref"], t, kw["n_test"]) - target  # noqa: E731
    if f(lo) * f(hi) > 0:
        return None
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if (f(lo) <= 0) == (f(mid) <= 0):
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


for c0 in real["checks"]:
    if c0["rule"] != "band|z":
        continue
    for target in (3.5, -3.5, 2.5, -2.5):
        v = solve(c0["ref"], target)
        if v is None:
            continue
        r, c = planted(c0["stat"], v)
        expect = bool(c["ok_band"]) if abs(target) > kw["share_z_max"] else True
        e = {"stat": c0["stat"], "target_z": target, "value": v, "z": c["z"], "lo": c["lo"], "hi": c["hi"], "inside_old_band": c["ok_band"],
             "ok": c["ok"], "passed_by": c["passed_by"], "expected_ok": expect}
        out["planted_z"].append(e)
        if abs(c["z"] - target) > 1e-6 or c["ok"] != expect or r["ok"] != expect:
            bad.append(f"planted {c0['stat']} z {target:+g}: ok {c['ok']} instead of {expect}")
pz = out["planted_z"]
n35 = sum(1 for e in pz if abs(e["target_z"]) == 3.5 and not e["inside_old_band"] and not e["ok"])
n25 = sum(1 for e in pz if abs(e["target_z"]) == 2.5 and not e["inside_old_band"] and e["ok"])
in35 = sum(1 for e in pz if abs(e["target_z"]) == 3.5 and e["inside_old_band"])
out["summary"] = {"z35_outside_band_alarm": n35, "z25_outside_band_pass": n25, "z35_inside_old_band_pass": in35}
if n35 == 0 or n25 == 0:
    bad.append(f"too few planted cases outside the old band (|z| 3.5: {n35}, |z| 2.5: {n25})")
if not ("identical[d1-d2]" in [e["stat"] for e in pz if e["target_z"] == 3.5 and not e["ok"]]):
    bad.append("identical[d1-d2] with |z| 3.5 does not fire")
out["ok"] = not bad
out["problems"] = bad
Path(sys.argv[3]).write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
print(f"  real statistics: widened rule {'held' if real['ok'] else 'FAILED'} (passed via z: {out['real']['passed_by_z']}; max |z| "
      f"{out['real']['max_abs_z']:.2f}); the same statistics under the original band rule: failed {out['band_rule_on_same_aggregates']['failed']}")
for e in out["fixed"]:
    print(f"  planted {e['stat']} = {e['value']:g}: {'fires' if e['alarm'] else 'DOES NOT FIRE'}"
          + (f" (z {e['z']:+.1f})" if e["z"] is not None else " (old band only)"))
print(f"  planted z per share (10 shares × ±3.5/±2.5, reachable {len(pz)}): |z| 3.5 outside the old band -> alarm in {n35}/"
      f"{sum(1 for e in pz if abs(e['target_z']) == 3.5 and not e['inside_old_band'])}; |z| 2.5 outside -> passed in {n25}/"
      f"{sum(1 for e in pz if abs(e['target_z']) == 2.5 and not e['inside_old_band'])}; |z| 3.5 inside the old band (passes, the rule only widens): {in35}")
print("  positive controls: " + ("all passed" if not bad else "FAILED: " + "; ".join(bad)) + f" -> {sys.argv[3]}")
raise SystemExit(4 if bad else 0)
PY

# 4. real file swap: notes file of draw d2 in place of d1
echo "== 4. file swap (notes file d2 as d1) =="
python -I - "$STAGE2" "$TMP/stage2_swapcli.json" "$NOTES1" "$NOTES2" <<'PY' || exit 4
import json
import sys

s2 = json.load(open(sys.argv[1]))
assert s2["test_notes"][0]["file"] == sys.argv[3] and s2["test_notes"][1]["file"] == sys.argv[4]
s2["test_notes"][0]["file"] = sys.argv[4]  # file of d2, test ranking still that of d1
open(sys.argv[2], "w").write(json.dumps(s2, indent=1, ensure_ascii=False) + "\n")
PY
if ms --name "$STEM" "${SRC[@]}" --out "$TMP/sub_swapcli.csv" --suffix _swapcli --stage2 "$TMP/stage2_swapcli.json" \
     --stage2-report "$TMP/guards_swapcli.json" > "$TMP/swapcli.log"; then
  echo "ABORT: file swap via the command line NOT detected (file written)" >&2; exit 4
fi
if [ -f "$TMP/sub_swapcli.csv" ] || ! grep -q "test notes" "$TMP/swapcli.log"; then
  echo "ABORT: file swap via the command line not caught by the provenance check" >&2; cat "$TMP/swapcli.log" >&2; exit 4
fi
echo "  (i) command line: aborted without a file — $(grep -m1 -o 'test notes.*' "$TMP/swapcli.log" | cut -c1-200)"
python -I - "$STAGE2" "$TMP" "$NOTES1" "$NOTES2" "$RANK2" "${SRC[@]}" <<'PY' > "$TMP/swap.log" 2>&1 \
  || { echo "ABORT: file swap with consistent provenance not computed" >&2; grep -E "ABORT|Error|SystemExit|--stage2" "$TMP/swap.log" >&2; exit 4; }
import importlib.util
import sys
from pathlib import Path

stage2, tmp, n1, n2, r2 = sys.argv[1:6]
src = sys.argv[6:]
spec = importlib.util.spec_from_file_location("make_submission_swap", Path.cwd() / "tools" / "make_submission.py")
MS = importlib.util.module_from_spec(spec)
spec.loader.exec_module(MS)
orig, calls = MS._test_notes_texts, []


def swapped(root, rel_file, rank_spec, n_test):
    """File swap: the frozen notes of draw d2 (with their own consistent provenance) in place of d1."""
    if rel_file == n1:
        calls.append(rel_file)
        texts, info = orig(root, n2, r2, n_test)
        return texts, {**info, "swapped_in_for": rel_file}
    return orig(root, rel_file, rank_spec, n_test)


MS._test_notes_texts = swapped
sys.argv = ["make_submission.py", "--name", "s032_stage2", *src, "--out", f"{tmp}/sub_swap.csv", "--suffix", "_swap", "--stage2", stage2,
            "--stage2-report", f"{tmp}/guards_swap.json"]
rc = MS.main()
if calls != [n1]:
    raise SystemExit(f"swap not executed ({len(calls)} calls)")
raise SystemExit(rc)
PY
grep -E "SECOND STAGE|note guard" "$TMP/swap.log" | cut -c1-320 | sed 's/^/  /' || true
python -I - "$TMP/guards_swap.json" "$TMP/sub_swap.csv" "$FB_FILE_SHA" "$PRIV/swap_control.json" <<'PY' || exit 4
import hashlib
import json
import sys

g = json.load(open(sys.argv[1]))
gn = g["guard_notes"]
failed = {c["stat"]: c for c in gn["checks"] if not c["ok"]}
sha = hashlib.sha256(open(sys.argv[2], "rb").read()).hexdigest()
bad = []
# "widened" is the rule label recorded in stage2_s034.json
if gn["ok"] or g["feature_set"] != "fallback" or "identical[d1-d2]" not in failed or gn.get("rule") != "widened":
    bad.append(f"guard (a) under the widened rule did not fire on the swap (form {g['feature_set']}, failed {sorted(failed)})")
if sha != sys.argv[3]:
    bad.append(f"fallback file {sha[:16]}… != fallback-form file")
rep = {"feature_set": g["feature_set"], "failed": {k: {"test": c["test"], "ref": c.get("ref"), "z": c.get("z")} for k, c in failed.items()},
       "sub_sha256": sha, "equals_fallback_form_file": sha == sys.argv[3], "changed_top": g["guard_changed_top"]}
open(sys.argv[4], "w").write(json.dumps(rep, indent=1, ensure_ascii=False) + "\n")
print("  (ii) consistent provenance: guard (a) under the widened rule fires (failed " + ", ".join(
      f"{k} z {c.get('z'):+.1f}" if c.get("z") is not None else k for k, c in failed.items()) + f") -> fallback; file = fallback-form file byte-identical ({sha[:16]}…)"
      if not bad else "ABORT: " + "; ".join(bad))
raise SystemExit(4 if bad else 0)
PY

# 5. main file (form "main" required; guard (c) violated -> no file, exit 5)
echo "== 5. main file with stage2_s034.json =="
if [ -f "$OUT" ] || [ -f "$TESTMAT" ]; then  # replace only by a byte-identical rebuild
  BAK="$(mktemp -d "${TMPDIR:-/tmp}/s034_stage2_backup.XXXXXX")"
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
FORM="$(python -I -c "import json;print(json.load(open('$GUARDS'))['feature_set'])")"
if [ "$FORM" != main ]; then
  restore
  echo "ABORT: note guard (a) fails under the widened rule as well -> form $FORM instead of main — no file (statistics in $GUARDS)" >&2; exit 5
fi
python -I - "$GUARDS" "$TMP/guards_null.json" <<'PY' || { restore; exit 4; }
import json
import sys

g, gnull = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
gn = g["guard_notes"]
bad = []
# "widened" is the rule label recorded in stage2_s034.json
if gn.get("rule") != "widened" or not gn["ok"] or g["null"] or g["n_features"] != 43 or not g["guard_changed_top"]["ok"]:
    bad.append(f"main file: rule {gn.get('rule')}, guard {gn['ok']}, null model {g['null']}, {g['n_features']} features, (c) {g['guard_changed_top']['ok']}")
if gn["test"] != gnull["guard_notes"]["test"] or gn["checks"] != gnull["guard_notes"]["checks"]:
    bad.append("test aggregates of the main file != those of the positive controls (step 2)")
print("  main form (main, 43 features), test aggregates = step 2" if not bad else "ABORT: " + "; ".join(bad))
raise SystemExit(4 if bad else 0)
PY
if [ -n "$BAK" ] && [ -f "$BAK/$(basename "$OUT")" ]; then
  if ! cmp -s "$OUT" "$BAK/$(basename "$OUT")"; then
    restore
    echo "ABORT: rebuild differs from the existing $OUT — backup restored, investigate the cause" >&2; exit 4
  fi
  echo "  existing $OUT rebuilt byte-identically"
fi

# 6. check of the file and comparison (aggregates only)
echo "== 6. check and comparison =="
check "$FB_FILE_SHA" "$FB_FILE"
check "$REF_BASE_SHA" "$REF_BASE"
python -I - "$OUT" "$FB_FILE" "$REF_BASE" "$GUARDS" "$PRIV/compare_s034.json" <<'PY' || exit 4
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
for path, label in ((sys.argv[2], "fallback-form file (sub_s032_stage2.csv)"), (sys.argv[3], "file without second stage (sub_s029_best_q397m3.csv, β 0.2)")):
    ref = read(path)
    b = new.loc[ref.index]
    top = float((ref.str.split().str[0] != b.str.split().str[0]).mean())
    top3 = float((ref != b).mean())
    rep["vs"][Path(path).name] = {"share_other_top1": top, "share_other_top3_ordered": top3, "rows_other_top1": int(round(top * len(ref)))}
    print(f"  against {label}: different top code in {top:.4f} of the rows ({int(round(top * len(ref)))}), different first three (with order) in {top3:.4f}")
g = json.load(open(sys.argv[4]))
gc, gn = g["guard_changed_top"], g["guard_novel_task_top5"]
rep["guard_changed_top"], rep["guard_novel_task_top5"] = gc, gn
rep["share_top3_changed"], rep["share_top3_changed_oof"] = g["share_top3_changed"], g["share_top3_changed_oof"]
rep["guard_notes_checks"] = g["guard_notes"]["checks"]
print(f"  form: {'MAIN FORM (main)' if not g['fallback'] else 'FALLBACK'} — note guard held under {g['guard_notes'].get('rule')}")
print(f"  {'statistic':26s} {'test':>8s} {'ref':>8s} {'old band':>17s} {'z':>6s}  passed via")
for c in g["guard_notes"]["checks"]:
    z = "—" if c["z"] is None else f"{c['z']:+.2f}"
    refv = "—" if c["ref"] is None else f"{c['ref']:.4f}"
    print(f"  {c['stat']:26s} {c['test']:8.4f} {refv:>8s} [{c['lo']:.4f}, {c['hi']:.4f}] {z:>6s}  {c['passed_by']}")
print(f"  guard (c): new top code from the reranker {gc['test']:.4f} (OOF main {gc['oof']:.4f}, band {gc['factor'][0] * gc['oof']:.4f}…"
      f"{gc['factor'][1] * gc['oof']:.4f}) -> {'ok' if gc['ok'] else 'VIOLATED'}; (b) \"new for the task\" among the first 5: test {gn['test']:.4f}, "
      f"OOF {gn['oof']:.4f}; first three changed test {g['share_top3_changed']:.4f}, OOF {float(g['share_top3_changed_oof'] or 0.0):.4f}")
Path(sys.argv[5]).write_text(json.dumps(rep, indent=1, ensure_ascii=False) + "\n")
PY

# 7. CV of the main form against the deployed fallback form (training rows; nested row vectors from c118 oof, step 0)
echo "== 7. CV main form against the deployed fallback form (training rows) =="
python -I - "artifacts/eda/private/stage2/c118_oof_rows.npz" "artifacts/eda/private/stage2/c118_oof.json" "$PRIV/cv_main_vs_fallback.json" <<'PY' || exit 4
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

comp = Path.cwd()
sys.path[:0] = [str(comp), str(comp / "src")]
import metric  # noqa: E402
from vlm import common as C  # noqa: E402

spec = importlib.util.spec_from_file_location("c107_cv", comp / "eda" / "c107_q397_full.py")
C107 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(C107)
ctx = C107.context(C.find_data_root())
draws = C107.question_draws(ctx["q_code"])
z, res = np.load(sys.argv[1]), json.load(open(sys.argv[2]))
folds = ctx["folds"]
out, bad = {"note": "row vectors of the nested recomputation (c118 oof); main and fallback applied without abstention (as deployed: "
                    "the deployment gate applies both); σ_q over tasks (10,000 draws, seed as in c107), σ_z row bootstrap (metric.paired_bootstrap)",
            "forms": {}}, []
label = {"fold": "per fold", "pre": "without last step", "set": "one set"}
for f in ("fold", "pre", "set"):
    m, fb, fba, own = z[f"main_{f}_forced"], z[f"fallback_{f}_forced"], z[f"fallback_{f}_abstain"], z[f"own_{f}"]
    if abs(m.mean() - res["sets"]["main"][f]["map_forced"]) > 1e-12 or abs(fb.mean() - res["sets"]["fallback"][f]["map_forced"]) > 1e-12:
        bad.append(f"row vectors [{f}] do not match c118_oof.json")
    d = m - fb
    cb = C107.cluster_boot(d, ctx["q_code"], draws)
    pb = metric.paired_bootstrap(m, fb)
    du = C107.units(m) - C107.units(fb)
    e = {"main_map": float(m.mean()), "fallback_map": float(fb.mean()), "delta": float(d.mean()), "units": int(du.sum()),
         "fold_units": [int(du[folds == k].sum()) for k in range(5)], "se_q": cb["se_q"], "z_q": cb["z_q"], "ci95_q": cb["ci95_q"],
         "se_rows": pb["sigma"], "z_rows": pb["z"], "p_main_better_rows": pb["p_a_better"], "rows_differ": int((d != 0).sum()),
         "main_vs_own_recipe": float(m.mean() - own.mean()), "fallback_vs_own_recipe": float(fb.mean() - own.mean()),
         "main_minus_fallback_with_abstention": float(m.mean() - fba.mean())}
    out["forms"][f] = e
    print(f"  {label[f]:22s} main − fallback {e['delta']:+.5f} ({e['units']:+d}/9000; per fold {e['fold_units']}) ± {e['se_q']:.5f} over tasks "
          f"(z {e['z_q']:+.2f}) | ± {e['se_rows']:.5f} over rows (z {e['z_rows']:+.2f}) | rows differing {e['rows_differ']} | against the base "
          f"recipe (β 0.2): main {e['main_vs_own_recipe']:+.5f}, fallback {e['fallback_vs_own_recipe']:+.5f}")
out["mean_of_three_forms"] = float(np.mean([out["forms"][f]["delta"] for f in ("fold", "pre", "set")]))
print(f"  mean of the three forms: {out['mean_of_three_forms']:+.5f} (expected in advance ≈ +0.001)")
Path(sys.argv[3]).write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
raise SystemExit(4 if bad else 0)
PY

# 8. checksums
echo "== 8. checksums =="
check "$OUT_SHA" "$OUT"
sha256sum "$STACK" "$STAGE2_32" "$STAGE2" "$FB_FILE" "$OUT" "$TESTMAT"
echo "s034_stage2_main best: $OUT built (main form under the widened rule of the note guard)"
