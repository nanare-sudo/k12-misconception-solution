#!/usr/bin/env bash
# PRIMARY CANDIDATE primary_m3x4 (four-draw rule: expected-value rule DEPLOY; primary slot): primary_m3px27 (tools/builds/primary_m3px27.sh)
# with the fourth M3 draw d4 — channel m3x4_full = mean of the four draws d1..d4 (fixed order; OOF: d1–d3 as m3x3_full + d4 = m3d4_oof_rp;
# test: d1–d3 of the full fit as in primary_m3px27 + d4 = m3d4te_rp) in place of m3x3_full; the second-stage features still read only d1–d3
# (field draws_first of the deployment file, tools/make_submission.stage2_draws_first). Blend refitted (eda/c135_primary_m3x4.py stack: θ on
# log loss from all 1,500 rows, β rule as in s032_stage2), second stage refitted (c135 fit: chain main form -> widened rule -> guard form ->
# I1 form -> final form in the primary_m3px27 recipe with the swapped channel; deployment form of the training rows = standard cross-fit with
# β per fold as in primary_m3px27). Order unchanged: blend -> margin alignment (one set) -> second stage (+ guard) -> REL-01. Rotation list
# ORIENT=repro.
#
#     bash tools/builds/primary_m3x4.sh                     # complete; if the d4 test side is missing: exit 3 after the copy null point, with the command for the final build
#     PRIMARY_M3X4_REGRESSION=full …                        # step 0 with PRIMARY_M3PX27_REGRESSION=full (inside it the build of safety_ndr) instead of primary_m3px27 alone
#     PRIMARY_M3X4_REGRESSION=0 …                           # step 0 as checksums only (debugging; a deployment build requires step 0)
#
# STEPS: 0 regression check: build of primary_m3px27 complete (primary_m3px27.sh, inside it primary_final, primary_i1, the guard form, s034n,
#   s033n, s032n) byte-identical; 1 inputs (checksums; verdict of the four-draw rule from c135 rule if the report file is present — in the
#   offline package the pins suffice); 2 channel m3x4_full (c135 oof: a9 mean of the four draws, the first three = m3x3_full bit-identical);
#   3 blend (c135 stack); 4 second stage (c135 fit: chain, MM rows = evaluation rows R4_0 of the bank bit-identical) and field comparison with
#   stage2_primary_m3px27.json; 5 copy null point of the training side (c135 copy: d4 removed -> blend = stack_primary_m3px27, chain rows =
#   primary_m3px27 chain, files a–e = those of primary_m3px27); 6 copy null point of the test side (d4 removed: three test draws under the new
#   path -> CSV and test matrix of primary_m3px27 byte-identical); 7 test guard d4 (c135 test-guard: 3,498/3,498 exact at the ranking files,
#   settings equal to the OOF draw d4, NONE within the band of the d1–d3 test sides, training regime); 8 null model -> file without second stage
#   byte-identical; 9 main file (guard (c), form main_guard_i1_mm required); 10 check and comparison with primary_m3px27 (aggregates only);
#   11 sensitivity with fixed β (aggregates only, private); 12 checksums (the first build prints them -> pin OUT_SHA/TESTMAT_SHA here -> second
#   build byte-identical).
# MODE OF THE CHECK STEPS: oof/stack/fit always run with --save = writes missing files and requires the same content for existing ones
#   (never overwrites a differing file); no step requires private report files except the products of step 0. Completeness of the test side
#   at the ranking files (test_proto_rank.done.npy), not at status.json.
# EXIT: 2 usage; 3 input missing (also: d4 test side not fetched yet); 4 checksum/recomputation/null point; 5 test guard, form or guard (c)
#   violated.
#
# Only aggregates and checksums are printed. Nothing is submitted from here.
set -euo pipefail
[ $# -eq 0 ] || { echo "usage: $0   (no arguments; PRIMARY_M3X4_REGRESSION=m3px27|full|0)" >&2; exit 2; }
cd "$(dirname "$0")/../.."
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}" OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export ORIENT=repro
STEM="primary_m3x4"
STEM_COPY="primary_m3x4_copy"
# stored candidate primary_m3px27 (tools/builds/primary_m3px27.sh)
M3PX27_FILE="artifacts/submissions/sub_primary_m3px27.csv"; M3PX27_FILE_SHA="23101e74aa530b9c280087442b15f72bdf25a0c4627f0207e92c648e83a2176a"
M3PX27_MAT="artifacts/test_primary_m3px27_primary_m3px27.npy"; M3PX27_MAT_SHA="f3988d9c18c0d2485d085f7b4ab160f368f8da13b381a6bab48115e7b11446ff"
M3PX27_STACK="artifacts/stack_primary_m3px27.json"; M3PX27_STACK_SHA="ebdd585e4650922d59972b3a84eb85c44af4f2dc469dd723ef2100ab7d797a3c"
M3PX27_STAGE2="artifacts/stage2_primary_m3px27.json"; M3PX27_STAGE2_SHA="9321d597b175d672fe5b31bf284ce984fbce149cb2e2cd8941a31aa8070d2500"
M3PX27_MID="artifacts/oof_m3px27xm.npy"; M3PX27_MID_SHA="47e128eed7865deec8a5818cdbd514ed29246dec79a56f7da59d86ed3897cfe8"
FINAL_FILE="artifacts/submissions/sub_primary_final.csv"; FINAL_FILE_SHA="d9c68c460c6da06d4975513911e11c4541d70d07b78f132566a3e43e7ad60b24"
NDR_FILE="artifacts/submissions/sub_safety_ndr.csv"; NDR_FILE_SHA="b8f17ce3e5fe1ed6dbf22febf9b153ec7c23870af123fa60d4342979d67270f6"
# inputs of the channel
M3X3_OOF="artifacts/oof_m3x3_full.npy"; M3X3_OOF_SHA="13b3b03ecf64161067a324d48eb1d85922dd9908167fa99ef35b29a7f1a0467d"
M3X3_META="artifacts/meta_m3x3_full.json"; M3X3_META_SHA="eb434a931a3564fe63c2dcd05274b00a04757dff30e223316661d74602e99313"
M3X3_ROWS="data/derived/m3x3_full_rows.npy"; M3X3_ROWS_SHA="c83e7d6eb3bd6a2aa38bb7c9594f24e282d781b8f144c5f403c011b2004b6aed"  # row list (all 1,500), also that of m3x4_full
RULE="artifacts/eda/private/primary_m3x4/c135_rule.json"; RULE_SHA=""  # verdict of the four-draw rule (c135 rule); optional in the package (the pins of the deployment files suffice)
# results (pinned after the first build)
CH_OOF="artifacts/oof_m3x4_full.npy"; CH_OOF_SHA="96ad8725bbaa9d274376a0bb68549abc4051878171ebb572f7707962600399a6"
CH_META="artifacts/meta_m3x4_full.json"; CH_META_SHA=""
STACK="artifacts/stack_${STEM}.json"; STACK_SHA=""
STAGE2="artifacts/stage2_${STEM}.json"; STAGE2_SHA=""
PRIV="artifacts/eda/private/primary_m3x4"
SENS="$PRIV/sens/stage2_primary_m3x4_fixbeta.json"; SENS_SHA=""
COPY_STAGE2="$PRIV/copy/stage2_m3x4k_e.json"
D4_RUN="m3d4te_rp"
D4_DONE="artifacts/runs/${D4_RUN}/${D4_RUN}/m3d4_test_rank/test_proto_rank.done.npy"  # ranking of the d4 test side (step 7 checks the completeness)
OUT="artifacts/submissions/sub_${STEM}.csv"; OUT_SHA="76da786dfdbe6a6ddcbcc93070f385e18870cb5cdaea895d92104aa3565f9973"  # candidate primary_m3x4 (first build with the d4 test side; pinned)
SUFFIX="_m3x4"
TESTMAT="artifacts/test_${STEM}${SUFFIX}.npy"; TESTMAT_SHA="30f74c8f72746f34cc423df7218bed60e7cf9b121b504a27e34606dbb3c97504"  # likewise
GUARDS="$PRIV/guards_primary_m3x4.json"
RANK1="runs/m3_fullte_rp/m3_test_rank/test"
RANK2="runs/m3d2te_rp/m3d2_test_rank/test"
RANK3="runs/m3d3te_rp/m3d3_test_rank/test"
RANK4="runs/${D4_RUN}/m3d4_test_rank/test"
MID_TEST_RANK="runs/m3px27te_rp/m3px27_test_rank/test"
Q397M3_TEST="runs/q397m3te_rp/q397m3_test_rank/test"
FIRST=0
check() { echo "$1  $2" | sha256sum -c --quiet || { echo "ABORT: $2 differs (checksum)" >&2; exit 4; }; }
pin() {  # deployment checksum: pinned -> check; empty (first build) -> print it, then enter it here
  if [ -n "$1" ]; then check "$1" "$2"; else FIRST=1; echo "  FIRST BUILD — checksum $(sha256sum "$2" | cut -d' ' -f1)  $2 (pin it in tools/builds/primary_m3x4.sh)"; fi
}
. tools/builds/orient_list.sh  # rotation list of the v1r test side: always repro here
orient_select
[ "$ORIENT" = repro ] || { echo "ABORT: this build path only runs with ORIENT=repro" >&2; exit 2; }
mkdir -p "$PRIV"

# 0. regression check of the stored builds (make_submission is extended additively: stage2_draws_first only with the field draws_first)
echo "== 0. regression check (PRIMARY_M3X4_REGRESSION=${PRIMARY_M3X4_REGRESSION:-m3px27}) =="
case "${PRIMARY_M3X4_REGRESSION:-m3px27}" in
  m3px27) bash tools/builds/primary_m3px27.sh || { rc=$?; echo "ABORT: build of primary_m3px27 (code $rc)" >&2; exit "$rc"; } ;;
  full) PRIMARY_M3PX27_REGRESSION=full bash tools/builds/primary_m3px27.sh || { rc=$?; echo "ABORT: build of primary_m3px27 with safety_ndr (code $rc)" >&2; exit "$rc"; } ;;
  0) echo "  skipped (checksums only; no deployment build)" ;;
  *) echo "usage: PRIMARY_M3X4_REGRESSION=m3px27|full|0" >&2; exit 2 ;;
esac
for pair in "$M3PX27_FILE_SHA $M3PX27_FILE" "$M3PX27_MAT_SHA $M3PX27_MAT" "$M3PX27_STACK_SHA $M3PX27_STACK" "$M3PX27_STAGE2_SHA $M3PX27_STAGE2" "$M3PX27_MID_SHA $M3PX27_MID" \
            "$FINAL_FILE_SHA $FINAL_FILE" "$NDR_FILE_SHA $NDR_FILE"; do
  check ${pair% *} ${pair#* }
done
echo "  stored files unchanged: primary_m3px27 (CSV, test matrix, blend, deployment file, mean channel), primary_final, safety_ndr"

TMP="$(mktemp -d "${TMPDIR:-/tmp}/primary_m3x4.XXXXXX")"
BAK=""
cleanup() {
  python -I eda/c135_primary_m3x4.py clean-copy >/dev/null 2>&1 || true
  rm -rf -- "${TMP:?}"
  rm -f -- "artifacts/test_${STEM_COPY}_m3x4k.npy" "artifacts/test_${STEM}_m3x4base.npy" "artifacts/test_${STEM}_m3x4null.npy" "artifacts/test_${STEM}_m3x4sens.npy"
  [ -n "$BAK" ] && rm -rf -- "$BAK"
  return 0
}
trap cleanup EXIT

# 1. inputs
echo "== 1. inputs =="
check "$M3X3_OOF_SHA" "$M3X3_OOF"
check "$M3X3_META_SHA" "$M3X3_META"
check "$M3X3_ROWS_SHA" "$M3X3_ROWS"
if [ -f "$RULE" ]; then
  pin "$RULE_SHA" "$RULE"
  python -I - "$RULE" <<'PY' || exit 4
import json
import sys

e = json.load(open(sys.argv[1]))
d = e.get("decision") or {}
if d.get("verdict") != "DEPLOY" or e.get("missing"):
    print(f"ABORT: c135 rule gave {d.get('verdict')!r} (missing splits {e.get('missing')}) — no deployment under the four-draw rule")
    raise SystemExit(4)
v = d["values"]
print(f"  c135 rule (four-draw rule (1)–(2)): verdict {d['verdict']} — 20 splits G {float(v['splits_mean_G']):+.5f} against copy {float(v['splits_mean_G_copy']):+.5f}; "
      f"original {float(v['orig_m3x4_vs_m3x3']):+.5f} against copy {float(v['orig_copy']):+.5f}")
PY
else
  echo "  verdict c135 rule not in the tree ($RULE) — in the offline package the pinned deployment files apply (steps 2–4)"
fi

# 2. channel m3x4_full (training rows): a9 mean of the four draws, the first three = m3x3_full bit-identical; row list derived/m3x3_full_rows.npy
echo "== 2. channel m3x4_full (c135 oof) =="
python -I eda/c135_primary_m3x4.py oof --save | cut -c1-320 || { rc=$?; echo "ABORT: c135 oof (code $rc)" >&2; exit "$rc"; }
pin "$CH_OOF_SHA" "$CH_OOF"
pin "$CH_META_SHA" "$CH_META"

# 3. blend (θ on log loss from all 1,500 rows, β rule; rank_only m3x4_full with four draws; rows, folds, last step as in primary_m3px27)
echo "== 3. blend (c135 stack) =="
python -I eda/c135_primary_m3x4.py stack --save | cut -c1-320 || { rc=$?; echo "ABORT: c135 stack (code $rc)" >&2; exit "$rc"; }
pin "$STACK_SHA" "$STACK"

# 4. second stage (chain with the swapped channel; MM rows = evaluation rows R4_0 bit-identical; sensitivity with fixed β)
echo "== 4. second stage (c135 fit) =="
python -I eda/c135_primary_m3x4.py fit --save | grep -vE "^\s+\[(main|fallback)\]" | cut -c1-320 || { rc=$?; echo "ABORT: c135 fit (code $rc)" >&2; exit "$rc"; }
pin "$STAGE2_SHA" "$STAGE2"
pin "$SENS_SHA" "$SENS"
python -I - "$M3PX27_STAGE2" "$STAGE2" "$STACK" <<'PY' || exit 4
import hashlib
import json
import sys

a, b = (json.load(open(p)) for p in sys.argv[1:3])
ren = lambda f: f.replace("m3x3_full_", "m3x4_full_", 1) if f.startswith("m3x3_full_") else ("m3x4_full" if f == "m3x3_full" else f)  # noqa: E731
bad = []
want = []
for k in a:
    want.append(k)
    if k == "draws_channel":
        want.append("draws_first")
if list(b) != [*want, "m3x4_build"]:
    bad.append("order/set of the fields differs from stage2_primary_m3px27.json (+ draws_first, m3x4_build)")
if b["stack"] != sys.argv[3] or b["stack_sha256"] != hashlib.sha256(open(sys.argv[3], "rb").read()).hexdigest():
    bad.append(f"blend file {b['stack']}")
if b["channels"] != [ren(c) for c in a["channels"]]:
    bad.append(f"channels {b['channels']}")
if b["draws_channel"] != "m3x4_full" or b["draws_first"] != 3:
    bad.append(f"draws_channel {b['draws_channel']} / draws_first {b.get('draws_first')}")
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
for k in ("guard_build", "i1_build", "mm_build"):
    keep = ("oof", "splits", "rule_oof", "fallback")
    if {x: y for x, y in b[k].items() if x not in keep} != {x: y for x, y in a[k].items() if x not in keep}:
        bad.append(f"{k} differs (except provenance)")
s = b["m3x4_build"]
if s.get("kind") != "deploy" or not (s.get("oof_rows") or {}).get("bit_identical") or not all(v["ok"] for v in s["null_point_i"].values()):
    bad.append("m3x4_build: evaluation rows not bit-identical or copy null point (i) violated")
df = (b.get("m3px27_build") or {}).get("deploy_forms") or {}
print(f"  field comparison with stage2_primary_m3px27.json: same fields (+ draws_first after draws_channel, + m3x4_build), channel name m3x3_full -> "
      f"m3x4_full in channels and features, draws_channel m3x4_full with draws_first 3, test notes, note guard, K, λ, mech27 and recipe fields equal; main "
      f"model main_guard_i1_mm (64, guard on; OOF new top code {m['oof_changed_top_share']:.4f}), fallback 40; deployment form standard cross-fit "
      f"(β per fold {df.get('beta_per_fold')}); MM rows = evaluation R4_0 bit-identical" if not bad else "ABORT: " + "; ".join(bad))
raise SystemExit(4 if bad else 0)
PY

# 5. copy null point of the training side (d4 removed: channel m3x4k = m3x3_full; blend = stack_primary_m3px27, chain rows = primary_m3px27 chain, files a–e = primary_m3px27)
echo "== 5. copy null point of the training side (c135 copy) =="
python -I eda/c135_primary_m3x4.py copy --keep | grep -vE "^\s+\[(main|fallback)\]" | cut -c1-320 || { rc=$?; echo "ABORT: c135 copy (code $rc)" >&2; exit "$rc"; }

# shared sources of the test side (as in tools/builds/primary_m3px27.sh with ORIENT=repro), without the M3 channel
python -I tools/orient_list.py check --mode repro --spec "$V1R_TEST" --json "$PRIV/check_orient_primary_m3x4.json" || exit $?
BASE=(--test "$V1R_TEST" --fill "v1r=v1" --test "v1=v1te_a_t4/v1_test_a/test+v1te_b_t4/v1_test_b/test"
      --test "zs27q=zs27q_test0_rp/zs27q_test0/test+zs27q_test1_rp/zs27q_test1/test" --fill "think18nb=zs27q"
      --test "think18nb=zs27v_te_rp/think_te/test/think@derived/think_rows/test.npy,zs27v_n3_rp/think_nte/test/think@derived/think_rows/test_tb_q.npy"
      --proto "m3px27xm=runs/pr27te_rp/proto_te/proto" --rank "m3px27xm=$MID_TEST_RANK")
TAIL=(--rank-only "q397m3=$Q397M3_TEST" --rank-none-band-for "q397m3=0.27,0.35"
      --align 0.75 --align-min 3 --blank-rows derived/blank/test_rows.npy)
SRC_COPY=("${BASE[@]}" --rank-only "m3x4k=$RANK1,$RANK2,$RANK3" "${TAIL[@]}")
SRC=("${BASE[@]}" --rank-only "m3x4_full=$RANK1,$RANK2,$RANK3,$RANK4" "${TAIL[@]}")
ms() { PYTHONDONTWRITEBYTECODE=1 python -I tools/make_submission.py "$@" 2>&1 | grep -vi '^warning' | grep -E "SECOND STAGE|note guard|written|ABORT|Traceback|Error|--stage2|--rank|MEAN" | cut -c1-320; return "${PIPESTATUS[0]}"; }

# 6. copy null point of the test side: the same path with the three test draws d1–d3 (d4 removed; channel m3x4k, copy blend and copy deployment file) -> primary_m3px27
echo "== 6. copy null point of the test side (d4 removed) =="
ms --name "$STEM_COPY" "${SRC_COPY[@]}" --out "$TMP/sub_copy.csv" --suffix "_m3x4k" --stage2 "$COPY_STAGE2" --stage2-report "$TMP/guards_copy.json" \
  || { echo "ABORT: copy run of the test side not computed" >&2; exit 4; }
cmp -s "$TMP/sub_copy.csv" "$M3PX27_FILE" || { echo "ABORT: copy null point violated — CSV != primary_m3px27" >&2; exit 4; }
cmp -s "artifacts/test_${STEM_COPY}_m3x4k.npy" "$M3PX27_MAT" || { echo "ABORT: copy null point violated — test matrix != primary_m3px27" >&2; exit 4; }
echo "  copy null point: CSV and test matrix byte-identical to primary_m3px27 ($(sha256sum "$M3PX27_FILE" | cut -c1-16)… / $(sha256sum "$M3PX27_MAT" | cut -c1-16)…)"
python -I eda/c135_primary_m3x4.py clean-copy >/dev/null
rm -f -- "artifacts/test_${STEM_COPY}_m3x4k.npy"

# 7. d4 test side: present? technical checks (aggregates only)
echo "== 7. test guard d4 (c135 test-guard) =="
if [ ! -f "$D4_DONE" ]; then
  echo "TEST SIDE MISSING: $D4_DONE (run $D4_RUN not finished or not completely fetched)."
  echo "  Everything else is built and checked (steps 0–6). Final build after fetching: bash tools/builds/primary_m3x4.sh"
  echo "  (the first build prints the checksums of $OUT and $TESTMAT -> pin OUT_SHA/TESTMAT_SHA here -> second build byte-identical)"
  exit 3
fi
rc=0; python -I eda/c135_primary_m3x4.py test-guard --rank "$RANK4" || rc=$?
if [ "$rc" = 3 ]; then
  echo "TEST SIDE INCOMPLETE: $RANK4 (ranking files not yet 3,498/3,498 exact; run $D4_RUN not finished yet or not completely fetched)."
  echo "  Everything else is built and checked (steps 0–6). Final build after fetching: bash tools/builds/primary_m3x4.sh"
  exit 3
fi
[ "$rc" = 0 ] || { echo "ABORT: test guard d4 (code $rc)" >&2; exit "$rc"; }

# 8. null model (weight 0) -> file without second stage byte-identical (computes the 64 features with m3x4_full, d1–d3 and mech27 of the test side)
echo "== 8. null model =="
python -I eda/c135_primary_m3x4.py null --source "$STAGE2" --out "$TMP/null.json" >/dev/null
ms --name "$STEM" "${SRC[@]}" --out "$TMP/sub_base.csv" --suffix "_m3x4base" || { echo "ABORT: file without second stage not computed" >&2; exit 4; }
ms --name "$STEM" "${SRC[@]}" --out "$TMP/sub_null.csv" --suffix "_m3x4null" --stage2 "$TMP/null.json" --stage2-report "$TMP/guards_null.json" \
  || { echo "ABORT: null model not computed" >&2; exit 4; }
cmp -s "$TMP/sub_base.csv" "$TMP/sub_null.csv" || { echo "ABORT: null model changes the file" >&2; exit 4; }
cmp -s "artifacts/test_${STEM}_m3x4base.npy" "artifacts/test_${STEM}_m3x4null.npy" || { echo "ABORT: null model changes the test matrix" >&2; exit 4; }
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
print(f"  null model: file without second stage byte-identical (CSV and test matrix); 64 features computed with m3x4_full (draw features d1–d3), mech27 "
      f"{mm.get('share_mech'):.4f}; note guard (widened rule) held" if not bad else "ABORT: " + "; ".join(bad))
raise SystemExit(4 if bad else 0)
PY

# 9. main file (form main with main_guard_i1_mm and the guard required; guard (c) violated -> no file, exit 5)
echo "== 9. main file =="
if [ -f "$OUT" ] || [ -f "$TESTMAT" ]; then  # only replace byte-identically
  BAK="$(mktemp -d "${TMPDIR:-/tmp}/primary_m3x4_backup.XXXXXX")"
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
src = side.get("sources") or {}
notes = [x for x in src.get("m3px27xm") or [] if isinstance(x, dict) and "notes" in x]
if len(notes) != 1 or notes[0].get("rank_compare") != "sft":
    bad.append(f"sources of the channel m3px27xm without note guard/comparison form sft ({len(notes)})")
m4 = (src.get("m3x4_full") or [{}])[0]
if m4.get("draws") != 4 or len(str(m4.get("rank_only", "")).split(",")) != 4 or m4.get("share_all_draws") != 1.0:
    bad.append(f"channel m3x4_full: {m4.get('draws')} draws, share with all draws {m4.get('share_all_draws')}")
print(f"  main form main_guard_i1_mm (64 features, guard on: {g['rows_protected']} test rows keep the base order); guard (c) new top code "
      f"{g['guard_changed_top']['test']:.4f} (OOF {g['guard_changed_top']['oof']:.4f}) ok; mech27 {g['mech27']['share_mech']:.4f}; note guard = step 8; "
      f"m3x4_full from 4 test draws on all rows" if not bad else "ABORT: " + "; ".join(bad))
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

# 10. check of the file and comparison with primary_m3px27 (aggregates only)
echo "== 10. check and comparison with primary_m3px27 (aggregates only) =="
check "$M3PX27_FILE_SHA" "$M3PX27_FILE"
python -I - "$OUT" "$M3PX27_FILE" "$GUARDS" "$PRIV/compare_primary_m3x4.json" <<'PY' || exit 4
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
     "share_top_none_primary_m3px27": float((t_ref == "NONE").mean())}
print(f"  vs. primary_m3px27: other top code in {e['rows_other_top1']} of {e['rows']} rows ({e['share_other_top1']:.4f}), other first three (ordered) in "
      f"{e['rows_other_top3_ordered']} ({e['share_other_top3_ordered']:.4f}), as a set {e['rows_other_top3_as_set']}; NONE at the top "
      f"{e['share_top_none_primary_m3px27']:.4f} -> {e['share_top_none_new']:.4f}")
g = json.load(open(sys.argv[3]))
rep = {"check": stats, "vs_primary_m3px27": e, **{k: g[k] for k in ("guard_changed_top", "guard_novel_task_top5", "share_top3_changed", "share_top3_changed_oof",
                                                                     "rows_protected", "mech27")}}
gc = g["guard_changed_top"]
print(f"  guard (c): new top code {gc['test']:.4f} (OOF {gc['oof']:.4f}, band {gc['factor'][0] * gc['oof']:.4f}…{gc['factor'][1] * gc['oof']:.4f}) -> "
      f"{'ok' if gc['ok'] else 'VIOLATED'}; guarded {g['rows_protected']} test rows")
Path(sys.argv[4]).write_text(json.dumps(rep, indent=1, ensure_ascii=False) + "\n")
PY

# 11. sensitivity (report only): main model on the deployment form with the fixed β of the blend file instead of the standard cross-fit
echo "== 11. sensitivity with fixed β (aggregates only, private) =="
ms --name "$STEM" "${SRC[@]}" --out "$TMP/sub_sens.csv" --suffix "_m3x4sens" --stage2 "$SENS" --stage2-report "$TMP/guards_sens.json" \
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
rm -f -- "artifacts/test_${STEM}_m3x4sens.npy"

# 12. checksums
echo "== 12. checksums =="
pin "$OUT_SHA" "$OUT"
pin "$TESTMAT_SHA" "$TESTMAT"
sha256sum "$CH_OOF" "$CH_META" "$STACK" "$STAGE2" "$OUT" "$TESTMAT" "${OUT%.csv}.json" "$GUARDS"
if [ "$FIRST" = 1 ]; then
  echo "primary_m3x4: FIRST BUILD — enter the printed checksums and run the build path again (second build byte-identical)"
else
  echo "primary_m3x4: $OUT built (candidate primary_m3x4, primary line with the fourth M3 draw; nothing is submitted from here)"
fi
