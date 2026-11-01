#!/usr/bin/env bash
# Build path (kept for provenance; it produced stack_s029_best_m3x3.json, which the later build paths check by checksum):
# Qwen3.5-397B-A17B as the SIXTH channel q397 — the earlier primary (best) or inductive (safe) recipe unchanged plus the channel
# in the teacher protocol from the 397B (named codes with probabilities, floor 0.05, no gate, no debiasing). Test channels exactly
# as for the earlier files of the line (offline Kaggle-T4 versions of v1 and zs27t), plus --proto q397=runs/<q397 test run>/proto_te/proto.
#   best  v1r, zs27q, think18nb × 1.4 (floor 0.01), pr27xm (0.05; mean with ranking), M3 (0.05; pure ranking), q397 (0.05);
#         β 0.2; margin alignment 0.75 for classes with at least 3 rows (transductive, primary line only); REL-01 for blank images
#         blend file artifacts/stack_s029_best.json — θ from all 1,500 rows, cross value 0.82811 (earlier recipe 0.81989; +0.0082 ± 0.0029)
#   safe  v1r, zs27t, think18sb (0.01), pr27xm (0.05), M3 (0.05), q397 (0.05); β 0.4; inductive alignment 0.75; REL-01
#         blend file artifacts/stack_s029_safe.json — θ from all 1,500 rows (earlier recipe: 600), cross value 0.82067 (+0.0087 ± 0.0031)
# Blend files, cross values and checks: eda/c110_q397_deploy.py (stack --save/--verify).
#
#     bash tools/builds/s029_q397.sh best|safe [--m3 m3x2_full_cv01|m3x3_full] [--beta-novel] [--rel-readable]
#
#   --m3 m3x3_full   M3 as the mean of THREE draws instead of two: requires oof_m3x3_full (tools/builds/s029_m3x3.sh or
#                    eda/a9_m3_draws_oof.py, --verify) and a deployment verdict for the line in c108 (IGNORE_VERDICT=1 for checking
#                    only); the blend file stack_s029_<line>_m3x3[_bn].json is created on the first call (c110 stack --save),
#                    afterwards --verify.
#   --beta-novel     SWITCH (default off): arithmetic share β_new for codes that were never the root for the task in training
#                    (field arith_beta_novel; best β_new 0.6: cross value 0.83067 = +0.0026 ± 0.0018 against 0.82811; safe 0.7:
#                    +0.0020 ± 0.0018)
#   --rel-readable   SWITCH (default off): REL-01 at rank 1 also for the test rows for which pr27 reports "readable = no"
#                    (list derived/blank/test_rows_readable.npy: 6 blank + 5 new rows; cross value on training rows ±0)
#
# 397B test run chosen automatically: q397te_rp (4×H200) OR q397te8_rp (8×H100), whichever has been fetched (both: Q397_RUN=<run>).
# CHECKS before make_submission: checksums (OOF artifacts, source job, blend file), oof_q397 bit-identical to the channel of the
# decision (c110 verify-oof), blend file = recomputation (c110 stack --verify: anchors, cross value exact, θ check, oof_mixture),
# test side (c110 test-status: ≥ 97 % valid, NONE at the top within the training band ± 0.06, model and settings as in the Tinker
# measurement), fidelity check PASSED (artifacts/eda/private/q397_deploy/c109_parity.json), ranking runs complete (c95 --status).
#
# DRY RUN (code check only): PLACEHOLDER=1 uses the pr27 test side (runs/pr27te_rp/proto_te/proto) as q397, skips the model,
# settings and fidelity checks, writes into a throw-away directory and deletes the file and test matrix at the end — NEVER submit.
#
# Only aggregates and checksums are printed. Nothing is submitted from here.
set -euo pipefail
cd "$(dirname "$0")/../.."
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}" OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
usage() { echo "usage: $0 best|safe [--m3 m3x2_full_cv01|m3x3_full] [--beta-novel] [--rel-readable]" >&2; exit 2; }
LINE="${1:-}"
[ $# -gt 0 ] && shift
case "$LINE" in best|safe) ;; *) usage ;; esac
M3="m3x2_full_cv01"; BETA_NOVEL=0; READABLE=0
while [ $# -gt 0 ]; do
  case "$1" in
    --m3) [ $# -ge 2 ] || usage; M3="$2"; shift 2 ;;
    --beta-novel) BETA_NOVEL=1; shift ;;
    --rel-readable) READABLE=1; shift ;;
    *) usage ;;
  esac
done
case "$M3" in m3x2_full_cv01|m3x3_full) ;; *) usage ;; esac
PLACEHOLDER="${PLACEHOLDER:-0}"
STEM="s029_${LINE}"
[ "$M3" = m3x3_full ] && STEM="${STEM}_m3x3"
[ "$BETA_NOVEL" = 1 ] && STEM="${STEM}_bn"
SUFFIX=""; BLANK="derived/blank/test_rows.npy"
[ "$READABLE" = 1 ] && { SUFFIX="_rd"; BLANK="derived/blank/test_rows_readable.npy"; }
STACK="artifacts/stack_${STEM}.json"
STACK_FLAGS=(--line "$LINE" --m3 "$M3")
[ "$BETA_NOVEL" = 1 ] && STACK_FLAGS+=(--beta-novel)
RANK1="runs/m3_fullte_rp/m3_test_rank/test"
RANK2="runs/m3d2te_rp/m3d2_test_rank/test"
RANK3="runs/m3d3te_rp/m3d3_test_rank/test"
RANK_XM="runs/n1_rp/rank_test/test"
V1_T4="v1=v1te_a_t4/v1_test_a/test+v1te_b_t4/v1_test_b/test"
case "$LINE" in
  best) REF="artifacts/submissions/sub_cv01_best_l2.csv"; REF_SHA="1c8c4bb12122e3feeef07d4488e79339635078ab667c29a57192560dcba4e346"; REF_ID="the earlier primary-line file" ;;
  safe) REF="artifacts/submissions/sub_v1r_zs27t_think18sb_pr27xm_m3x2_ar_ia_t4.csv"; REF_SHA="80ea14fe28449a4b7be66bb0eeb8622f4870a4629105ebe6dc7a3859c36d5557"; REF_ID="the earlier inductive-line file" ;;
esac
check() { echo "$1  $2" | sha256sum -c --quiet || { echo "ABORT: $2 differs (checksum)" >&2; exit 4; }; }

# 397B test side: the fetched run (or the placeholder)
if [ "$PLACEHOLDER" = 1 ]; then
  Q397_TEST="runs/pr27te_rp/proto_te/proto"
  echo "PLACEHOLDER: q397 test side := $Q397_TEST (pr27, NOT the 397B) — code check only, the file is deleted, NEVER submit" >&2
else
  if [ -n "${Q397_RUN:-}" ]; then
    RUNS=("$Q397_RUN")
  else
    RUNS=()
    for R in q397te_rp q397te8_rp; do [ -f "artifacts/runs/$R/$R/proto_te/proto_meta.json" ] && RUNS+=("$R"); done
  fi
  case "${#RUNS[@]}" in
    1) Q397_TEST="runs/${RUNS[0]}/proto_te/proto" ;;
    0) echo "ABORT: no fetched 397B test side (artifacts/runs/q397te_rp|q397te8_rp/<run>/proto_te/proto_meta.json) — wait for the run or fetch it" >&2; exit 3 ;;
    *) echo "ABORT: both 397B test runs exist (${RUNS[*]}) — choose the deployed one with Q397_RUN=<run>" >&2; exit 3 ;;
  esac
fi
echo "s029_q397 $LINE: blend $STACK (M3 $M3$([ "$BETA_NOVEL" = 1 ] && echo ', β_new')), q397 test side $Q397_TEST, REL-01 list $BLANK"

# 0. checksums: channel q397 (source job and OOF artifact, c110 source + a8), M3, pr27xm, submitted reference file
check 6935614248a22da250054dd14b180fc5875832ddba7afc49c4b3da0337503280 artifacts/runs/q397_tinker/q397_tinker/proto_t/proto_labels.jsonl
check 57eee5101f00d46d9b7a1c1fc68cf2554245b7cb2ecd6ec1c4bcd77e023d4279 artifacts/runs/q397_tinker/q397_tinker/proto_t/proto_meta.json
check c5bcd665a0cd5246b78dcd67012d6831e0e51ecb553666d60e6f164bf519fadf artifacts/oof_q397.npy
check f591eb385fb7fecad986937488d5e9ac5c180bc0d597f19f71750340917388dc artifacts/meta_q397.json
check e08f2deefe90a9b3b903668222cb51aa3a07f6dd80705e8f2c078ff3b2302ef1 artifacts/oof_m3x2_full_cv01.npy
check b15d3c16ad412875092a5b7b4d7cf2203172a733583e0c5a2fc4912d76f04f13 artifacts/meta_m3x2_full_cv01.json
check bf3d1b58abfc57eb64139418ed7646f45ba9f879791661658035171b9a2f0d03 artifacts/oof_pr27xm.npy
check "$REF_SHA" "$REF"
case "$STEM" in  # blend files built in advance (c110 stack --save); the m3x3 versions are created on the first call
  s029_best) check 29d2a79450848e0d61c7367cce431431684d9d4a646836af7317c2731b3ae20d "$STACK" ;;
  s029_safe) check 883f2e7a37ce3b61a1b7020a81bfddfbeec4b4a7f847ab3807232bceee19d3ef "$STACK" ;;
  s029_best_bn) check 8d2513e43b6d715503d0f9e172a805f4d34a8dcd2687e47717c2308e20280399 "$STACK" ;;
  s029_safe_bn) check 73f163b0cabfc9ef64a9956ab808b15d58e4df3566858b67ff487e8e65ca9cab "$STACK" ;;
esac
python -I eda/c110_q397_deploy.py verify-oof || { echo "ABORT: oof_q397 is not the channel of the c107 decision" >&2; exit 4; }

# 1. M3 from three draws (only --m3 m3x3_full): artifact = mean of its sources, verdict of the line in c108, test draw 3
if [ "$M3" = m3x3_full ]; then
  EVAL="artifacts/eda/private/m3_draws_eval/c108_m3_three_draws.json"
  python -I eda/a9_m3_draws_oof.py --verify m3x3_full || { echo "ABORT: oof_m3x3_full missing or not the mean of its sources (tools/builds/s029_m3x3.sh)" >&2; exit 4; }
  python -I - "$EVAL" "$LINE" "${IGNORE_VERDICT:-0}" <<'PY'
import hashlib
import io
import json
import sys
from pathlib import Path

import numpy as np

path, line, ignore_verdict = Path(sys.argv[1]), sys.argv[2], sys.argv[3] == "1"
if not path.is_file():
    raise SystemExit(f"ABORT: {path} missing — run python -I eda/c108_m3_three_draws.py first")
ev = json.loads(path.read_text())
if ev.get("mode") != "real":
    raise SystemExit(f"ABORT: {path} comes from a check run (mode {ev.get('mode')!r})")
meta = json.loads(Path("artifacts/meta_m3x3_full.json").read_text())
if (ev.get("specs") or {}).get("m3x3_source") != (meta.get("rank") or {}).get("source"):
    raise SystemExit("ABORT: c108 evaluated sources other than those named in meta_m3x3_full.json")
buf = io.BytesIO()
np.save(buf, np.asarray(np.load("artifacts/oof_m3x3_full.npy"), dtype=np.float64))
if hashlib.sha256(buf.getvalue()).hexdigest() != ev.get("m3x3_npy_sha256"):
    raise SystemExit("ABORT: oof_m3x3_full.npy is not the mean evaluated by c108")
dec = ev["decision"][line]
print(f"  c108 ({ev['created_utc']}): {line} -> {'DEPLOY' if dec['deploy'] else 'NO DEPLOYMENT'} (guard {'violated' if ev['decision']['guard_violated'] else 'held'})")
if not dec["deploy"]:
    if not ignore_verdict:
        print(f"ABORT: c108 gave NO DEPLOYMENT for {line} — IGNORE_VERDICT=1 builds for checking only, never for submission", file=sys.stderr)
        raise SystemExit(6)
    print("  WARNING: IGNORE_VERDICT=1 — file for checking only, do NOT submit", file=sys.stderr)
PY
  python -I eda/a9_m3_draws_oof.py --gen-status "$RANK1" --table test --min-complete 0.3 || { echo "ABORT: generation behind $RANK1" >&2; exit 3; }
  python -I eda/a9_m3_draws_oof.py --gen-status "$RANK2" --table test --min-complete 0.5 || { echo "ABORT: generation behind $RANK2" >&2; exit 3; }
  python -I eda/a9_m3_draws_oof.py --gen-status "$RANK3" --table test --min-complete "${D3_MIN_COMPLETE:-0.5}" || { echo "ABORT: generation behind $RANK3" >&2; exit 3; }
  for R in "$RANK1" "$RANK2" "$RANK3"; do
    python -I eda/c99_m3_deploy.py --variant "$R" --channel m3p5 --require-gold-only || { echo "ABORT: training regime of $R not gold-only" >&2; exit 5; }
  done
fi

# 2. blend file = recomputation (anchors, cross value, θ check, oof_mixture; safe: inductive class weights)
if [ -f "$STACK" ]; then
  python -I eda/c110_q397_deploy.py stack "${STACK_FLAGS[@]}" --verify || { echo "ABORT: $STACK is not the recomputation" >&2; exit 4; }
else
  case "$STEM" in
    *_m3x3*) python -I eda/c110_q397_deploy.py stack "${STACK_FLAGS[@]}" --save ;;
    *) echo "ABORT: $STACK missing (built in advance; rebuild only deliberately: python -I eda/c110_q397_deploy.py stack ${STACK_FLAGS[*]} --save)" >&2; exit 4 ;;
  esac
fi

# 3. 397B test side (aggregates only) and fidelity check
if [ "$PLACEHOLDER" = 1 ]; then
  python -I eda/c110_q397_deploy.py test-status "$Q397_TEST" --placeholder || { rc=$?; echo "ABORT: placeholder test side" >&2; exit "$rc"; }
else
  python -I eda/c110_q397_deploy.py test-status "$Q397_TEST" || { rc=$?; echo "ABORT: 397B test side not deployable (code $rc)" >&2; exit "$rc"; }
  python -I - <<'PY' || { echo "ABORT: fidelity check missing or not passed — do not deploy the test side" >&2; exit 5; }
import json
import sys
from pathlib import Path

f = Path("artifacts/eda/private/q397_deploy/c109_parity.json")
if not f.is_file():
    print(f"{f} missing — python -I eda/c109_q397_parity.py runs/q397p_rp/proto_p/proto (or q397p8_rp)", file=sys.stderr)
    raise SystemExit(5)
r = json.loads(f.read_text())
print(f"  fidelity check (c109, training rows): {r['verdict']} — valid {r['valid_own']}/{r['rows']}, input tokens within 1 % in "
      f"{r['token_within_1pct']:.3f} of the rows, same top {r['same_top']:.3f}, MAP@3 own {r['map3_own']:.3f} / Tinker {r['map3_tinker']:.3f}")
raise SystemExit(0 if r.get("verdict") == "PASSED" else 5)
PY
fi

# 4. ranking runs complete
RANKS=("$RANK1" "$RANK2" "$RANK_XM"); M3_TEST="$RANK1,$RANK2"
[ "$M3" = m3x3_full ] && { RANKS=("$RANK1" "$RANK2" "$RANK3" "$RANK_XM"); M3_TEST="$RANK1,$RANK2,$RANK3"; }
for R in "${RANKS[@]}"; do
  python -I eda/c95_rank_mean.py --status "$R" --table test || { echo "ABORT: $R missing or incomplete" >&2; exit 3; }
done

# 5. REL-01 list (switch --rel-readable): blank images ∪ pr27 "readable = no" (test rows only counted)
if [ "$READABLE" = 1 ]; then
  python -I eda/c110_q397_deploy.py readable --save || { echo "ABORT: REL-01 list" >&2; exit 4; }
  check 2b2e10699e032b3706691abe0163b9132719c162dc7d2ac6f81add777bc73b1d "data/$BLANK"
fi

# 6. test file
if [ "$PLACEHOLDER" = 1 ]; then
  TMP="$(mktemp -d "${TMPDIR:-/tmp}/s029_placeholder.XXXXXX")"
  OUT="$TMP/sub_${STEM}${SUFFIX}_PLACEHOLDER.csv"
  TESTNPY="artifacts/test_${STEM}${SUFFIX}_PLACEHOLDER.npy"
  trap 'rm -rf -- "${TMP:?}"; rm -f -- "${TESTNPY:?}"; echo "PLACEHOLDER: file, JSON and test matrix deleted" >&2' EXIT
  OUTARGS=(--suffix "${SUFFIX}_PLACEHOLDER" --out "$OUT")
else
  OUT="artifacts/submissions/sub_${STEM}${SUFFIX}.csv"
  OUTARGS=(--suffix "$SUFFIX")
fi
if [ "$LINE" = best ]; then
  PYTHONDONTWRITEBYTECODE=1 python tools/make_submission.py --name "$STEM" \
    --test "v1r=v1rot_t4/rot_test/test@derived/orient/test_c80_rows.npy" --fill "v1r=v1" --test "$V1_T4" \
    --test "zs27q=zs27q_test0_rp/zs27q_test0/test+zs27q_test1_rp/zs27q_test1/test" --fill "think18nb=zs27q" \
    --test "think18nb=zs27v_te_rp/think_te/test/think@derived/think_rows/test.npy,zs27v_n3_rp/think_nte/test/think@derived/think_rows/test_tb_q.npy" \
    --proto "pr27xm=runs/pr27te_rp/proto_te/proto" --rank "pr27xm=$RANK_XM" \
    --proto "q397=$Q397_TEST" \
    --rank-only "$M3=$M3_TEST" \
    --align 0.75 --align-min 3 --blank-rows "$BLANK" "${OUTARGS[@]}" 2>&1 | grep -vi '^warning' | cut -c1-300
else
  PYTHONDONTWRITEBYTECODE=1 python tools/make_submission.py --name "$STEM" \
    --test "v1r=v1rot_t4/rot_test/test@derived/orient/test_c80_rows.npy" --fill "v1r=v1" --test "$V1_T4" \
    --test "zs27t=px27_t4zt0/zs27t_test_t4/test+px27_t4zt1/zs27t_test_t4/test" --fill "think18sb=zs27t" \
    --test "think18sb=zs27v_n3_rp/think_nte/test/think@derived/think_rows/test_tb_t.npy" \
    --proto "pr27xm=runs/pr27te_rp/proto_te/proto" --rank "pr27xm=$RANK_XM" \
    --proto "q397=$Q397_TEST" \
    --rank-only "$M3=$M3_TEST" \
    --align-inductive 0.75 --align-min 3 --blank-rows "$BLANK" "${OUTARGS[@]}" 2>&1 | grep -vi '^warning' | cut -c1-300
fi
[ -f "$OUT" ] || { echo "ABORT: $OUT was not written" >&2; exit 4; }

# 7. reference file unchanged; deviation only as an aggregate
check "$REF_SHA" "$REF"
python -I - "$REF" "$OUT" "$REF_ID" <<'PY'
import sys

import pandas as pd

a = pd.read_csv(sys.argv[1], dtype=str, keep_default_na=False).set_index("trace_id")["root_code"]
b = pd.read_csv(sys.argv[2], dtype=str, keep_default_na=False).set_index("trace_id")["root_code"].loc[a.index]
top_a, top_b = a.str.split().str[0], b.str.split().str[0]
print(f"  against {sys.argv[3]}: different top code in {(top_a != top_b).mean():.4f} of the rows, different first three (with order) in {(a != b).mean():.4f}")
PY
sha256sum "$STACK" "$OUT"
if [ "$PLACEHOLDER" = 1 ]; then
  echo "PLACEHOLDER RUN: code path checked (two --proto channels, --rank, --rank-only, last step) — the file is NOT a candidate and is deleted"
else
  echo "s029_q397 $LINE: $OUT built"
fi
