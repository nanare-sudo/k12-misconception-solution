#!/usr/bin/env bash
# Build path of the M3 channel as the MEAN OF THREE DRAWS (channel m3x3_full) instead of two (m3x2_full_cv01) — otherwise
# exactly the earlier recipes of the primary line (best: v1r, zs27q, think18nb × 1.4, pr27xm, M3; floors 0.01/0.05/0.05;
# β 0.2; margin alignment 0.75; REL-01) and of the inductive line (safe: v1r, zs27t, think18sb, pr27xm, M3; β 0.4; inductive
# alignment 0.75; REL-01), test channels as before (offline Kaggle-T4 versions of v1 and zs27t). The safe line of this script
# is the submitted inductive-line file. The third draw is SAMPLED (temperature 0.6, top_p 0.95, top_k 20, seed 3; 5,200 tokens):
# OOF m3d3_oof_rp (per fold the adapter without that fold), test m3d3te_rp (adapter of the full fit). The first two draws as
# before: d1 greedy 3,200 tokens (OOF m3_5f, test m3_fullte_rp), d2 greedy 5,200 tokens (OOF m3long + cv01_m3d2_f034, test
# m3d2te_rp). Weights of ALL channels from the 1,500 rows with three draws (eda/c99_m3_deploy.py --save: a2.fit_mix on log
# loss, then think18nb × 1.4 — the check in step 3 reproduces θ of the earlier primary-line blend file with m3x2_full_cv01
# bit-identically), then arith_beta and the field cv from the c108 evaluation (cross-fit per fold with β and the last step).
#
# ORDER after fetching both runs (m3d3_oof_rp, m3d3te_rp):
#   1. python -I eda/c108_m3_three_draws.py        evaluation and verdict -> artifacts/eda/private/m3_draws_eval/c108_m3_three_draws.json
#   2. bash tools/builds/s029_m3x3.sh best|safe     this build path (aborts if c108 did not give a deployment for the line)
#
# STEPS: 0 test draws complete (ranking exact; generation above the tipping threshold: d1 0.3, d2 0.5, d3 D3_MIN_COMPLETE,
#   default 0.5 as for d2) and training regime gold-only per test draw (c99 --variant against m3p5: the same five fold adapters
#   as d2 and d3 — c108 checks the fold integrity of all OOF jobs); 1 verdict from c108 (real run, same sources, line deployed;
#   IGNORE_VERDICT=1 builds anyway for checking — never for submission); 2 OOF artifact m3x3_full via eda/a9_m3_draws_oof.py
#   (bit-identical to the evaluated mean; an existing artifact only with --verify); 3 blend file (c99 --save, check against the
#   earlier primary-line weights, arith_beta, cv, safe: checksum of the inductive class weights recomputed); 4 make_submission with
#   THREE test draws in the order of the OOF draws (rank_only_draws_guard: per draw ranking and generation against ITS OOF draw
#   — d1 3,200/T 0, d2 5,200/T 0, d3 5,200/T 0.6 differ pairwise, a swapped order aborts); 5 comparison with the earlier
#   two-draw file of the line (aggregates).
#
#     bash tools/builds/s029_m3x3.sh best|safe
#
# Only aggregates and checksums are printed. Nothing is submitted from here.
set -euo pipefail
cd "$(dirname "$0")/../.."
LINE="${1:?best or safe}"
CH="m3x3_full"
D1="runs/m3_5f/m3_rank/bench"
D2="runs/m3long_comb/m3l_rank/bench+runs/cv01_m3d2_f034/m3l_rank_f0/bench+runs/cv01_m3d2_f034/m3l_rank_f3/bench+runs/cv01_m3d2_f034/m3l_rank_f4/bench"
D3="runs/m3d3_oof_rp/m3d3_rank_f0/bench+runs/m3d3_oof_rp/m3d3_rank_f1/bench+runs/m3d3_oof_rp/m3d3_rank_f2/bench+runs/m3d3_oof_rp/m3d3_rank_f3/bench+runs/m3d3_oof_rp/m3d3_rank_f4/bench"
RANK1="runs/m3_fullte_rp/m3_test_rank/test"
RANK2="runs/m3d2te_rp/m3d2_test_rank/test"
RANK3="runs/m3d3te_rp/m3d3_test_rank/test"
RANK_XM="runs/n1_rp/rank_test/test"
ROWS="derived/${CH}_rows.npy"
N_DRAWS=3  # number of draws in the channel (d1, d2, d3)
EVAL="artifacts/eda/private/m3_draws_eval/c108_m3_three_draws.json"
V1_T4="v1=v1te_a_t4/v1_test_a/test+v1te_b_t4/v1_test_b/test"
PR27XM_SHA="bf3d1b58abfc57eb64139418ed7646f45ba9f879791661658035171b9a2f0d03"  # artifacts/oof_pr27xm.npy (deployed sources)
D3_MIN_COMPLETE="${D3_MIN_COMPLETE:-0.5}"
case "$LINE" in
  best) BASE="v1r_zs27q_think18nb_pr27xm_${CH}_w14"; BETA=0.2; SUFFIX="_ra_t4"
        REF="artifacts/submissions/sub_cv01_best_l2.csv"; REF_SHA="1c8c4bb12122e3feeef07d4488e79339635078ab667c29a57192560dcba4e346" ;;  # earlier primary-line file (two draws)
  safe) BASE="v1r_zs27t_think18sb_pr27xm_${CH}"; BETA=0.4; SUFFIX="_ia_t4"
        REF="artifacts/submissions/sub_v1r_zs27t_think18sb_pr27xm_m3x2_ar_ia_t4.csv"; REF_SHA="80ea14fe28449a4b7be66bb0eeb8622f4870a4629105ebe6dc7a3859c36d5557" ;;  # earlier inductive-line file (two draws)
  *) echo "usage: $0 best|safe" >&2; exit 2 ;;
esac
NAME="${BASE}_ar"
OUT="artifacts/submissions/sub_${NAME}${SUFFIX}.csv"
echo "s029 $LINE: channel $CH (three draws) -> artifacts/stack_${NAME}.json -> $OUT"
# 0. test draws complete and gold-only (aggregates only)
for R in "$RANK1" "$RANK2" "$RANK3" "$RANK_XM"; do
  python -I eda/c95_rank_mean.py --status "$R" --table test || {
    echo "ABORT: $R missing or incomplete — wait for the run or fetch it" >&2; exit 3; }
done
python -I eda/a9_m3_draws_oof.py --gen-status "$RANK1" --table test --min-complete 0.3 || { echo "ABORT: generation behind $RANK1" >&2; exit 3; }
python -I eda/a9_m3_draws_oof.py --gen-status "$RANK2" --table test --min-complete 0.5 || { echo "ABORT: generation behind $RANK2" >&2; exit 3; }
python -I eda/a9_m3_draws_oof.py --gen-status "$RANK3" --table test --min-complete "$D3_MIN_COMPLETE" || {
  echo "ABORT: generation behind $RANK3 incomplete or below the tipping threshold $D3_MIN_COMPLETE" >&2; exit 3; }
for R in "$RANK1" "$RANK2" "$RANK3"; do
  python -I eda/c99_m3_deploy.py --variant "$R" --channel m3p5 --require-gold-only || {
    echo "ABORT: training regime of $R not gold-only or different from the fold adapters (see above)" >&2; exit 5; }
done
# 1. verdict of the c108 evaluation (real run, the same sources, line deployed)
python -I - "$EVAL" "$LINE" "$D1,$D2,$D3" "${IGNORE_VERDICT:-0}" <<'PY'
import json
import sys
from pathlib import Path

path, line, src, ignore_verdict = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4] == "1"
if not Path(path).is_file():
    raise SystemExit(f"ABORT: {path} missing — run python -I eda/c108_m3_three_draws.py first")
ev = json.loads(Path(path).read_text())
if ev.get("mode") != "real":
    raise SystemExit(f"ABORT: {path} comes from a check run (mode {ev.get('mode')!r})")
if (ev.get("specs") or {}).get("m3x3_source") != src:
    raise SystemExit(f"ABORT: c108 evaluated different sources ({(ev.get('specs') or {}).get('m3x3_source')})")
dec = ev["decision"][line]
d = ev["lines"][line]["delta"]
print(f"  c108 ({ev['created_utc']}): {line} Δ {d['mean']:+.5f} ± {d['se_q']:.5f} (tasks), cross m3x3 {ev['lines'][line]['cross_m3x3']:.5f}; "
      f"guard {'violated' if ev['decision']['guard_violated'] else 'held'} -> {'DEPLOY' if dec['deploy'] else 'NO DEPLOYMENT'}")
if not dec["deploy"]:
    if not ignore_verdict:
        print(f"ABORT: c108 gave NO DEPLOYMENT for {line} — nothing built (IGNORE_VERDICT=1 builds for checking only, never for submission)", file=sys.stderr)
        raise SystemExit(6)
    print("  WARNING: IGNORE_VERDICT=1 — file for checking only, do NOT submit", file=sys.stderr)
PY
# 2. OOF artifact: mean of three draws on all 1,500 rows (a9; d1 bit-identical to oof_m3p5 via --ref), bit-identical to the evaluation
if [ -f "artifacts/meta_${CH}.json" ]; then
  python -I eda/a9_m3_draws_oof.py --verify "$CH" || { echo "ABORT: oof_${CH} is not the mean of its sources or the row list does not match" >&2; exit 4; }
else
  python -I eda/a9_m3_draws_oof.py "$CH" --draw "$D1" --draw "$D2" --draw "$D3" --rows-out "$ROWS" --ref m3p5 2>&1 | cut -c1-300
fi
python -I - "$EVAL" "$CH" "$D1,$D2,$D3" "$N_DRAWS" <<'PY'
import hashlib
import io
import json
import sys
from pathlib import Path

import numpy as np

ev, ch, src, nd = json.loads(Path(sys.argv[1]).read_text()), sys.argv[2], sys.argv[3], int(sys.argv[4])
meta = json.loads(Path(f"artifacts/meta_{ch}.json").read_text())
buf = io.BytesIO()
np.save(buf, np.asarray(np.load(f"artifacts/oof_{ch}.npy"), dtype=np.float64))
sha = hashlib.sha256(buf.getvalue()).hexdigest()
rk = meta["rank"]
if rk["source"] != src or len(rk.get("draws") or []) != nd or rk.get("rows_by_draws") != {str(nd): 1500}:
    raise SystemExit(f"ABORT: meta_{ch}.json names other sources or not {nd} draws on all rows ({rk.get('rows_by_draws')})")
if sha != ev["m3x3_npy_sha256"]:
    raise SystemExit(f"ABORT: oof_{ch}.npy ({sha[:16]}…) is not the mean evaluated by c108 ({ev['m3x3_npy_sha256'][:16]}…)")
print(f"  oof_{ch}.npy = evaluated mean (SHA-256 of the values {sha[:16]}…); row list {rk['rows_all_draws_file']} ({rk['rows_all_draws']} rows)")
PY
# 3. blend file: weights of all channels on the 1,500 rows (c99 --save; log artifacts/eda/private/m3_deploy/c99_m3_deploy_${CH}_${LINE}.*)
echo "${PR27XM_SHA}  artifacts/oof_pr27xm.npy" | sha256sum -c --quiet || { echo "ABORT: oof_pr27xm.npy differs from the checked state" >&2; exit 4; }
python -I eda/c99_m3_deploy.py --channel "$CH" --rows "$ROWS" --lines "$LINE" --save 2>&1 | cut -c1-300
python -I - "artifacts/stack_${BASE}.json" "artifacts/stack_${NAME}.json" "$BETA" "$LINE" "$EVAL" "$CH" "${IGNORE_VERDICT:-0}" "$N_DRAWS" <<'PY'
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

comp = Path.cwd()
sys.path[:0] = [str(comp), str(comp / "src")]
import labels  # noqa: E402
from vlm import common as C  # noqa: E402


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


src, dst, beta, line, eval_path, ch, ignore_verdict = Path(sys.argv[1]), Path(sys.argv[2]), float(sys.argv[3]), sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7] == "1"
nd = int(sys.argv[8])
s = json.loads(src.read_text())
ev = json.loads(Path(eval_path).read_text())
if s.get("arith_beta"):
    raise SystemExit(f"{src.name} already carries an arithmetic share")
if s.get("rows") != 1500 or (s.get("rank_only") or {}).get(ch, {}).get("draws") != nd:
    raise SystemExit(f"{src.name}: not fitted on 1,500 rows with {nd} draws")
c99 = load("s029_c99", comp / "eda" / "c99_m3_deploy.py")
art = comp / "artifacts"
if line == "best":  # check: the same path with the channel from two draws reproduces the weights of the earlier primary-line blend file bit-identically
    ctx = c99.context()
    names = ["v1r", "zs27q", "think18nb", "pr27xm", "m3x2_full_cv01"]
    mats = {nm: np.load(art / f"oof_{nm}.npy") for nm in names}
    spec = c99.line_setup("best", "m3x2_full_cv01", 0.05, art)
    _, logs, _, ti = c99.variant_logs(spec, mats, ctx["seen"], True)
    th, _ = c99.fit_theta(logs, ctx["y"], np.arange(ctx["n"]), ti, spec["factor"])
    ref = json.loads((art / "stack_cv01_best_l2.json").read_text())["theta"]
    if not np.array_equal(th, np.asarray(ref, dtype=float)):
        raise SystemExit(f"check against the earlier primary-line weights failed: θ {th.tolist()} instead of {ref}")
    print("  check: c99 with m3x2_full_cv01 on 1,500 rows reproduces θ of stack_cv01_best_l2.json bit-identically")
out = dict(s)
out["arith_beta"] = beta
out["cv"] = float(ev["lines"][line]["cross_m3x3"])
out["cv_rows"] = 1500
# provenance fields of the blend file: cv_note, note and the verdict of the three-draw evaluation (m3_draws_eval)
out["cv_note"] = (f"cross-fit per fold on 1,500 rows WITH the arithmetic share β = {beta:g} and the last step as deployed "
                  f"(eda/c108_m3_three_draws.py, {ev['created_utc']}); against m3x2_full_cv01 in the same procedure "
                  f"{ev['lines'][line]['delta']['mean']:+.5f} ± {ev['lines'][line]['delta']['se_q']:.5f} (tasks). Base OOFs not fully nested in the meta fold.")
out["note"] = f"{s.get('note', '')} | arithmetic share β = {beta:g} (tools/builds/s029_m3x3.sh {line}); derived from {src.name}"
out["m3_draws_eval"] = {"eval": eval_path, "eval_created_utc": ev["created_utc"], "m3x3_npy_sha256": ev["m3x3_npy_sha256"], "decision": ev["decision"][line],
                        "guard_violated": ev["decision"]["guard_violated"], "ignore_verdict": ignore_verdict}
if "inductive_check" in s:  # class weights from the blend WITH β (as for the earlier inductive-line blend file); make_submission recomputes them
    ms = load("s029_ms", comp / "tools" / "make_submission.py")
    c21 = load("s029_c21", comp / "eda" / "c21_margin_align.py")
    root = C.find_data_root()
    train = C.load_table(root, "train")
    classes = labels.load_classes(root / "competition_data" / "taxonomy_v1.csv")
    seen = labels.seen_mask(train["root_code"], classes)
    counts = np.bincount(labels.encode(train["root_code"], classes), minlength=len(classes))
    ic = s["inductive_check"]
    target = np.where(counts >= ic["min_rows"], counts, 0) / len(train)
    P = ms.oof_mixture(out, seen, art, n_rows=len(train))
    w = c21.fit_weights(P[ms.stack_fit_rows(out, len(train))], target)
    used = target > 0
    out["inductive_check"] = {**ic, "weights_sha256": hashlib.sha256(np.ascontiguousarray(w, dtype=np.float64).tobytes()).hexdigest()[:16],
                              "weights_q10_q50_q90": [round(float(q), 3) for q in np.quantile(w[used], [0.1, 0.5, 0.9])],
                              "note": ic["note"] + f"; blend with the arithmetic share β = {beta:g}"}
    print(f"  inductive class weights recomputed: checksum {out['inductive_check']['weights_sha256']} (without the share {ic['weights_sha256']})")
dst.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n")
print(f"  {dst.name}: θ " + " · ".join(f"{n} {v:.3f}" for n, v in zip(out["channels"], out["theta"])) + f"; arith_beta {beta:g}, cv {out['cv']:.5f}")
PY
# 4. test file: sources as for the earlier files of the line, M3 as the mean of three test draws (order = OOF draws)
if [ "$LINE" = best ]; then
  python tools/make_submission.py --name "$NAME" \
    --test "v1r=v1rot_t4/rot_test/test@derived/orient/test_c80_rows.npy" --fill v1r=v1 --test "$V1_T4" \
    --test zs27q=zs27q_test0_rp/zs27q_test0/test+zs27q_test1_rp/zs27q_test1/test --fill "think18nb=zs27q" \
    --test "think18nb=zs27v_te_rp/think_te/test/think@derived/think_rows/test.npy,zs27v_n3_rp/think_nte/test/think@derived/think_rows/test_tb_q.npy" \
    --proto pr27xm=runs/pr27te_rp/proto_te/proto --rank "pr27xm=$RANK_XM" \
    --rank-only "$CH=$RANK1,$RANK2,$RANK3" \
    --align 0.75 --align-min 3 --blank-rows derived/blank/test_rows.npy --suffix "$SUFFIX" 2>&1 | cut -c1-300
else
  python tools/make_submission.py --name "$NAME" \
    --test "v1r=v1rot_t4/rot_test/test@derived/orient/test_c80_rows.npy" --fill v1r=v1 --test "$V1_T4" \
    --test zs27t=px27_t4zt0/zs27t_test_t4/test+px27_t4zt1/zs27t_test_t4/test --fill "think18sb=zs27t" \
    --test "think18sb=zs27v_n3_rp/think_nte/test/think@derived/think_rows/test_tb_t.npy" \
    --proto pr27xm=runs/pr27te_rp/proto_te/proto --rank "pr27xm=$RANK_XM" \
    --rank-only "$CH=$RANK1,$RANK2,$RANK3" \
    --align-inductive 0.75 --align-min 3 --blank-rows derived/blank/test_rows.npy --suffix "$SUFFIX" 2>&1 | cut -c1-300
fi
# 5. reference file unchanged; deviation only as an aggregate
echo "${REF_SHA}  ${REF}" | sha256sum -c --quiet || { echo "ERROR: reference file changed ($REF)" >&2; exit 4; }
python -I - "$REF" "$OUT" <<'PY'
import sys

import pandas as pd

a = pd.read_csv(sys.argv[1], dtype=str, keep_default_na=False).set_index("trace_id")["root_code"]
b = pd.read_csv(sys.argv[2], dtype=str, keep_default_na=False).set_index("trace_id")["root_code"].loc[a.index]
top_a, top_b = a.str.split().str[0], b.str.split().str[0]
print(f"  against the reference file (two draws): different top code in {(top_a != top_b).mean():.4f} of the rows, "
      f"different first three (with order) in {(a != b).mean():.4f}")
PY
sha256sum "artifacts/stack_${NAME}.json" "$OUT"
