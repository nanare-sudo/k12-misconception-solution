#!/usr/bin/env bash
# Primary line with q397m3 as the sixth channel (the primary-line file that the second stage starts from):
# the six-channel recipe of tools/builds/s029_q397.sh best --m3 m3x3_full (v1r, zs27q, think18nb × 1.4, pr27xm, M3 as the mean
# of three draws; floors 0.01/0.05/0.05/0.05; β 0.2; margin alignment 0.75 for classes with at least 3 rows, transductive,
# primary line only; REL-01 for blank photos), but the sixth channel is q397m3 INSTEAD OF q397: M3 scores all codes behind the
# notes of Qwen3.5-397B (pure ranking channel, floor 0.05). OOF: per fold the M3 adapter trained without that fold, behind the
# 397B notes of the training rows (runs/q397m3_oof_rp/q397m3_rank_f0…f4; eda/c117_q397m3_deploy.py oof, bit-identical to the
# channel of the c115 evaluation); test: M3 of the final fit (m3_fullte_rp/m3_train_full) behind the notes of our own 397B run
# (runs/q397m3te_rp/q397m3_test_rank/test, notes derived/sft/q397m3_test_notes.jsonl from runs/q397te_rp/proto_te/proto).
# Blend file artifacts/stack_s029_best_q397m3.json: θ from all 1,500 rows, cross value 0.83267 (with q397 instead: 0.82922;
# Δ +0.0034 ± 0.0028 over tasks; eda/c110_q397_deploy.py stack --replace-q397).
# All other test channels byte-identical to the six-channel file with q397 (tools/builds/s029_q397.sh best --m3 m3x3_full).
#
#     bash tools/builds/s029_q397m3.sh best        # -> artifacts/submissions/sub_s029_best_q397m3.csv
#
# Primary line only. Exit 3 while the q397m3 test side is missing or incomplete (nothing written); 4 = checksum or
# recomputation; 5 = guard of the test side violated; 6 = c108 verdict for m3x3 missing; 1 = a check inside a called
# Python script that stops with a message.
# STEPS: 0 q397m3 test side present and complete (c95 --status), 397B test run chosen; 1 checksums (base, M3, pr27xm, q397 for
#   the cross-check, notes, reference files); 2 OOF artifact q397m3 (c117 oof --save on the first call, otherwise --verify;
#   checksum fixed); 3 M3 from three draws (a9 --verify, c108 verdict, generation and training regime of the three test draws);
#   4 blend file (c110 stack --replace-q397 q397m3: cross value = replacement value of c115 exactly, cross-check
#   stack_s029_best_m3x3.json character by character; checksum fixed); 5 397B test side (c110 test-status) and fidelity check;
#   6 guard of the q397m3 test side (c117 test-guard: settings per OOF shard, 397B notes with source and protocol, training
#   regime of the final fit against the fold adapters, gold-only); 7 ranking runs complete; 8 make_submission (the note guard
#   rank.notes applies there once more; NONE band for q397m3 0.27–0.35 instead of 0.30–0.38, OOF 0.310 ± 0.04, fixed in
#   advance); 9 comparison with the two reference files (aggregates only). An existing output file is only replaced by a
#   byte-identical rebuild (otherwise it is restored and the script exits with 4).
# DRY RUN (code check only): PLACEHOLDER=1 uses the d3 test ranking (runs/m3d3te_rp/m3d3_test_rank/test) as q397m3 — the note
#   guard skips the test notes (c117 test-guard --placeholder, make_submission --rank-only-placeholder); file, JSON and test
#   matrix go to a throw-away directory and are deleted at the end — NEVER submit.
# ROTATION LIST (tools/builds/orient_list.sh): ORIENT=repro (default) reads for v1r the rotation list of the released orientation
#   network, derived/orient_repro/test_c80_rows.npy (tools/orient_list.py check before, verify-sub after make_submission), and
#   writes the intermediate file artifacts/submissions/sub_s032n_q397m3.csv (test matrix test_s029_best_q397m3_s032n.npy;
#   reference of copy null point (a) of tools/builds/s033n_stage2.sh); all other inputs and guards as with ORIENT=legacy, plus a
#   comparison with the legacy-list file as aggregates. ORIENT=legacy rebuilds the historical file sub_s029_best_q397m3.csv
#   with the first-version rotation list byte-identically; the released-list build needs that file first.
#
# Only aggregates and checksums are printed. Nothing is submitted from here.
set -euo pipefail
cd "$(dirname "$0")/../.."
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}" OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
usage() { echo "usage: $0 best   (PLACEHOLDER=1 for the dry run)" >&2; exit 2; }
[ $# -eq 1 ] && [ "$1" = best ] || usage
PLACEHOLDER="${PLACEHOLDER:-0}"
STEM="s029_best_q397m3"
CH="q397m3"
M3="m3x3_full"
STACK="artifacts/stack_${STEM}.json"
Q397M3_REAL="runs/q397m3te_rp/q397m3_test_rank/test"
Q397M3_STANDIN="runs/m3d3te_rp/m3d3_test_rank/test"
RANK1="runs/m3_fullte_rp/m3_test_rank/test"
RANK2="runs/m3d2te_rp/m3d2_test_rank/test"
RANK3="runs/m3d3te_rp/m3d3_test_rank/test"
RANK_XM="runs/n1_rp/rank_test/test"
V1_T4="v1=v1te_a_t4/v1_test_a/test+v1te_b_t4/v1_test_b/test"
EVAL="artifacts/eda/private/m3_draws_eval/c108_m3_three_draws.json"
REF_Q397="artifacts/submissions/sub_s029_best_m3x3.csv"; REF_Q397_SHA="704d8ae8e6eb93051aa4fbe8ec954da143c1c68ff751fc6fe0063c681c063b18"
REF_5CH="artifacts/submissions/sub_v1r_zs27q_think18nb_pr27xm_m3x3_full_w14_ar_ra_t4.csv"; REF_5CH_SHA="22ccc68c7d833904c994a648c52d8d3a1165984e21f81dc1b2dd87a69600c82a"
OOF_SHA="5583b46fe0fcb1f075e62ecd689ea10bc5d1a392cf69d7f62b334dc96946f5d2"    # artifacts/oof_q397m3.npy (= channel of the c115 evaluation)
META_SHA="baf1ceb345f00eaf4d699d59406807d0d741c086cd496a149a90728f8a12b36a"   # artifacts/meta_q397m3.json (c117 oof)
STACK_SHA="4910a2ac4be346da19fbfd0776711b51e7bbaea5e125d7f2423854d37b26d1f3"  # artifacts/stack_s029_best_q397m3.json (c110)
NOTES_TEST_SHA="4b29f057d50c7decfc3db73ddb37182d26b51326e3af05683d4665a56f20eb60"  # data/derived/sft/q397m3_test_notes.jsonl
# NONE malfunction band for q397m3 only (OOF 0.310 ± 0.04; = eda/c117_q397m3_deploy.NONE_BAND), fixed in advance — the shared
# M3 band 0.30–0.38 (--rank-none-band) would lie only 0.01 below this channel's OOF value; it still applies to all other ranking channels
NONE_BAND_Q397M3="0.27,0.35"
check() { echo "$1  $2" | sha256sum -c --quiet || { echo "ABORT: $2 differs (checksum)" >&2; exit 4; }; }
. tools/builds/orient_list.sh  # rotation list of the v1r test side: ORIENT=repro (default, released list) | legacy (first-version list)
orient_select
RSUF=""  # suffix of the output files: empty = legacy-list file
if [ "$ORIENT" = repro ]; then
  RSUF="_s032n"
  LEGACY_OUT="artifacts/submissions/sub_s029_best_q397m3.csv"; LEGACY_OUT_SHA="768b7778ea049caeb23455ecad26b9bc90ebff52bee3cb97e356aac7cadf07e8"  # legacy-list file of this build
fi

# 0. q397m3 test side present and complete (otherwise exit 3, nothing written); 397B test run (source of the test notes)
if [ "$PLACEHOLDER" = 1 ]; then
  Q397M3_TEST="$Q397M3_STANDIN"
  echo "PLACEHOLDER: q397m3 test side := $Q397M3_TEST (d3, NOT M3 behind the 397B notes) — code check only, the file is deleted, NEVER submit" >&2
else
  Q397M3_TEST="$Q397M3_REAL"
fi
python -I eda/c95_rank_mean.py --status "$Q397M3_TEST" --table test || {
  echo "ABORT: q397m3 test side $Q397M3_TEST missing or incomplete — wait for the run or fetch it" >&2; exit 3; }
if [ -n "${Q397_RUN:-}" ]; then
  RUNS=("$Q397_RUN")
else
  RUNS=()
  for R in q397te_rp q397te8_rp; do [ -f "artifacts/runs/$R/$R/proto_te/proto_meta.json" ] && RUNS+=("$R"); done
fi
case "${#RUNS[@]}" in
  1) Q397_TEST="runs/${RUNS[0]}/proto_te/proto" ;;
  0) echo "ABORT: no fetched 397B test side (artifacts/runs/q397te_rp|q397te8_rp/<run>/proto_te/proto_meta.json)" >&2; exit 3 ;;
  *) echo "ABORT: both 397B test runs exist (${RUNS[*]}) — choose the deployed one with Q397_RUN=<run>" >&2; exit 3 ;;
esac
echo "s029_q397m3 best: blend $STACK, q397m3 test side $Q397M3_TEST, 397B test side (source of the test notes) $Q397_TEST"

# 1. checksums: M3, pr27xm, q397 (cross-check), source job of the 397B training notes, six-channel blend with q397, notes, reference files
check 13b3b03ecf64161067a324d48eb1d85922dd9908167fa99ef35b29a7f1a0467d artifacts/oof_m3x3_full.npy
check bf3d1b58abfc57eb64139418ed7646f45ba9f879791661658035171b9a2f0d03 artifacts/oof_pr27xm.npy
check c5bcd665a0cd5246b78dcd67012d6831e0e51ecb553666d60e6f164bf519fadf artifacts/oof_q397.npy
check f591eb385fb7fecad986937488d5e9ac5c180bc0d597f19f71750340917388dc artifacts/meta_q397.json
check 6935614248a22da250054dd14b180fc5875832ddba7afc49c4b3da0337503280 artifacts/runs/q397_tinker/q397_tinker/proto_t/proto_labels.jsonl
check 9c98234642e51f16a77f6deb746b81d8f73523c706a69f1a5690a76a63ef2d97 artifacts/stack_s029_best_m3x3.json
check 5a8fb7358bf9a25bcd4f6b12b28c02b31e5ad345a3ee21f9d94bc182bf180d52 data/derived/sft/q397m3_train_notes.jsonl
[ "$PLACEHOLDER" = 1 ] || check "$NOTES_TEST_SHA" data/derived/sft/q397m3_test_notes.jsonl
check "$REF_Q397_SHA" "$REF_Q397"
check "$REF_5CH_SHA" "$REF_5CH"
[ "$ORIENT" = repro ] && check "$LEGACY_OUT_SHA" "$LEGACY_OUT"
python -I eda/c110_q397_deploy.py verify-oof || { echo "ABORT: oof_q397 is not the channel of the c107 decision" >&2; exit 4; }

# 2. OOF artifact q397m3: pure ranking of the five shards, bit-identical to the channel of the c115 evaluation (fold fidelity, notes, settings)
if [ -f "artifacts/meta_${CH}.json" ]; then
  python -I eda/c117_q397m3_deploy.py oof --verify || { rc=$?; echo "ABORT: oof_${CH} is not the recomputation from the shards" >&2; exit "$rc"; }
else
  python -I eda/c117_q397m3_deploy.py oof --save || { rc=$?; echo "ABORT: oof_${CH} not built" >&2; exit "$rc"; }
fi
check "$OOF_SHA" "artifacts/oof_${CH}.npy"
check "$META_SHA" "artifacts/meta_${CH}.json"

# 3. M3 from three draws as in the six-channel file with q397: artifact = mean of its sources, verdict of the line in c108, test draws complete, gold-only
python -I eda/a9_m3_draws_oof.py --verify "$M3" || { echo "ABORT: oof_${M3} missing or not the mean of its sources" >&2; exit 4; }
python -I - "$EVAL" <<'PY'
import hashlib
import io
import json
import sys
from pathlib import Path

import numpy as np

path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(f"ABORT: {path} missing — stored verdict of the M3 evaluation (frozen input, see docs/CODEMAP.md §10)")
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
dec = ev["decision"]["best"]
print(f"  c108 ({ev['created_utc']}): best -> {'DEPLOY' if dec['deploy'] else 'NO DEPLOYMENT'} (guard {'violated' if ev['decision']['guard_violated'] else 'held'})")
if not dec["deploy"]:
    print("ABORT: c108 gave NO DEPLOYMENT for the primary line", file=sys.stderr)
    raise SystemExit(6)
PY
python -I eda/a9_m3_draws_oof.py --gen-status "$RANK1" --table test --min-complete 0.3 || { echo "ABORT: generation behind $RANK1" >&2; exit 3; }
python -I eda/a9_m3_draws_oof.py --gen-status "$RANK2" --table test --min-complete 0.5 || { echo "ABORT: generation behind $RANK2" >&2; exit 3; }
python -I eda/a9_m3_draws_oof.py --gen-status "$RANK3" --table test --min-complete "${D3_MIN_COMPLETE:-0.5}" || { echo "ABORT: generation behind $RANK3" >&2; exit 3; }
for R in "$RANK1" "$RANK2" "$RANK3"; do
  python -I eda/c99_m3_deploy.py --variant "$R" --channel m3p5 --require-gold-only || { echo "ABORT: training regime of $R not gold-only" >&2; exit 5; }
done

# 4. blend file = recomputation (cross value = replacement value of c115 exactly, six-channel recipe with q397 = anchor, cross-check stack_s029_best_m3x3.json, oof_mixture)
if [ -f "$STACK" ]; then
  python -I eda/c110_q397_deploy.py stack --line best --m3 "$M3" --replace-q397 "$CH" --verify || { rc=$?; echo "ABORT: $STACK is not the recomputation" >&2; exit "$rc"; }
else
  python -I eda/c110_q397_deploy.py stack --line best --m3 "$M3" --replace-q397 "$CH" --save || { rc=$?; echo "ABORT: $STACK not built" >&2; exit "$rc"; }
fi
check "$STACK_SHA" "$STACK"

# 5. 397B test side (source of the test notes; aggregates only) and fidelity check
python -I eda/c110_q397_deploy.py test-status "$Q397_TEST" || { rc=$?; echo "ABORT: 397B test side not deployable (code $rc)" >&2; exit "$rc"; }
python -I - <<'PY' || { echo "ABORT: fidelity check missing or not passed" >&2; exit 5; }
import json
import sys
from pathlib import Path

f = Path("artifacts/eda/private/q397_deploy/c109_parity.json")
if not f.is_file():
    print(f"{f} missing — stored verdict of the 397B parity check (frozen input, see docs/CODEMAP.md §10)", file=sys.stderr)
    raise SystemExit(5)
r = json.loads(f.read_text())
print(f"  fidelity check (c109, training rows): {r['verdict']} — valid {r['valid_own']}/{r['rows']}, same top {r['same_top']:.3f}")
raise SystemExit(0 if r.get("verdict") == "PASSED" else 5)
PY

# 6. guard of the q397m3 test side: settings per OOF shard, 397B notes (source = chosen 397B test run, protocol), training regime
if [ "$PLACEHOLDER" = 1 ]; then
  python -I eda/c117_q397m3_deploy.py test-guard "$Q397M3_TEST" --placeholder || { rc=$?; echo "ABORT: guard (placeholder) code $rc" >&2; exit "$rc"; }
else
  python -I eda/c117_q397m3_deploy.py test-guard "$Q397M3_TEST" --notes-source "$Q397_TEST" || { rc=$?; echo "ABORT: q397m3 test side not deployable (code $rc)" >&2; exit "$rc"; }
fi

# 7. ranking runs complete (M3 test draws, pr27xm)
for R in "$RANK1" "$RANK2" "$RANK3" "$RANK_XM"; do
  python -I eda/c95_rank_mean.py --status "$R" --table test || { echo "ABORT: $R missing or incomplete" >&2; exit 3; }
done

# 8. test file (channels as in the six-channel file with q397, q397m3 instead of --proto q397)
[ "$ORIENT" = repro ] && { python -I tools/orient_list.py check --mode repro --spec "$V1R_TEST" || exit $?; }
if [ "$PLACEHOLDER" = 1 ]; then
  TMP="$(mktemp -d "${TMPDIR:-/tmp}/s029_q397m3_placeholder.XXXXXX")"
  OUT="$TMP/sub_${STEM}${RSUF}_PLACEHOLDER.csv"
  TESTNPY="artifacts/test_${STEM}${RSUF}_PLACEHOLDER.npy"
  trap 'rm -rf -- "${TMP:?}"; rm -f -- "${TESTNPY:?}"; echo "PLACEHOLDER: file, JSON and test matrix deleted" >&2' EXIT
  OUTARGS=(--suffix "${RSUF}_PLACEHOLDER" --out "$OUT" --rank-only-placeholder "$CH")
else
  OUT="artifacts/submissions/sub_${STEM}.csv"
  OUTARGS=()
  if [ "$ORIENT" = repro ]; then  # intermediate file with the released rotation list: own file and test matrix; the legacy-list file stays untouched
    OUT="artifacts/submissions/sub_s032n_q397m3.csv"
    OUTARGS=(--suffix "$RSUF" --out "$OUT")
  fi
  if [ -f "$OUT" ]; then  # replace only by a byte-identical rebuild: back up, compare after the build
    BAK="$(mktemp -d "${TMPDIR:-/tmp}/s029_q397m3_backup.XXXXXX")"
    trap 'rm -rf -- "${BAK:?}"' EXIT  # the backup holds test identifiers: never leave it behind
    for F in "$OUT" "${OUT%.csv}.json" "artifacts/test_${STEM}${RSUF}.npy"; do [ -f "$F" ] && cp -p "$F" "$BAK/"; done
  fi
fi
PYTHONDONTWRITEBYTECODE=1 python -I tools/make_submission.py --name "$STEM" \
  --test "$V1R_TEST" --fill "v1r=v1" --test "$V1_T4" \
  --test "zs27q=zs27q_test0_rp/zs27q_test0/test+zs27q_test1_rp/zs27q_test1/test" --fill "think18nb=zs27q" \
  --test "think18nb=zs27v_te_rp/think_te/test/think@derived/think_rows/test.npy,zs27v_n3_rp/think_nte/test/think@derived/think_rows/test_tb_q.npy" \
  --proto "pr27xm=runs/pr27te_rp/proto_te/proto" --rank "pr27xm=$RANK_XM" \
  --rank-only "$M3=$RANK1,$RANK2,$RANK3" \
  --rank-only "$CH=$Q397M3_TEST" --rank-none-band-for "$CH=$NONE_BAND_Q397M3" \
  --align 0.75 --align-min 3 --blank-rows derived/blank/test_rows.npy "${OUTARGS[@]}" 2>&1 | grep -vi '^warning' | cut -c1-300
[ -f "$OUT" ] || { echo "ABORT: $OUT was not written" >&2; exit 4; }
if [ "$PLACEHOLDER" != 1 ] && [ -n "${BAK:-}" ]; then
  if ! cmp -s "$OUT" "$BAK/$(basename "$OUT")"; then
    for F in "$OUT" "${OUT%.csv}.json"; do [ -f "$BAK/$(basename "$F")" ] && cp -p "$BAK/$(basename "$F")" "$F"; done
    [ -f "$BAK/test_${STEM}${RSUF}.npy" ] && cp -p "$BAK/test_${STEM}${RSUF}.npy" "artifacts/test_${STEM}${RSUF}.npy"
    echo "ABORT: rebuild differs from the existing $OUT — backup restored, investigate the cause" >&2; exit 4
  fi
  echo "  existing $OUT rebuilt byte-identically"
fi
[ "$ORIENT" = repro ] && { python -I tools/orient_list.py verify-sub --mode repro "${OUT%.csv}.json" || exit $?; }

# 9. reference files unchanged; deviation only as an aggregate
check "$REF_Q397_SHA" "$REF_Q397"
check "$REF_5CH_SHA" "$REF_5CH"
python -I - "$OUT" "$REF_Q397" "$REF_5CH" <<'PY'
import sys

import pandas as pd


def read(p):
    return pd.read_csv(p, dtype=str, keep_default_na=False).set_index("trace_id")["root_code"]


new = read(sys.argv[1])
for path, label in ((sys.argv[2], "six-channel file with q397"), (sys.argv[3], "five-channel file (no sixth channel)")):
    ref = read(path)
    b = new.loc[ref.index]
    print(f"  against {label}: different top code in {(ref.str.split().str[0] != b.str.split().str[0]).mean():.4f} of the rows, "
          f"different first three (with order) in {(ref != b).mean():.4f}")
PY
if [ "$ORIENT" = repro ]; then  # legacy-list file unchanged; difference from it only as aggregates (split by the rows whose list membership differs)
  check "$LEGACY_OUT_SHA" "$LEGACY_OUT"
  python -I tools/orient_list.py compare "$LEGACY_OUT" "$OUT" --label "primary line without second stage, first-version rotation list"
fi
sha256sum "$STACK" "$OUT"
if [ "$PLACEHOLDER" = 1 ]; then
  echo "PLACEHOLDER RUN: code path checked (two --rank-only channels, note guard with placeholder, last step) — the file is NOT a candidate and is deleted"
elif [ "$ORIENT" = repro ]; then
  sha256sum "${OUT%.csv}.json" "artifacts/test_${STEM}${RSUF}.npy"
  echo "s029_q397m3 best (ORIENT=repro): $OUT built (intermediate file with the released rotation list, reference of copy null point (a) of s033n_stage2.sh; not meant for submission)"
else
  echo "s029_q397m3 best: $OUT built"
fi
