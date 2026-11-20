#!/usr/bin/env bash
# Run the GPU model steps of the final submissions again and write A and B from the new outputs.
#
#   bash tools/gpu_inference.sh rescore <package-dir> [--out DIR] [--dry-run]
#   bash tools/gpu_inference.sh full    <package-dir> [--out DIR] [--dry-run]
#
# rescore   every scoring and ranking step on the test photos, behind the frozen texts of the package. No text is
#           generated, and no vLLM environment is needed.
# full      every step, with new reasoning texts, protocol answers, M3 texts and 397B answers.
# Run times and the hardware of each step: docs/VERIFICATION.md, section 3.
#
# Before a real run:
#   1. Run "bash reproduce.sh <package-dir>" once in this repository. It sets up artifacts/ and data/derived/ and writes
#      the reference files artifacts/submissions/check_A.csv and check_B.csv.
#   2. Put the competition photos under data/competition_data/images/ and data/unlabelled.csv next to the other CSV
#      files.
#   3. Install the GPU environment of docs/VERIFICATION.md, section 2. Mode full also needs the vLLM environment in
#      /workspace/vllm_env with the marker file .k12_ready (docs/INFERENCE.md, section 10.7).
#   4. Download the base models at their pinned revisions into one folder and set K12_MODELS to it. Merged M3 models
#      are written there too, one at a time.
# The script sets K12_DATA to data/ of this repository and K12_ADAPTERS to <package-dir>/weights, and it switches off
# every download and usage report of the libraries.
#
# Steps:
#   1. Check every input (tools/gpu_inference_inputs.py); stop before the first GPU step if one is missing.
#   2. Make data/derived/ a writable copy of the package folder, because the steps below write new notes into it.
#   3. Copy the delivered M3 adapter to <out>/m3_fullte_rp/m3_train_full/; every M3 configuration merges it from there.
#      Mode rescore also copies the frozen texts of the package into <out>/.
#   4. Run the configurations with the model runner, then copy each finished job folder to
#      artifacts/runs/<run>/<run>/<job>/, where the CPU tools read it. Mode full then exports the new protocol notes
#      and 397B notes and runs the M3 rankings behind them.
#   5. Freeze the M3 texts as the rankings scored them (tools/freeze_m3_texts.py). Mode full re-records the checksums of
#      the new protocol answers (tools/repin_test_side.py --write); mode rescore checks that they did not change.
#   6. Write artifacts/submissions/new_B.csv and new_A.csv with the two make_submission calls of reproduce.sh and
#      compare them with check_B.csv and check_A.csv. Only counts and shares are printed.
#
# A configuration that finished is not run again when the script is called again with the same output folder (marker
# files in <out>/.done/). The default output folder is runs/gpu_<mode>/ in this repository. --dry-run prints every
# command and checks every input; it needs no GPU and no base models. The configurations name the GPUs they use
# (field gpu of each worker); to split the work over several machines, run the printed commands by hand.
# Afterwards artifacts/ and data/derived/ hold the new outputs. To run reproduce.sh on the frozen outputs again, move
# both folders away first; reproduce.sh then sets them up again from the package.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$REPO"

usage() { sed -n '4,5p' "$0" | sed 's/^# \{0,1\}//'; }
die() { echo "ERROR: $*" >&2; exit 1; }

MODE="" PKG_ARG="" OUT_ARG="" DRY=0
while [ $# -gt 0 ]; do
  case "$1" in
    rescore|full) [ -z "$MODE" ] || { usage >&2; exit 2; }; MODE="$1" ;;
    --out) [ $# -ge 2 ] || { usage >&2; exit 2; }; OUT_ARG="$2"; shift ;;
    --out=*) OUT_ARG="${1#--out=}" ;;
    --dry-run) DRY=1 ;;
    -h|--help) usage; exit 0 ;;
    -*) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
    *) [ -z "$PKG_ARG" ] || { usage >&2; exit 2; }; PKG_ARG="$1" ;;
  esac
  shift
done
[ -n "$MODE" ] && [ -n "$PKG_ARG" ] || { usage >&2; exit 2; }
[ -d "$PKG_ARG" ] || die "package folder not found: $PKG_ARG"
PKG="$(cd "$PKG_ARG" && pwd -P)"
OUT="$(realpath -m "${OUT_ARG:-$REPO/runs/gpu_$MODE}")"
case "$OUT/" in "$PKG"/*) die "the output folder lies inside the package folder; choose another --out" ;; esac

# Fixed environment: data of this repository, our adapters from the package, no downloads, no usage reports, local
# addresses only (without VLLM_HOST_IP, vLLM looks up its own address with a UDP connect to a public address).
export K12_DATA="$REPO/data" K12_ADAPTERS="$PKG/weights"
export PYTHONPATH="$REPO/src:$REPO"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 \
       VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 VLLM_DO_NOT_TRACK=1 VLLM_HOST_IP=127.0.0.1 \
       GLOO_SOCKET_IFNAME=lo NCCL_SOCKET_IFNAME=lo
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 PYTHONUTF8=1
unset ORIENT

# The configurations of each mode, in the order in which they run.
GPU_STEPS_RESCORE=(v1te_a_t4 v1te_b_t4 test_side/v1rot_t4 zs27q_test0_rp zs27q_test1_rp
  rescore/zs27v_te_rp rescore/zs27v_n3_rp
  rescore/m3_fullte_rp rescore/m3d2te_rp rescore/m3d3te_rp rescore/m3d4te_rp rescore/m3hte_a_rp rescore/m3hte_b_rp
  m3px27te_rp q397m3te_rp)
GPU_STEPS_FULL=(v1te_a_t4 v1te_b_t4 test_side/v1rot_t4 zs27q_test0_rp zs27q_test1_rp
  test_side/zs27v_te_rp test_side/zs27v_n3_rp pr27te_rp
  test_side/m3_fullte_rp m3d2te_rp m3d3te_rp m3d4te_rp m3hte_a_rp m3hte_b_rp
  q397p_rp q397te_rp)
GPU_STEPS_FULL_AFTER_NOTES=(m3px27te_rp q397m3te_rp)   # M3 behind the new protocol notes and 397B notes
# Frozen texts that mode rescore copies from the package (the rescore configurations read them).
FROZEN_TEXTS=(zs27v_te_rp/gen_te zs27v_n3_rp/gen_nte m3_fullte_rp/m3_test_gen m3d2te_rp/m3d2_test_gen
  m3d3te_rp/m3d3_test_gen m3d4te_rp/m3d4_test_gen m3hte_a_rp/m3h_test_gen m3hte_b_rp/m3h_test_gen)
# Job folders that the CPU tools read, per mode (<run>/<job>).
PUT_RESCORE=(v1te_a_t4/v1_test_a v1te_b_t4/v1_test_b v1rot_t4/rot_test zs27q_test0_rp/zs27q_test0 zs27q_test1_rp/zs27q_test1
  zs27v_te_rp/think_te zs27v_n3_rp/think_nte m3_fullte_rp/m3_test_rank m3d2te_rp/m3d2_test_rank m3d3te_rp/m3d3_test_rank
  m3d4te_rp/m3d4_test_rank m3hte_a_rp/m3h_test_rank m3hte_b_rp/m3h_test_rank m3px27te_rp/m3px27_test_rank
  q397m3te_rp/q397m3_test_rank)
PUT_FULL=(v1te_a_t4/v1_test_a v1te_b_t4/v1_test_b v1rot_t4/rot_test zs27q_test0_rp/zs27q_test0 zs27q_test1_rp/zs27q_test1
  zs27v_te_rp/gen_te zs27v_te_rp/think_te zs27v_n3_rp/gen_nte zs27v_n3_rp/think_nte pr27te_rp/proto_te
  m3_fullte_rp/m3_test_gen m3_fullte_rp/m3_test_rank m3d2te_rp/m3d2_test_gen m3d2te_rp/m3d2_test_rank
  m3d3te_rp/m3d3_test_gen m3d3te_rp/m3d3_test_rank m3d4te_rp/m3d4_test_gen m3d4te_rp/m3d4_test_rank
  m3hte_a_rp/m3h_test_gen m3hte_a_rp/m3h_test_rank m3hte_b_rp/m3h_test_gen m3hte_b_rp/m3h_test_rank
  q397p_rp/proto_p q397te_rp/proto_te)
PUT_FULL_AFTER_NOTES=(m3px27te_rp/m3px27_test_rank q397m3te_rp/q397m3_test_rank)
# M3 draws d1-d4: generation and ranking of each test draw, and the frozen note file that the second stage reads.
FREEZE=(
  "runs/m3_fullte_rp/m3_fullte_rp/m3_test_gen/gen runs/m3_fullte_rp/m3_fullte_rp/m3_test_rank/test derived/sft/m3_d1_test_frozen_notes.jsonl"
  "runs/m3d2te_rp/m3d2te_rp/m3d2_test_gen/gen runs/m3d2te_rp/m3d2te_rp/m3d2_test_rank/test derived/sft/m3_d2_test_frozen_notes.jsonl"
  "runs/m3d3te_rp/m3d3te_rp/m3d3_test_gen/gen runs/m3d3te_rp/m3d3te_rp/m3d3_test_rank/test derived/sft/m3_d3_test_frozen_notes.jsonl"
  "runs/m3d4te_rp/m3d4te_rp/m3d4_test_gen/gen runs/m3d4te_rp/m3d4te_rp/m3d4_test_rank/test derived/sft_k2/m3_d4_test_frozen_notes.jsonl"
)

# The arguments of the two calls that write the final submissions (the same as in reproduce.sh; a test checks it).
B_ARGS=(
  --name primary_m3x4
  --test "v1r=v1rot_t4/rot_test/test@derived/orient_repro/test_c80_rows.npy" --fill "v1r=v1"
  --test "v1=v1te_a_t4/v1_test_a/test+v1te_b_t4/v1_test_b/test"
  --test "zs27q=zs27q_test0_rp/zs27q_test0/test+zs27q_test1_rp/zs27q_test1/test"
  --fill "think18nb=zs27q"
  --test "think18nb=zs27v_te_rp/think_te/test/think@derived/think_rows/test.npy,zs27v_n3_rp/think_nte/test/think@derived/think_rows/test_tb_q.npy"
  --proto "m3px27xm=runs/pr27te_rp/proto_te/proto" --rank "m3px27xm=runs/m3px27te_rp/m3px27_test_rank/test"
  --rank-only "m3x4_full=runs/m3_fullte_rp/m3_test_rank/test,runs/m3d2te_rp/m3d2_test_rank/test,runs/m3d3te_rp/m3d3_test_rank/test,runs/m3d4te_rp/m3d4_test_rank/test"
  --rank-only "q397m3=runs/q397m3te_rp/q397m3_test_rank/test" --rank-none-band-for "q397m3=0.27,0.35"
  --align 0.75 --align-min 3 --blank-rows derived/blank/test_rows.npy
  --stage2 artifacts/stage2_primary_m3x4.json
)
A_ARGS=(
  --name primary_m3x5h
  --test "v1r=v1rot_t4/rot_test/test@derived/orient_repro/test_c80_rows.npy" --fill "v1r=v1"
  --test "v1=v1te_a_t4/v1_test_a/test+v1te_b_t4/v1_test_b/test"
  --test "zs27q=zs27q_test0_rp/zs27q_test0/test+zs27q_test1_rp/zs27q_test1/test"
  --fill "think18nb=zs27q"
  --test "think18nb=zs27v_te_rp/think_te/test/think@derived/think_rows/test.npy,zs27v_n3_rp/think_nte/test/think@derived/think_rows/test_tb_q.npy"
  --proto "m3px27xm=runs/pr27te_rp/proto_te/proto" --rank "m3px27xm=runs/m3px27te_rp/m3px27_test_rank/test"
  --rank-only "m3x5h=runs/m3_fullte_rp/m3_test_rank/test,runs/m3d2te_rp/m3d2_test_rank/test,runs/m3d3te_rp/m3d3_test_rank/test,runs/m3d4te_rp/m3d4_test_rank/test,runs/m3hte_a_rp/m3h_test_rank/test+runs/m3hte_b_rp/m3h_test_rank/test@derived/charsize/test_h20_rows.npy"
  --rank-only "q397m3=runs/q397m3te_rp/q397m3_test_rank/test" --rank-none-band-for "q397m3=0.27,0.35"
  --align 0.75 --align-min 3 --blank-rows derived/blank/test_rows.npy
  --stage2 artifacts/stage2_primary_m3x5h.json
)

show() { printf '+'; printf ' %q' "$@"; printf '\n'; }   # print a command, quoted for copy and paste
cmd() {  # cmd <command...>: print a command, then run it (not in a dry run)
  show "$@"
  [ "$DRY" = 1 ] || "$@" || die "this step failed: $*"
}
run() {  # run <configuration>: one configuration with the model runner; skipped if it finished in an earlier call
  local cfg="$1" mark="$OUT/.done/${1//\//__}"
  if [ "$DRY" = 0 ] && [ -f "$mark" ]; then
    echo "   $cfg finished earlier ($(cat "$mark")); not run again"
    return 0
  fi
  show python -u -m vlm.runner --config "$REPO/configs/vlm/$cfg.yaml" --out "$OUT"
  [ "$DRY" = 1 ] && return 0
  (cd "$REPO/src" && python -u -m vlm.runner --config "$REPO/configs/vlm/$cfg.yaml" --out "$OUT") \
    || die "configuration $cfg failed (worker logs: $OUT/<run>/worker*.log)"
  mkdir -p "$OUT/.done"
  date -u +%Y-%m-%dT%H:%MZ > "$mark"
}
replace_dir() {  # replace_dir <source folder> <target parent>: copy the folder there; an existing copy is removed first
  local name
  name="$(basename "$1")"
  if [ "$DRY" = 0 ]; then
    [ -d "$1" ] || die "not computed: $1"
    mkdir -p "$2"
    case "$(realpath "$2")/" in "$PKG"/*) die "$2 lies inside the package folder; the package must not be written to" ;; esac
  fi
  show mkdir -p "$2"
  cmd rm -rf "${2:?}/$name"
  cmd cp -r "$1" "$2/"
  cmd chmod -R u+w "$2/$name"
}
put() {  # put <run>/<job> ...: replace these job folders where the CPU tools read them (artifacts/runs/<run>/<run>/<job>/)
  local d
  for d in "$@"; do
    replace_dir "$OUT/${d%%/*}/${d#*/}" "$REPO/artifacts/runs/${d%%/*}/${d%%/*}"
    [ "$DRY" = 1 ] || echo "$(date -u +%Y-%m-%dT%H:%MZ) $d" >> "$OUT/put.log"
  done
}

echo "== $MODE: package $PKG, output folder $OUT$([ "$DRY" = 1 ] && echo ', dry run')"
[ -f "$PKG/MANIFEST.sha256" ] && [ -d "$PKG/artifacts/runs" ] && [ -d "$PKG/weights" ] \
  || die "$PKG is not the private package (MANIFEST.sha256, artifacts/runs/ and weights/ expected)"

echo "== 1. Inputs"
SETUP=()  # problems of the repository setup (fatal in a real run, listed in a dry run)
[ -d artifacts ] && [ ! -L artifacts ] || SETUP+=("artifacts/ is not set up: run bash reproduce.sh $PKG first")
[ ! -L artifacts/runs ] || SETUP+=("artifacts/runs is a link; it must be a writable copy (the run folders are replaced)")
for x in A B; do
  [ -f "artifacts/submissions/check_$x.csv" ] || SETUP+=("artifacts/submissions/check_$x.csv is missing: run bash reproduce.sh $PKG first")
done
[ -e data/derived ] || SETUP+=("data/derived is missing: run bash reproduce.sh $PKG first")
[ -f "$OUT/.k12_mode" ] && [ "$(cat "$OUT/.k12_mode")" != "$MODE" ] \
  && SETUP+=("$OUT holds a run of mode $(cat "$OUT/.k12_mode"); choose another --out")
if [ "$MODE" = rescore ]; then
  STEPS=("${GPU_STEPS_RESCORE[@]}")
else
  STEPS=("${GPU_STEPS_FULL[@]}" "${GPU_STEPS_FULL_AFTER_NOTES[@]}")
fi
CHECK=(python -I tools/gpu_inference_inputs.py --package "$PKG")
[ -n "${K12_MODELS:-}" ] && CHECK+=(--models "$K12_MODELS")
[ "$DRY" = 1 ] || CHECK+=(--strict-runtime)
INPUTS_OK=1
"${CHECK[@]}" "${STEPS[@]}" || INPUTS_OK=0
for p in ${SETUP[@]+"${SETUP[@]}"}; do echo "    setup: $p"; done
if [ "$DRY" = 1 ]; then
  [ -n "${K12_MODELS:-}" ] || echo "    note: K12_MODELS is not set (needed for a real run)"
  echo "   The commands below run from $REPO with this environment:"
  show cd "$REPO"
  show export "K12_MODELS=${K12_MODELS:-/path/to/base/models}" "K12_DATA=$K12_DATA" "K12_ADAPTERS=$K12_ADAPTERS" "PYTHONPATH=$PYTHONPATH" \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 VLLM_NO_USAGE_STATS=1 \
    DO_NOT_TRACK=1 VLLM_DO_NOT_TRACK=1 VLLM_HOST_IP=127.0.0.1 GLOO_SOCKET_IFNAME=lo NCCL_SOCKET_IFNAME=lo \
    OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 PYTHONUTF8=1
else
  [ "$INPUTS_OK" = 1 ] || die "inputs are missing (see the list above)"
  [ ${#SETUP[@]} -eq 0 ] || die "the repository is not set up (see the list above)"
  [ -n "${K12_MODELS:-}" ] && [ -d "$K12_MODELS" ] || die "set K12_MODELS to the folder of the base models"
  ngpu=$(python -I -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 0)
  [ "$ngpu" -gt 0 ] || die "no GPU visible to torch"
  echo "   $ngpu GPUs visible; base models in $K12_MODELS ($(df -Ph "$K12_MODELS" | awk 'NR==2 {print $4}') free there)"
  mkdir -p "$OUT"
  echo "$MODE" > "$OUT/.k12_mode"
fi

echo "== 2. data/derived as a writable copy"
if [ -L data/derived ]; then
  SRC_DERIVED="$(readlink -f data/derived)"
  cmd rm -rf data/derived.partial
  cmd cp -r "$SRC_DERIVED" data/derived.partial
  cmd chmod -R u+w data/derived.partial
  cmd rm data/derived
  cmd mv data/derived.partial data/derived
elif [ -d data/derived ]; then
  [ "$DRY" = 1 ] || [ -w data/derived ] || die "data/derived/ is not writable"
  echo "   data/derived/ is a folder; used as it is"
fi

echo "== 3. Delivered M3 adapter$([ "$MODE" = rescore ] && echo ' and frozen texts')"
M3_DST="$OUT/m3_fullte_rp/m3_train_full"
if [ "$DRY" = 0 ] && [ -d "$M3_DST" ]; then
  cmp -s "$M3_DST/adapter/adapter_model.safetensors" "$PKG/weights/m3_full/adapter/adapter_model.safetensors" \
    || die "$M3_DST holds another adapter than the package; remove it or choose another --out"
  echo "   $M3_DST holds the delivered adapter"
else
  cmd mkdir -p "$OUT/m3_fullte_rp"
  cmd cp -r "$PKG/weights/m3_full" "$M3_DST"
  cmd chmod -R u+w "$M3_DST"
fi
if [ "$MODE" = rescore ]; then
  for d in "${FROZEN_TEXTS[@]}"; do
    replace_dir "$PKG/artifacts/runs/${d%%/*}/${d%%/*}/${d#*/}" "$OUT/${d%%/*}"
  done
fi

echo "== 4. GPU steps"
if [ "$MODE" = rescore ]; then
  for c in "${GPU_STEPS_RESCORE[@]}"; do run "$c"; done
  put "${PUT_RESCORE[@]}"
else
  for c in "${GPU_STEPS_FULL[@]}"; do run "$c"; done
  put "${PUT_FULL[@]}"
  # Notes behind which M3 ranks: the new protocol answers of the 27B and the new 397B answers (CPU). The token counts
  # in the meta files are statistics only; --tokenizer '' leaves them out, so no tokenizer is needed here.
  cmd python tools/export_proto_notes.py --name test_pr27 --from pr27te_rp/proto_te/proto --allow-test
  if [ "$DRY" = 1 ]; then
    echo "+ python -I tools/make_m3px27_notes.py --table test --tokenizer '' --px27-sha \"\$(sha256sum data/derived/proto_notes_test/test_pr27.jsonl | cut -d' ' -f1)\""
  else
    cmd python -I tools/make_m3px27_notes.py --table test --tokenizer '' \
      --px27-sha "$(sha256sum data/derived/proto_notes_test/test_pr27.jsonl | cut -d' ' -f1)"
  fi
  cmd python -I tools/make_q397m3_notes.py --test-from runs/q397te_rp/proto_te/proto --tokenizer ''
  for c in "${GPU_STEPS_FULL_AFTER_NOTES[@]}"; do run "$c"; done
  put "${PUT_FULL_AFTER_NOTES[@]}"
fi

echo "== 5. Frozen M3 texts and pinned checksums"
for entry in "${FREEZE[@]}"; do
  read -r gen rank out <<< "$entry"
  cmd python -I tools/freeze_m3_texts.py --gen "$gen" --rank "$rank" --out "$out"
done
if [ "$MODE" = full ]; then
  cmd python -I tools/repin_test_side.py --write
else
  cmd python -I tools/repin_test_side.py --check
fi

echo "== 6. Submissions from the new outputs"
cmd mkdir -p artifacts/eda/private/check artifacts/submissions
cmd python -I tools/make_submission.py "${B_ARGS[@]}" --out artifacts/submissions/new_B.csv --suffix _new_B \
  --stage2-report artifacts/eda/private/check/guards_new_B.json
cmd python -I tools/make_submission.py "${A_ARGS[@]}" --out artifacts/submissions/new_A.csv --suffix _new_A \
  --stage2-report artifacts/eda/private/check/guards_new_A.json
COMPARE=(python -I - artifacts/submissions/new_A.csv artifacts/submissions/check_A.csv
         artifacts/submissions/new_B.csv artifacts/submissions/check_B.csv)
echo "+ ${COMPARE[*]}   # the short program at the end of tools/gpu_inference.sh; counts and shares only"
if [ "$DRY" = 1 ]; then
  echo "dry run finished; inputs $([ "$INPUTS_OK" = 1 ] && echo complete || echo INCOMPLETE)"
  [ "$INPUTS_OK" = 1 ] || exit 1
  exit 0
fi
"${COMPARE[@]}" <<'PY'
import hashlib
import sys

import pandas as pd

PINS = {"A": "4505c15950167650a66cdc60eb9c153eebfb9ed6192277bd283045f3797354e7",
        "B": "76da786dfdbe6a6ddcbcc93070f385e18870cb5cdaea895d92104aa3565f9973"}
for x, (new_f, ref_f) in zip("AB", (sys.argv[1:3], sys.argv[3:5])):
    new, ref = (pd.read_csv(p, dtype=str, keep_default_na=False).set_index("trace_id")["root_code"].str.split() for p in (new_f, ref_f))
    ref = ref.reindex(new.index)
    sha = hashlib.sha256(open(new_f, "rb").read()).hexdigest()
    print(f"{x}: rows {len(new)}, same top-1 code {float((new.str[0] == ref.str[0]).mean()):.4f}, "
          f"same three codes {float((new.map(frozenset) == ref.map(frozenset)).mean()):.4f}, "
          f"same line {float((new.str.join(' ') == ref.str.join(' ')).mean()):.4f}; SHA-256 {sha[:16]}…"
          f"{' = the submitted file' if sha == PINS[x] else ''}")
PY
echo "done: artifacts/submissions/new_A.csv and new_B.csv"
