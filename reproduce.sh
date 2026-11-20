#!/usr/bin/env bash
# Rebuild both final submissions on a CPU from the private package and compare them with the submitted files.
#
#   bash reproduce.sh <package-dir>                   quick check, about 2 minutes
#   bash reproduce.sh <package-dir> --full            quick check, then the full build chain, about 2 hours more
#   bash reproduce.sh <package-dir> --skip-manifest   leave out the check of the package against MANIFEST.sha256
#
# <package-dir> is the unpacked private package, the folder with MANIFEST.sha256. The competition files must be in
# data/ of this repository (links are fine; no photos are needed):
#   data/train.csv, data/test.csv                 from the root of the competition download
#   data/competition_data/taxonomy_v1.csv
#   data/competition_data/sample_submission.csv
#
# Steps:
#   1. Check the Python version and the main package versions against env/requirements-cpu.txt. A different version
#      gives a warning; a missing package stops the script.
#   2. Check the four competition files by SHA-256 and the package with "sha256sum -c MANIFEST.sha256".
#   3. Copy <package-dir>/artifacts to artifacts/ and make the copy writable; the steps below write next to these files.
#      An existing artifacts/ is used if every package file in it is unchanged.
#   4. Link data/derived to <package-dir>/data/derived. This folder is only read. An existing link is replaced; an
#      existing folder is used if every package file in it is unchanged.
#   5. Check the fold file artifacts/folds.npy against configs/folds.parquet.
#   6. Quick check: two calls of tools/make_submission.py write artifacts/submissions/check_B.csv and check_A.csv from
#      the frozen model outputs and the delivered blend and second-stage files. They refit nothing, so the result
#      does not depend on the CPU type.
#   7. With --full: five build scripts refit or verify every blend and second-stage file, compare each one bit for bit
#      with the delivered file and write A and B again. This needs an x86-64 CPU with AVX2 and FMA; the script sets the
#      OpenBLAS kernel of our reference CPU (OPENBLAS_CORETYPE=Haswell).
#
# The script never changes a file of the package. It uses two BLAS threads, as in all our runs. Logs are written to
# artifacts/eda/private/check/ and artifacts/eda/private/reproduce/. Exit code: 0 if every checksum matches, 2 wrong
# arguments, 3 an input is missing or wrong, 4 a check failed.
set -euo pipefail

A_SHA=4505c15950167650a66cdc60eb9c153eebfb9ed6192277bd283045f3797354e7  # submission A: Kaggle 58353348, 2026-11-14 19:22 UTC
B_SHA=76da786dfdbe6a6ddcbcc93070f385e18870cb5cdaea895d92104aa3565f9973  # submission B: Kaggle 58139357, 2026-11-08 19:38 UTC
# The competition files we used (docs/PROVENANCE.md, section 2).
DATA_FILES=(
  "data/train.csv 38583e0fc020638fca6be5456df4a140d573c536eb07c6e1cacb02ce05e9175f"
  "data/test.csv 703ec890aaa83f1e749e300ff40c376415cfe18a0b3bdb4ce14c81ecbb37e0ea"
  "data/competition_data/taxonomy_v1.csv fd1b612149a1fae44bb93e14ad06738cc085ef88e8db53f2fc7550449f005523"
  "data/competition_data/sample_submission.csv ee7dd719d927427e375d6e2146bc7f970daf673fef9c991dc19b47b2b43de2aa"
)

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
cd "$REPO"

usage() { sed -n '4,6p' "$0" | sed 's/^# \{0,1\}//'; }
fail_input() { echo "ERROR: $*" >&2; exit 3; }
fail_check() { echo "ERROR: $*" >&2; exit 4; }

PKG_ARG="" FULL=0 SKIP_MANIFEST=0
for arg in "$@"; do
  case "$arg" in
    --full) FULL=1 ;;
    --skip-manifest) SKIP_MANIFEST=1 ;;
    -h|--help) usage; exit 0 ;;
    -*) echo "unknown option: $arg" >&2; usage >&2; exit 2 ;;
    *) [ -z "$PKG_ARG" ] || { echo "more than one package folder given" >&2; usage >&2; exit 2; }; PKG_ARG="$arg" ;;
  esac
done
[ -n "$PKG_ARG" ] || { usage >&2; exit 2; }
[ -d "$PKG_ARG" ] || fail_input "package folder not found: $PKG_ARG"
PKG="$(cd "$PKG_ARG" && pwd -P)"
[ -f "$PKG/MANIFEST.sha256" ] && [ -d "$PKG/artifacts" ] && [ -d "$PKG/data/derived" ] \
  || fail_input "$PKG is not the private package (MANIFEST.sha256, artifacts/ and data/derived/ expected)"
command -v sha256sum >/dev/null || fail_input "sha256sum not found (GNU coreutils)"
command -v python >/dev/null || fail_input "python not found; activate the Python 3.13 environment of docs/VERIFICATION.md, section 2"

# Fixed settings: two BLAS threads as in our runs; data/ of this repository (K12_DATA would point the tools elsewhere).
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2
unset K12_DATA ORIENT
T0=$SECONDS
elapsed() { local s=$((SECONDS - $1)); printf '%dm%02ds' $((s / 60)) $((s % 60)); }

echo "== 1. Python environment"
python -I - <<'PY' || fail_input "a package of env/requirements-cpu.txt is missing (see above)"
import importlib.metadata as md
import platform
import re
import sys

pins = {}
for line in open("env/requirements-cpu.txt", encoding="utf-8"):
    m = re.match(r"^([A-Za-z0-9_.-]+)==([^\s#;]+)", line.strip())
    if m:
        pins[m.group(1).lower()] = m.group(2)
print(f"   Python {platform.python_version()}")
if sys.version_info[:2] != (3, 13):
    print("   WARNING: the reference environment is Python 3.13 (env/requirements-cpu.txt)")
missing = []
for name in ("numpy", "scipy", "pandas", "scikit-learn", "pyarrow", "pyyaml", "pillow"):
    try:
        have = md.version(name)
    except md.PackageNotFoundError:
        missing.append(name)
        continue
    note = "" if have == pins.get(name) else f"   WARNING: env/requirements-cpu.txt pins {pins.get(name)}"
    print(f"   {name} {have}{note}")
if missing:
    print(f"   missing: {', '.join(missing)} (python -m pip install -r env/requirements-cpu.txt)")
    sys.exit(1)
PY

echo "== 2. Competition files and package"
for entry in "${DATA_FILES[@]}"; do
  f=${entry%% *} want=${entry##* }
  if [ ! -f "$f" ]; then
    echo "ERROR: $f is missing. Put the competition files into data/ of this repository (links are fine):" >&2
    echo "  data/train.csv and data/test.csv (from the root of the competition download)" >&2
    echo "  data/competition_data/taxonomy_v1.csv and data/competition_data/sample_submission.csv" >&2
    exit 3
  fi
  got=$(sha256sum "$f" | cut -d' ' -f1)
  [ "$got" = "$want" ] || fail_input "$f has SHA-256 ${got:0:16}… instead of ${want:0:16}…. Use the file from the root of the competition download; competition_data/train.csv and competition_data/test.csv carry the topic names in Indonesian and give other results."
done
echo "   data/: the four competition files match"
if [ "$SKIP_MANIFEST" = 1 ]; then
  echo "   package check skipped (--skip-manifest)"
else
  t=$SECONDS
  (cd "$PKG" && sha256sum -c --quiet --strict MANIFEST.sha256) \
    || fail_input "the package does not match its MANIFEST.sha256 (files listed above); unpack it again"
  echo "   package: all $(wc -l < "$PKG/MANIFEST.sha256") files of MANIFEST.sha256 match ($(elapsed $t))"
fi

# Compare a folder of this repository with the package entries under the same path (prints failing paths only).
same_as_package() {
  local part="$1" out
  if out=$(grep -E "^[0-9a-f]{64}  $part/" "$PKG/MANIFEST.sha256" | sha256sum -c --quiet --strict - 2>&1); then
    return 0
  fi
  echo "$out" | head -n 10 >&2
  echo "   ($(printf '%s\n' "$out" | grep -c 'FAILED') files differ or are missing)" >&2
  return 1
}

echo "== 3. artifacts/"
[ -L artifacts ] && fail_input "artifacts is a link. The steps below write into it, so it must be a writable copy: remove the link and run again."
if [ ! -e artifacts ]; then
  free_kb=$(df -Pk "$REPO" | awk 'NR==2 {print $4}')
  [ "$free_kb" -ge 3000000 ] || fail_input "less than 3 GB free on $(df -P "$REPO" | awk 'NR==2 {print $6}'); the copy needs about 2.1 GB"
  t=$SECONDS
  rm -rf artifacts.partial
  cp -r "$PKG/artifacts" artifacts.partial
  chmod -R u+w artifacts.partial
  mv artifacts.partial artifacts
  echo "   copied from the package and made writable ($(elapsed $t))"
else
  same_as_package artifacts || fail_input "artifacts/ exists, but does not match the package. This happens after tools/gpu_inference.sh replaced run folders, or if artifacts/ comes from another package. Move it away (for example: mv artifacts artifacts.old) and run again; the script then copies the package again."
  chmod -R u+w artifacts
  echo "   artifacts/ exists and every package file in it is unchanged; it is used as it is"
fi
case "$(readlink -f artifacts)/" in "$PKG"/*) fail_input "artifacts/ lies inside the package folder; the package must not be written to" ;; esac

echo "== 4. data/derived"
if [ -L data/derived ]; then
  if [ "$(readlink -f data/derived)" != "$(readlink -f "$PKG/data/derived")" ]; then
    rm data/derived
    ln -s "$PKG/data/derived" data/derived
    echo "   link replaced: data/derived -> $PKG/data/derived"
  else
    echo "   data/derived -> $PKG/data/derived"
  fi
elif [ -d data/derived ]; then
  same_as_package data/derived || fail_input "data/derived/ is a folder that does not match the package. This happens after tools/gpu_inference.sh wrote new notes into it. Move it away (for example: mv data/derived data/derived.old) and run again; the script then links the package."
  echo "   data/derived/ is a folder; every package file in it is unchanged; it is used as it is"
else
  ln -s "$PKG/data/derived" data/derived
  echo "   linked: data/derived -> $PKG/data/derived"
fi

# Run a command once more if it ended by a signal (exit code above 128), for example after a crash at interpreter exit
# on a busy machine. The steps are deterministic; a second run decides.
again_on_signal() {
  local rc=0
  "$@" || rc=$?
  if [ "$rc" -gt 128 ]; then
    echo "   NOTE: the command ended by signal $((rc - 128)); running it once more: ${*:1:3}" >&2
    rc=0
    "$@" || rc=$?
  fi
  return "$rc"
}

echo "== 5. Folds"
rc=0
again_on_signal python -I tools/make_folds.py --check || rc=$?
case $rc in
  0) ;;
  3) python -I tools/make_folds.py ;;
  4) fail_check "artifacts/folds.npy does not match configs/folds.parquet" ;;
  *) fail_check "tools/make_folds.py --check ended with exit code $rc" ;;
esac

# The arguments of the two calls that write the final submissions (the same in tools/gpu_inference.sh; a test checks it).
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

STATUS=0
verify() {  # verify <A|B> <file> <expected SHA-256>
  local got
  got=$(sha256sum "$2" | cut -d' ' -f1)
  if [ "$got" = "$3" ]; then
    echo "OK $1 $got"
  else
    echo "FAIL $1 $got (expected $3, file $2)"
    STATUS=4
  fi
}
write_file() {  # write_file <A|B> <what> <make_submission arguments...>: one call, output to a log file
  local x="$1" what="$2" log="artifacts/eda/private/check/make_submission_check_$1.log" t=$SECONDS
  shift 2
  echo "   writing $x ($what) ..."
  if ! again_on_signal python -I tools/make_submission.py "$@" --out "artifacts/submissions/check_$x.csv" --suffix "_check_$x" \
       --stage2-report "artifacts/eda/private/check/guards_$x.json" > "$log" 2>&1; then
    tail -n 25 "$log" >&2
    fail_check "the call for $x stopped (full output: $log)"
  fi
  grep -m1 'NOTE: the command ended by signal' "$log" >&2 || true
  echo "   $(grep '^written:' "$log" | tail -n 1 | sed -E 's/, check .*//') ($(elapsed $t))"
}

echo "== 6. Quick check: the two calls that write the final submissions"
mkdir -p artifacts/eda/private/check artifacts/submissions
write_file B "submission of 2026-11-08" "${B_ARGS[@]}"
write_file A "submission of 2026-11-14" "${A_ARGS[@]}"
verify A artifacts/submissions/check_A.csv "$A_SHA"
verify B artifacts/submissions/check_B.csv "$B_SHA"

if [ "$FULL" = 1 ] && [ "$STATUS" != 0 ]; then
  echo "The build chain is not started, because the quick check failed."
elif [ "$FULL" = 1 ]; then
  echo "== 7. Full proof: the build chain"
  if [ -r /proc/cpuinfo ]; then
    grep -qw avx2 /proc/cpuinfo && grep -qw fma /proc/cpuinfo \
      || fail_input "the build chain needs an x86-64 CPU with AVX2 and FMA; the quick check above does not depend on the CPU"
    if grep -qw avx512f /proc/cpuinfo; then
      echo "   NOTE: this CPU has AVX-512. The chain was verified on a CPU without it; numpy's own vector code may differ."
      echo "   If a refit check stops, run again with NPY_DISABLE_CPU_FEATURES=\"X86_V4 AVX512_ICL AVX512_SPR\" (not tested)."
    fi
  fi
  export OPENBLAS_CORETYPE="${OPENBLAS_CORETYPE:-Haswell}"
  echo "   OPENBLAS_CORETYPE=$OPENBLAS_CORETYPE; logs in artifacts/eda/private/reproduce/; each script also writes its"
  echo "   first step, the rebuild of the predecessor chain, to artifacts/eda/private/<script>/regression_*.log"
  LOGDIR=artifacts/eda/private/reproduce
  mkdir -p "$LOGDIR"
  build() {  # build <n> <log name> <command...>
    local n="$1" name="$2" t=$SECONDS
    shift 2
    echo "-- $n/5 $*"
    "$@" 2>&1 | tee "$LOGDIR/${n}_$name.log" || fail_check "step $n/5 stopped: $* (log: $LOGDIR/${n}_$name.log)"
    echo "-- $n/5 done ($(elapsed $t))"
  }
  build 1 s029_m3x3 env ORIENT=legacy bash tools/builds/s029_m3x3.sh safe          # historical files, used as references
  build 2 s034_stage2_main env ORIENT=legacy bash tools/builds/s034_stage2_main.sh best  #   by the builds below
  build 3 safety_ndr bash tools/builds/safety_ndr.sh    # safety-line files; the primary chain checks them by checksum
  build 4 primary_m3px27 bash tools/builds/primary_m3px27.sh
  build 5 primary_m3x5h bash tools/builds/primary_m3x5h.sh   # writes B (primary_m3x4) and the intermediate form, then A
  verify A artifacts/submissions/sub_primary_m3x5h.csv "$A_SHA"
  verify B artifacts/submissions/sub_primary_m3x4.csv "$B_SHA"
fi

if [ "$STATUS" = 0 ]; then
  echo "All checks passed ($(elapsed $T0))."
else
  echo "A checksum differs (see FAIL above; $(elapsed $T0))."
fi
exit "$STATUS"
