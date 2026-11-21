# Verification guide

*Version 0.6, 2026-11-21.*

This page answers the host's checklist for the solution package. Section 1 answers the last item of the list: which
submission is which. Sections 2 to 8 follow the README items of the list in the host's order. Sections 9 and 10 cover
the private package and the licence. The team details and the download link of the package are in our email. The
write-up is [WRITEUP.md](WRITEUP.md), and [GLOSSARY.md](GLOSSARY.md) explains our internal names.

The code that we deliver is the tag `v0.6` of this repository; every command on this page refers to it. All commands run
from the repository root unless a block says otherwise. No step on this page calls an external API.

## 1. Which submission is which

Both selected submissions come from our primary line, the design that maximises cross-validated MAP@3, and the code
reproduces both. The build of A runs the build of B as its first step. On the way it also builds an intermediate form,
which we submitted on 2026-11-13 but did not select.

| | Submission A | Submission B |
|---|---|---|
| Submitted on Kaggle (UTC) | 2026-11-14 19:22 | 2026-11-08 19:38 |
| Kaggle submission reference | 58353348 | 58139357 |
| Public leaderboard | 0.84266 | 0.84450 |
| Content | B plus nine reranker columns and a fifth M3 draw. Eight columns come from the frozen protocol answers of the 27B and the 397B, one from the taxonomy sections seen per topic in the training labels. The fifth draw reads the photo at 2,048 px, on the 2,459 test rows with an estimated character height of at most 20 px | Six channels, with M3 as the mean of four draws. Then the alignment over the test set and the reranker, which keeps `NONE` first where the blend puts it first |
| Build script | `tools/builds/primary_m3x5h.sh` | `tools/builds/primary_m3x4.sh` |
| Output file | `artifacts/submissions/sub_primary_m3x5h.csv` (115,293 bytes) | `artifacts/submissions/sub_primary_m3x4.csv` (115,259 bytes) |
| SHA-256 | `4505c15950167650a66cdc60eb9c153eebfb9ed6192277bd283045f3797354e7` | `76da786dfdbe6a6ddcbcc93070f385e18870cb5cdaea895d92104aa3565f9973` |
| Cross-validation (per fold / as one set) | 0.8426 / 0.8404 | 0.8436 / 0.8380 |

Both files were uploaded to Kaggle as `submission.csv`. The method of both is in
[METHOD.md §4.8](METHOD.md#48-m3-with-four-and-five-draws-final-submissions) and
[§9.12](METHOD.md#912-extra-columns-from-frozen-protocol-answers-intermediate-form-and-submission-a).

## 2. Environment

We ran every check on Linux (Ubuntu 24.04, x86-64). On Windows, use WSL2 with Ubuntu. macOS is untested; `reproduce.sh`
needs GNU coreutils such as `sha256sum` and `readlink -f`. The GPU modes need Linux with NVIDIA GPUs.

| Environment | Used for | Python | CUDA | Main pins |
|---|---|---|---|---|
| CPU | `assemble`, blend and second-stage fits, tests | 3.13.5 | none | numpy 2.4.2, scipy 1.17.1, pandas 2.3.3, scikit-learn 1.8.0, pyarrow 23.0.1, PyYAML 6.0.3, Pillow 11.3.0; torch 2.11.0 and transformers 5.15.1, which shared helpers import and use on the CPU |
| GPU base | Training, adapter merge, exact scoring | 3.12, Docker image `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` (Ubuntu 24.04) | 12.8 (torch 2.8.0+cu128) | transformers 5.15.1, peft 0.21.2, accelerate 1.15.0, bitsandbytes 0.50.2, flash-linear-attention 0.5.2, fla-core 0.5.2, triton 3.4.0 |
| vLLM | Text generation through a local server on 127.0.0.1 | 3.12, a separate virtual environment made with `uv` | 13.0 user-space libraries from the wheels (`cuda-toolkit` 13.0.3.0, `nvidia-cuda-runtime` 13.0.96); the driver must support CUDA 13 | vllm 0.31.0, transformers 5.17.0, torch 2.13.0, flashinfer-python 0.7.0.post1, triton 3.7.1 |
| Kaggle T4 | 9B scorer on T4 × 2 in fp16 | 3.13 (Kaggle GPU image) | 12.8 (torch 2.11.0+cu128 of the image) | accelerate 1.15.0, bitsandbytes 0.50.2, peft 0.21.2, transformers 5.15.1 |

The complete pinned package lists are [`env/requirements-cpu.txt`](../env/requirements-cpu.txt),
[`env/requirements-gpu.txt`](../env/requirements-gpu.txt), [`env/requirements-vllm.txt`](../env/requirements-vllm.txt)
and [`env/requirements-kaggle-t4.txt`](../env/requirements-kaggle-t4.txt). Each file names its Python and CUDA version
and its install command in the header. The pins carry no hashes. Sources of all pins:
[INFERENCE.md §4](INFERENCE.md#4-environments), [§10.4](INFERENCE.md#104-software-environment) and
[TRAINING.md §2](TRAINING.md#2-environment).

### CPU

```bash
python3.13 -m venv .venv && . .venv/bin/activate
python -m pip install -r env/requirements-cpu.txt
python -m pip install pytest==8.4.2          # only for the tests (§5.1)
```

pip may warn that `shellingham` 1.5.0 is yanked; this is harmless. `requirements-cpu.txt` holds the default PyPI build
of torch with its CUDA wheels, as on our machine. The environment then takes 5.4 GB. The CPU build of torch needs less,
2.3 GB in all. For it, install the same pins without the `cuda-*` and `nvidia-*` lines:

```bash
python3.13 -m venv .venv && . .venv/bin/activate
grep -vE '^(cuda-|nvidia-)' env/requirements-cpu.txt > .venv/requirements-cpu-only.txt
python -m pip install -r .venv/requirements-cpu-only.txt --extra-index-url https://download.pytorch.org/whl/cpu
python -m pip install pytest==8.4.2
```

On 2026-11-21 the quick check, the full proof and the tests passed with this build, torch 2.11.0+cpu. We used Python
3.13.5 from Miniconda, and we have not tested other Python versions. If your system has no Python 3.13, install it with
Miniconda or uv.

### GPU base

Start from the Docker image above. Remove its `torchao` package first, because its older version conflicts with the
pinned packages, then install the pins:

```bash
python -m pip uninstall -y torchao
python -m pip install -r env/requirements-gpu.txt
```

### vLLM

The vLLM environment is needed for mode `full` and for training. Create it at `/workspace/vllm_env`, the default of the
generation jobs; the field `server.python` of a generation job names another place. Put an empty marker file
`.k12_ready` into it. The runner then uses the environment as it is and installs nothing.

```bash
uv venv --python 3.12 /workspace/vllm_env
uv pip install --python /workspace/vllm_env/bin/python -r env/requirements-vllm.txt
touch /workspace/vllm_env/.k12_ready
```

### Kaggle T4

The `_t4` configurations of the 9B scorer expect two GPUs with 16 GB each, as on Kaggle T4 × 2, and run in fp16. We ran
them only in Kaggle notebooks with internet off, on the Kaggle GPU image of the day. We did not record the digest of
that image. `env/requirements-kaggle-t4.txt` lists the four packages that we installed on top of it from offline wheels.
On other machines with two GPUs they run like the other configurations, but we have not tested this; the scores may then
differ slightly (section 8).

### Runs without network access

Install both GPU environments and download the base models first. Then set these variables in every shell that runs a
model step, and cut the network:

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 \
       VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 VLLM_DO_NOT_TRACK=1 VLLM_HOST_IP=127.0.0.1 \
       GLOO_SOCKET_IFNAME=lo NCCL_SOCKET_IFNAME=lo
export PIP_NO_INDEX=1 UV_OFFLINE=1          # an accidental install attempt then fails at once
```

With these settings the runner downloads nothing, and no library reports usage. Without `VLLM_HOST_IP`, vLLM looks up
its own address with a UDP connect to a public address.

## 3. Hardware and run times

"Measured" values come from the logs of our runs. "Estimate" values are sums or extrapolations of measured times per row
or per step. No complete `rescore` or `full` run has been made in one go; the steps ran on several machines, partly in
parallel.

### Training

Details: [TRAINING.md](TRAINING.md).

| Step | Hardware | Time |
|---|---|---|
| 9B LoRA: five fold adapters and the full fit | 1 × A40 48 GB, bf16 | 0.8–1.0 s per step; full fit 4,500 steps in 59 min (measured); about 6 GPU hours in total (estimate) |
| M3 LoRA: five fold adapters and the full fit | 1 × RTX PRO 6000 Blackwell 96 GB, bf16, peak about 72 GB | 1.75–1.80 s per step: about 70 min per fold (2,400 steps) and 90 min for the full fit (3,000 steps) (measured); about 8 GPU hours in total |
| Orientation network | CPU | minutes (measured) |
| Out-of-fold outputs of the untrained 27B channels | A40 for the 4-bit scorer (3.1–3.7 s per row); RTX PRO 6000 for reasoning (generation 2.1 s, scoring 1.6–1.7 s per row) and protocol answers (3.3–3.5 s per row) | about 1.5 h per channel and draw on 1,500 rows (estimate) |
| Out-of-fold M3 draws and M3 rankings conditioned on notes | RTX PRO 6000 | about 3–4.5 h per draw on 1,500 rows, about 4–6 h for draw h at 2,048 px (estimate) |
| Blend weights, second stages | CPU, 2 BLAS threads | seconds to minutes per fit; the meta-splits of one rule about 8 min (measured) |

### Test inference

All 3,498 test rows. Details: [INFERENCE.md §7](INFERENCE.md#7-run-times-cost-and-package-size) and
[§10.8](INFERENCE.md#108-run-time-and-cost-of-the-reference-run).

| Step | Hardware | Time |
|---|---|---|
| 9B scorer, plain and rotation-corrected | Kaggle T4 × 2, fp16 | about 6 s per row (early measurement) |
| Untrained 27B scorer | 1 × A40 48 GB, 4-bit | 3.1–3.7 s per row, about 3.4 h |
| Reasoning draws g and n, on the gated rows | 1 × RTX PRO 6000 96 GB | about 2.5 h each |
| Protocol answers of the 27B | 1 × RTX PRO 6000 | 3.3–3.5 s per row |
| M3 draws d1–d4: merge, generation, ranking | 1 × RTX PRO 6000 | about 7–8 h each (d4: 7.4 h, measured) |
| M3 draw h on the 2,459 listed rows at 2,048 px (A only) | 1 × RTX PRO 6000 | two halves of about 4 h (measured) |
| M3 rankings conditioned on the protocol notes and on the 397B notes | 1 × RTX PRO 6000 | 4.9 s and about 3.4 s per row |
| 397B notes | 4 × H200 141 GB, tensor parallel 4, FP8 weights | download 17 min, server start about 7.5 min, 3,498 rows in about 87 min; about 2.05 h in total (measured) |
| `assemble`, quick check (§5.1) | CPU, 2 BLAS threads | 1–2 min in total, of which the two calls take about 30 s each (measured) |
| `assemble`, full proof (§5.1) | CPU, 2 BLAS threads | 1 h 32 min to 2 h 23 min in six runs (measured) |

### Totals

These are estimates. In mode `full` the test-set inference of A needs about 55 h on one 96 GB card and 3.4 h on an
A40. It also needs about 6 h on Kaggle T4 × 2 and about 2 h on 4 × H200. The steps can run on several cards in
parallel. At the prices we paid for rented GPUs this is about 200 USD. B needs the same without draw h, which is about
8 h less. Mode `rescore` needs roughly 25–30 h on one 96 GB card plus the A40 and T4 time above. The GPU script runs no
397B step in this mode; the 397B parity probe `q397p_rp` on 4 × H200 is optional.

### RAM and disk

The CPU machine has 4 cores (8 threads) and 15 GB RAM; it is the reference CPU of §5.1, and other jobs shared it during
the timed runs. The CPU check needs about 14 GB of free disk. Of these, the environment takes 5.4 GB, or 2.3 GB with
the CPU build of torch. The zip takes 2.8 GB, the unpacked package 3.63 GB and the copy of `artifacts/` 2.2 GB. The full
proof used at most 1.1 GB of RAM.

The 4 × H200 machine reported 304 GiB of host memory. We did not record the system RAM of the rented single-GPU machines
and of the Kaggle notebooks. The 27B and M3 steps in bf16 ran on one 96 GB card and the 4-bit scorer on a 48 GB A40.
The 9B scorer ran on two 16 GB T4 cards and the 397B on four 141 GB H200 cards. As a check on a second hardware class,
three rankings of our second design ran in bf16 on one A100 80 GB, among them two M3 rankings.

Disk: 200 GB for the 27B and M3 runs and 600 GB for the 397B run, whose weights alone take 406.2 GB. Each M3
configuration merges the adapter into its own model folder under `$K12_MODELS` and removes that folder after its last
job, so only one merged model exists at a time. A merge writes the changed weight files and links the others; we
estimate about 18.5 GB of new files. A configuration for the 397B on 8 × H100 exists but
has never run ([INFERENCE.md §10.12](INFERENCE.md#1012-open-points)).

## 4. Where to place the data

```
data/                          competition download; the model runs read the folder named by K12_DATA
├── train.csv  test.csv  unlabelled.csv       root CSV files, with English topic names
├── competition_data/
│   ├── taxonomy_v1.csv  sample_submission.csv
│   └── images/train/  images/test/  images/questions/    GPU steps and row-list checks; images/unlabelled/ only for the
│                                                         orientation network and the image statistics
├── derived/                   from the private package: frozen derived inputs
└── models/Qwen3.5-0.8B/       optional: tokenizer for the training-side builders; three tests need this model
artifacts/                     from the private package: frozen channel outputs (runs/), OOF matrices, blend and
                               second-stage files, orientation network, reference submission files
$K12_MODELS/Qwen3.5-9B/  Qwen3.8-27B/  Qwen3.5-397B-A17B-FP8/      base models at the pinned revisions (GPU steps)
```

The CPU check of §5.1 needs only `train.csv`, `test.csv`, `taxonomy_v1.csv` and `sample_submission.csv`. The scripts
read the root CSV files; the copies under `competition_data/` carry Indonesian topic names and are not read. The model
runner checks the row order of `test.csv` against a pinned digest.

The folder `data/models/Qwen3.5-0.8B/` is optional. The builders of the M3 training targets and of the 397B notes use
the tokenizer of this model. Two tests need its tokenizer, and one more test also loads its weights. To get it, run in
the CPU environment:

```bash
python -c "from huggingface_hub import snapshot_download as d; d('Qwen/Qwen3.5-0.8B', revision='2fc06364715b967f1860aea9cf38778875588b17', local_dir='data/models/Qwen3.5-0.8B')"
```

`reproduce.sh` copies the package's `artifacts/` and makes the copy writable, because the build scripts write their
outputs and reports next to the frozen files. It links `data/derived/` to the package, because the CPU check only reads
it. The GPU script of §5.2 replaces this link by a writable copy, because it writes new notes there. For the training
steps of section 6, use a writable copy as well. Full tree of the private part:
[INFERENCE.md §3](INFERENCE.md#3-data-layout). File checksums of the base models:
[weights_manifest.json](weights_manifest.json).

## 5. (a) Inference from the trained weights to the submission file

| Mode | What is recomputed | Hardware | Expected result |
|---|---|---|---|
| `assemble` | Blend, alignment, reranker, `REL-01` rule and the file, from the frozen channel outputs | CPU | Byte-identical to the submitted files |
| `rescore` | Every scoring and ranking step on the test photos, conditioned on the frozen texts; no text is generated | GPUs of section 3, without the vLLM environment | Expected: identical on the reference hardware and number format, statistically equivalent on other hardware (section 8). A and B are written from the new outputs. The script ran without network access on 8 test rows each, not yet on all rows (§5.2) |
| `full` | Everything, including new reasoning texts, protocol answers, M3 generations and 397B notes | GPUs of section 3 | A new draw of every text: within about ±0.003 MAP@3 of the submitted private score (1 sd), with no expected systematic shift; about 91–93 % of rows keep the top-1 code (section 8.1). A and B are written from the new outputs. The script ran without network access on 8 test rows each, not yet on all rows (§5.2) |

### 5.1 `assemble` (CPU)

Use the CPU environment of section 2 and the competition data of section 4.

#### Quick check, about 2 minutes

```bash
bash reproduce.sh <package-dir>
```

The script does the following, in this order:
1. It checks the Python version and the versions of numpy, scipy, pandas, scikit-learn, pyarrow, PyYAML and Pillow. A
   different version gives a warning; a missing package stops the script.
2. It checks the four competition files by their SHA-256 and every file of the package against `MANIFEST.sha256`.
   `--skip-manifest` skips the package check.
3. It copies the package's `artifacts/` to `artifacts/` and makes the copy writable. An existing `artifacts/` folder is
   used only if every package file in it is unchanged.
4. It links `data/derived` to the package's `data/derived`, which the CPU check only reads. An existing `data/derived/`
   folder is used only if every package file in it is unchanged. The script never changes a file of the package.
5. It checks the fold array `artifacts/folds.npy` against `configs/folds.parquet`.
6. It runs the last step of the build scripts of B and A: two calls of `tools/make_submission.py`. They read the frozen
   channel outputs and the delivered blend and second-stage files, run every guard on the test-set aggregates, and write
   `artifacts/submissions/check_B.csv` and `check_A.csv`.
7. It compares both files with the checksums of section 1.

The output ends like this; only the time differs:

```
OK A 4505c15950167650a66cdc60eb9c153eebfb9ed6192277bd283045f3797354e7
OK B 76da786dfdbe6a6ddcbcc93070f385e18870cb5cdaea895d92104aa3565f9973
All checks passed (2m01s).
```

A deviation prints a line that starts with `FAIL`. The exit code is 0 if every checksum matches, 2 for wrong arguments,
3 if an input is missing or wrong and 4 if a check failed. The logs of the two calls are in
`artifacts/eda/private/check/`. The two calls refit nothing. With another OpenBLAS kernel forced they wrote the same
bytes. Their full argument lists are in `reproduce.sh` and in [INFERENCE.md §5](INFERENCE.md#5-mode-assemble).

#### Full proof, about 1.5 to 2.5 hours

```bash
bash reproduce.sh <package-dir> --full
```

After the quick check, the script runs the build chain itself. It needs an x86-64 CPU with AVX2 and FMA, and it writes
one log per build script to `artifacts/eda/private/reproduce/`. You do not type the commands below; they show what the
script runs, so that you can rerun a single step. The script sets two BLAS threads, removes `K12_DATA` and `ORIENT` from
the environment and forces the OpenBLAS kernel of the reference CPU, unless you set `OPENBLAS_CORETYPE` yourself. To
rerun one step by hand, activate the CPU environment and set the same variables:

```bash
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 OPENBLAS_CORETYPE=Haswell
unset K12_DATA ORIENT
ORIENT=legacy bash tools/builds/s029_m3x3.sh safe          # earlier files that the later builds compare against
ORIENT=legacy bash tools/builds/s034_stage2_main.sh best
bash tools/builds/safety_ndr.sh                            # safety-line files; the primary chain checks their checksums
bash tools/builds/primary_m3px27.sh                        # the predecessor of B
bash tools/builds/primary_m3x5h.sh                         # builds B, the intermediate form and A
```

The first two calls rebuild earlier files with the rotation list of the first orientation network (`ORIENT=legacy`).
`s029` and `s034` are numbers of our earlier Kaggle submissions ([GLOSSARY.md](GLOSSARY.md)). The third call builds the
files of the safety line, our second design, which we did not select. The primary chain checks these files only by their
checksums. The last two calls build the primary chain; each build script there first runs the build of its
predecessor.

Each build script checks its pinned inputs and refits or verifies its blend and second-stage files. It compares them bit
for bit with the delivered files, runs its copy null points and stops at the first deviation. A copy null point is a
control: a new component with weight 0 must rebuild the previous file byte for byte. An existing output file is replaced
only by a byte-identical rebuild. At the end, `reproduce.sh` compares the files `sub_primary_m3x5h.csv` and
`sub_primary_m3x4.csv` with the checksums of A and B in section 1 and prints the two `OK` lines again.

The chain prints about 1,850 lines of diagnostics. About 18 of them contain the word `FAILED` inside the line. They come
from controls that must fail, such as the positive controls of the guards and a swapped note file, and from earlier
files that use their fallback form by design. A deviation is a line that starts with `FAIL`, an exit code other than 0,
or a missing `All checks passed` line.

#### Reference CPU

We made all pinned CPU fits on an Intel Core i5-8250U, an x86-64 CPU with AVX2 and FMA but without AVX-512. On it, the
OpenBLAS libraries of the numpy 2.4.2 and scipy 1.17.1 wheels select the kernel `Haswell`.
`OPENBLAS_VERBOSE=2 python -I -c "import scipy.optimize"` prints one line for each of these two libraries; on the
reference CPU both say `Haswell`. With another kernel the refitted weights differ by about 1e-8, and the chain stops at
its first refit check; we tested this by forcing `Sandybridge`. `OPENBLAS_CORETYPE=Haswell` forces the reference kernel
on any x86-64 CPU with AVX2 and FMA. On CPUs with AVX-512, numpy's own vector code may differ as well. Setting
`NPY_DISABLE_CPU_FEATURES="X86_V4 AVX512_ICL AVX512_SPR"` before the run should make it follow the reference CPU, but we
have not tested this on such a CPU. On a CPU without AVX2, use the quick check.

If the chain stops on another CPU, the log of the step names the check that failed. A stop at a refit check means that a
refitted file is not bit-identical on this CPU. It does not concern the two submission files: the quick check refits
nothing and writes both byte for byte.

#### Evidence

We made the submitted files with our internal development code. This repository is a cleaned copy of that code, with
English names and comments. Both files were rebuilt byte for byte on a Kaggle CPU with internet disabled, with the
development code. On 2026-11-17 the code of this repository rebuilt them in two fresh clones that read only the
competition data and the private package. All 44 CSV and matrix files of the chain were byte-identical. On 2026-11-21 we
checked this version in two fresh clones with the delivered zip. In both, the quick check passed, in 1 min 21 s and
1 min 14 s, and `bash reproduce.sh <package-dir> --full` passed, in 1 h 33 min and 1 h 32 min. The second clone used the
CPU build of torch. Both runs shared the reference CPU with other jobs.

#### Tests

Run the tests after the full proof, with the path of the package in `K12_PACKAGE`:

```bash
K12_PACKAGE=<package-dir> python -m pytest tests -q
```

On 2026-11-21, after the full proof in a fresh clone, 685 tests passed and 15 skipped without `K12_PACKAGE` and without
the optional 0.8B model of section 4. Five of these 15 need these inputs. The two tests of the GPU dry run need
`K12_PACKAGE` and the competition photos under `data/competition_data/images/`. Two tests need the tokenizer of the
0.8B model, and one more also its weights. With these inputs in place the five pass too, so 690 pass and 10 skip. The
other 10 skip by design. Six of them check files of earlier development builds, which the package does not contain. One
needs a helper script that is not in this repository, one the per-image statistics table and one the first version of
the orientation network, whose weights were not stored. One runs only while a test-set output is still missing. Before
the full proof more tests skip, because they read files that the chain writes. No test should fail.

### 5.2 `rescore` and `full` (GPU)

`tools/gpu_inference.sh` runs the GPU model steps of both submissions again and then writes A and B from the new
outputs. In mode `rescore` it recomputes every scoring and ranking step, conditioned on the frozen texts of the package;
no text is generated, and no vLLM environment is needed. In mode `full` it also generates every text anew.

#### Before you start

1. Run the quick check of §5.1 once. It sets up `artifacts/` and `data/derived/` and writes `check_A.csv` and
   `check_B.csv`, the reference files for the comparison at the end.
2. Put the competition photos under `data/competition_data/images/` and `unlabelled.csv` next to the other CSV files in
   `data/` (section 4).
3. Install the GPU base environment of section 2. For `full`, also install the vLLM environment.
4. Download the base models at the pinned revisions into one folder and set `K12_MODELS` to it. The folder must be
   writable, because each M3 configuration writes its merged model there. Only one merged model exists at a time; we
   estimate about 18.5 GB of new files for it. In the GPU base environment:

   ```bash
   export K12_MODELS=/path/to/models
   python -c "from huggingface_hub import snapshot_download as d; d('Qwen/Qwen3.5-9B', revision='c202236235762e1c871ad0ccb60c8ee5ba337b9a', local_dir='$K12_MODELS/Qwen3.5-9B')"
   python -c "from huggingface_hub import snapshot_download as d; d('Qwen/Qwen3.8-27B', revision='1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0', local_dir='$K12_MODELS/Qwen3.8-27B')"
   # only for mode full, 406.2 GB:
   python -c "from huggingface_hub import snapshot_download as d; d('Qwen/Qwen3.5-397B-A17B-FP8', revision='ea5b4f81096f3901c91dea97f81324302495781d', local_dir='$K12_MODELS/Qwen3.5-397B-A17B-FP8', max_workers=32)"
   ```

   The model runner uses the same call to download a missing model, but `tools/gpu_inference.sh` switches downloads
   off, so download the models first. The dry run below checks only that each model folder has a `config.json`.
   [weights_manifest.json](weights_manifest.json) lists the size and SHA-256 of every file.
5. Cut the network. The script sets the offline variables of section 2 itself, except `PIP_NO_INDEX` and
   `UV_OFFLINE`; set these two yourself if an install attempt should fail at once. It also sets `K12_DATA` to `data/`
   and `K12_ADAPTERS` to the package's `weights/` folder.

#### Run

```bash
bash tools/gpu_inference.sh rescore <package-dir> --dry-run    # prints every command and checks every input
bash tools/gpu_inference.sh rescore <package-dir>              # or: bash tools/gpu_inference.sh full <package-dir>
```

`--dry-run` needs no GPU and no base models, so you can run it on a CPU first. `--out DIR` sets the folder for the raw
outputs of the model runs; the default is `runs/gpu_<mode>/` in the repository. A configuration that has finished is not
run again when you call the script again with the same output folder. Each configuration names the GPUs it uses (field
`gpu` of each worker). To split the work over several machines, run the printed commands by hand.

The script has run as a dry run, and its CPU steps have run on the frozen outputs. Its GPU jobs have run without
network access on 8 test rows each (Network, below); a run over all test rows has not been done. Its
configurations hold, unchanged, the jobs of the runs that produced our test-set outputs.

#### What the script runs

The model runner runs every job of a configuration file, and our reference configurations also hold training jobs. The
script therefore runs copies that hold only the jobs for the test rows: `configs/vlm/test_side/` for `full` and
`configs/vlm/rescore/` for `rescore`; `rescore` also uses `test_side/v1rot_t4`, which generates no text. The jobs are
unchanged and keep their run and job names, so that their outputs land where the CPU tools read them. `test_side/` has
no training job and no job on training rows, except short pre-check jobs. `rescore/` has only the merge, scoring and
ranking jobs.
`tools/make_inference_configs.py` writes both folders from the reference configurations, and
`tests/test_inference_configs.py` checks that every job in them equals the job of the same name in the reference
configuration.

**Do not run `configs/vlm/m3_fullte_rp.yaml` itself for inference.** Its jobs `m3_smoke_full` and `m3_train_full`
retrain M3 for about 1.5 h and overwrite the delivered adapter, from which every M3 configuration merges.

The script works in six steps:
1. It checks every input and stops before the first GPU step if one is missing.
2. It replaces the link `data/derived` by a writable copy of the package folder, because the later steps write new notes
   into it.
3. It copies the delivered M3 adapter to `<out>/m3_fullte_rp/m3_train_full/`, where every M3 configuration merges it
   from. In `rescore` it also copies the frozen texts of the reasoning and M3 draws into the output folder, so that the
   ranking jobs read them.
4. It runs the configurations listed below with the model runner. It then copies each finished job folder to
   `artifacts/runs/<run>/<run>/<job>/`, where the CPU tools read it.
5. It freezes the M3 notes of the draws d1–d4 again, as their new rankings scored them (`tools/freeze_m3_texts.py`). In
   `full` it
   re-records the checksums of the new protocol and 397B answers with `tools/repin_test_side.py --write`; in `rescore`
   it checks with `--check` that they did not change.
6. It runs the two calls of the quick check with the output names `artifacts/submissions/new_A.csv` and `new_B.csv`. It
   prints counts and shares only: per file the number of rows, the shares of rows with the same top-1 code, the same
   three codes and the same row of the file as `check_A.csv` or `check_B.csv`, and its SHA-256.

The configurations of `rescore`, in this order:
- 9B scorer, plain and rotation-corrected: `v1te_a_t4`, `v1te_b_t4`, `test_side/v1rot_t4`.
- Untrained 27B scorer: `zs27q_test0_rp`, `zs27q_test1_rp`.
- Scoring conditioned on the reasoning draws g and n: `rescore/zs27v_te_rp`, `rescore/zs27v_n3_rp`.
- M3 rankings of the draws d1–d4: `rescore/m3_fullte_rp`, `rescore/m3d2te_rp`, `rescore/m3d3te_rp`, `rescore/m3d4te_rp`.
- M3 ranking of draw h, for A only: `rescore/m3hte_a_rp`, `rescore/m3hte_b_rp`.
- M3 rankings conditioned on the frozen protocol notes and on the frozen 397B notes: `m3px27te_rp`, `q397m3te_rp`.

The configurations of `full`, in this order:
- 9B scorer and untrained 27B scorer as in `rescore`.
- Reasoning draws g and n, and the protocol answers of the 27B: `test_side/zs27v_te_rp`, `test_side/zs27v_n3_rp`,
  `pr27te_rp`.
- M3 draws d1–d4 with merge, generation and ranking: `test_side/m3_fullte_rp`, `m3d2te_rp`, `m3d3te_rp`, `m3d4te_rp`.
  Draw h for A: `m3hte_a_rp`, `m3hte_b_rp`.
- 397B parity probe and 397B notes on the test rows, on 4 × H200: `q397p_rp`, `q397te_rp`.
- On the CPU, the notes from the new answers: `tools/export_proto_notes.py`, `tools/make_m3px27_notes.py` and
  `tools/make_q397m3_notes.py`.
- M3 rankings conditioned on the new notes: `m3px27te_rp`, `q397m3te_rp`.

After a GPU run, `artifacts/` and `data/derived/` hold the new outputs and no longer match the package. To run the quick
check on the frozen outputs again, move both folders away first, for example with `mv artifacts artifacts.gpu` and
`mv data/derived data/derived.gpu`. `reproduce.sh` then sets them up from the package again.

#### Reference hardware

We expect bit-identical rescored files on the hardware of section 3. That is the RTX PRO 6000 Blackwell for the 27B and
M3 steps, the A40 for the 4-bit scorer, Kaggle T4 × 2 in fp16 for the 9B and 4 × H200 for the 397B. On other hardware
the scores differ slightly. The build scripts of §5.1 then stop by design, because they pin our test matrices. The two
calls of the quick check verify the frozen texts but pin no score file, so they accept the new scores and write A and B.
For each row whose generation stopped before "Root:", the M3 ranking jobs first complete the finding with greedy
decoding. This concerns 1,798 of the 3,498 rows for d1, 1,236 for d2, 1,266 for d3 and 1,241 for d4. On other hardware
this completion is new, and the freeze step records it. Section 8 gives the expected effect.

#### What stays pinned

The deployment files of A and B record the checksums of our protocol answers (`pr27te_rp`; for A also the 397B answers
of `q397te_rp`). `meta_m3px27xm.json` records the checksum of the protocol notes. After a `full` run,
`tools/repin_test_side.py --write` re-records exactly these fields and keeps the old files under
`artifacts/repin_backup/`. Nothing is refitted. `tools/repin_test_side.py --check` only reports.

#### Delivered weights

The runner reads the base models from `$K12_MODELS` and our adapters from the package's `weights/` folder. The 9B
configurations need the adapter `v1_full` and, for their pre-check job `check_f1`, the fold adapter `v1_folds/v1_f1`.
The M3 full-fit adapter `weights/m3_full` must lie at `<out>/m3_fullte_rp/m3_train_full/`, where every M3 configuration
merges it from; step 3 of the script puts it there. The 9B configurations expect two GPUs, as on Kaggle T4 × 2 (worker
`gpu: "0,1"` with `max_memory` per card; section 2). The keys `kernel_sources` and `dataset_sources` in the T4
configurations name our private Kaggle resources; the runner ignores them.

#### Row lists

Both modes use the frozen row lists of the package. You can check each list on its own; the commands print counts and
checksums only. The first three read the photos. The first two also need `opencv-python-headless==4.13.0.92`, which is
not in `env/requirements-cpu.txt`.
- Rotation list:
  `python -I tools/orient_net.py infer --from-images --tables test --weights artifacts/orient/orient_cnn.pt --out data/orient_check`
  recomputes it from the photos. `data/orient_check/test_c80_rows.npy` must have the SHA-256
  `aa6515c8bf17bf593c5c90aa90d9240e9992d84f650c5445a37a5f015dbbff1d`, the frozen list in `data/derived/orient_repro/`.
  Give `--weights` exactly as shown, because the path is written into the list's meta file.
- Character-height list of draw h: it needs the per-image statistics, which are not in the package. Run
  `python -I eda/i5_01_image_stats.py`; it reads all 55,117 photos, `images/unlabelled/` included, in about 20 minutes
  with three worker processes. Then run
  `K12_DATA=<folder> python -I tools/make_charsize_rows.py --max-height 20 --name h20`, where `<folder>` holds links to
  `train.csv` and `test.csv`. In `<folder>/derived/charsize/`, the training list must have the SHA-256
  `11afce9af96fca09032153b566ee81725f466ab927e782fcdad4324b55d18e77` and the test list
  `9b1a1837bc2cf399e2e221c49def4b4ae11d666b5346ecb1eddc076622c8340a`. Do not write into the package's
  `data/derived/charsize/`. Its meta file pins the checksum of the statistics table, which is not bit-reproducible, and
  `tools/builds/primary_m3x5h.sh` checks that meta file.
- Blank-photo list: `python -I tools/make_blank_list.py check` computes the lists of blank training and test photos and
  compares them with the frozen lists in `data/derived/blank/`; the test list has 6 rows. `write --out <folder>` writes
  new lists and never touches the frozen ones. A photo counts as blank if, converted to grey scale and resized to 256 ×
  256, its mean is below 12 and its standard deviation below 8.
- Gate lists of the reasoning draws: `python -I tools/replay/gates.py check --table test --line all --sources a40`. It
  reads `artifacts/oof_v1.npy` and the A40 test outputs `v1_test_rp` and `zs27t_all_rp` of the package and reports each
  of the four lists of test rows as bit-identical or not.

#### Ranking of the untrained 27B conditioned on the protocol notes

This ranking (`n1_rp`) is read only by the fallback forms and by checks of the build chain, not by the two calls. To
recompute it, run `cd src && python -u -m vlm.runner --config ../configs/vlm/test_side/n1_rp.yaml --out <out>`.

#### Network

The CPU assembly and the 9B scorer on Kaggle T4 have run without network access. On 2026-11-21 we also ran the GPU
jobs of both modes without network access, on 8 test rows each. This was 21 steps on one RTX PRO 6000 96 GB, with the
code of the tag `v0.6`, the pinned environments of section 2 and the files of the delivered zip. DNS was switched off
and every process was traced with `strace`; no step tried to open a network connection. The vLLM server started with
empty caches, the prompts had the same token counts as in our runs, and every LoRA merge gave the same weights.

- Mode `rescore`, conditioned on the frozen texts: every row kept its top-1 code. 8 of the 11 steps were bit-identical
  with our outputs, and the other three differed by at most 0.035 in a probability. A repeated step was bit-identical.
  The three steps whose original ran on an A40 or a Kaggle T4 kept the top-1 code on 23 of 24 rows.
- Mode `full`, with newly generated texts: the top-1 code stayed the same on 35 of 40 rows.

This test started the same configuration files as `tools/gpu_inference.sh`, with its own driver. Not covered: the
script itself as one run, a run over all 3,498 test rows, and the 397B path, whose inputs were only checked for
completeness ([ROADMAP.md](ROADMAP.md)). Section 2 lists the settings for a run without network access.

#### The 397B path

The 397B path, with its pass conditions and the settings for a run without network access, is described in
[INFERENCE.md §10.7](INFERENCE.md#107-order-of-the-steps).

## 6. (b) Training from scratch

This section rebuilds every trained and fitted part, but it does not end in a submission file. After a retraining every
pinned checksum changes, and the build scripts stop at the first one by design (end of this section).

Training writes into `artifacts/` and `data/derived/`, so use writable copies of both package folders, not links. The
steps below rebuild every trained and fitted part, in this order. [TRAINING.md](TRAINING.md) gives the commands, inputs
and checks of each step; its sections are given in brackets. GPU steps use the model runner as in
[TRAINING.md §2](TRAINING.md#2-environment). Below, `run <name>` stands for
`cd src && python -u -m vlm.runner --config ../configs/vlm/<name>.yaml --out <out>`.

1. Folds (§3): `python -I tools/make_folds.py --recompute`. It recomputes 5 folds stratified by root code with seed 42
   and confirms `configs/folds.parquet`.
2. 9B adapters (§4): `run ft_v1a_rp; run ft_sum1_rp; run ft_v1b_rp; run ft_v1c_rp` for folds 0–4 and the full fit, then
   the OOF assembly with `eda/a6_assemble_oof.py`.
3. Orientation network and rotated 9B reading (§5): `tools/orient_net.py maps`, `train --seed 0 --threads 2` and
   `infer`, then `run v1rot_t4`. This run needs all five 9B fold adapters `v1_f0` … `v1_f4` under
   `$K12_ADAPTERS/v1_folds/`. Its pre-check job `smoke` and its fold jobs read the frozen first-version training lists
   `data/derived/orient/train_c50.json` and `train_c50_rows.npy`; that first version cannot be recomputed, as section 8
   explains.
4. Teacher pool labels (§6): frozen files from a paid API; not rerun.
5. Untrained 27B scorers (§7): `run zs27q_oof_rp; run zs27q2_f1_rp; run zs27t_all_rp`.
6. Reasoning draws (§8): the package holds the frozen gate lists of the training rows. After you retrain steps 2 and 5,
   rebuild them with `python -I tools/replay/gates.py write --out data/derived/think_rows_new --table train --line all`
   and copy `train.npy`, `train_n3.npy`, `train_tb_q.npy` and `train_tb_t.npy` into `data/derived/think_rows/`. The
   reasoning configurations and the OOF assembly read that folder; `write` never writes into it, and `--line all` also
   reads `data/derived/orient/train_v50_rows.npy`. Then run draw g with `zs27v_all_rp`, `zs27v_f2_rp`, `zs27v_cut_rp`
   and `zs27v_te_rp`, and draw n with `zs27v_n_rp` and `zs27v_n3_rp`.
7. Protocol channel (§9): `run pr27_rp; run px27_rp`. The row lists `data/derived/proto_rows/train_sample.npy`,
   `train_rest.npy` and `train_api27.npy` split the 1,500 training rows into the jobs `proto_a` with 860 rows and
   `proto_b` with 640 rows, which the later scripts read; `train_api27` only makes 70 of them run first. Export the
   notes with `tools/export_proto_notes.py`. The M3 ranking conditioned on these notes runs with `run m3px27_oof_rp`
   after step 8.
8. M3 (§10): the six training-target files are frozen, and the recipe of each adapter records their checksums.
   `python -I tools/make_m3_targets.py --full --check` rebuilds them in memory and compares. A rebuild with `--full`
   needs the tokenizer of `Qwen/Qwen3.5-0.8B` at `data/models/Qwen3.5-0.8B`, or `--tokenizer <path>`. Then
   `run m3_pilot_rp; run m3_f034_rp; run m3_fullte_rp` for the fold adapters with draw d1 and the full fit, and the OOF
   draws `m3long_rp`, `cv01_m3d2_f034`, `m3d3_oof_rp`, `m3d4_oof_rp`, `m3ha_oof_rp` and `m3hb_oof_rp`. The combined fold
   folders `runs/m3_5f` and `runs/m3long_comb` are delivered frozen; their combination script is not included, as
   section 8 lists.
9. 397B on the training rows (§11): the answers come from a hosted API and are frozen, not rerun. Rebuild the notes with
   `python -I tools/make_q397m3_notes.py --tokenizer '' --no-upload --out derived_check/q397m3_train_notes.jsonl`. The
   path is relative to `data/`. The file must have the SHA-256
   `5a8fb7358bf9a25bcd4f6b12b28c02b31e5ad345a3ee21f9d94bc182bf180d52`, the same as
   `data/derived/sft_q397m3/q397m3_train_notes.jsonl`. Without these switches the tool needs the tokenizer and
   overwrites the frozen files. Then `run q397m3_oof_rp`.
10. Blends and second stages on the CPU (§12, §13, §16): the `eda/` commands listed there, in that order, ending with
    `eda/c135_primary_m3x4.py`, `eda/c151_primary_cols.py` and `eda/c166_primary_m3x5h.py`.

GPU training is not bit-reproducible, so a retrained adapter is a new model with a new test file
([TRAINING.md §15](TRAINING.md#15-reproducibility-of-training)). The step scripts of step 10 also check the checksums of
the files they read from earlier steps. For example, `eda/c135_primary_m3x4.py` checks `oof_m3px27xm.npy` and
`stack_primary_m3px27.json`. After a retraining these files change, and the scripts stop at that check. A retrained
pipeline therefore needs new pins, and every pre-registered decision rule has to be evaluated again on the new OOF
matrices. We have not built a path from a retrained pipeline to a submission file.

## 7. Seeds

| Component | Seed or setting |
|---|---|
| Fold split | `kaggle_core.folds.make_folds`, seed 42 (fold hash `07ec381cc58b2a80`) |
| 9B and M3 LoRA training | seed 42 |
| Orientation network | `--seed 0`; fixed sample of 10,000 pool photos with seed 20261010 |
| Reasoning draws, protocol answers, 397B answers, M3 draws d1 and d2 | greedy decoding; rows in a fixed shuffled order (`order_seed`) |
| Exact scoring and rankings (9B scorer, untrained 27B scorer, all rankings conditioned on texts) | no sampling |
| M3 draws d3, d4 and h | sampled with temperature 0.6, top-p 0.95, top-k 20; seeds 3, 4 and 5 |
| 397B parity rows | seed 20261030 |
| Blend and second stages | deterministic fits (L-BFGS-B); the meta-splits used for decisions have fixed seeds and are not needed to build |

All texts that we generated ourselves were generated with vLLM 0.31.0 and 64 concurrent requests.
Greedy decoding in vLLM is still not deterministic across runs, because the requests are batched differently. The 397B
answers on the training rows and the teacher pseudo-labels are frozen API outputs (section 8).

## 8. What cannot be rerun exactly

| Part | Why | How the solution handles it |
|---|---|---|
| Generated texts: reasoning, protocol answers, M3 draws, 397B notes | vLLM batches parallel requests differently from run to run. Two greedy runs with the same prompt agree on the top code in about 84 % of rows. Our local 397B path agrees with the hosted one on the top code in 78 % of 64 training rows | The texts are frozen with checksums, and `assemble` and `rescore` read them. A `full` rerun is a new draw (section 8.1) |
| Teacher pseudo-labels of the unlabelled pool | Paid, non-deterministic API (Claude Haiku 5.5) | Frozen and delivered. The test inference reads them as a file, as the per-question task block in two prompts, and calls no API |
| 397B answers on the training rows | Produced through the Tinker sampling API; the provider's revision and number format were not recorded | Frozen and delivered. They enter only fitted weights and the training rows of six reranker columns of A. On the test rows the 397B runs locally in its FP8 release |
| GPU training (9B, M3) | Kernels and bf16 sums are not deterministic | The trained adapters are delivered with checksums |
| First version of the orientation network | Its training run was not seeded, and its weights were not stored | Its rotation lists of the training rows are delivered frozen. The test rows use the released, seeded network, whose list `tools/orient_net.py infer` recomputes (§5.2) |
| Rescoring on other hardware | Other kernels and number formats | Statistically equivalent. The 9B on T4 in fp16 against A40 in bf16 changes the channel's top code on about 9 % of rows and the blend's on 1–2 %. M3 rankings on T4 in 4-bit agree on the top code in 0.971 (d1) and 0.978 (d2, half of the rows) |
| CPU fits on another CPU class | Another OpenBLAS kernel changes the refitted blend and second-stage weights by about 1e-8; the build scripts require them bit for bit and stop | Bit-identical on the reference CPU with two BLAS threads (§5.1). Elsewhere, force that kernel or use the quick check, which refits nothing and reproduces both files byte for byte |
| Steps whose script is not in this repository | Listed in [PROVENANCE.md §9](PROVENANCE.md#9-open-items-of-this-list): among them the combination of the fold runs of draws d1 and d2, the 397B parity check and the Bayesian step of one M3 decision rule | Their outputs are delivered frozen; no input was produced by hand. The band gate of the reasoning draws was first written by ad-hoc scripts; `tools/replay/gates.py` reproduces all gate lists bit for bit (§5.2) |

None of these parts is a step at test time that needs an API or a person. A rerun of the test inference depends only on
the text generation of the open models (first row).

### 8.1 How close a `full` rerun should come

Assembling the files from the frozen inputs is deterministic. Text generation is not: vLLM serves 64 concurrent
requests, and the order of floating-point sums depends on the batch, so greedy and seeded continuations change alike. A
`full` rerun therefore replaces every generated text by a new draw: the 27B reasoning texts and protocol answers, the
four M3 draws of B or the five of A, and the 397B notes. It is assembled with the steps of §5.2. The M3 notes are frozen
again as their new rankings scored them, `tools/repin_test_side.py --write` re-records the checksums of the new protocol
and 397B answers, and the two calls of §5.1 write A and B.

We measured the effect on the 1,500 training rows, with the blend and second-stage weights held fixed. We replaced every
text for which we hold a second draw: the protocol answers, the reasoning texts on two of the five folds and the three
M3 draws that the reranker features read. This moved the row scores by an amount that corresponds to ±0.0015–0.0017
MAP@3 (1 sd) on 2,498 rows. We extrapolated to all generated texts: the M3 mean channel from single-draw swaps, the 397B
and protocol rankings from a stand-in calibrated on a real second draw. We therefore expect a rerun to land within about
±0.003 MAP@3 (1 sd) of the submitted file's private score, for both submissions, with no expected systematic shift. The
95 % range is about ±0.006; over our assumptions the sd ranges from 0.0026 to 0.0044.

You can check a rerun without labels. About 91–93 % of the rows should keep the top-1 code of the submitted file, and
about two thirds the same three codes; `tools/gpu_inference.sh` prints both shares. A clearly lower share points to
another generation set-up or to the fallback form below. Re-scoring the frozen texts on other GPUs in bf16 adds at most
about ±0.001.

One discrete risk remains. The note guard checks 13 statistics of the M3 test notes against their range on the three
training draws. Examples are how many codes the notes mention and how often two draws wrote word-identical notes. If
one statistic is out of range, the build switches to the fallback form without note features
([METHOD.md §9.2](METHOD.md#92-test-side-guards)). The two greedy M3 draws might produce word-identical notes more
often than in our runs, for example under a more deterministic batching set-up. The guard would then switch, and the
fallback form scored about 0.004–0.007 lower in cross-validation. With the documented set-up (vLLM 0.31, 64 concurrent
requests) we put the chance at roughly 1–2 %; this is a rough estimate, not a measurement.

Seeds: section 7.

## 9. Private weights and inputs

Download: the private link in our email; it is not stored in this repository. Everything in the package is trained on or
derived from the competition data, so it is not public.

The package is one zip file with 572 indexed files and three files about the package: `README_PACKAGE.md`,
`PACKAGE_INDEX.tsv` and `MANIFEST.sha256`. Unpacked, it takes 3.63 GB. `MANIFEST.sha256` holds the checksum of every
other file. `PACKAGE_INDEX.tsv` lists each indexed file with its size, SHA-256 and role, and whether it is derived from
test rows. 161 files with 543.2 MB in total are derived from test rows. `README_PACKAGE.md` in the package says where
each folder goes; `reproduce.sh` sets this up for the CPU check.

| Role | Files | Size | What it is |
|---|---:|---:|---|
| `assemble` | 464 | 2.32 GB | Read by the CPU build: frozen channel outputs, OOF matrices, earlier blend, second-stage and reference files, stored reports and frozen derived inputs. Also the five M3 fold adapters, whose recipes and checksums the build checks |
| `assemble-generated` | 11 | 2.1 MB | Files that the build scripts write and pin themselves; they are included so that the scripts verify them instead of recomputing them |
| `rescore` | 40 | 258.5 MB | Frozen inputs of the GPU modes that the CPU build does not read: the texts that the rankings read, the teacher pseudo-labels of the unlabelled pool, row lists of the model runs, the 397B parity probe |
| `training` | 31 | 69.9 MB | Inputs of the training steps of section 6, such as the M3 training targets, the raw 397B answers on the training rows and `artifacts/oof_v1.npy`. `README_PACKAGE.md`, section 5, lists them all |
| `check` | 9 | 4.3 MB | Inputs of the checks of §5.2: the A40 test outputs from which the gate lists are computed, and the frozen blank-photo list of the training rows |
| `weights` | 17 | 969.4 MB | Our trained weights (below) |

| Weights | Content | Base model |
|---|---|---|
| `weights/m3_full/` | M3 adapter of the full fit (300.5 MB) and its training recipe `sft_recipe.json` | Qwen3.8-27B |
| `weights/v1_full/` | 9B adapter of the full fit (110.9 MB) | Qwen3.5-9B |
| `weights/v1_folds/v1_f0` … `v1_f4` | The five 9B fold adapters, used for the out-of-fold outputs; `v1_f1` also serves the pre-check job of the 9B test runs | Qwen3.5-9B |
| `weights/orient_cnn/` | Orientation network and its description (3.4 MB) | none; trained from scratch on the CPU |
| `artifacts/runs/m3_pilot_rp/m3_pilot_rp/m3_train_f1/` and `…/m3_train_f2/`; `artifacts/runs/m3_f034_rp/m3_f034_rp/m3_train_f0/`, `…/m3_train_f3/` and `…/m3_train_f4/` | The five M3 fold adapters (300.5 MB each), used for the out-of-fold outputs. The CPU build checks their recipes and checksums | Qwen3.8-27B |

The M3 full-fit adapter and the orientation network lie both under `weights/` and under `artifacts/`; these are the same
files. The base models are public and not in the package: Qwen3.5-9B, Qwen3.8-27B and Qwen3.5-397B-A17B-FP8 at the
pinned revisions, 481.1 GB together. [weights_manifest.json](weights_manifest.json) lists every file with its checksum.
Sizes against the host's limits (481.53 GB as deployed, 440.94 GB at 8 bit, 434.4 B parameters) are in
[MODEL_SIZES.md](MODEL_SIZES.md). [PROVENANCE.md](PROVENANCE.md) gives the origin of every input; its file counts
describe the inputs traced in our builds, including the competition CSV files, and are not a list of the package.

## 10. Licence and external components

Our code and documentation are licensed under the Apache License 2.0 ([LICENSE](../LICENSE)).
[THIRD_PARTY.md](../THIRD_PARTY.md) lists every pre-trained model, dataset, API and package with link, version and
licence. The API cost of one inference run on the test set is 0 USD, because the test inference calls no API. The
competition data are not part of this repository.
