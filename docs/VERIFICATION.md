# For the verification team

*Status: 2026-11-17. This page answers the host's checklist for the solution package in order and points to the
documents that hold the details. Code, configurations and documentation are in this repository; the trained weights and
all frozen intermediate files are in a private package (section 9). No step below calls an external API.*

## 1. Which submission is which

Both selected submissions come from the primary line, and the code reproduces both. The build of A runs the build of B
(and of the intermediate form submitted on 2026-11-13, which is not selected) as its first steps.

| | Primary submission A | Primary submission B |
|---|---|---|
| Submitted on Kaggle (UTC) | 2026-11-14 19:22 | 2026-11-08 19:38 |
| Kaggle submission reference | 58353348 | 58139357 |
| Public leaderboard | 0.84266 | 0.84450 |
| Content | B + nine second-stage columns from frozen protocol answers + a fifth M3 draw (photo at 2,048 px) on the 2,459 test rows with small handwriting | Six channels; M3 as the mean of four draws; alignment over the test set; reranker with NONE guard |
| Build script | `tools/builds/primary_m3x5h.sh` | `tools/builds/primary_m3x4.sh` |
| Output file | `artifacts/submissions/sub_primary_m3x5h.csv` (115,293 bytes) | `artifacts/submissions/sub_primary_m3x4.csv` (115,259 bytes) |
| SHA-256 | `4505c15950167650a66cdc60eb9c153eebfb9ed6192277bd283045f3797354e7` | `76da786dfdbe6a6ddcbcc93070f385e18870cb5cdaea895d92104aa3565f9973` |
| Cross-validation (per fold / as one set) | 0.8426 / 0.8404 | 0.8436 / 0.8380 |

Both files were uploaded to Kaggle as `submission.csv`. A needs, besides B's inputs, the run folders of the fifth M3 draw,
the two row lists of the character-height rule and the frozen column sources ([INFERENCE.md §3](INFERENCE.md#3-data-layout),
"Additional inputs of the final submissions"). Method of both: [METHOD.md §4.8, §9.12](METHOD.md#48-m3-with-four-and-five-draws-final-submissions).

## 2. Environment

| Environment | Used for | Python | CUDA | Main pins |
|---|---|---|---|---|
| CPU | `assemble`, blend and second-stage fits, tests | 3.13.5 | — | numpy 2.4.2, scipy 1.17.1, pandas 2.3.3, scikit-learn 1.8.0, pyarrow 23.0.1, PyYAML 6.0.3, Pillow 11.3.0; torch 2.11.0 and transformers 5.15.1 (imported by shared helpers, CPU only); pytest 8.4.2 |
| GPU base | Training, adapter merge, exact scoring | 3.12 (Docker image `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`, Ubuntu 24.04) | 12.8 (torch 2.8.0+cu128) | transformers 5.15.1, peft 0.21.2, accelerate 1.15.0, bitsandbytes 0.50.2, flash-linear-attention 0.5.2, fla-core 0.5.2, triton 3.4.0 |
| vLLM | Text generation (a local server bound to 127.0.0.1) | 3.12, separate virtual environment created with `uv` | 13.0 user-space libraries from the wheels (`cuda-toolkit` 13.0.3.0, `nvidia-cuda-runtime` 13.0.96); the driver must support CUDA 13 (the driver version of the rented machines was not recorded) | vllm 0.31.0, transformers 5.17.0, torch 2.13.0, flashinfer-python 0.7.0.post1, triton 3.7.1 |
| Kaggle T4 | 9B scorer on T4×2 (fp16) | 3.13 (Kaggle GPU image) | 12.8 (torch 2.11.0+cu128 of the image) | accelerate 1.15.0, bitsandbytes 0.50.2, peft 0.21.2, transformers 5.15.1 |

Requirement files with the complete pinned package lists: [`env/requirements-cpu.txt`](../env/requirements-cpu.txt), [`env/requirements-gpu.txt`](../env/requirements-gpu.txt), [`env/requirements-vllm.txt`](../env/requirements-vllm.txt) and [`env/requirements-kaggle-t4.txt`](../env/requirements-kaggle-t4.txt) (pins without hashes). All pins with their sources:
[INFERENCE.md §4](INFERENCE.md#4-environments), [§10.4](INFERENCE.md#104-software-environment) and
[TRAINING.md §2](TRAINING.md#2-environment). Runs without network access set `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`,
`VLLM_NO_USAGE_STATS=1` and `DO_NOT_TRACK=1`.

## 3. Hardware and run times

"Measured" values come from the logs of our runs; "estimate" values are sums or extrapolations of measured per-row or
per-step times. No complete `rescore` or `full` run has been made in one go: the steps ran on several machines, partly in
parallel.

**Training** (details: [TRAINING.md](TRAINING.md))

| Step | Hardware | Time |
|---|---|---|
| 9B LoRA: five fold adapters and the full fit | 1 × A40 48 GB, bf16 | 0.8–1.0 s per step, full fit 4,500 steps in 59 min (measured); about 6 GPU hours in total (estimate) |
| M3 LoRA: five fold adapters and the full fit | 1 × RTX PRO 6000 Blackwell 96 GB, bf16, peak about 72 GB | 1.75–1.80 s per step: about 70 min per fold (2,400 steps) and 90 min for the full fit (3,000 steps) (measured); about 8 GPU hours in total |
| Orientation network | CPU | minutes (measured) |
| Out-of-fold outputs of the untrained 27B channels | A40 (4-bit scorer, 3.1–3.7 s per row); RTX PRO 6000 (reasoning: generation 2.1 s, scoring 1.6–1.7 s per row; protocol answers 3.3–3.5 s per row) | about 1.5 h per channel and draw on 1,500 rows (estimate) |
| Out-of-fold M3 draws and M3 rankings behind notes | RTX PRO 6000 | about 3–4.5 h per draw on 1,500 rows (estimate; the 2,048-px draw is the slowest) |
| Blend weights, second stages | CPU, 2 BLAS threads | seconds to minutes per fit; the meta-splits of a rule about 8 min (measured) |

**Test inference** (3,498 rows; details: [INFERENCE.md §7](INFERENCE.md#7-run-times-cost-and-package-size), [§10.8](INFERENCE.md#108-run-time-and-cost-of-the-reference-run))

| Step | Hardware | Time |
|---|---|---|
| 9B scorer, plain and rotation-corrected | Kaggle T4×2, fp16 | about 6 s per row (early measurement) |
| 27B zero-shot scorer with task block | 1 × A40 48 GB, 4-bit | 3.1–3.7 s per row, about 3.4 h |
| Reasoning draws g and n (gated rows) | 1 × RTX PRO 6000 96 GB | about 2.5 h each |
| Protocol answers of the 27B | 1 × RTX PRO 6000 | 3.3–3.5 s per row |
| M3 draws d1–d4 (merge, generation, ranking) | 1 × RTX PRO 6000 | about 7–8 h each (d4: 7.4 h, measured) |
| M3 draw h, 2,459 listed rows at 2,048 px (submission A only) | 1 × RTX PRO 6000 | two halves of about 4 h (measured) |
| M3 behind the protocol notes; M3 behind the 397B notes | 1 × RTX PRO 6000 | 4.9 s and about 3.4 s per row |
| 397B notes | 4 × H200 141 GB, tensor parallel 4, FP8 weights | download 17 min, server start about 7.5 min, 3,498 rows in about 87 min; about 2.05 h in total (measured) |
| `assemble` | CPU, 2 BLAS threads | about 45 min for the chain of A on our machine; 3,987 s for both files in the netless Kaggle CPU rebuild (measured) |

**Totals (estimate).** Test side of submission A in mode `full`: about 55 h on one 96 GB card (the steps can run on
several cards in parallel), 3.4 h on an A40, about 6 h on Kaggle T4×2 and about 2 h on 4 × H200; at the RunPod prices we
paid, about 0.2 kUSD. Submission B needs the same without draw h (about 8 h less). Mode `rescore` (the scoring and ranking
parts only) needs roughly 25–30 h on one 96 GB card plus the A40 and T4 time above; 4 × H200 only for the optional 397B
parity check.

**RAM and disk.** The CPU machine has 8 cores and 15 GB RAM. The 4 × H200 machine reported 304 GiB of host memory; the
system RAM of the rented single-GPU machines and of the Kaggle notebooks was not recorded. Disk: 200 GB for the 27B and M3
runs, 600 GB for the 397B run (its weights alone take 406.2 GB). Other hardware for the 397B (for example 8 × H100) has a
configuration but has never run ([INFERENCE.md §10.12](INFERENCE.md#1012-open-points)).

## 4. Where to place the data

```
data/                         competition download (or any folder named by K12_DATA for the model runs)
├── train.csv  test.csv  unlabelled.csv      the root CSV files (English topic names)
├── competition_data/
│   ├── taxonomy_v1.csv  sample_submission.csv
│   └── images/train/  images/test/  images/questions/  (images/unlabelled/ only to retrain the orientation network)
└── derived/                  private package: frozen derived inputs
artifacts/                    private package: frozen channel outputs (runs/), OOF matrices, blend and second-stage files,
                              orientation network, reference submission files
$K12_MODELS/Qwen3.5-9B/  Qwen3.8-27B/  Qwen3.5-397B-A17B-FP8/      base models at the pinned revisions
```

The scripts read the **root** CSV files; the copies under `competition_data/` carry Indonesian topic names and are not
read. The model runner checks the row order of `test.csv` against a pinned digest. Full tree of the private part:
[INFERENCE.md §3](INFERENCE.md#3-data-layout). File checksums of the base models: [weights_manifest.json](weights_manifest.json).

## 5. (a) Inference from the trained weights to the submission file

| Depth | What is recomputed | Hardware | Expected result |
|---|---|---|---|
| `assemble` | Blend, alignment, reranker, `REL-01` rule and the file, from the frozen channel outputs | CPU | Byte-identical to the submitted files |
| `rescore` | Every scoring and ranking step on the test photos, behind the frozen texts | GPUs of section 3; 4 × H200 only for the optional 397B parity check | Identical on the reference hardware and number format; statistically equivalent on other hardware (section 8) |
| `full` | Everything, including new reasoning texts, protocol answers, M3 generations and 397B notes | GPUs of section 3 | A new draw of every text: within about ±0.003 MAP@3 of the submitted private score (1 sd; 95 % within about ±0.006), no expected systematic shift; about 91–93 % of rows keep the top-1 code (section 8.1) |

### 5.1 `assemble` (CPU)

```bash
# Python 3.13 environment of section 2; competition data under data/, private package under data/derived/ and artifacts/
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2
python -I tools/make_folds.py --check || python -I tools/make_folds.py   # artifacts/folds.npy from configs/folds.parquet
ORIENT=legacy bash tools/builds/s029_m3x3.sh safe          # historical files (first-version rotation list),
ORIENT=legacy bash tools/builds/s034_stage2_main.sh best   #   used as references by the builds below
bash tools/builds/safety_ndr.sh       # safety-line files; the primary chain checks them by checksum
bash tools/builds/primary_m3px27.sh   # candidate of version 0.3 (also run inside the next step)
bash tools/builds/primary_m3x5h.sh    # builds B (primary_m3x4) and the intermediate form, then A
sha256sum artifacts/submissions/sub_primary_m3x5h.csv artifacts/submissions/sub_primary_m3x4.csv
#   4505c15950167650a66cdc60eb9c153eebfb9ed6192277bd283045f3797354e7  sub_primary_m3x5h.csv   (A)
#   76da786dfdbe6a6ddcbcc93070f385e18870cb5cdaea895d92104aa3565f9973  sub_primary_m3x4.csv    (B)
```

Every script verifies its pinned inputs, refits or verifies its blend and second-stage files, runs its copy null points
and stops at the first deviation; an existing output file is replaced only by a byte-identical rebuild. The checksums of the
JSON files the scripts write themselves are pinned as well (learned on the first build with this code); the CSV and
test-matrix checksums are checked on every run ([INFERENCE.md §2, §5](INFERENCE.md#5-mode-assemble)). Evidence: both files were rebuilt byte for
byte on a Kaggle CPU without network access from the frozen inputs, with the development code from which this repository
was translated; and on 2026-11-17 with the code of this repository, in two fresh clones that read only the competition data
and the private input package: both files, the intermediate form, the three test matrices and all 44 CSV and matrix files of
the chain were byte-identical between the two clones and matched the pinned checksums, with no change to the code.

### 5.2 `rescore` and `full` (GPU)

The configurations below produced the frozen test-side outputs; they are run one after another with the model runner.
There is no single command per mode yet (a stage-wise runner is planned, [INFERENCE.md §6](INFERENCE.md#6-modes-rescore-and-full)).

```bash
cd /workspace/k12/src                      # repository cloned to /workspace/k12, competition data in its data/ folder
export K12_DATA=/workspace/k12/data K12_MODELS=/workspace/models K12_ADAPTERS=/workspace/adapters
export PYTHONPATH="$PWD:$PWD/.." HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
OUT=/workspace/out                         # one output root for all runs
run() { python -u -m vlm.runner --config ../configs/vlm/$1.yaml --out "$OUT"; }
run v1te_a_t4; run v1te_b_t4; run v1rot_t4             # 9B scorer, Kaggle T4x2 (fp16)
run zs27q_test0_rp; run zs27q_test1_rp                 # 27B zero-shot scorer, A40 (4-bit)
run zs27v_te_rp; run zs27v_n3_rp; run pr27te_rp        # reasoning draws g, n; protocol answers (RTX PRO 6000)
run m3_fullte_rp                                       # M3 draw d1 (see the note on the delivered adapter below)
run m3d2te_rp; run m3d3te_rp; run m3d4te_rp            # M3 draws d2-d4
run m3hte_a_rp; run m3hte_b_rp                         # M3 draw h (submission A only)
run q397p_rp; run q397te_rp                            # 397B parity probe and test side (4 x H200)
cd ..   # CPU steps; copy each $OUT/<run> to artifacts/runs/<run>/ first (the layout these tools and the assembly read)
python tools/export_proto_notes.py --name test_pr27 --from pr27te_rp/proto_te/proto --allow-test
python -I tools/make_m3px27_notes.py --table test
python -I tools/make_q397m3_notes.py --test-from runs/q397te_rp/proto_te/proto
cd src && run m3px27te_rp && run q397m3te_rp && cd ..  # M3 behind the protocol notes and behind the 397B notes
python tools/freeze_m3_texts.py --gen runs/m3_fullte_rp/m3_fullte_rp/m3_test_gen/gen \
  --rank runs/m3_fullte_rp/m3_fullte_rp/m3_test_rank/test --out derived/sft/m3_d1_test_frozen_notes.jsonl
#   likewise for d2 and d3 (derived/sft/) and d4 (derived/sft_k2/m3_d4_test_frozen_notes.jsonl)
```

- **Delivered weights.** Base models under `$K12_MODELS` at the pinned revisions; the 9B adapter `v1_full` (and the fold-1
  adapter `v1_f1`, which the pre-check job of the 9B configurations loads) under `$K12_ADAPTERS`; the M3 full-fit adapter
  with its `sft_recipe.json` at `$OUT/m3_fullte_rp/m3_train_full/` (folder `adapter/` inside), where all M3 configurations
  merge it from. `m3_fullte_rp` also contains the two jobs that trained this adapter (`m3_smoke_full`, `m3_train_full`):
  to use the delivered adapter, run a copy of the file without these two jobs.
- **Row lists.** The rotation list is recomputed with `python -I tools/orient_net.py infer --from-images --tables test
  --weights artifacts/orient/orient_cnn.pt --out data/derived/orient_repro`; the list of draw h with
  `python -I tools/make_charsize_rows.py --max-height 20 --name h20`. The gate lists of the reasoning draws and the
  blank-photo list are taken as frozen inputs ([INFERENCE.md §6.1](INFERENCE.md#61-test-side-steps)).
- **`rescore`.** Copy the frozen run folders of the private package to `$OUT` and delete the folders of the scoring and
  ranking jobs; the generation jobs then find every row done and generate nothing. They record the model path in their
  recipe, so use the paths of the reference runs (`K12_MODELS=/workspace/models`).
- **The 397B path** with its pass conditions and the settings for a run without network access:
  [INFERENCE.md §10.7](INFERENCE.md#107-order-of-the-steps).
- **Network.** With `HF_HUB_OFFLINE=1` the runner never downloads weights. It creates the vLLM environment from PyPI on
  first use unless that environment exists and carries the marker `.k12_ready` ([INFERENCE.md §10.7](INFERENCE.md#107-order-of-the-steps),
  step 2), so for a run without network the environment is installed beforehand. The Kaggle T4 steps and the CPU
  assembly have run without network access; the GPU steps on the rented machines have not yet
  ([ROADMAP.md](ROADMAP.md), step 2).
- **Assembly of new outputs.** On the reference hardware the rescored files are expected to be bit-identical, and §5.1 runs
  unchanged. Otherwise the build scripts stop by design, because they and the deployment files of A and B pin the
  checksums of the frozen test-side files. A mode that re-records these checksums without refitting anything is not built
  yet ([INFERENCE.md §9](INFERENCE.md#9-running-on-a-new-test-set), item 4); until then a new draw can be assembled with
  `tools/make_submission.py` only in the forms without second stage (§9, item 6).

## 6. (b) Training from scratch

In this order; commands, inputs and checks per step are in [TRAINING.md](TRAINING.md) (sections in brackets). GPU steps
use `run` as in section 5.2.

1. **Folds** (§3): `python -I tools/make_folds.py --recompute` (5 folds, seed 42, stratified by root code; it confirms
   `configs/folds.parquet`).
2. **9B adapters** (§4): `run ft_v1a_rp; run ft_sum1_rp; run ft_v1b_rp; run ft_v1c_rp` (folds 0–4, full fit), then the OOF
   assembly with `eda/a6_assemble_oof.py`.
3. **Orientation network and rotated 9B reading** (§5): `tools/orient_net.py maps`, `train --seed 0 --threads 2`,
   `infer`; `run v1rot_t4`.
4. **Teacher pool labels** (§6): frozen files (paid API; not rerun).
5. **27B zero-shot scorers** (§7): `run zs27q_oof_rp; run zs27q2_f1_rp; run zs27t_all_rp`.
6. **Reasoning draws** (§8): gates with `tools/replay/gates.py write`; draws g (`zs27v_all_rp`, `zs27v_f2_rp`,
   `zs27v_cut_rp`, `zs27v_te_rp`) and n (`zs27v_n_rp`, `zs27v_n3_rp`).
7. **Protocol channel** (§9): `run pr27_rp; run px27_rp`; notes with `tools/export_proto_notes.py`; M3 behind the notes
   with `run m3px27_oof_rp` after step 8.
8. **M3** (§10): training targets (six frozen files, pinned in each adapter's recipe; how they are built: §10.1), then
   `run m3_pilot_rp; run m3_f034_rp; run m3_fullte_rp` (fold adapters with draw d1, full fit), the OOF draws
   `m3long_rp`, `cv01_m3d2_f034`, `m3d3_oof_rp`, `m3d4_oof_rp`, `m3ha_oof_rp`, `m3hb_oof_rp`.
9. **397B training side** (§11): frozen answers (hosted API; not rerun), notes with `tools/make_q397m3_notes.py`,
   `run q397m3_oof_rp`.
10. **Blends and second stages on the CPU** (§12, §13, §16): the `eda/` commands listed there, in that order, ending with
    `eda/c135_primary_m3x4.py`, `eda/c151_primary_cols.py` and `eda/c166_primary_m3x5h.py`. After a retraining the pinned
    checksums of the build scripts no longer apply: every pre-registered rule is evaluated again on the new OOF matrices.

GPU training is not bit-reproducible; a retrained adapter is a new model with a new test file
([TRAINING.md §15](TRAINING.md#15-reproducibility-of-training)).

## 7. Seeds

| Component | Seed or setting |
|---|---|
| Fold split | `kaggle_core.folds.make_folds`, seed 42 (fold hash `07ec381cc58b2a80`) |
| 9B and M3 LoRA training | seed 42 |
| Orientation network | `--seed 0`; fixed sample of 10,000 pool photos with seed 20261010 |
| Reasoning draws, protocol answers, 397B answers, M3 draws d1 and d2 | greedy decoding; rows in a fixed shuffled order (`order_seed`) |
| Exact scoring and rankings (9B scorer, 27B zero-shot scorer, all rankings behind texts) | no sampling |
| M3 draws d3, d4, h | sampled with temperature 0.6, top-p 0.95, top-k 20; seeds 3, 4, 5 |
| 397B parity rows | seed 20261030 |
| Blend and second stages | deterministic fits (L-BFGS-B); the meta-splits used for decisions have fixed seeds and are not needed to build |

Greedy decoding in vLLM is not deterministic across runs, because requests are batched differently (section 8).

## 8. What cannot be rerun exactly

| Part | Why | How the solution handles it |
|---|---|---|
| Generated texts (reasoning, protocol answers, M3 draws, 397B notes) | vLLM batches parallel requests differently from run to run; two greedy runs with the same prompt agree on the top code in about 84 % of rows; the local 397B path agrees with the hosted one in 78 % | Texts are frozen with checksums; `assemble` and `rescore` read them. A `full` rerun is a new draw; expected deviation in section 8.1 |
| Teacher pseudo-labels of the unlabelled pool | Paid, non-deterministic API (Claude Haiku 5.5) | Frozen files, delivered. The test inference reads them as a file (the per-question task block in two prompts) and calls no API |
| 397B answers on the training rows | Produced through the Tinker sampling API; the provider's revision and number format are not recorded | Frozen, delivered. They enter only fitted weights and the training side of six columns; the test side runs locally with the FP8 release |
| GPU training (9B, M3) | Kernels and bf16 sums are not deterministic | The trained adapters are delivered with checksums |
| Rescoring on other hardware | Other kernels and number formats | Statistically equivalent: 9B on T4 in fp16 against A40 in bf16 changes the channel's top code on about 9 % of rows and the blend's on 1–2 %; M3 rankings on T4 in 4-bit agree on the top code in 0.971 (d1) and 0.978 (d2, half the rows) |
| CPU fits on another CPU class | Last bits of fitted weights can differ | Bit-identical on the same CPU class with two BLAS threads |
| Steps whose script is not in this repository | Listed in [PROVENANCE.md §9](PROVENANCE.md#9-open-items-of-this-list), among them the blank-photo list, the combination of the fold runs of draws d1 and d2 and the band part of the reasoning gates | Their outputs are delivered frozen; no input was produced by hand |

None of these parts is a step at test time that needs an API or a person: a rerun of the test inference depends only on the
generation of the open models (first row).

### 8.1 How close a `full` rerun should come

Assembling the submission files from the frozen inputs is deterministic (rebuilt byte-identically on a CPU without network
access). What cannot be rerun exactly is the generation of texts: the 27B reasoning texts and protocol answers, the four
(submission B) or five (submission A) M3 note draws and the 397B notes. vLLM serves 64 concurrent requests, and the
batch-dependent floating-point order changes greedy and seeded continuations alike. A full rerun therefore replaces every
generated text by a new draw.

We measured this on the 1,500 training rows, with blend and second-stage weights held fixed. Replacing every text for which
we hold another draw (the protocol answers, the reasoning texts on two of the five folds, the three M3 feature draws) moved
the row scores by an amount that corresponds to ±0.0015–0.0017 MAP@3 (1 sd) on 2,498 rows. Extrapolated to all generated
texts (the M3 mean channel from single-draw swaps, the 397B and protocol rankings from a stand-in calibrated on a real
second draw), we expect a rerun to land within about **±0.003 MAP@3 (1 sd; 95 % within about ±0.006; range over our
assumptions 0.0026–0.0044)** of the submitted file's private score, for both submissions, with no expected systematic
shift. About 91–93 % of the rows should keep the same top-1 code and about two thirds the same three codes; this can be
checked against the submitted files without labels. Re-scoring the frozen texts on other GPUs in bf16 adds at most about
±0.001.

One discrete risk: if the two greedy M3 draws produce word-identical notes more often than in our runs (for example under a
more deterministic batching setup), the documented note guard switches the build automatically to the fallback form without
note features, which scored about 0.004–0.007 lower in cross-validation.

Seeds: section 7.

## 9. Private weights and inputs

Download: the private link in our email to the host (not stored in this repository). Not public: everything in it is trained on or derived from the competition data.

| Content | Size |
|---|---:|
| Our adapters: M3 full fit (300.5 MB), 9B `v1_full` (110.9 MB); orientation network (3.4 MB) | 0.41 GB |
| Fold adapters of M3 and the 9B (training side, pre-check jobs) | 2.06 GB |
| Frozen inputs of `assemble` for both submissions: channel outputs, frozen texts and notes, row lists, OOF matrices, blend and second-stage files, stored reports, reference files (494 files; 155 derived from test rows) | 2.66 GB |
| Teacher pseudo-labels of the unlabelled pool (35 MB; read by `rescore` and `full`) and the six M3 training targets (56 MB; retraining). The 397B answers on the training rows are among the frozen inputs above | 0.09 GB |

The base models are public and not in the package: Qwen3.5-9B, Qwen3.8-27B and Qwen3.5-397B-A17B-FP8 at the pinned
revisions, 481.1 GB together, every file with its checksum in [weights_manifest.json](weights_manifest.json). Sizes against
the host's limits (481.53 GB as deployed, 440.94 GB at 8 bit, 434.4 B parameters): [MODEL_SIZES.md](MODEL_SIZES.md). Origin
of every file: [PROVENANCE.md](PROVENANCE.md).

## 10. Licence and external components

Code and documentation: Apache License 2.0 ([LICENSE](../LICENSE)). Every pre-trained model, API and package with link,
version and licence, and the API cost of a test inference run (0 USD): [THIRD_PARTY.md](../THIRD_PARTY.md). The competition
data are not part of this repository.
