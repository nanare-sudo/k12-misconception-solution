# Inference

*Status: 2026-11-01 (version 0). Items marked **TODO** are being built; see [ROADMAP.md](ROADMAP.md).*

The final solution is built to run **without network access and without any human step**. No external API is called at
inference time, and all models in the inference path are open-weight models whose weights, including our trained
adapters, are part of the delivered package. Smoke tests without network access for every model step are on the
[roadmap](ROADMAP.md).

**Contents:**
1. [Three modes](#1-three-modes)
2. [Lines and expected checksums](#2-lines-and-expected-checksums)
3. [Data layout](#3-data-layout)
4. [Environments](#4-environments)
5. [Mode `assemble`](#5-mode-assemble)
6. [Modes `rescore` and `full`](#6-modes-rescore-and-full)
7. [Run times, cost and package size](#7-run-times-cost-and-package-size)
8. [Operating rules](#8-operating-rules)

---

## 1. Three modes

| Mode | What is recomputed | Expected result | Hardware |
|---|---|---|---|
| `assemble` | Pre-checks, blend, alignment, reranker, `REL-01` rule and the output file, from the frozen channel outputs | **Byte-identical** to the submitted file | CPU, minutes |
| `rescore` | Every scoring and ranking step on all test images, behind the frozen reasoning texts, notes and gates | Identical on the reference hardware; statistically equivalent on other hardware | Kaggle T4×2, A40, RTX PRO 6000 (96 GB); 4×H200 for the 397B parity check |
| `full` | Everything from the photos and CSV files, including new reasoning texts, protocol answers, M3 generations and 397B notes | Statistically equivalent: every generated text is a new draw | As `rescore`, plus 4×H200 for the 397B notes |

`assemble` is deterministic on the CPU. The scoring steps of `rescore` are deterministic for a given text, model and
hardware. Text generation is not reproducible line by line: two greedy runs with the same prompt agree on the top code in
only 84 % of rows, because the server batches requests differently. The texts are therefore frozen with checksums, and
`rescore` scores behind them.

## 2. Lines and expected checksums

| Line | Build script | Output file | Bytes | SHA-256 |
|---|---|---|---:|---|
| Strictly inductive line | `bash tools/builds/s029_m3x3.sh safe` | `artifacts/submissions/sub_v1r_zs27t_think18sb_pr27xm_m3x3_full_ar_ia_t4.csv` | 115,240 | `edd9fb0070efb3e3734437df59d00505c03cc4d99aa7247d8b969c2160034546` |
| Primary line, reranker fallback form (submitted) | `bash tools/builds/s032_stage2.sh best` | `artifacts/submissions/sub_s032_stage2.csv` | 115,494 | `96bafc6fa4c0fb89a32e45fe24816454685b137d0c8264171bfcea93ea371809` |
| Primary line, reranker main form (candidate) | `bash tools/builds/s034_stage2_main.sh best` | `artifacts/submissions/sub_s034_stage2_main.csv` | 115,357 | `d9dc8b480b3c87c968426c2c16b264b69ca1b1bf533049105875dc666ec8f7ef` |

The blend and second-stage files used by these lines are listed with their checksums in
[TRAINING.md](TRAINING.md#12-blend-weights). The script names carry historical build labels; the frozen artifacts refer to
them, so they are kept.

The build scripts of the primary line are chained. The main-form script first rebuilds the fallback form. The
fallback-form script first rebuilds the earlier primary-line file without reranker and checks earlier files of the line by
checksum. Every script stops at the first failing check.

> **TODO.** A single entry point, `python -I tools/replay/assemble.py --line <line>`, is being added. It reads the line
> definition from `configs/lines/<line>.yaml` (sources, floors, last step, expected checksums), verifies every input
> checksum, calls `tools/make_submission.py` and compares the output checksum. It removes the need to run the chain of
> earlier builds. The build scripts stay in the repository as the record of how each file was produced.

## 3. Data layout

```
data/                                   K12_DATA may point to another folder with the same layout
├── train.csv  test.csv  unlabelled.csv          competition data (root CSV files)
├── competition_data/
│   ├── taxonomy_v1.csv  sample_submission.csv
│   └── images/train/  images/test/  images/questions/  (images/unlabelled/ only for training the orientation network)
└── derived/                                     private: frozen derived inputs
    ├── blank/        blank-photo list
    ├── orient/       rotation lists (c50: photos turned upright; c80: rows using the upright reading)
    ├── think_rows/   gate lists of the reasoning channels
    ├── proto_notes_test/   frozen protocol notes of the test rows
    ├── sft/          frozen M3 test texts (draws d1–d3) and 397B notes for M3
    └── teacher/      teacher pool labels (primary line only)
artifacts/                              private
├── oof_<channel>.npy  meta_<channel>.json        OOF matrices and metadata of every channel
├── stack_<name>.json                             blend files
├── stage2_<name>.json                            second-stage files
├── runs/<run>/<run>/<job>/…                      frozen channel outputs (test-side scores, rankings, texts)
└── submissions/                                  written by the assembly
```

The competition data come from the competition page. All other inputs are private: the frozen derived inputs and channel
outputs are in the private Kaggle dataset `k12-derived`, and the trained adapters, the orientation network and the
blend/second-stage files are in the private Kaggle model `k12-weights`. They are available to the competition host for
verification ([DATA_AND_COMPLIANCE.md](DATA_AND_COMPLIANCE.md)). Every file is referenced by relative path and SHA-256,
so the code finds it unchanged.

> **TODO.** A manifest of all frozen inputs per line (`solution/frozen_manifest.json`) and a weights manifest
> (`solution/weights_manifest.json`, with base-model revisions and file checksums) are being generated in this
> repository. The private dataset is being completed with the evaluation reports that the build scripts read.

## 4. Environments

| Lock file | Environment | Main pins |
|---|---|---|
| `solution/env/cpu.lock` | Assembly on the CPU | Python 3.13.5, numpy 2.4.2, scipy 1.17.1, pandas 2.3.3, scikit-learn 1.8.0, torch 2.11.0 (CPU, imported by a shared helper) |
| `solution/env/base.lock` | GPU path: training, merge, exact scoring | Python 3.12, torch 2.8.0+cu128, transformers 5.15.1, peft 0.21.2, accelerate 1.15.0, flash-linear-attention 0.5.2, bitsandbytes 0.50.2 |
| `solution/env/vllm.lock` | Text generation (separate virtual environment) | vllm 0.31.0, transformers 5.17.0, torch 2.13.0, flashinfer-python 0.7.0.post1 |
| `solution/env/t4.lock` | Kaggle T4 path | accelerate 1.15.0, bitsandbytes 0.50.2, peft 0.21.2, transformers 5.15.1 on top of the Kaggle GPU image |

**Offline installation.**
- `kernels/wheelhouse/fetch.py` builds a wheelhouse for these locks on a machine with network access but without
  competition data.
- The target machine then installs with `pip install --no-index --require-hashes --find-links <wheelhouse> -r <lock>`.
- For the vLLM environment the same works with `uv pip install --offline …`.

> **TODO.** Complete hashes for every pin are written by the wheelhouse build. Until then, the locks document the exact
> versions.

**Settings for runs without network access:** `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`; vLLM usage statistics
disabled (`VLLM_NO_USAGE_STATS=1`, `DO_NOT_TRACK=1`); the vLLM server bound to 127.0.0.1. **TODO:** the offline runner
sets these and verifies the base-model files against the weights manifest.

## 5. Mode `assemble`

```bash
# 1. Python 3.13 environment with the versions of solution/env/cpu.lock
# 2. Competition data under data/, private inputs under data/derived/ and artifacts/ (section 3)
bash tools/builds/s029_m3x3.sh safe          # strictly inductive line
bash tools/builds/s032_stage2.sh best        # primary line, reranker fallback form
bash tools/builds/s034_stage2_main.sh best   # primary line, reranker main form
sha256sum artifacts/submissions/sub_v1r_zs27t_think18sb_pr27xm_m3x3_full_ar_ia_t4.csv \
          artifacts/submissions/sub_s032_stage2.csv artifacts/submissions/sub_s034_stage2_main.csv
```

Each build script first verifies the checksums of its inputs and recomputes the blend file from the OOF artifacts. It then
calls `tools/make_submission.py`, which:
1. loads the frozen test-side output of every channel;
2. averages the draws of a channel;
3. applies floors, blend and arithmetic share;
4. applies the alignment of the line;
5. for the primary line, reranks the top five;
6. puts `REL-01` first for blank photos;
7. writes the CSV file, a JSON record (sources, checksums, aggregates) and the test probability matrix.

The checks of [VALIDATION.md](VALIDATION.md#6-checks-before-a-submission) run along the way. Only aggregates are printed.

The final `make_submission` call of each line is shown for reference. Its arguments name the frozen channel outputs.
- A source `<run>/<job>/<tag>` (after `--test`) or `runs/<run>/<job>/<tag>` (after `--proto`, `--rank`, `--rank-only`)
  resolves to `artifacts/runs/<run>/<run>/<job>/`.
- Several shards of one draw are joined with `+`, and several draws of one channel with `,`.
- Row lists after `@` and after `--blank-rows` are relative to the data folder.

```bash
# Strictly inductive line
python tools/make_submission.py --name v1r_zs27t_think18sb_pr27xm_m3x3_full_ar \
  --test v1r=v1rot_t4/rot_test/test@derived/orient/test_c80_rows.npy --fill v1r=v1 \
  --test v1=v1te_a_t4/v1_test_a/test+v1te_b_t4/v1_test_b/test \
  --test zs27t=px27_t4zt0/zs27t_test_t4/test+px27_t4zt1/zs27t_test_t4/test \
  --fill think18sb=zs27t \
  --test think18sb=zs27v_n3_rp/think_nte/test/think@derived/think_rows/test_tb_t.npy \
  --proto pr27xm=runs/pr27te_rp/proto_te/proto --rank pr27xm=runs/n1_rp/rank_test/test \
  --rank-only m3x3_full=runs/m3_fullte_rp/m3_test_rank/test,runs/m3d2te_rp/m3d2_test_rank/test,runs/m3d3te_rp/m3d3_test_rank/test \
  --align-inductive 0.75 --align-min 3 \
  --blank-rows derived/blank/test_rows.npy --suffix _ia_t4

# Primary line (main form; the fallback form uses --stage2 artifacts/stage2_s032.json and the default output name)
python tools/make_submission.py --name s032_stage2 \
  --test v1r=v1rot_t4/rot_test/test@derived/orient/test_c80_rows.npy --fill v1r=v1 \
  --test v1=v1te_a_t4/v1_test_a/test+v1te_b_t4/v1_test_b/test \
  --test zs27q=zs27q_test0_rp/zs27q_test0/test+zs27q_test1_rp/zs27q_test1/test \
  --fill think18nb=zs27q \
  --test think18nb=zs27v_te_rp/think_te/test/think@derived/think_rows/test.npy,zs27v_n3_rp/think_nte/test/think@derived/think_rows/test_tb_q.npy \
  --proto pr27xm=runs/pr27te_rp/proto_te/proto --rank pr27xm=runs/n1_rp/rank_test/test \
  --rank-only m3x3_full=runs/m3_fullte_rp/m3_test_rank/test,runs/m3d2te_rp/m3d2_test_rank/test,runs/m3d3te_rp/m3d3_test_rank/test \
  --rank-only q397m3=runs/q397m3te_rp/q397m3_test_rank/test --rank-none-band-for q397m3=0.27,0.35 \
  --align 0.75 --align-min 3 --blank-rows derived/blank/test_rows.npy \
  --stage2 artifacts/stage2_s034.json --stage2-report <guard-report.json> \
  --out artifacts/submissions/sub_s034_stage2_main.csv --suffix _s034main
```

The two second-stage files contain identical weights. They differ only in the rule of the notes guard. With the original
rule the guard selects the fallback form, and with the corrected rule the main form ([METHOD.md](METHOD.md#9-second-stage-conditional-logit-reranker)).

## 6. Modes `rescore` and `full`

### 6.1 Test-side steps

Every model step is a configuration in `configs/vlm/`, run as described in [TRAINING.md](TRAINING.md#2-environment)
(`python -u -m vlm.runner --config … --out …`). In `rescore` only the scoring and ranking jobs run, behind the frozen
texts, notes and gate lists. In `full` the generation jobs run as well.

| Step | Channel | Configuration (job) or script | Reference hardware | Lines |
|---|---|---|---|---|
| Blank photos | all | `tools/make_blank_list.py` | CPU | both |
| Rotation lists | `v1r` | `python -I tools/orient_net.py infer --from-images --tables test --weights <orient_cnn.pt> --out <folder>` | CPU | both |
| 9B scorer | `v1` | `v1te_a_t4`, `v1te_b_t4` (two shards of the test set) | Kaggle T4×2, fp16 | both |
| Rotation-corrected 9B | `v1r` | `v1rot_t4` (`rot_test`) | Kaggle T4×2, fp16 | both |
| 27B zero-shot, task block | `zs27q` | `zs27q_test0_rp`, `zs27q_test1_rp` | A40 48 GB, 4-bit | primary |
| 27B zero-shot, topic block | `zs27t` | `px27_t4zt0`, `px27_t4zt1` | Kaggle T4×2, 4-bit | inductive |
| Gates | reasoning | `python -I tools/replay/gates.py check --table test --line <line>` | CPU | both |
| Reasoning draw g | `think18nb` | `zs27v_te_rp` (`think_te`) | RTX PRO 6000 96 GB | primary |
| Reasoning draw n | `think18nb`, `think18sb` | `zs27v_n3_rp` (`think_nte`) | RTX PRO 6000 96 GB | both |
| Protocol answers | `pr27xm` | `pr27te_rp` (`proto_te`) | RTX PRO 6000 96 GB | both |
| Protocol notes | `pr27xm` | `tools/export_proto_notes.py` | CPU | both |
| Exact ranking behind the notes | `pr27xm` | `n1_rp` (`rank_test`) | RTX PRO 6000 96 GB | both |
| M3 merge and draw d1 | `m3x3_full` | `m3_fullte_rp` | RTX PRO 6000 96 GB | both |
| M3 draws d2 and d3 | `m3x3_full` | `m3d2te_rp`, `m3d3te_rp` | RTX PRO 6000 96 GB | both |
| Frozen M3 texts | `m3x3_full`, reranker | `tools/freeze_m3_texts.py` | CPU | both |
| 397B protocol notes | `q397m3` | `q397te_rp` (`proto_te`) | 4×H200, tensor parallel 4, FP8 | primary |
| 397B parity check (64 training rows) | `q397m3` | `q397p_rp`, then `eda/c109_q397_parity.py` | 4×H200 | primary |
| Notes for M3 | `q397m3` | `tools/make_q397m3_notes.py` | CPU | primary |
| M3 behind the 397B notes | `q397m3` | `q397m3te_rp` | RTX PRO 6000 96 GB | primary |

**Gates.** The frozen gate lists were computed from the A40 versions of the 9B and `zs27t` test outputs
(`v1_test_rp`, `zs27t_all_rp`), while the submitted files use the offline Kaggle T4 versions of these two channels.
Recomputing the gates from the T4 outputs therefore shifts some rows between gates. In `rescore` the gate lists are taken
as frozen inputs, and `gates.py check` confirms them against the A40 outputs.

### 6.2 Stage order

> **TODO.** A stage-wise runner (`tools/replay/run_inference.sh <line> --mode assemble|rescore|full`) is being added.

| Stage | Content | Rule |
|---|---|---|
| S0 | Pre-checks: GPU, driver, disk, network level (with a positive control), checksums of all package files | Any deviation stops the run |
| S1 | Environments from the wheelhouse | Installed versions equal the locks |
| S2 | Blank-photo list; rotation lists from the stored orientation network | Byte-identical |
| S3 | Fast channels: 9B, rotation-corrected 9B, `zs27q` or `zs27t` | Byte-identical on the reference hardware |
| S4 | Gates | Byte-identical if S3 was |
| S5 | M3 adapter merge, once for all draws | Checksums of the merged shards |
| S6 | Texts: reasoning draws g and n, protocol answers, M3 draws d1–d3, 397B notes | Only aggregates against the reference (`full` only) |
| S7 | Export of the protocol notes and M3 texts | Byte-identical in `rescore` |
| S8 | Exact rankings | Reference hardware: identical |
| S9 | Assembly | `assemble`: checksum equals the submitted file |

In `full` mode, S5 and S6 run first because they are the most fragile steps.

### 6.3 Reference hardware and expected agreement

| Steps | Reference hardware |
|---|---|
| 9B scorer, rotation-corrected 9B, `zs27t` | Kaggle T4×2, fp16 (4-bit for `zs27t`), without network access |
| `zs27q` | A40 48 GB, 4-bit |
| All 27B bf16 steps and vLLM text generation | RTX PRO 6000 Blackwell, 96 GB |
| 397B notes | 4×H200, tensor parallel 4, FP8 weights, vLLM 0.31 |

- **Same hardware and number format:** the scores are bit-identical.
- **A40 in bf16 against T4 in fp16 (9B scorer):** the channel's top code changes on about 9 % of rows, the blend's top
  code on 1–2 %.
- **Evidence on a second hardware class** (aggregates only):

  | Step | Status |
  |---|---|
  | 9B test side on T4 | complete |
  | `zs27t` on T4 | top-code agreement 0.944 |
  | M3 draw d1 ranking on T4 in 4-bit | agreement 0.971 |
  | M3 draw d2 ranking on T4 | half done, agreement 0.978 on 1,750 rows |
  | M3 draw d3 ranking, reasoning ranking | pending |

- **Rerun of a generated text:** a rerun of the protocol answers changed the blend's top code on about 8–9 % of rows of
  an error-heavy sample, without a change in MAP@3.

## 7. Run times, cost and package size

**Test side per channel** (3,498 rows unless stated):

| Channel | Hardware | Run time | Cost |
|---|---|---|---|
| 9B scorer with exact scoring | Kaggle T4×2, fp16 | about 6 s per row (early measurement) | free quota |
| `zs27q` | A40, 4-bit | 3.1–3.7 s per row, about 3.4 h | about 2–3 USD |
| `zs27t` | Kaggle T4, 4-bit | 17–19 s per row, about 17 T4 hours | free quota |
| Reasoning, one draw (gated rows) | RTX PRO 6000 | about 2.5 h | about 6.3 USD |
| Protocol answers | RTX PRO 6000 | 3.3–3.5 s per row | about 8 USD |
| M3, one draw (generation and ranking) | RTX PRO 6000 | about 7–8 h | about 17–20 USD |
| M3 ranking only | Kaggle T4, 4-bit | 26.5 s per row, about 26 T4 hours per draw | free quota |
| 397B notes | 4×H200 | 1.49 s per row | — |
| M3 behind the 397B notes | RTX PRO 6000 | about 3.4 s per row | about 10 USD |

**Totals.** The following estimates are preliminary. They were made for the two lines before the 397B channel and the
reranker were added and will be updated (**TODO**).
- `rescore` of both lines: about 9.4–10.6 h on an RTX PRO 6000, 3.5 h on an A40 and 25 T4 hours, about 25–29 USD.
- `full` of both lines: about 29.7 h on an RTX PRO 6000, 3.5 h on an A40 and 25 T4 hours, about 76–79 USD.
- The 397B steps add about 27–42 USD.

**Package size.** About 84–87 GB without the 397B; about 490 GB with it. The 397B FP8 weights alone take about 406 GB
(378 GiB). The base models are referenced by pinned revision and file checksums; M3 is shipped as an adapter (300.5 MB)
and merged at run time.

## 8. Operating rules

- **Full test set only for the primary line.** The transductive alignment is defined over the complete test set (3,498
  rows). The primary line is never assembled from a subset of rows.
- **Fixed draws.** M3 uses exactly three draws with fixed settings. Each test draw must match the generation settings of
  its OOF draw; the checks reject any mismatch.
- **Aggregates only.** Model outputs for test rows are written to files and never printed. Logs contain counts, shares
  and checksums.
- **Fault bands are not tuning knobs.** The bands for the share of `NONE` at the top and for the rate of changed top codes
  detect technical faults. They are fixed in advance and never used to choose between variants.
