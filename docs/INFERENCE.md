# Inference

*Status: 2026-11-07 (version 0.2). Planned items are marked **Planned** and listed in [ROADMAP.md](ROADMAP.md).*

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

The current candidates for the two final submissions are the first two rows; the next rows are their fallbacks and the
files of version 0.1. All of them use the rotation list of the released orientation network (`ORIENT=repro`, the default
of the build scripts; see below).

| Line and form | Build script | Output file in `artifacts/submissions/` | Bytes | SHA-256 |
|---|---|---|---:|---|
| **Primary line, final form: I1 form with the minor-code × mechanical-match column** (candidate) | `bash tools/builds/primary_final.sh` | `sub_primary_final.csv` | 115,282 | `d9c68c460c6da06d4975513911e11c4541d70d07b78f132566a3e43e7ad60b24` |
| **Safety line, alignment over the test set, second stage with interactions and NONE guard** (candidate) | `OVERRIDE_CV_II=override_cv_condition_ii bash tools/builds/safety_final.sh` | `sub_safety_final.csv` | 115,500 | `736d87b3cb61494aeb6dcac29bc248a909138c6e85e3939f37a01180984e78f7` |
| Primary line, I1 form: guard form with row × candidate interactions (fallback of the primary candidate) | `bash tools/builds/primary_i1.sh` | `sub_primary_i1.csv` | 115,233 | `60b63d69bcdd5970cafaf4d5645440d8d375953871d6dd688ae5091b25256bee` |
| Strictly inductive line with second stage and NONE guard (fallback of the safety candidate; candidate of version 0.1) | `bash tools/builds/s036_safe_stage2.sh` | `sub_s036_safe_stage2.csv` | 115,508 | `ff45c2f72d21fd9d6b227fc47bad10b4b4092cd79008734da917e2ba3bcefd13` |
| Primary line, guard form: reranker main form with NONE guard and general-code identifiers (candidate of version 0.1) | `bash tools/builds/s035_stage2_guard.sh` | `sub_s035_stage2_guard.csv` | 115,318 | `d4dce1aaa5cfbd1fd5668a9e7073a47734c0c5aacda7b7c2020255005a81f360` |
| Primary line, reranker main form | `bash tools/builds/s034n_stage2_main.sh` | `sub_s034n_stage2_main.csv` | 115,359 | `8b8892e474ace44048d6c02eba7622288599990f5f5f0814cbefaeb6bd9ce242` |
| Strictly inductive line without second stage | `bash tools/builds/s030n_safe.sh` | `sub_s030n_safe.csv` | 115,238 | `8cca7bbfe9b8c9dce47d6153d59712575bb7b8ad6dea8a286f7ef530df448b67` |
| Primary line, reranker fallback form | `bash tools/builds/s033n_stage2.sh` | `sub_s033n_stage2.csv` | 115,498 | `76aba64fbc431b9ad403a5dec9fb2cefdb7fe2e76260149c96176459c4cd8dfe` |
| Primary line without second stage (intermediate file) | built by `s033n_stage2.sh` | `sub_s032n_q397m3.csv` | 115,404 | `d1667f6d2fb118844b72ecd8abd85220754a709d468519b7a047a893bbea12f1` |

**Names of the version-0.2 files.** The three new build scripts, their deployment files (`artifacts/stage2_<name>.json`),
their output files (`sub_<name>.csv`) and their report folders are named by role: `primary_i1`, `primary_final`,
`safety_final`. The earlier scripts keep their historical build labels (`s029` … `s036`), because frozen artifacts and
their checksums refer to them.

**Override of the safety candidate.** `safety_final.sh` needs `OVERRIDE_CV_II=override_cv_condition_ii`. The
cross-validation part of one condition of its pre-registered rule was missed by one sixth in 9,000; the decision to build
the file anyway is recorded in [`decisions/override_cv_condition_ii.json`](../decisions/override_cv_condition_ii.json),
whose SHA-256 is pinned in `eda/c123_safety_set.py` ([METHOD.md](METHOD.md#99-second-stage-of-the-safety-line-with-alignment-over-the-test-set-final-safety-candidate)).
Without the variable the script stops with exit code 5 after the rule; any other value stops it with exit code 2. The
override affects only that condition; all other conditions, the copy null points, the guards and the checksums stay in
force.

**Historical files with the first-version rotation list** (`ORIENT=legacy`). The earlier submissions were built with the
rotation list of an unseeded first version of the orientation network, whose weights were not stored. The build scripts
rebuild these files byte-identically from the frozen list; the released-list builds above compare against them and need
them first.

| Line and form | Build script | Output file in `artifacts/submissions/` | Bytes | SHA-256 |
|---|---|---|---:|---|
| Strictly inductive line | `ORIENT=legacy bash tools/builds/s029_m3x3.sh safe` | `sub_v1r_zs27t_think18sb_pr27xm_m3x3_full_ar_ia_t4.csv` | 115,240 | `edd9fb0070efb3e3734437df59d00505c03cc4d99aa7247d8b969c2160034546` |
| Primary line without second stage | built by `s032_stage2.sh` | `sub_s029_best_q397m3.csv` | 115,406 | `768b7778ea049caeb23455ecad26b9bc90ebff52bee3cb97e356aac7cadf07e8` |
| Primary line, reranker fallback form | `ORIENT=legacy bash tools/builds/s032_stage2.sh best` | `sub_s032_stage2.csv` | 115,494 | `96bafc6fa4c0fb89a32e45fe24816454685b137d0c8264171bfcea93ea371809` |
| Primary line, reranker main form (never submitted in this form) | `ORIENT=legacy bash tools/builds/s034_stage2_main.sh best` | `sub_s034_stage2_main.csv` | 115,357 | `d9dc8b480b3c87c968426c2c16b264b69ca1b1bf533049105875dc666ec8f7ef` |
| Six-channel primary file with the 397B answer channel (the script always uses the first-version list and does not read `ORIENT`) | `bash tools/builds/s029_q397.sh best --m3 m3x3_full` | `sub_s029_best_m3x3.csv` | 115,479 | `704d8ae8e6eb93051aa4fbe8ec954da143c1c68ff751fc6fe0063c681c063b18` |

The released list differs from the first-version list on 24 of the 3,498 test photos (344 rows instead of 350); all 344
rows were already computed by the existing rotation run with the same rotation, so no new GPU computation is involved.
The choice is made by the sourced helper `tools/builds/orient_list.sh`. With `ORIENT=repro` the scripts reject the
first-version list (by path and by checksum) and run `tools/orient_list.py check` before and `verify-sub` after the
assembly ([VALIDATION.md](VALIDATION.md#6-checks-before-a-submission)).

The blend files used by these lines are listed with their checksums in [TRAINING.md](TRAINING.md#12-blend-weights), the
second-stage files in [TRAINING.md](TRAINING.md#13-second-stages). The script names carry historical build labels; the
frozen artifacts refer to them, so they are kept.

**The build scripts are chained.**
- `primary_final.sh` runs `primary_i1.sh`, which runs `s035_stage2_guard.sh`.
- `s035_stage2_guard.sh` runs `s034n_stage2_main.sh`, which runs `s034_stage2_main.sh best` with `ORIENT=repro` (main
  form). That script first runs `s032_stage2.sh best` (fallback form, the same steps as `s033n_stage2.sh`), which first
  runs `s029_q397m3.sh best` (file without second stage).
- `safety_final.sh` runs `s036_safe_stage2.sh`, which runs `s030n_safe.sh`, which runs `s029_m3x3.sh safe` with
  `ORIENT=repro`.
- Every script stops at the first failing check.
- An existing output CSV file is replaced only by a byte-identical rebuild; otherwise the previous files are restored and
  the script aborts. The exceptions are `s029_m3x3.sh` with `ORIENT=legacy` and `s029_q397.sh`, which overwrite their
  outputs. Test matrices written next to an unchanged CSV file are not compared; pinned are the test matrices of the
  inductive file without second stage (by `s036_safe_stage2.sh`), of the strictly inductive file with second stage (by
  `safety_final.sh`), of the guard form and of the I1 form (by `primary_i1.sh` and `primary_final.sh`).

> **Planned.** A single entry point that reads a line definition (sources, floors, last step, expected checksums), verifies
> every input checksum and calls `tools/make_submission.py` directly, without the chain of earlier builds. The build
> scripts stay in the repository as the record of how each file was produced.

## 3. Data layout

```
data/                                   K12_DATA may point to another folder with the same layout
├── train.csv  test.csv  unlabelled.csv          competition data (root CSV files)
├── competition_data/
│   ├── taxonomy_v1.csv  sample_submission.csv
│   └── images/train/  images/test/  images/questions/  (images/unlabelled/ only for training the orientation network)
└── derived/                                     private: frozen derived inputs
    ├── blank/            blank-photo list
    ├── orient/           first-version rotation lists (c50: photos turned upright; c80: rows using the upright reading)
    ├── orient_repro/     rotation lists of the released orientation network (test_c80_rows.npy, test_c80.json, .meta.json)
    ├── think_rows/       gate lists of the reasoning channels
    ├── sft/              frozen M3 test texts (draws d1–d3), 397B notes for M3, M3 training targets
    ├── sft_q397m3/  sft_q397m3te/   copies of the 397B notes for M3 as read by the model runs (primary line)
    ├── open_teacher/     frozen 397B answers on the training rows (primary line)
    ├── m3x3_full_rows.npy  training rows with all three M3 draws (rewritten identically by s029_m3x3.sh)
    ├── proto_notes/  proto_notes_test/   frozen protocol notes of the training and test rows (model runs only)
    └── teacher/          teacher pool labels (primary line; model runs only)
artifacts/                              private
├── folds.npy                                     fold array (tools/make_folds.py); its checksum is recorded by the second stage
├── oof_<channel>.npy  meta_<channel>.json        OOF matrices and metadata of every channel
├── stack_<name>.json                             blend files
├── stage2_<name>.json                            second-stage files
├── orient/orient_cnn.json  orient_cnn.pt         description and weights of the released orientation network
├── runs/<run>/<run>/<job>/…                      frozen channel outputs (test-side scores, rankings, texts)
├── eda/private/<topic>/                          stored reports read by the build scripts and reports they write
└── submissions/                                  earlier files used as references, and the files written by the assembly
```

Folders marked *model runs only* are read by `rescore` and `full`, not by `assemble`.

**Inputs per line** (in addition to the competition CSV files and `taxonomy_v1.csv`). The list was taken from a trace of
every file the build scripts open in a fresh copy of the repository with the frozen inputs; it covers the assembly and
all checks and fits that the scripts run on the way. "Safety line" covers both of its forms (the strictly inductive form
and the final candidate with the alignment over the test set); they read the same frozen inputs.

| Input | Safety line | Primary line |
|---|---|---|
| `data/derived/orient_repro/*`, `artifacts/orient/orient_cnn.json` (+ `orient_cnn.pt`, checked if present) | yes | yes |
| `data/derived/orient/test_c80_rows.npy`, `test_c50.json` (checked: rejection of the old list, rotation of the run) | yes | yes |
| `artifacts/runs/v1rot_t4/…/rot_test/` (rotation run), `v1te_a_t4`, `v1te_b_t4` | yes | yes |
| `artifacts/runs/px27_t4zt0`, `px27_t4zt1` (`zs27t`), `zs27v_n3_rp` (reasoning draw n) | yes | `zs27v_n3_rp` only |
| `artifacts/runs/zs27q_test0_rp`, `zs27q_test1_rp`, `zs27v_te_rp` | — | yes |
| `artifacts/runs/pr27te_rp`, `n1_rp` (protocol channel), `m3_fullte_rp`, `m3d2te_rp`, `m3d3te_rp` (M3 draws) | yes | yes |
| Training-side rankings, read when the M3 channel and the protocol mean channel are re-verified: `artifacts/runs/m3_pilot_rp`, `m3_f034_rp`, `m3_5f`, `m3long_rp`, `m3long_comb`, `cv01_m3d2_f034`, `m3d3_oof_rp`, `px27_rp` | yes | yes |
| `artifacts/runs/q397te_rp`, `q397m3te_rp`, `q397m3_oof_rp`, `q397_tinker`, `pr27_rp`; `data/derived/sft/q397m3_*`, `data/derived/sft_q397m3/`, `data/derived/sft_q397m3te/`; `data/derived/open_teacher/` | — | yes |
| `data/derived/sft/m3_d{1,2,3}_test_frozen_notes.jsonl` and their meta files (frozen M3 texts, read by the second stage) | — | yes |
| Answers of the protocol channel as text, for the mechanical comparison of the final answer with the key (final primary form): `artifacts/runs/pr27_rp/pr27_rp/proto_a/`, `proto_b/` (training rows), `artifacts/runs/pr27te_rp/pr27te_rp/proto_te/proto_raw.jsonl` (test rows, pinned by SHA-256) | — | yes |
| Decision file of the override (repository file): `decisions/override_cv_condition_ii.json` | yes (final candidate) | — |
| `data/derived/think_rows/*`, `data/derived/blank/test_rows.npy` | yes | yes |
| `artifacts/oof_*.npy`, `meta_*.json` of the channels and of the earlier channels the checks compare with (`m3x2_full_cv01`, `m3p5`; in the primary line also `q397`), `artifacts/folds.npy` | yes | yes |
| Earlier blend files read as references: `stack_v1r_zs27t_think18sb_pr27xm.json` | yes | — |
| Earlier blend files read as references: `stack_v1r_zs27t_think18sb_pr27xm_m3x2_ar.json`, `stack_s029_best_m3x3.json`, `stack_cv01_best_l2.json` | — | yes |
| Stored reports: `eda/private/m3_draws_eval/c108_m3_three_draws.json` | yes | yes |
| Stored reports: `eda/private/q397_deploy/c109_parity.json`, `eda/private/q397m3_eval/c115_q397m3_eval.json`, `eda/private/stage2/crosscheck_rows.npz` | — | yes |
| Earlier files as references (`submissions/`): `sub_v1r_zs27t_think18sb_pr27xm_m3x2_ar_ia_t4.csv` | yes | — |
| Earlier files as references (`submissions/`): `sub_s029_best_m3x3.csv`, `sub_v1r_zs27q_think18nb_pr27xm_m3x3_full_w14_ar_ra_t4.csv` | — | yes |

The files that the build scripts write and pin themselves are not listed: blend and second-stage files, the meta-split
files and the probe files of the safety line. The private package contains them, so that the scripts verify them instead
of recomputing them; without them the scripts recompute them and require the pinned checksums
([TRAINING.md](TRAINING.md#13-second-stages)).

The historical six-channel file (`s029_q397.sh`) additionally reads `submissions/sub_cv01_best_l2.csv`; the historical
five-channel file (`ORIENT=legacy s029_m3x3.sh best`) reads `stack_v1r_zs27q_think18nb_pr27xm_w14.json` and
`submissions/sub_cv01_best_l2.csv`.

The build scripts of the safety line also read the OOF matrices of the teacher-block channels `zs27q` and `think18nb`,
but only to verify their checksums (a shared loading function checks all base channels); these channels enter neither
the blend nor the test side of the safety line.

The competition data come from the competition page. All other inputs are private and available to the competition host
for verification ([DATA_AND_COMPLIANCE.md](DATA_AND_COMPLIANCE.md)): the frozen derived inputs, the channel outputs, the
OOF matrices with their metadata, the stored reports and the earlier submission files used as references in the private
Kaggle dataset `k12-derived`, and the trained adapters, the orientation network and the blend and second-stage files in
the private Kaggle model `k12-weights`. Every file is referenced by relative path, the pinned files also by SHA-256
([VALIDATION.md](VALIDATION.md#6-checks-before-a-submission)). The private copies are being brought to the state
of this version (updated text fields of the stored reports, the released rotation lists and the new deployment files);
see [ROADMAP.md](ROADMAP.md).

> **Planned.** A machine-readable manifest of all frozen inputs per line and a weights manifest (base-model revisions and
> file checksums).

## 4. Environments

The pinned versions are listed here; lock files with hashes and an offline wheelhouse are **planned**.

| Environment | Main pins |
|---|---|
| Assembly on the CPU | Python 3.13.5, numpy 2.4.2, scipy 1.17.1, pandas 2.3.3, scikit-learn 1.8.0, pyarrow 23.0.1 (fold table), PyYAML 6.0.3 (run configurations), Pillow 11.3.0, torch 2.11.0 and transformers 5.15.1 (imported by shared model helpers; everything runs on the CPU); pytest 8.4.2 for the tests |
| GPU path: training, merge, exact scoring | Python 3.12, torch 2.8.0+cu128, transformers 5.15.1, peft 0.21.2, accelerate 1.15.0, flash-linear-attention 0.5.2, bitsandbytes 0.50.2 |
| Text generation (separate virtual environment) | vllm 0.31.0, transformers 5.17.0, torch 2.13.0, flashinfer-python 0.7.0.post1 |
| Kaggle T4 path | accelerate 1.15.0, bitsandbytes 0.50.2, peft 0.21.2, transformers 5.15.1 on top of the Kaggle GPU image |

**Settings for runs without network access:** `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`; vLLM usage statistics
disabled (`VLLM_NO_USAGE_STATS=1`, `DO_NOT_TRACK=1`); the vLLM server bound to 127.0.0.1.

**Kaggle T4 runs.** The configurations `v1te_a_t4`, `v1te_b_t4` and `v1rot_t4` attach private notebook outputs with the
9B weights and the offline wheels, a private dataset with the adapters and the private dataset `k12-derived`;
`px27_t4zt0` and `px27_t4zt1` attach the notebook output with the wheels and read the 27B weights from a public mirror
([THIRD_PARTY.md](../THIRD_PARTY.md)).

## 5. Mode `assemble`

```bash
# 1. Python 3.13 environment with the versions of section 4
# 2. Competition data under data/, private inputs under data/derived/ and artifacts/ (section 3)
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2
python -I tools/make_folds.py --check || python -I tools/make_folds.py   # artifacts/folds.npy from configs/folds.parquet

# 3. Historical files with the first-version rotation list (references of the released-list builds)
ORIENT=legacy bash tools/builds/s029_m3x3.sh safe          # strictly inductive line
ORIENT=legacy bash tools/builds/s034_stage2_main.sh best   # primary line: file without second stage, fallback form, main form

# 4. Final candidates and their fallbacks (released rotation list)
bash tools/builds/primary_final.sh        # primary line; builds sub_s032n…, sub_s033n…, sub_s034n…, sub_s035…, sub_primary_i1 on the way
OVERRIDE_CV_II=override_cv_condition_ii bash tools/builds/safety_final.sh   # safety line; builds sub_s030n_safe, sub_s036… on the way
sha256sum artifacts/submissions/sub_primary_final.csv artifacts/submissions/sub_safety_final.csv
```

Each build script first verifies the checksums of its pinned inputs and recomputes (or verifies) its blend and
second-stage files from the OOF artifacts. It then calls `tools/make_submission.py`, which:
1. loads the frozen test-side output of every channel;
2. averages the draws of a channel;
3. applies floors, blend and arithmetic share;
4. applies the alignment of the line;
5. reranks the top five with the second stage, including the NONE guard;
6. puts `REL-01` first for blank photos;
7. writes the CSV file, a JSON record (sources, checksums, aggregates) and the test probability matrix.

The checks of [VALIDATION.md](VALIDATION.md#6-checks-before-a-submission) run along the way. Only aggregates are printed.
Each candidate script additionally runs its copy null points: with weight 0, the second stage reproduces the file without
it byte for byte, and the scripts of version 0.2 also show that the new columns with weight 0 reproduce the file of the
predecessor form. The three scripts of version 0.2 recompute their meta-split files (about 8 minutes each) only if they
are absent; the private package contains them, and the scripts check them against their pinned checksums. Besides the submission files, the build scripts rewrite some of their own reports under
`artifacts/eda/private/` (new time stamps), and `s029_m3x3.sh` rewrites the two blend files of its line with identical
content (in both `ORIENT` modes); one test rewrites a stored report of the 397B deployment in the same way. Of these
files only the blend file of the inductive line, `stack_v1r_zs27t_think18sb_pr27xm_m3x3_full_ar.json`, is pinned (by
`s030n_safe.sh` and `s036_safe_stage2.sh`). A build that stops with an error can leave reports that describe the
failed attempt; the next successful run rewrites them.

The final `make_submission` call of each candidate is shown for reference. Its arguments name the frozen channel outputs.
- A source `<run>/<job>/<tag>` (after `--test`) or `runs/<run>/<job>/<tag>` (after `--proto`, `--rank`, `--rank-only`)
  resolves to `artifacts/runs/<run>/<run>/<job>/`.
- Several shards of one draw are joined with `+`, and several draws of one channel with `,`.
- Row lists after `@` and after `--blank-rows` are relative to the data folder.

```bash
# Safety line, final candidate (safety_final.sh, step 7)
python -I tools/make_submission.py --name v1r_zs27t_think18sb_pr27xm_m3x3_full_ar \
  --test v1r=v1rot_t4/rot_test/test@derived/orient_repro/test_c80_rows.npy --fill v1r=v1 \
  --test v1=v1te_a_t4/v1_test_a/test+v1te_b_t4/v1_test_b/test \
  --test zs27t=px27_t4zt0/zs27t_test_t4/test+px27_t4zt1/zs27t_test_t4/test \
  --fill think18sb=zs27t \
  --test think18sb=zs27v_n3_rp/think_nte/test/think@derived/think_rows/test_tb_t.npy \
  --proto pr27xm=runs/pr27te_rp/proto_te/proto --rank pr27xm=runs/n1_rp/rank_test/test \
  --rank-only m3x3_full=runs/m3_fullte_rp/m3_test_rank/test,runs/m3d2te_rp/m3d2_test_rank/test,runs/m3d3te_rp/m3d3_test_rank/test \
  --align 0.75 --align-min 3 --blank-rows derived/blank/test_rows.npy \
  --stage2 artifacts/stage2_safety_final.json --stage2-report artifacts/eda/private/safety_set/guards_safety_final.json \
  --out artifacts/submissions/sub_safety_final.csv --suffix _safety_final

# Primary line, final form (primary_final.sh, step 9)
python -I tools/make_submission.py --name s032_stage2 \
  --test v1r=v1rot_t4/rot_test/test@derived/orient_repro/test_c80_rows.npy --fill v1r=v1 \
  --test v1=v1te_a_t4/v1_test_a/test+v1te_b_t4/v1_test_b/test \
  --test zs27q=zs27q_test0_rp/zs27q_test0/test+zs27q_test1_rp/zs27q_test1/test \
  --fill think18nb=zs27q \
  --test think18nb=zs27v_te_rp/think_te/test/think@derived/think_rows/test.npy,zs27v_n3_rp/think_nte/test/think@derived/think_rows/test_tb_q.npy \
  --proto pr27xm=runs/pr27te_rp/proto_te/proto --rank pr27xm=runs/n1_rp/rank_test/test \
  --rank-only m3x3_full=runs/m3_fullte_rp/m3_test_rank/test,runs/m3d2te_rp/m3d2_test_rank/test,runs/m3d3te_rp/m3d3_test_rank/test \
  --rank-only q397m3=runs/q397m3te_rp/q397m3_test_rank/test --rank-none-band-for q397m3=0.27,0.35 \
  --align 0.75 --align-min 3 --blank-rows derived/blank/test_rows.npy \
  --out artifacts/submissions/sub_primary_final.csv --suffix _primary_final \
  --stage2 artifacts/stage2_primary_final.json --stage2-report artifacts/eda/private/stage2_mm/guards_primary_final.json
```

- **Fallback forms.** The I1 form of the primary line uses `--stage2 artifacts/stage2_primary_i1.json`
  (`--out …/sub_primary_i1.csv --suffix _primary_i1`; `primary_i1.sh` runs this call through `eda/c121_stage2_i1.py submit`,
  which gives the same file), the guard form `--stage2 artifacts/stage2_s035.json` (`--out …/sub_s035_stage2_guard.csv
  --suffix _s035guard`), the main form without the NONE guard `--stage2 artifacts/stage2_s034.json` (`--out
  …/sub_s034n_stage2_main.csv --suffix _s034n`), the fallback form `--stage2 artifacts/stage2_s032.json`. The strictly
  inductive form of the safety line replaces `--align 0.75` by `--align-inductive 0.75` and uses `--stage2
  artifacts/stage2_s036_safe.json` (`--out …/sub_s036_safe_stage2.csv --suffix _ia_t4_s036`); without second stage it omits
  `--stage2` (`--out …/sub_s030n_safe.csv --suffix _ia_t4_s030n`).
- **Second-stage files.** `stage2_s032.json` and `stage2_s034.json` contain identical weights and differ only in the rule
  of the note guard: with the original rule the guard selects the fallback form, with the widened rule the main form.
  `stage2_s035.json` equals `stage2_s034.json` except for the main model, which uses the 48 features of `main_guard` and
  the NONE guard; `stage2_primary_i1.json` equals `stage2_s035.json` except for the main model (`main_guard_i1`, 63
  features), and `stage2_primary_final.json` equals `stage2_primary_i1.json` except for the main model (`main_guard_i1_mm`,
  64 features) and the field `mech27`, which names the frozen test answers of the protocol channel with their SHA-256.
  `stage2_s036_safe.json` is the second stage of the strictly inductive form: it names `align_mode: inductive`, so
  `make_submission` requires `--align-inductive 0.75 --align-min 3` and uses the fallback feature set without notes.
  `stage2_safety_final.json` names `align_mode: set`, so `make_submission` requires `--align 0.75 --align-min 3` and runs the
  same path without notes, with the feature set `fallback_i1`
  ([METHOD.md](METHOD.md#9-second-stage-conditional-logit-reranker)).
- **First-version list.** The historical files use `--test v1r=v1rot_t4/rot_test/test@derived/orient/test_c80_rows.npy`
  with otherwise the same arguments and their own output names.

## 6. Modes `rescore` and `full`

### 6.1 Test-side steps

Every model step is a configuration in `configs/vlm/`, run as described in [TRAINING.md](TRAINING.md#2-environment)
(`python -u -m vlm.runner --config … --out …`). In `rescore` only the scoring and ranking jobs run, behind the frozen
texts, notes and gate lists. In `full` the generation jobs run as well.

| Step | Channel | Configuration (job) or script | Reference hardware | Lines |
|---|---|---|---|---|
| Blank photos | all | frozen list `data/derived/blank/test_rows.npy` (its script is not part of this repository) | CPU | both |
| Rotation lists | `v1r` | `python -I tools/orient_net.py infer --from-images --tables test --weights artifacts/orient/orient_cnn.pt --out data/derived/orient_repro` | CPU | both |
| 9B scorer | `v1` | `v1te_a_t4`, `v1te_b_t4` (two shards of the test set) | Kaggle T4×2, fp16 | both |
| Rotation-corrected 9B | `v1r` | `v1rot_t4` (`rot_test`) | Kaggle T4×2, fp16 | both |
| 27B zero-shot, task block | `zs27q` | `zs27q_test0_rp`, `zs27q_test1_rp` | A40 48 GB, 4-bit | primary |
| 27B zero-shot, topic block | `zs27t` | `px27_t4zt0`, `px27_t4zt1` | Kaggle T4×2, 4-bit | inductive |
| Gates | reasoning | frozen gate lists `data/derived/think_rows/` (checked with `python -I tools/replay/gates.py check --table test --line <line>` where the A40 outputs are available, see below) | CPU | both |
| Reasoning draw g | `think18nb` | `zs27v_te_rp` (`think_te`) | RTX PRO 6000 96 GB | primary |
| Reasoning draw n | `think18nb`, `think18sb` | `zs27v_n3_rp` (`think_nte`) | RTX PRO 6000 96 GB | both |
| Protocol answers | `pr27xm` | `pr27te_rp` (`proto_te`) | RTX PRO 6000 96 GB | both |
| Protocol notes | `pr27xm` | frozen notes `data/derived/proto_notes_test/test_pr27.jsonl` (exported from the answers; the export script is not part of this repository) | CPU | both |
| Exact ranking behind the notes | `pr27xm` | `n1_rp` (`rank_test`) | RTX PRO 6000 96 GB | both |
| M3 merge and draw d1 | `m3x3_full` | `m3_fullte_rp` | RTX PRO 6000 96 GB | both |
| M3 draws d2 and d3 | `m3x3_full` | `m3d2te_rp`, `m3d3te_rp` | RTX PRO 6000 96 GB | both |
| Frozen M3 texts | `m3x3_full`, second stage | `tools/freeze_m3_texts.py` | CPU | both |
| 397B protocol notes | `q397m3` | `q397te_rp` (`proto_te`) | 4×H200, tensor parallel 4, FP8 | primary |
| 397B parity check (64 training rows) | `q397m3` | `q397p_rp`; the stored verdict `eda/private/q397_deploy/c109_parity.json` is read by the build scripts | 4×H200 | primary |
| Notes for M3 | `q397m3` | `tools/make_q397m3_notes.py` | CPU | primary |
| M3 behind the 397B notes | `q397m3` | `q397m3te_rp` | RTX PRO 6000 96 GB | primary |

- **Rotation lists.** `--weights` must be given exactly as `artifacts/orient/orient_cnn.pt`: the network writes this path
  into the meta file of the list, and `tools/orient_list.py check` requires it there.
- **Gates.** The frozen gate lists were computed from the A40 versions of the 9B and `zs27t` test outputs (`v1_test_rp`,
  `zs27t_all_rp`), while the submitted files use the offline Kaggle T4 versions of these two channels. Recomputing the
  gates from the T4 outputs therefore shifts some rows between gates. In `rescore` the gate lists are taken as frozen
  inputs. `gates.py check` confirms them only where the A40 outputs are available; these two runs are not part of the
  private dataset.

### 6.2 Stage order

> **Planned.** A stage-wise runner for the three modes.

| Stage | Content | Rule |
|---|---|---|
| S0 | Pre-checks: GPU, driver, disk, network level (with a positive control), checksums of all package files | Any deviation stops the run |
| S1 | Environments | Installed versions equal the pins of section 4 |
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
- **CPU fits of the assembly** (blend weights, second stage) are bit-identical on the reference CPU class with two BLAS
  threads; on another CPU class the last bits of fitted weights, and with them single ranks, can differ.

## 7. Run times, cost and package size

**Assembly on the CPU** (two BLAS threads, measured in a fresh copy of the repository with the frozen inputs, from which
every file the build paths write had been deleted): the two historical builds take about 1.3 minutes (strictly inductive
line) and 5.6 minutes (primary line, main form, including the fallback form and the file without second stage); the guard
form of the primary line, including the chain of its released-list builds, about 8 minutes; the strictly inductive form
with second stage, including its released-list build, about 2 minutes. The build scripts added in version 0.2 run their
predecessors again and were measured with a second copy of the repository running in parallel: `primary_i1.sh` about 18
minutes and `primary_final.sh` about 20 minutes, each including 4–7 minutes for its 20 meta-splits when these are absent;
the probe run of variant b of the safety line about 9 minutes (including its meta-splits) and `safety_final.sh` with the
override about 5 minutes. With the complete private package (meta-splits and deployment files present, so the scripts
verify them), the README quick start took about 32 minutes.

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
second stages were added and will be updated.
- `rescore` of both lines: about 9.4–10.6 h on an RTX PRO 6000, 3.5 h on an A40 and 25 T4 hours, about 25–29 USD.
- `full` of both lines: about 29.7 h on an RTX PRO 6000, 3.5 h on an A40 and 25 T4 hours, about 76–79 USD.
- The 397B steps add about 27–42 USD.

**Package size.** About 84–87 GB without the 397B; about 490 GB with it. The 397B FP8 weights alone take about 406 GB
(378 GiB). The base models are referenced by pinned revision and file checksums; M3 is shipped as an adapter (300.5 MB)
and merged at run time.

## 8. Operating rules

- **Full test set.** The alignment over the test set is defined over the complete test set (3,498 rows). The final
  candidates of both lines are never assembled from a subset of rows; only the strictly inductive form of the safety line
  processes each row on its own.
- **Override only by the recorded decision.** The safety candidate is built only with `OVERRIDE_CV_II=override_cv_condition_ii`
  and only while `decisions/override_cv_condition_ii.json` matches the checksum pinned in `eda/c123_safety_set.py`.
- **Fixed draws.** M3 uses exactly three draws with fixed settings. Each test draw must match the generation settings of
  its OOF draw; the checks reject any mismatch.
- **Fixed rotation list.** The released-list builds reject the first-version list; the first-version list is used only to
  rebuild the historical files.
- **Aggregates only.** Model outputs for test rows are written to files and never printed. Logs contain counts, shares
  and checksums.
- **Fault bands are not tuning knobs.** The bands for the share of `NONE` at the top and for the rate of changed top codes
  detect technical faults. They are fixed in advance and never used to choose between variants.
