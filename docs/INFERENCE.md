# Inference

Reference document. To reproduce the submissions, start with [VERIFICATION.md](VERIFICATION.md).

*Status: 2026-11-21 (version 0.6). Parts that are not built are marked **Not built**; the open steps are listed in
[ROADMAP.md](ROADMAP.md).*

The final solution is built to run **without network access and without any human step**. No external API is called at
inference time. All models in the inference path are open-weight models. The base models are pinned by revision and
file checksum, and our trained adapters are part of the delivered package. The CPU assembly and the 9B scorer on Kaggle
T4 have run without network access; the other GPU model steps have not yet ([ROADMAP.md](ROADMAP.md), step 2).

**Contents:**
1. [Three modes](#1-three-modes)
2. [Lines and expected checksums](#2-lines-and-expected-checksums)
3. [Data layout](#3-data-layout)
4. [Environments](#4-environments)
5. [Mode `assemble`](#5-mode-assemble)
6. [Modes `rescore` and `full`](#6-modes-rescore-and-full)
7. [Run times, cost and package size](#7-run-times-cost-and-package-size)
8. [Operating rules](#8-operating-rules)
9. [Running on a new test set](#9-running-on-a-new-test-set)
10. [Recomputing the 397B channel on the test rows](#10-recomputing-the-397b-channel-on-the-test-rows)

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

**The two final submissions (version 0.4).** Both come from the primary line. Each build script runs the build of its
predecessor as its step 0 and checks that predecessor's files byte for byte, so `primary_m3x5h.sh` builds all three files
below and the whole earlier chain.

| Submission | Build script | Output file in `artifacts/submissions/` | Bytes | SHA-256 of the file | SHA-256 of the test matrix |
|---|---|---|---:|---|---|
| **Primary submission A**: five M3 draws (draw h with the photo at 2,048 px on the rows with small script), nine extra second-stage columns | `bash tools/builds/primary_m3x5h.sh` | `sub_primary_m3x5h.csv` | 115,293 | `4505c15950167650a66cdc60eb9c153eebfb9ed6192277bd283045f3797354e7` | `artifacts/test_primary_m3x5h_m3x5h.npy`: `12c72dccc5995a90f6f406355ec7526a87a8c742dd5402945f003a69c073424a` |
| Intermediate form (submitted, not a final slot): four M3 draws, nine extra second-stage columns | `bash tools/builds/primary_cols.sh` | `sub_primary_cols.csv` | 115,268 | `53ec296976834959caea500217aa8e01904fbdc8be40a0542f63a462e4251926` | `artifacts/test_primary_m3x4_cols.npy`: `84cf58aac011166aa5c242f94d1b6c19a5e7f7d6892d60157185cdfb1148ec95` |
| **Primary submission B**: four M3 draws, no extra columns | `bash tools/builds/primary_m3x4.sh` | `sub_primary_m3x4.csv` | 115,259 | `76da786dfdbe6a6ddcbcc93070f385e18870cb5cdaea895d92104aa3565f9973` | `artifacts/test_primary_m3x4_m3x4.npy`: `30f74c8f72746f34cc423df7218bed60e7cf9b121b504a27e34606dbb3c97504` |

**Chain.** `primary_m3x5h.sh` → step 0 `primary_cols.sh` → step 0 `primary_m3x4.sh` → step 0 `primary_m3px27.sh` (the
candidate of version 0.3) → `primary_final.sh` → `primary_i1.sh` → `s035_stage2_guard.sh` → `s034n_stage2_main.sh` →
`s034_stage2_main.sh best` → `s032_stage2.sh best` → `s029_q397m3.sh best`. The environment variables
`PRIMARY_M3X5H_REGRESSION=0`, `PRIMARY_COLS_REGRESSION=0` and `PRIMARY_M3X4_REGRESSION=0` skip the step 0 of the respective
script and only check the checksums of the predecessor's files (for debugging; a deployment build runs step 0).

**What is pinned.** The three output files and their test matrices are pinned to the values of the submitted files above,
as are all content files the scripts read: the frozen channel outputs, the row lists, the digests over the frozen column
sources and over the files of draw h, and the OOF matrices `oof_m3x4_full.npy` and `oof_m3x5h.npy`. The JSON files that the
version-0.4 code writes itself carry English descriptive fields and the renamed keys of this repository; their numbers are
those of the files from which the submitted files were built (a field-by-field comparison shows no difference in any
non-text value), their checksums are not. They were learned on the first build with the code of this repository
(2026-11-17), when the scripts printed each of them as "FIRST BUILD — checksum …", and are pinned since; a second build in
a fresh copy of the repository from the private input package reproduced every file byte for byte
([ROADMAP.md](ROADMAP.md)). The private input package contains these files, so that the scripts verify them; without them
the scripts write them and require the same checksums.

| File (under `artifacts/`) | Written by | SHA-256 |
|---|---|---|
| `oof_m3x4_full.npy` | `eda/c135_primary_m3x4.py oof` (pinned since version 0.4) | `96ad8725bbaa9d274376a0bb68549abc4051878171ebb572f7707962600399a6` |
| `meta_m3x4_full.json` | `eda/c135_primary_m3x4.py oof` | `acd88b8744493b364ff14142eae1f9e59ca9351ec917e52bc607cc653c48be00` |
| `stack_primary_m3x4.json` | `eda/c135_primary_m3x4.py stack` | `1e90852a68f53efd6dfd0a51730d6035e78345dafc5c0730b59a976de369420f` |
| `stage2_primary_m3x4.json` | `eda/c135_primary_m3x4.py fit` | `7259ce1446a85aa700448f6c283170e119c414b24b0aea7bd243d67b6090c58a` |
| `eda/private/primary_m3x4/sens/stage2_primary_m3x4_fixbeta.json` | `eda/c135_primary_m3x4.py fit` (sensitivity with fixed β) | `3350e3ab2206333c7f6b7bba378a102aae96865d5f2ff10f18de4347eec0d4d3` |
| `stage2_primary_cols.json` | `eda/c151_primary_cols.py fit` | `102f98ce9c50c955753305a1795b808136d098c82c5fa1d25a0fde6f3e988a7f` |
| `oof_m3x5h.npy` | `eda/c166_primary_m3x5h.py oof` (pinned since version 0.4) | `388546e508a603fc993bc52faf887e3831895cd57513887c3d21aed7bf78cc2f` |
| `meta_m3x5h.json` | `eda/c166_primary_m3x5h.py oof` | `a2979b02d58572252575c202cd759d0f80b003824ed421d09945ce98ab9188f3` |
| `stack_primary_m3x5h.json` | `eda/c166_primary_m3x5h.py stack` | `50b46374e9703131301bb771bf46c967a8f9b3b02fdf994b8e7f9bbb1545ed66` |
| `stage2_primary_m3x5h_base.json` | `eda/c166_primary_m3x5h.py fit-base` | `4e3ad9d8c6310701ac214793c5cdd172e00f9828607567bbe376843d2a32030d` |
| `stage2_primary_m3x5h.json` | `eda/c166_primary_m3x5h.py fit` | `572ff297dc04b7d0e4a9a28b477af74f532bcd742371717d10e172daad894d05` |

**Earlier candidates and fallbacks** (versions 0.1–0.3). All of them use the rotation list of the released orientation
network (`ORIENT=repro`, the default of the build scripts; see below).

| Line and form | Build script | Output file in `artifacts/submissions/` | Bytes | SHA-256 |
|---|---|---|---:|---|
| **Primary line, final form with the mean channel `m3px27xm` (M3 ranking behind the protocol notes)** (candidate) | `bash tools/builds/primary_m3px27.sh` | `sub_primary_m3px27.csv` | 115,234 | `23101e74aa530b9c280087442b15f72bdf25a0c4627f0207e92c648e83a2176a` |
| **Safety line, second stage with note mentions, example similarity and regex note columns** (candidate) | `bash tools/builds/safety_ndr.sh` | `sub_safety_ndr.csv` | 115,173 | `b8f17ce3e5fe1ed6dbf22febf9b153ec7c23870af123fa60d4342979d67270f6` |
| Primary line, final form with `pr27xm`: I1 form with the minor-code × mechanical-match column (fallback of the primary candidate; candidate of version 0.2) | `bash tools/builds/primary_final.sh` | `sub_primary_final.csv` | 115,282 | `d9c68c460c6da06d4975513911e11c4541d70d07b78f132566a3e43e7ad60b24` |
| Safety line, second stage with note mentions and example similarity (fallback of the safety candidate) | `bash tools/builds/safety_nd.sh` | `sub_safety_nd.csv` | 115,210 | `561a3c45ca3579763c44780b1f0cb009a6b75d5c730d4cbe42c4d11ce7e45025` |
| Safety line, alignment over the test set, second stage with interactions and NONE guard (candidate of version 0.2) | `OVERRIDE_CV_II=override_cv_condition_ii bash tools/builds/safety_final.sh` | `sub_safety_final.csv` | 115,500 | `736d87b3cb61494aeb6dcac29bc248a909138c6e85e3939f37a01180984e78f7` |
| Primary line, I1 form: guard form with row × candidate interactions | `bash tools/builds/primary_i1.sh` | `sub_primary_i1.csv` | 115,233 | `60b63d69bcdd5970cafaf4d5645440d8d375953871d6dd688ae5091b25256bee` |
| Strictly inductive line with second stage and NONE guard (fallback of the safety line under the strictest reading of the rules; candidate of version 0.1) | `bash tools/builds/s036_safe_stage2.sh` | `sub_s036_safe_stage2.csv` | 115,508 | `ff45c2f72d21fd9d6b227fc47bad10b4b4092cd79008734da917e2ba3bcefd13` |
| Primary line, guard form: reranker main form with NONE guard and general-code identifiers (candidate of version 0.1) | `bash tools/builds/s035_stage2_guard.sh` | `sub_s035_stage2_guard.csv` | 115,318 | `d4dce1aaa5cfbd1fd5668a9e7073a47734c0c5aacda7b7c2020255005a81f360` |
| Primary line, reranker main form | `bash tools/builds/s034n_stage2_main.sh` | `sub_s034n_stage2_main.csv` | 115,359 | `8b8892e474ace44048d6c02eba7622288599990f5f5f0814cbefaeb6bd9ce242` |
| Strictly inductive line without second stage | `bash tools/builds/s030n_safe.sh` | `sub_s030n_safe.csv` | 115,238 | `8cca7bbfe9b8c9dce47d6153d59712575bb7b8ad6dea8a286f7ef530df448b67` |
| Primary line, reranker fallback form | `bash tools/builds/s033n_stage2.sh` | `sub_s033n_stage2.csv` | 115,498 | `76aba64fbc431b9ad403a5dec9fb2cefdb7fe2e76260149c96176459c4cd8dfe` |
| Primary line without second stage (intermediate file) | built by `s033n_stage2.sh` | `sub_s032n_q397m3.csv` | 115,404 | `d1667f6d2fb118844b72ecd8abd85220754a709d468519b7a047a893bbea12f1` |

**Names of the files of version 0.4.** Role names as before: `primary_m3x4` (submission B: the four-draw M3 channel
`m3x4_full`), `primary_cols` (the intermediate form with the extra columns) and `primary_m3x5h` (submission A: the M3
channel `m3x5h` with the fifth draw on the listed rows); report folders `artifacts/eda/private/primary_m3x4/`,
`primary_cols/` and `primary_m3x5h/`. Channel and run names are unchanged.

**Names of the files of versions 0.2 and 0.3.** The build scripts added in these versions, their deployment files
(`artifacts/stage2_<name>.json`), their output files (`sub_<name>.csv`) and their report folders are named by role:
`primary_i1`, `primary_final`, `safety_final` (version 0.2) and `safety_nd`, `safety_ndr`, `primary_m3px27` (version 0.3;
report folders `safety_nd/`, `safety_ndr/`, `primary_m3px27/`, and `m3px27_eval/` for the stored evaluation of the channel
`m3px27`). The earlier scripts keep their historical build labels (`s029` … `s036`), because frozen artifacts and their
checksums refer to them.

**Override in the safety chain.** `safety_final.sh` needs `OVERRIDE_CV_II=override_cv_condition_ii`; `safety_nd.sh` sets it
when it runs `safety_final.sh` in its step 0. The cross-validation part of one condition of the pre-registered rule of the
`safety_final` form was missed by one sixth in 9,000; the decision to build the file anyway is recorded in [`decisions/override_cv_condition_ii.json`](../decisions/override_cv_condition_ii.json),
whose SHA-256 is pinned in `eda/c123_safety_set.py` ([METHOD.md](METHOD.md#99-second-stage-of-the-safety-line-with-alignment-over-the-test-set)).
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
- `primary_m3px27.sh` runs `primary_final.sh` (`PRIMARY_M3PX27_REGRESSION=full` runs `safety_ndr.sh` instead, which contains
  it; `PRIMARY_M3PX27_REGRESSION=0` only checks the checksums of the earlier files) and then checks the files of `safety_nd.sh`
  and `safety_ndr.sh` by their checksums.
- `safety_ndr.sh` runs `safety_nd.sh`, which runs `safety_final.sh` (with the override) and `primary_final.sh`
  (`SAFETY_NDR_NO_REGRESSION=1` and `SAFETY_ND_NO_REGRESSION=1` skip this step 0 and are meant for debugging only).
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
  `safety_final.sh`), of the guard form and of the I1 form (by `primary_i1.sh` and `primary_final.sh`), and those of the
  three version-0.3 files and of their predecessors (by `safety_ndr.sh` and `primary_m3px27.sh`; `safety_nd.sh` compares the
  test matrix of its copy null point with that of `safety_final`).

**One command.** `bash reproduce.sh <package-dir>` checks the private package against its `MANIFEST.sha256`, sets up
`artifacts/` and `data/derived/`, runs the two `make_submission` calls of submissions B and A (§5) and compares both files
with the checksums above. With `--full` it also runs the build chain of §5. Details:
[VERIFICATION.md §5.1](VERIFICATION.md#51-assemble-cpu).

> **Not built.** An entry point that reads a line definition (sources, floors, last step, expected checksums) and calls
> `tools/make_submission.py` for any line without the chain of earlier builds. The build scripts stay in the repository as
> the record of how each file was produced.

## 3. Data layout

```
data/                                   K12_DATA may point to another folder with the same layout
├── train.csv  test.csv  unlabelled.csv          competition data (root CSV files)
├── competition_data/
│   ├── taxonomy_v1.csv  sample_submission.csv
│   └── images/train/  images/test/  images/questions/  (images/unlabelled/ only for training the orientation network)
└── derived/                                     private: frozen derived inputs
    ├── blank/            blank-photo list
    ├── charsize/         row lists "character height ≤ 20 px" of draw h (submission A)
    ├── orient/           first-version rotation lists (c50: photos turned upright; c80: rows using the upright reading)
    ├── orient_repro/     rotation lists of the released orientation network (test_c80_rows.npy, test_c80.json, .meta.json)
    ├── think_rows/       gate lists of the reasoning channels
    ├── sft/              frozen M3 test texts and training notes (draws d1–d3), 397B notes for M3, M3 training targets
    ├── sft_k2/           frozen M3 test texts of draw d4
    ├── sft_q397m3/  sft_q397m3te/   copies of the 397B notes for M3 as read by the model runs (primary line)
    ├── sft_m3px27/  sft_m3px27te/   protocol notes in the M3 format, behind which M3 ranks for `m3px27xm` (primary line)
    ├── open_teacher/     frozen 397B answers on the training rows (primary line); row list of the 397B parity probe
    ├── proto_rows/       row lists of the protocol-answer runs on the training rows (model runs only)
    ├── m3x3_full_rows.npy  training rows with all three M3 draws (read only; s029_m3x3.sh writes it only when
    │                       artifacts/meta_m3x3_full.json is absent)
    ├── proto_notes/  proto_notes_test/   frozen protocol notes of the training rows (two draws) and of the test rows
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

Folders marked *model runs only* are read by `rescore` and `full`, not by `assemble`. Since version 0.3 the protocol notes are
also read by `assemble`: by the regex note columns of the safety candidate and by the checks of the primary candidate.

**Inputs per line** (in addition to the competition CSV files and `taxonomy_v1.csv`). The list was taken from a trace of
every file the build scripts open in a fresh copy of the repository with the frozen inputs; it covers the assembly and
all checks and fits that the scripts run on the way. "Safety line" covers both of its forms (the strictly inductive form
and the final candidate with the alignment over the test set); they read the same frozen inputs. The private input package
holds every file that the build of both final submissions reads (traced file by file). It also holds the files that the
build scripts pin, the frozen inputs of the model steps, the inputs of the training side and of the checks in
[VERIFICATION.md §5.2](VERIFICATION.md#52-rescore-and-full-gpu), the M3 training targets and our weights. Its
`PACKAGE_INDEX.tsv` lists every file with size, SHA-256, role and whether it is derived from test rows;
[VERIFICATION.md §9](VERIFICATION.md#9-private-weights-and-inputs) gives the counts per role.

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
| `data/derived/sft/m3_d{1,2,3}_test_frozen_notes.jsonl` and their meta files (frozen M3 texts, read by the second stage) | yes (forms with note features) | yes |
| `data/derived/sft/m3_d{1,2,3}_train_frozen_notes.jsonl` and their meta files (frozen M3 training notes: reference of the example similarity; rebuilt from the training-side rankings for the check) | yes (forms with note features) | — |
| Frozen protocol notes: `data/derived/proto_notes/train_pr27.jsonl`, `train_pr27d2.jsonl`, `data/derived/proto_notes_test/test_pr27.jsonl` and their meta files; for their re-export check the raw answers `artifacts/runs/pr27_rp/pr27_rp/proto_a/`, `proto_b/`, `pr27s2_rp/pr27s2_rp/proto_a2/`, `pr27s3_rp/pr27s3_rp/proto_b2/`, `pr27te_rp/pr27te_rp/proto_te/` | yes (final candidate) | `train_pr27.jsonl`, `test_pr27.jsonl` (checksums, final candidate) |
| Fourth M3 draw on the training rows `artifacts/runs/m3d4_oof_rp/` (draw swap of the rule; read only when the draw-swap file `eda/private/safety_ndr/c130_swap.json` is absent and recomputed) | yes (final candidate) | — |
| Channel `m3px27xm`: `data/derived/sft_m3px27/`, `data/derived/sft_m3px27te/`, `artifacts/runs/m3px27_oof_rp/` (five shards), `artifacts/runs/m3px27te_rp/` (test ranking), `artifacts/oof_m3px27.npy`, `meta_m3px27.json`, the stored evaluation `eda/private/m3px27_eval/c132_m3px27_eval.json` and `…_rows.npz` | — | yes (final candidate) |
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

**Additional inputs of the final submissions (version 0.4)**, on top of the primary line above (producer and network
status: [PROVENANCE.md](PROVENANCE.md); every file with size and SHA-256: `PACKAGE_INDEX.tsv` of the private input
package):

| Input | Submission B (`primary_m3x4`) | Intermediate (`primary_cols`) | Submission A (`primary_m3x5h`) |
|---|---|---|---|
| Fourth M3 draw: `artifacts/runs/m3d4_oof_rp/` (training rows, five folds), `artifacts/runs/m3d4te_rp/` (test rows), frozen test texts `data/derived/sft_k2/m3_d4_test_frozen_notes.jsonl` | yes | yes | yes |
| `data/derived/m3x3_full_rows.npy` (rows with all M3 draws; all 1,500) | yes | yes | yes |
| Column sources (training side): protocol answers `artifacts/runs/pr27_rp/pr27_rp/proto_a/`, `proto_b/` (`proto_raw.jsonl`, `proto_labels.jsonl`, `proto_meta.json`), 397B answers through the hosted API `artifacts/runs/q397_tinker/q397_tinker/proto_t/proto_labels.jsonl`, the frozen notes `data/derived/proto_notes/train_pr27.jsonl` and `data/derived/sft_q397m3/q397m3_train_notes.jsonl` (made from `data/derived/open_teacher/qwen35-397b/raw.jsonl` by `tools/make_q397m3_notes.py`; the build reads the notes, not `raw.jsonl`) and their meta files | — | yes | yes |
| Column sources (test side, pinned by SHA-256 in the deployment file): `artifacts/runs/pr27te_rp/pr27te_rp/proto_te/` and `artifacts/runs/q397te_rp/q397te_rp/proto_te/` (answers and parsed answers), `data/derived/proto_notes_test/test_pr27.jsonl`, `data/derived/sft/q397m3_test_notes.jsonl` and their meta files | — | yes | yes |
| Draw h (photo at 2,048 px): `artifacts/runs/m3ha_oof_rp/`, `m3hb_oof_rp/` (training rows), `artifacts/runs/m3hte_a_rp/`, `m3hte_b_rp/` (test rows, two shards) | — | — | yes |
| Row lists "character height ≤ 20 px": `data/derived/charsize/train_h20_rows.npy`, `test_h20_rows.npy`, `charsize_h20.meta.json` (`tools/make_charsize_rows.py`) | — | — | yes |
| Files of the predecessor (CSV, test matrix, blend and second-stage file; checked by checksum) | `primary_m3px27` | `primary_m3x4` | `primary_cols`, `primary_m3x4` |

The files that the build scripts write and pin themselves are not listed: blend and second-stage files, the meta-split
files and the probe files of the safety line. The private package contains them, so that the scripts verify them instead
of recomputing them; without them the scripts recompute them and require the pinned checksums
([TRAINING.md](TRAINING.md#13-second-stages)). Steps 2–4 of `primary_m3px27.sh` (mean channel, blend, second stage)
always recompute; files that are already present must be byte-identical to the recomputation.

The historical six-channel file (`s029_q397.sh`) additionally reads `submissions/sub_cv01_best_l2.csv`; the historical
five-channel file (`ORIENT=legacy s029_m3x3.sh best`) reads `stack_v1r_zs27q_think18nb_pr27xm_w14.json` and
`submissions/sub_cv01_best_l2.csv`.

The build scripts of the safety line also read the OOF matrices of the teacher-block channels `zs27q` and `think18nb`,
but only to verify their checksums (a shared loading function checks all base channels); these channels enter neither
the blend nor the test side of the safety line.

The competition data come from the competition page. All other inputs are private. They are delivered to the competition
host for verification in the private input package (download link in our email;
[VERIFICATION.md](VERIFICATION.md#9-private-weights-and-inputs), [DATA_AND_COMPLIANCE.md](DATA_AND_COMPLIANCE.md)): the
frozen derived inputs, the channel outputs, the OOF matrices with their metadata, the stored reports, the earlier
submission files used as references, the blend and second-stage files, our trained adapters and the orientation network
(`weights/`). Every file is referenced by relative path, the pinned files also by SHA-256
([VALIDATION.md](VALIDATION.md#6-checks-before-a-submission)).

Machine-readable lists: `PACKAGE_INDEX.tsv` and `MANIFEST.sha256` in the private package (every file with size, SHA-256
and role), [PROVENANCE.md](PROVENANCE.md) and [weights_manifest.json](weights_manifest.json).

## 4. Environments

The main pins are listed here. The complete pin set of each environment is in `env/requirements-cpu.txt`,
`env/requirements-gpu.txt`, `env/requirements-vllm.txt` and `env/requirements-kaggle-t4.txt` (each with its Python and CUDA
version). Lock files with hashes for every pin and an offline wheelhouse are **not built** ([ROADMAP.md](ROADMAP.md),
step 4).

| Environment | Main pins |
|---|---|
| Assembly on the CPU | Python 3.13.5, numpy 2.4.2, scipy 1.17.1, pandas 2.3.3, scikit-learn 1.8.0, pyarrow 23.0.1 (fold table), PyYAML 6.0.3 (run configurations), Pillow 11.3.0, torch 2.11.0 and transformers 5.15.1 (imported by shared model helpers; everything runs on the CPU); pytest 8.4.2 for the tests |
| GPU path: training, merge, exact scoring | Python 3.12, torch 2.8.0+cu128, transformers 5.15.1, peft 0.21.2, accelerate 1.15.0, flash-linear-attention 0.5.2, bitsandbytes 0.50.2 |
| Text generation (separate virtual environment) | vllm 0.31.0, transformers 5.17.0, torch 2.13.0, flashinfer-python 0.7.0.post1 |
| Kaggle T4 path | accelerate 1.15.0, bitsandbytes 0.50.2, peft 0.21.2, transformers 5.15.1 on top of the Kaggle GPU image |

**Settings for runs without network access:** `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `HF_DATASETS_OFFLINE=1`,
`HF_HUB_DISABLE_TELEMETRY=1`, `VLLM_NO_USAGE_STATS=1`, `DO_NOT_TRACK=1`, `VLLM_DO_NOT_TRACK=1`,
`VLLM_HOST_IP=127.0.0.1`, `GLOO_SOCKET_IFNAME=lo` and `NCCL_SOCKET_IFNAME=lo`. Without `VLLM_HOST_IP`, vLLM looks up its own
address with a UDP connect to a public address. The vLLM server is bound to 127.0.0.1. Install the environments and
download the base models before the network is cut. With `PIP_NO_INDEX=1` and `UV_OFFLINE=1`, an accidental install
attempt fails at once.

**GPU base image.** Remove the `torchao` package of the base image ([TRAINING.md](TRAINING.md#2-environment)) before
installing the pins: `python -m pip uninstall -y torchao`. Our job script did this, because the older version in the
image conflicts with the pinned packages.

**Kaggle T4 runs.** The configurations `v1te_a_t4`, `v1te_b_t4` and `v1rot_t4` name private Kaggle resources of our
account in the keys `kernel_sources` and `dataset_sources`: notebook outputs with the 9B weights and the offline wheels,
and datasets with the adapters and the frozen inputs. `px27_t4zt0` and `px27_t4zt1` name the notebook output with the
wheels and read the 27B weights from a public mirror ([THIRD_PARTY.md](../THIRD_PARTY.md)). These keys only describe our
Kaggle notebook runs. The runner does not read them, so on other GPUs these steps run with the runner like the others.

## 5. Mode `assemble`

There are two ways to rebuild the files on the CPU. Both are run by `reproduce.sh`
([VERIFICATION.md §5.1](VERIFICATION.md#51-assemble-cpu)):
- **Quick check** (`bash reproduce.sh <package-dir>`, about two minutes): the two `make_submission` calls of submissions
  B and A shown below. They read the frozen channel outputs and the delivered blend and second-stage files and refit
  nothing.
- **Build chain** (`bash reproduce.sh <package-dir> --full`, about 2 h): the six commands of the following block.
  They refit or verify every blend and second-stage file and rebuild every predecessor in the chain of A and B.

```bash
# 1. Python 3.13 environment with the versions of section 4 (env/requirements-cpu.txt)
# 2. Competition data under data/, private inputs under data/derived/ and artifacts/ (section 3)
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2
export OPENBLAS_CORETYPE=Haswell   # OpenBLAS kernel of the reference CPU (section 6.3)
python -I tools/make_folds.py --check || python -I tools/make_folds.py   # artifacts/folds.npy from configs/folds.parquet

# 3. Historical files with the first-version rotation list (references of the released-list builds)
ORIENT=legacy bash tools/builds/s029_m3x3.sh safe          # strictly inductive line
ORIENT=legacy bash tools/builds/s034_stage2_main.sh best   # primary line: file without second stage, fallback form, main form

# 4. Candidates of version 0.3 and their fallbacks (released rotation list); the primary chain checks their files
bash tools/builds/safety_ndr.sh       # safety line; builds sub_s030n_safe, sub_s036…, sub_safety_final, the primary chain up to
                                      #   sub_primary_final, and sub_safety_nd on the way
bash tools/builds/primary_m3px27.sh   # primary line; runs primary_final.sh again and checks the files of the safety chain
sha256sum artifacts/submissions/sub_primary_m3px27.csv artifacts/submissions/sub_safety_ndr.csv

# 5. The two final submissions (version 0.4); primary_m3x5h.sh runs primary_cols.sh, which runs primary_m3x4.sh,
#    which runs primary_m3px27.sh (step 4 is therefore contained in this step)
bash tools/builds/primary_m3x5h.sh
sha256sum artifacts/submissions/sub_primary_m3x5h.csv artifacts/submissions/sub_primary_m3x4.csv
#   expected: 4505c15950167650…  sub_primary_m3x5h.csv (primary submission A)
#             76da786dfdbe6a6d…  sub_primary_m3x4.csv  (primary submission B)
```

Step 5 alone needs the safety chain only through the checksums of its files (`primary_m3px27.sh` checks them); in a
package without these files run step 4 first, or run steps 3–5 in this order. Every build checks the pinned checksums of
the JSON files the scripts write (section 2) together with the CSV and test-matrix checksums above.

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
it byte for byte, and the scripts of versions 0.2 and 0.3 also show that the new columns with weight 0 reproduce the file of
the predecessor form (`primary_m3px27.sh`: the channel `m3px27xm` with the ranking of the untrained 27B in place of the M3
ranking reproduces the predecessor file). The scripts of versions 0.2 and 0.3 recompute their meta-split files (about 8
minutes each), the draw swap of `safety_ndr.sh`, and the blend and second-stage files of `primary_m3px27.sh` only if they are
absent; the private package contains them, and the scripts check them against their pinned checksums. The safety scripts of
version 0.3 also run positive controls of their test-side guards (changed test notes in memory: a guard must fire, no file is
written). Besides the submission files, the build scripts rewrite some of their own reports under
`artifacts/eda/private/` (new time stamps), and `s029_m3x3.sh` rewrites the two blend files of its line with identical
content (in both `ORIENT` modes); one test rewrites a stored report of the 397B deployment in the same way. Of these
files only the blend file of the inductive line, `stack_v1r_zs27t_think18sb_pr27xm_m3x3_full_ar.json`, is pinned (by
`s030n_safe.sh` and `s036_safe_stage2.sh`). A build that stops with an error can leave reports that describe the
failed attempt; the next successful run rewrites them.

The last `make_submission` call of each build script is shown below. The calls of submissions B and A are the quick
check of [VERIFICATION.md §5.1](VERIFICATION.md#51-assemble-cpu), there with the output names `check_B.csv` and
`check_A.csv`. They refit nothing, so they reproduce both files byte for byte with any OpenBLAS kernel (section 6.3). The
arguments name the frozen channel outputs:
- A source `<run>/<job>/<tag>` (after `--test`) or `runs/<run>/<job>/<tag>` (after `--proto`, `--rank`, `--rank-only`)
  resolves to `artifacts/runs/<run>/<run>/<job>/`.
- Several shards of one draw are joined with `+`, and several draws of one channel with `,`.
- Row lists after `@` and after `--blank-rows` are relative to the data folder.

```bash
# Safety line, final candidate (safety_ndr.sh, step 9)
python -I tools/make_submission.py --name v1r_zs27t_think18sb_pr27xm_m3x3_full_ar \
  --test v1r=v1rot_t4/rot_test/test@derived/orient_repro/test_c80_rows.npy --fill v1r=v1 \
  --test v1=v1te_a_t4/v1_test_a/test+v1te_b_t4/v1_test_b/test \
  --test zs27t=px27_t4zt0/zs27t_test_t4/test+px27_t4zt1/zs27t_test_t4/test \
  --fill think18sb=zs27t \
  --test think18sb=zs27v_n3_rp/think_nte/test/think@derived/think_rows/test_tb_t.npy \
  --proto pr27xm=runs/pr27te_rp/proto_te/proto --rank pr27xm=runs/n1_rp/rank_test/test \
  --rank-only m3x3_full=runs/m3_fullte_rp/m3_test_rank/test,runs/m3d2te_rp/m3d2_test_rank/test,runs/m3d3te_rp/m3d3_test_rank/test \
  --align 0.75 --align-min 3 --blank-rows derived/blank/test_rows.npy \
  --stage2 artifacts/stage2_safety_ndr.json --stage2-report artifacts/eda/private/safety_ndr/guards_safety_ndr.json \
  --out artifacts/submissions/sub_safety_ndr.csv --suffix _safety_ndr

# Primary line, final candidate (primary_m3px27.sh, step 9)
python -I tools/make_submission.py --name primary_m3px27 \
  --test v1r=v1rot_t4/rot_test/test@derived/orient_repro/test_c80_rows.npy --fill v1r=v1 \
  --test v1=v1te_a_t4/v1_test_a/test+v1te_b_t4/v1_test_b/test \
  --test zs27q=zs27q_test0_rp/zs27q_test0/test+zs27q_test1_rp/zs27q_test1/test \
  --fill think18nb=zs27q \
  --test think18nb=zs27v_te_rp/think_te/test/think@derived/think_rows/test.npy,zs27v_n3_rp/think_nte/test/think@derived/think_rows/test_tb_q.npy \
  --proto m3px27xm=runs/pr27te_rp/proto_te/proto --rank m3px27xm=runs/m3px27te_rp/m3px27_test_rank/test \
  --rank-only m3x3_full=runs/m3_fullte_rp/m3_test_rank/test,runs/m3d2te_rp/m3d2_test_rank/test,runs/m3d3te_rp/m3d3_test_rank/test \
  --rank-only q397m3=runs/q397m3te_rp/q397m3_test_rank/test --rank-none-band-for q397m3=0.27,0.35 \
  --align 0.75 --align-min 3 --blank-rows derived/blank/test_rows.npy \
  --out artifacts/submissions/sub_primary_m3px27.csv --suffix _primary_m3px27 \
  --stage2 artifacts/stage2_primary_m3px27.json --stage2-report artifacts/eda/private/primary_m3px27/guards_primary_m3px27.json
```

```bash
# Primary submission B (primary_m3x4.sh, step 9)
python -I tools/make_submission.py --name primary_m3x4 \
  --test v1r=v1rot_t4/rot_test/test@derived/orient_repro/test_c80_rows.npy --fill v1r=v1 \
  --test v1=v1te_a_t4/v1_test_a/test+v1te_b_t4/v1_test_b/test \
  --test zs27q=zs27q_test0_rp/zs27q_test0/test+zs27q_test1_rp/zs27q_test1/test \
  --fill think18nb=zs27q \
  --test think18nb=zs27v_te_rp/think_te/test/think@derived/think_rows/test.npy,zs27v_n3_rp/think_nte/test/think@derived/think_rows/test_tb_q.npy \
  --proto m3px27xm=runs/pr27te_rp/proto_te/proto --rank m3px27xm=runs/m3px27te_rp/m3px27_test_rank/test \
  --rank-only m3x4_full=runs/m3_fullte_rp/m3_test_rank/test,runs/m3d2te_rp/m3d2_test_rank/test,runs/m3d3te_rp/m3d3_test_rank/test,runs/m3d4te_rp/m3d4_test_rank/test \
  --rank-only q397m3=runs/q397m3te_rp/q397m3_test_rank/test --rank-none-band-for q397m3=0.27,0.35 \
  --align 0.75 --align-min 3 --blank-rows derived/blank/test_rows.npy \
  --out artifacts/submissions/sub_primary_m3x4.csv --suffix _m3x4 \
  --stage2 artifacts/stage2_primary_m3x4.json --stage2-report artifacts/eda/private/primary_m3x4/guards_primary_m3x4.json

# Primary submission A (primary_m3x5h.sh, step 14): the M3 channel m3x5h has five draws; the fifth consists of two shards
# and counts only on the rows of the test list (draw with row list, field rank.row_list_draws of meta_m3x5h.json)
python -I tools/make_submission.py --name primary_m3x5h \
  --test v1r=v1rot_t4/rot_test/test@derived/orient_repro/test_c80_rows.npy --fill v1r=v1 \
  --test v1=v1te_a_t4/v1_test_a/test+v1te_b_t4/v1_test_b/test \
  --test zs27q=zs27q_test0_rp/zs27q_test0/test+zs27q_test1_rp/zs27q_test1/test \
  --fill think18nb=zs27q \
  --test think18nb=zs27v_te_rp/think_te/test/think@derived/think_rows/test.npy,zs27v_n3_rp/think_nte/test/think@derived/think_rows/test_tb_q.npy \
  --proto m3px27xm=runs/pr27te_rp/proto_te/proto --rank m3px27xm=runs/m3px27te_rp/m3px27_test_rank/test \
  --rank-only m3x5h=runs/m3_fullte_rp/m3_test_rank/test,runs/m3d2te_rp/m3d2_test_rank/test,runs/m3d3te_rp/m3d3_test_rank/test,runs/m3d4te_rp/m3d4_test_rank/test,runs/m3hte_a_rp/m3h_test_rank/test+runs/m3hte_b_rp/m3h_test_rank/test@derived/charsize/test_h20_rows.npy \
  --rank-only q397m3=runs/q397m3te_rp/q397m3_test_rank/test --rank-none-band-for q397m3=0.27,0.35 \
  --align 0.75 --align-min 3 --blank-rows derived/blank/test_rows.npy \
  --out artifacts/submissions/sub_primary_m3x5h.csv --suffix _m3x5h \
  --stage2 artifacts/stage2_primary_m3x5h.json --stage2-report artifacts/eda/private/primary_m3x5h/guards_primary_m3x5h.json
```

The intermediate form uses the arguments of submission B with `--stage2 artifacts/stage2_primary_cols.json` (`--out
…/sub_primary_cols.csv --suffix _cols`, report `artifacts/eda/private/primary_cols/guards_primary_cols.json`). Its second
stage and that of submission A carry the field `extra_columns`: `make_submission` then reads the frozen test-side column
sources named there (files pinned by SHA-256), computes the nine columns and the test guard per column and source, and
writes no file if the guard fails. The second stages of both final submissions carry `draws_first: 3`: their features
read the first three M3 draws (d1–d3) as on the training rows, while the channel averages four or five draws.

- **Fallback forms.** The final form of the primary line with `pr27xm` uses `--name s032_stage2`, `--proto
  pr27xm=runs/pr27te_rp/proto_te/proto --rank pr27xm=runs/n1_rp/rank_test/test` instead of the two `m3px27xm` arguments and
  `--stage2 artifacts/stage2_primary_final.json` (`--out …/sub_primary_final.csv --suffix _primary_final`). The form of the
  safety line with note mentions and example similarity uses `--stage2 artifacts/stage2_safety_nd.json` (`--out
  …/sub_safety_nd.csv --suffix _safety_nd`), the form before it `--stage2 artifacts/stage2_safety_final.json` (`--out
  …/sub_safety_final.csv --suffix _safety_final`). The I1 form of the primary line uses `--stage2 artifacts/stage2_primary_i1.json`
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
  ([METHOD.md](METHOD.md#9-second-stage-conditional-logit-reranker)). `stage2_safety_nd.json` equals it in form and adds the
  feature set `fallback_i1_nd` (58 features), the frozen M3 test and training notes (`test_notes`, `train_notes`, files and
  SHA-256) and the reference statistics of the note and similarity guards; `stage2_safety_ndr.json` adds the feature set
  `fallback_i1_ndr` (64), the frozen protocol test notes (`r_notes`, file and SHA-256) and the reference of the guard of the
  columns R. `stage2_primary_m3px27.json` has the fields of `stage2_primary_final.json` with the channel `m3px27xm` in place of
  `pr27xm` (in the channel list and the feature names), its own blend file and refitted models, and the field `m3px27_build`.
- **First-version list.** The historical files use `--test v1r=v1rot_t4/rot_test/test@derived/orient/test_c80_rows.npy`
  with otherwise the same arguments and their own output names.

## 6. Modes `rescore` and `full`

### 6.1 Test-side steps

Every model step is a configuration in `configs/vlm/`, run as described in [TRAINING.md](TRAINING.md#2-environment)
(`python -u -m vlm.runner --config … --out …`). The runner runs every job of a file. `configs/vlm/rescore/` holds the
scoring and ranking jobs only (`rescore`). `configs/vlm/test_side/` holds the test-side jobs with generation (`full`).
Both keep the run and job names of the reference runs, so that the outputs land where the CPU tools read them
([VERIFICATION.md §5.2](VERIFICATION.md#52-rescore-and-full-gpu)). `tools/make_inference_configs.py` writes the two
folders from the reference configurations. `tools/gpu_inference.sh rescore|full <package-dir>` runs the model steps of
the table in the order of [VERIFICATION.md §5.2](VERIFICATION.md#52-rescore-and-full-gpu) and freezes the new M3 texts.
In `full` it also exports the new notes and re-records the checksums of the new answers. It then writes A and B with
the two calls of §5. `--dry-run` only prints the commands and checks that every configuration and input folder exists.
The CPU rows that recompute or check a row list (blank photos, rotation, gates, character height) are separate commands.

| Step | Channel | Configuration (job) or script | Reference hardware | Lines |
|---|---|---|---|---|
| Blank photos | all | frozen list `data/derived/blank/test_rows.npy`; `python -I tools/make_blank_list.py check` recomputes it from the photos and compares (`write --out data/derived/blank_repro` writes a new list) | CPU | both |
| Rotation lists | `v1r` | `python -I tools/orient_net.py infer --from-images --tables test --weights artifacts/orient/orient_cnn.pt --out data/derived/orient_repro` | CPU | both |
| 9B scorer | `v1` | `v1te_a_t4`, `v1te_b_t4` (two shards of the test set) | Kaggle T4×2, fp16 | both |
| Rotation-corrected 9B | `v1r` | `test_side/v1rot_t4` (`rot_test`) | Kaggle T4×2, fp16 | both |
| 27B zero-shot, task block | `zs27q` | `zs27q_test0_rp`, `zs27q_test1_rp` | A40 48 GB, 4-bit | primary |
| 27B zero-shot, topic block | `zs27t` | `px27_t4zt0`, `px27_t4zt1` | Kaggle T4×2, 4-bit | inductive |
| Gates | reasoning | frozen gate lists `data/derived/think_rows/`, checked with `python -I tools/replay/gates.py check --table test --line all --sources a40` (inputs `artifacts/oof_v1.npy` and the A40 test outputs in the private package, see below) | CPU | both |
| Reasoning draw g | `think18nb` | `test_side/zs27v_te_rp` (`gen_te`, `think_te`; `full`), `rescore/zs27v_te_rp` (`think_te`; `rescore`) | RTX PRO 6000 96 GB | primary |
| Reasoning draw n | `think18nb`, `think18sb` | `test_side/zs27v_n3_rp` (`gen_nte`, `think_nte`; `full`), `rescore/zs27v_n3_rp` (`think_nte`; `rescore`) | RTX PRO 6000 96 GB | both |
| Protocol answers | `pr27xm` | `pr27te_rp` (`proto_te`; `full` only) | RTX PRO 6000 96 GB | both |
| Protocol notes | `pr27xm`, `m3px27xm`, second stage of the safety candidate | frozen notes `data/derived/proto_notes_test/test_pr27.jsonl` (`python tools/export_proto_notes.py --name test_pr27 --from pr27te_rp/proto_te/proto --allow-test`) | CPU | both |
| Exact ranking behind the notes | `pr27xm` | `test_side/n1_rp` (`rank_test`) | RTX PRO 6000 96 GB | both (primary: fallbacks) |
| Notes for M3 behind the protocol notes | `m3px27xm` | `tools/make_m3px27_notes.py` (test table) → `data/derived/sft_m3px27te/` | CPU | primary |
| M3 behind the protocol notes | `m3px27xm` | `m3px27te_rp` (full-fit adapter of M3) | RTX PRO 6000 96 GB | primary |
| M3 merge and draw d1 | `m3x3_full` | `test_side/m3_fullte_rp` (`full`), `rescore/m3_fullte_rp` (`rescore`); `m3d1te_rp` computes a new draw d1 in a fresh folder | RTX PRO 6000 96 GB | both |
| M3 draws d2 and d3 | `m3x3_full` | `m3d2te_rp`, `m3d3te_rp` (`full`); `rescore/m3d2te_rp`, `rescore/m3d3te_rp` (`rescore`) | RTX PRO 6000 96 GB | both |
| M3 draw d4 | `m3x4_full`, `m3x5h` | `m3d4te_rp` (`full`), `rescore/m3d4te_rp` (`rescore`) | RTX PRO 6000 96 GB | final submissions |
| Row list "character height ≤ 20 px" | `m3x5h` | `python -I tools/make_charsize_rows.py --max-height 20 --name h20` (stored per-image statistics of `eda/i5_01_image_stats.py`, which are not in the package; no photo is read by this step) | CPU | submission A |
| M3 draw h (photo at 2,048 px, listed rows only) | `m3x5h` | `m3hte_a_rp`, `m3hte_b_rp` (two halves of the list; `full`), `rescore/m3hte_a_rp`, `rescore/m3hte_b_rp` (`rescore`) | RTX PRO 6000 96 GB | submission A |
| Frozen M3 texts | `m3x3_full`, second stage | `tools/freeze_m3_texts.py` (test rows); the frozen training notes of the safety candidate's similarity reference are part of the training side (`eda/c129_safety_nd.py export-notes`) | CPU | both |
| 397B protocol notes | `q397m3` | `q397te_rp` (`proto_te`; `full` only) | 4×H200, tensor parallel 4, FP8 | primary |
| 397B parity check (64 training rows) | `q397m3` | `q397p_rp`; the stored verdict `eda/private/q397_deploy/c109_parity.json` is read by the build scripts | 4×H200 | primary |
| Notes for M3 | `q397m3` | `tools/make_q397m3_notes.py --test-from runs/q397te_rp/proto_te/proto` (`full` only) | CPU | primary |
| M3 behind the 397B notes | `q397m3` | `q397m3te_rp` | RTX PRO 6000 96 GB | primary |

- **Rotation lists.** `--weights` must be given exactly as `artifacts/orient/orient_cnn.pt`: the network writes this path
  into the meta file of the list, and `tools/orient_list.py check` requires it there.
- **Gates.** The frozen gate lists were computed from the A40 versions of the 9B and `zs27t` test outputs (`v1_test_rp`,
  `zs27t_all_rp`), while the submitted files use the offline Kaggle T4 versions of these two channels. Recomputing the
  gates from the T4 outputs therefore shifts some rows between gates. In `rescore` the gate lists are taken as frozen
  inputs. `gates.py check --sources a40` confirms them bit for bit; these two runs are in the private package (role
  `check`).

### 6.2 Stage order

The table gives the stages and the check after each stage. `tools/gpu_inference.sh` runs the model steps in the order of
[VERIFICATION.md §5.2](VERIFICATION.md#52-rescore-and-full-gpu), not stage by stage.

> **Not built.** A runner that runs the stages and their checks in one go.

| Stage | Content | Rule |
|---|---|---|
| S0 | Pre-checks: GPU, driver, disk, network level (with a positive control), checksums of all package files | Any deviation stops the run |
| S1 | Environments | Installed versions equal the pins of section 4 |
| S2 | Blank-photo list; rotation lists from the stored orientation network | Byte-identical |
| S3 | Fast channels: 9B, rotation-corrected 9B, `zs27q` or `zs27t` | Byte-identical on the reference hardware |
| S4 | Gates | Byte-identical if S3 was |
| S5 | M3 adapter merge, once for all draws | Checksums of the merged shards |
| S6 | Texts: reasoning draws g and n, protocol answers, M3 draws d1–d4 (and draw h on the listed rows for submission A), 397B notes | Only aggregates against the reference (`full` only) |
| S7 | Export of the protocol notes and M3 texts | Byte-identical in `rescore` |
| S8 | Exact rankings | Reference hardware: identical |
| S9 | Assembly | `assemble`: checksum equals the submitted file |

When the steps of `full` are run by hand, run S5 and S6 first: they are the most fragile steps.

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
  | M3 draw d2 ranking on T4 | agreement 0.978 on 1,750 of 3,498 rows (the run was not completed) |
  | M3 draw d3 ranking, reasoning ranking | not run on a second hardware class |

- **Rerun of a generated text:** a rerun of the protocol answers changed the blend's top code on about 8–9 % of rows of
  an error-heavy sample, without a change in MAP@3.
- **CPU fits of the assembly** (blend weights, second stage) are bit-identical on the reference CPU (Intel Core
  i5-8250U, OpenBLAS kernel `Haswell`) with two BLAS threads. With another OpenBLAS kernel the refitted weights differ by
  about 1e-8, and the build scripts stop at their first refit check. The two calls of §5 do not refit and still reproduce
  both files. Details: [VERIFICATION.md §5.1](VERIFICATION.md#51-assemble-cpu).

## 7. Run times, cost and package size

**Final submissions, code of this repository.** The two direct calls of section 5 take about 30 s per file. The build
chain of section 5 (six commands; `bash reproduce.sh <package-dir> --full` runs them) took 1 h 48 min, 1 h 52 min and
1 h 59 min in three runs on an Intel Core i5-8250U laptop (4 cores, 8 threads, 15 GB RAM, two BLAS threads, shared with
other jobs). The measurements below (versions 0.2–0.4) are history.

**Assembly on the CPU, earlier versions** (two BLAS threads, measured in a fresh copy of the repository with the frozen
inputs, from which every file the build paths write had been deleted): the two historical builds take about 1.3 minutes
(strictly inductive line) and 5.6 minutes (primary line, main form, including the fallback form and the file without
second stage); the guard form of the primary line, including the chain of its released-list builds, about 8 minutes; the
strictly inductive form with second stage, including its released-list build, about 2 minutes. The build scripts added
in version 0.2 run their predecessors again and were measured with a second copy of the repository running in parallel:
`primary_i1.sh` about 18 minutes and `primary_final.sh` about 20 minutes, each including 4–7 minutes for its 20
meta-splits when these are absent; the probe run of variant b of the safety line about 9 minutes (including its
meta-splits) and `safety_final.sh` with the override about 5 minutes. With the complete private package (meta-splits and
deployment files present, so the scripts verify them), the README quick start took about 32 minutes. The build scripts
added in version 0.3 were measured in the same way on a machine shared with other jobs, on which the version-0.2 scripts
ran two to four times slower than above: `safety_nd.sh` about 42 minutes and `safety_ndr.sh` about 78 minutes, each
without its step 0 and including its meta-splits (and, for `safety_ndr.sh`, the draw swap) when these are absent, and
`primary_m3px27.sh` about 8 minutes without its step 0 (its steps 2–4 recompute the mean channel, the blend and the
second stage on every run). On that machine the README quick start of version 0.3 took about 3.3 hours.

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
| M3 behind the protocol notes (`m3px27`) | RTX PRO 6000 | 4.9 s per row, about 4.8 h | about 12 USD |

**Totals.** For the final submissions see [VERIFICATION.md §3](VERIFICATION.md#3-hardware-and-run-times) (`rescore`
roughly 25–30 h, `full` about 55 h for A on one 96 GB card). The following earlier estimates were made for the two lines
before the 397B channel and the second stages were added:
- `rescore` of both lines: about 9.4–10.6 h on an RTX PRO 6000, 3.5 h on an A40 and 25 T4 hours, about 25–29 USD.
- `full` of both lines: about 29.7 h on an RTX PRO 6000, 3.5 h on an A40 and 25 T4 hours, about 76–79 USD.
- The 397B steps add about 27–42 USD.

**Final submissions (version 0.4), internal code.** The CPU build of submission A including its whole chain took about
45 minutes on the development machine (two BLAS threads; about 15 minutes without step 0) and 3,987 s in the netless
Kaggle CPU rebuild of both submissions from the frozen inputs, with the internal code version
([PROVENANCE.md](PROVENANCE.md)). Test side of the new draws: M3 draw d4 about 7.4 h on an RTX PRO 6000 (about 17 USD);
draw h on the 2,459 listed test rows in two halves of about 4 h each (generation about 5.8 s, ranking about 4.8 s per
row; about 21.5 USD together).

**Weights.** Counted as deployed (each base model once, adapters as adapters): **481.53 GB** (448.46 GiB), 30.5 GB below
512 GB; counted at 8 bit per stored value: **440.94 GB**; parameters: 397 B nominal (403.4 B stored values) for the largest
model, 434.4 B (440.9 B) for all models together, below 550 B. Details, the counting rule and the risks of other counts:
[MODEL_SIZES.md](MODEL_SIZES.md); file list with checksums: [weights_manifest.json](weights_manifest.json). The 397B FP8
weights alone take 406.20 GB (378.30 GiB; the runner's log prints GiB as "GB"). The base models are referenced by pinned
revision and file checksums; M3 is shipped as an adapter (300.5 MB) and merged at run time. Files and sizes of the
private input package: [VERIFICATION.md §9](VERIFICATION.md#9-private-weights-and-inputs).

## 8. Operating rules

- **Full test set.** The alignment over the test set is defined over the complete test set (3,498 rows). The final
  candidates of both lines are never assembled from a subset of rows; only the strictly inductive form of the safety line
  processes each row on its own.
- **Override only by the recorded decision.** The safety candidate is built only with `OVERRIDE_CV_II=override_cv_condition_ii`
  and only while `decisions/override_cv_condition_ii.json` matches the checksum pinned in `eda/c123_safety_set.py`.
- **Fixed draws.** M3 uses a fixed set of draws with fixed settings: three in the earlier candidates, four in submission
  B and five in submission A (the fifth only on the listed rows). Each test draw must match the generation settings of its
  OOF draw; the checks reject any mismatch.
- **Fixed rotation list.** The released-list builds reject the first-version list; the first-version list is used only to
  rebuild the historical files.
- **Aggregates only.** Model outputs for test rows are written to files and never printed. Logs contain counts, shares
  and checksums.
- **Guards of the note features only abort.** In the safety candidate and its fallback the note guard, the similarity guard,
  the guard of the regex columns and guard (c) never switch to another form: a failing guard stops the build without a file.
  Their thresholds are written in the deployment file and checked by the assembly.
- **Fault bands are not tuning knobs.** The bands for the share of `NONE` at the top and for the rate of changed top codes
  detect technical faults. They are fixed in advance and never used to choose between variants.

## 9. Running on a new test set

The checksums of §2 and the build scripts belong to the competition test set. On another set of test photos, everything
fitted on the 1,500 training rows stays as delivered, and only the test side is computed.

1. **What stays.** The adapters, the orientation network, the OOF matrices with their metadata, the blend files and the
   fitted weights of the deployment files (`artifacts/stage2_*.json`). Nothing is refitted.
2. **Test side.** The new `test.csv` and photos go into the layout of §3, and the test-side steps of §6.1 run for every
   channel of the line in the stage order of §6.2, in mode `full` (every generated text is a new draw). The three inputs
   that §6.1 takes as frozen lists are made anew: the rotation lists (`tools/orient_net.py infer`), the gate lists
   (`python -I tools/replay/gates.py write --out <folder> --table test --line all --sources t4`) and the blank-photo list
   (`python -I tools/make_blank_list.py write --out <folder>`).
3. **Assembly.** The build scripts are not used: they compare every input and output with the checksums of the competition
   test set. `tools/make_submission.py` is called directly, with the arguments of the reference calls of §5 or of the calls
   below; the run folders named there then hold the runs on the new photos.
4. **Records of the competition test set.** Three inputs of the final candidates record the SHA-256 of a frozen test-side
   file of the competition test set: the field `mech27.test` of `stage2_primary_m3px27.json` and `stage2_primary_final.json`
   (the protocol answers of the test rows), the field `r_notes.pr27_test` of `stage2_safety_ndr.json` (the protocol notes of
   the test rows) and the test-note fields of `meta_m3px27xm.json` (the notes behind which both rankings of the mean channel
   were computed). They tie these candidates to the test files on which they were checked; on any other test set they stop
   the assembly with an error, independently of the guards. Rewriting them for a new set has not been done or tested. The
   calls below read none of them. **The two final submissions carry more such records:** besides `mech27.test` (both), the
   second stage of submission A pins the SHA-256 of every test-side column source in `extra_columns.test.files_sha256`
   (protocol answers of the 27B and of the 397B, their parsed answers and notes; a check that cannot be skipped), and its
   channel metadata `meta_m3x5h.json` pins the test list of draw h (`rank.row_list_draws[].test`). On the competition test
   set, A and B are assembled from rescored or regenerated outputs with the two calls of §5. After new protocol answers or
   397B answers, `tools/repin_test_side.py --write` first re-records their checksums; it keeps the old files under
   `artifacts/repin_backup/` and refits nothing ([VERIFICATION.md §5.2](VERIFICATION.md#52-rescore-and-full-gpu)). It does
   not touch the test list of draw h. On a new test set that list is made anew with `tools/make_charsize_rows.py` from the
   image statistics of the new photos (the threshold of 20 px is fixed); recording the new list in `meta_m3x5h.json` is not
   built.
5. **Guards.** The second stages compare aggregates of the test rows with values from the training rows
   ([METHOD.md](METHOD.md#92-test-side-guards), §9.10 and §9.11). They are meant to stop on a set that does not look like
   the training rows:
   - Safety line (`stage2_safety_ndr.json`, `stage2_safety_nd.json`, `stage2_safety_final.json`): the note guard, the
     similarity guard, the guard of the regex columns and guard (c) only abort. The call ends with a non-zero exit code,
     writes no file and does not switch to another form.
   - Primary line: guard (c) aborts in the same way. The note guard (a) is the one documented exception: if it fails, the
     fallback model of the same deployment file, which reads no notes, is applied.
   - The fault bands of the channels (share of `NONE` at the top of a ranking, completeness of every test run, settings of
     the test runs against those of the training runs) abort as well.

   None of these guards can be switched off. If one stops the assembly, the line falls back to its file without second
   stage, by an explicit call.
6. **Files without second stage.** The blend of the line, the alignment and the `REL-01` rule, without `--stage2`. On the
   competition test set the three calls reproduce files that the build scripts check: the primary call gives
   `sub_s032n_q397m3.csv` byte for byte (the file under the second stages of the primary line with `pr27xm`; the mean
   channel `m3px27xm` is left out because of its record, item 4), the safety call gives the file without second stage under
   the safety candidates (the copy null point that every safety build recomputes), and the strictly inductive call gives
   `sub_s030n_safe.csv` byte for byte. The alignment of the first two calls assumes that the new set has the class shares of
   the training rows, as the stratified competition split has; the strictly inductive call scores each row on its own and
   makes no assumption about the set.

```bash
# Primary line without second stage
python -I tools/make_submission.py --name s029_best_q397m3 \
  --test v1r=v1rot_t4/rot_test/test@derived/orient_repro/test_c80_rows.npy --fill v1r=v1 \
  --test v1=v1te_a_t4/v1_test_a/test+v1te_b_t4/v1_test_b/test \
  --test zs27q=zs27q_test0_rp/zs27q_test0/test+zs27q_test1_rp/zs27q_test1/test --fill think18nb=zs27q \
  --test think18nb=zs27v_te_rp/think_te/test/think@derived/think_rows/test.npy,zs27v_n3_rp/think_nte/test/think@derived/think_rows/test_tb_q.npy \
  --proto pr27xm=runs/pr27te_rp/proto_te/proto --rank pr27xm=runs/n1_rp/rank_test/test \
  --rank-only m3x3_full=runs/m3_fullte_rp/m3_test_rank/test,runs/m3d2te_rp/m3d2_test_rank/test,runs/m3d3te_rp/m3d3_test_rank/test \
  --rank-only q397m3=runs/q397m3te_rp/q397m3_test_rank/test --rank-none-band-for q397m3=0.27,0.35 \
  --align 0.75 --align-min 3 --blank-rows derived/blank/test_rows.npy \
  --out artifacts/submissions/sub_primary_base.csv --suffix _primary_base

# Safety line without second stage (alignment over the test set)
python -I tools/make_submission.py --name v1r_zs27t_think18sb_pr27xm_m3x3_full_ar \
  --test v1r=v1rot_t4/rot_test/test@derived/orient_repro/test_c80_rows.npy --fill v1r=v1 \
  --test v1=v1te_a_t4/v1_test_a/test+v1te_b_t4/v1_test_b/test \
  --test zs27t=px27_t4zt0/zs27t_test_t4/test+px27_t4zt1/zs27t_test_t4/test --fill think18sb=zs27t \
  --test think18sb=zs27v_n3_rp/think_nte/test/think@derived/think_rows/test_tb_t.npy \
  --proto pr27xm=runs/pr27te_rp/proto_te/proto --rank pr27xm=runs/n1_rp/rank_test/test \
  --rank-only m3x3_full=runs/m3_fullte_rp/m3_test_rank/test,runs/m3d2te_rp/m3d2_test_rank/test,runs/m3d3te_rp/m3d3_test_rank/test \
  --align 0.75 --align-min 3 --blank-rows derived/blank/test_rows.npy \
  --out artifacts/submissions/sub_safety_base.csv --suffix _safety_base

# Strictly inductive line without second stage (each row on its own)
python -I tools/make_submission.py --name v1r_zs27t_think18sb_pr27xm_m3x3_full_ar \
  --test v1r=v1rot_t4/rot_test/test@derived/orient_repro/test_c80_rows.npy --fill v1r=v1 \
  --test v1=v1te_a_t4/v1_test_a/test+v1te_b_t4/v1_test_b/test \
  --test zs27t=px27_t4zt0/zs27t_test_t4/test+px27_t4zt1/zs27t_test_t4/test --fill think18sb=zs27t \
  --test think18sb=zs27v_n3_rp/think_nte/test/think@derived/think_rows/test_tb_t.npy \
  --proto pr27xm=runs/pr27te_rp/proto_te/proto --rank pr27xm=runs/n1_rp/rank_test/test \
  --rank-only m3x3_full=runs/m3_fullte_rp/m3_test_rank/test,runs/m3d2te_rp/m3d2_test_rank/test,runs/m3d3te_rp/m3d3_test_rank/test \
  --align-inductive 0.75 --align-min 3 --blank-rows derived/blank/test_rows.npy \
  --out artifacts/submissions/sub_inductive_base.csv --suffix _inductive_base
```

On the competition test set (checked in a fresh copy of the repository with the private package):

| Call | Output | SHA-256 |
|---|---|---|
| Primary line without second stage | `sub_primary_base.csv`, byte-identical to `sub_s032n_q397m3.csv` | `d1667f6d2fb118844b72ecd8abd85220754a709d468519b7a047a893bbea12f1` |
| Safety line without second stage | `sub_safety_base.csv`, the copy null point of the safety builds | `91a14773ac7ec367b1385f1d304b3b8a74b3d9598aecfd91fee49f62ee4e4bb2` |
| Strictly inductive line without second stage | `sub_inductive_base.csv`, byte-identical to `sub_s030n_safe.csv` | `8cca7bbfe9b8c9dce47d6153d59712575bb7b8ad6dea8a286f7ef530df448b67` |


## 10. Recomputing the 397B channel on the test rows

Applies to both final submissions of the primary line. Every figure below is taken from the configuration files, the
code or the logs of the reference run; items that have not been verified are listed in
[section 10.12](#1012-open-points) and are not claimed elsewhere in this section.

### 10.1 What the channel computes

The open model **Qwen3.5-397B-A17B** (official FP8 release) diagnoses every test row once, in the fixed marking protocol
that our own protocol channel uses. The prompt contains no labelled row.

| Part | Content |
|---|---|
| System text | Marking instructions and the full taxonomy (`src/vlm/proto.py`, `system_text`; default naming rule) |
| User message | Grade, topic and question text; the question diagram if there is one; the answer key; the photo of the working (`proto.user_parts`) |
| Images | Photo with its long side resized to at most 1,024 px, diagram to at most 448 px (Lanczos), sent losslessly as PNG (`vlm.common.open_image`, `proto.to_messages`) |
| First request | Thinking mode on (`chat_template_kwargs: {"enable_thinking": true}`), greedy (`temperature` 0, no seed), at most 2,500 new tokens for thinking and answer together. The answer is one JSON object with three codes and their probabilities |
| Completion request | Only for rows whose first answer is not a valid JSON object: thinking off, at most 1,500 new tokens; the prompt repeats the question with the last 16,000 characters of the model's own first-pass notes (`proto.FINISH_TEXT`) |
| Concurrency and order | 64 parallel requests; rows in a fixed shuffled order (`order_seed: 0`); an interrupted run resumes from its output files |
| Guards | Abort if fewer than half of the first 24 rows give a valid answer; the job fails if more than 10 % of rows stay without an answer after three attempts each |

Code: job kind `proto_gen` of the runner (`src/vlm/runner.py` → `src/vlm/proto.py`, `generate`), local vLLM server
(`src/vlm/think_vllm.py`, `start_server`). Configuration of the test side: `configs/vlm/q397te_rp.yaml`; parity probe on
64 training rows: `configs/vlm/q397p_rp.yaml`.

Where the outputs go:
- **Both submissions:** the first-pass thinking of each row is exported as a note (`tools/make_q397m3_notes.py`), and M3
  (the fine-tuned 27B) writes its finding behind that note and scores all 74 codes (channel `q397m3`, configuration
  `configs/vlm/q397m3te_rp.yaml`, one RTX PRO 6000).
- **Submission A (and the intermediate form) only:** the parsed answers (`proto_raw.jsonl`, parsed with the parser of the
  training-side measurement, checked against `proto_labels.jsonl`) also feed six columns of the second stage
  (`src/stage2_sources.py`).

The training side of the channel (the same protocol on the 1,500 training rows) was computed through a hosted sampling
API with the same open model; see [PROVENANCE.md](PROVENANCE.md). No API ever processed a test row.

### 10.2 Hardware

**Reference run:** one machine with **4 × NVIDIA H200** (141 GB each; 139.8 GiB visible to CUDA), tensor parallelism 4,
rented from RunPod Secure Cloud.

| Resource | Measured or configured in the reference run |
|---|---|
| GPU memory per card after loading | 93.35 GiB weights; 94.89 GiB consumed in total before the KV cache |
| KV cache per card | 32.57 GiB (memory utilisation 0.92 = 128.62 GiB of 139.8 GiB); 1,568,000 tokens in total, i.e. 98 concurrent requests of the maximal length 16,000 |
| Host memory | 304 GiB available when the weights were loaded |
| Disk | 600 GB container disk, no network volume; the weights take 406.2 GB |
| Base image | `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` (Ubuntu 24.04, Python 3.12.3, CUDA 12.8 toolkit) |
| Driver | must run the CUDA 13 user-space libraries of the vLLM environment (section 4); the exact driver version of the reference machine was not recorded |
| GPU interconnect | vLLM's custom all-reduce failed in this container (most likely no GPU peer-to-peer access); all-reduce over NCCL works |

Other hardware has **not** been run (section 10.12).

### 10.3 Model

| | |
|---|---|
| Repository | `Qwen/Qwen3.5-397B-A17B-FP8` (Hugging Face) |
| Revision | `ea5b4f81096f3901c91dea97f81324302495781d` |
| Files | 107 files, 406,198,638,888 bytes (406.20 GB = 378.30 GiB); of these 94 `*.safetensors` shards with 406,151,669,464 bytes |
| Stored tensors | 403,397,928,944 values: 400,719,609,856 in FP8 (E4M3), 2,678,310,448 in BF16, 8,640 in F32 (safetensors metadata of the revision) |
| Architecture | `Qwen3_5MoeForConditionalGeneration` (mixture of experts, image and text input) |
| Licence | Apache License 2.0 (licence file read at this revision) |
| File checksums | [weights_manifest.json](weights_manifest.json): every file with size and LFS SHA-256 (or git blob id for small files) |

The runner downloads the full snapshot of the revision into `$K12_MODELS/Qwen3.5-397B-A17B-FP8/` if it is missing; with
`HF_HUB_OFFLINE=1` it never downloads and stops if the folder is missing (`src/vlm/runner.py`, `fetch_model`). Its log
line reports the size in GiB although it writes "GB" (`378.3 GB` = 378.3 GiB = 406.2 GB).

### 10.4 Software environment

Two Python environments on the same machine:

| Environment | Content in the reference run |
|---|---|
| Runner (base) | The image's Python 3.12.3 with `transformers==5.15.1 peft==0.21.2 accelerate==1.15.0 flash-linear-attention==0.5.2` and the unpinned `bitsandbytes pyarrow pandas pyyaml pillow` (resolved: bitsandbytes 0.50.2, pyarrow 26.0.0, pandas 3.1.0, PyYAML 6.0.3, Pillow 11.0.0; torch 2.8.0+cu128 from the image). The recorded package list is identical to the source of the base lock file |
| vLLM server | Virtual environment `/workspace/vllm_env`, created with `uv venv --python <base interpreter>` and `uv pip install vllm==0.31.0 transformers==5.17.0` (`runner.ensure_vllm_env`). Resolved to 199 packages, among them vllm 0.31.0, torch 2.13.0, transformers 5.17.0, flashinfer-python 0.7.0.post1, triton 3.7.1, nvidia-nccl-cu13 2.29.7, cuda-toolkit 13.0.3.0, ninja 1.13.2, tokenizers 0.23.3, numpy 2.3.5, Pillow 13.0.0 |

**Lock files.** `env/requirements-vllm.txt` pins the same 199 packages (versions only; a private lock file carries hashes for
130 of them). The package list of the reference run differs from it in one package: `openai==3.42.0` (file: 3.41.0; a
dependency of vLLM, not used by the job). Hashes for every pin and an offline wheelhouse are not built (section
10.12).

### 10.5 Server start

The runner starts the server itself (`think_vllm.start_server`) from the `server` block of the job. Reconstructed from the
code and the configuration, and confirmed by the non-default arguments in the server log of the reference run:

```bash
env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0 VLLM_ALLREDUCE_USE_FLASHINFER=0 \
    PATH=/workspace/vllm_env/bin:$PATH \
  /workspace/vllm_env/bin/python -m vllm.entrypoints.openai.api_server \
    --model /workspace/models/Qwen3.5-397B-A17B-FP8 --served-model-name k12 \
    --host 127.0.0.1 --port 8000 --dtype auto --max-model-len 16000 \
    --gpu-memory-utilization 0.92 --limit-mm-per-prompt '{"image": 2}' --max-num-seqs 64 \
    --tensor-parallel-size 4 --disable-custom-all-reduce
```

| Setting | Why |
|---|---|
| `--tensor-parallel-size 4` | the FP8 weights need 93.35 GiB per card on four cards |
| `--disable-custom-all-reduce` | vLLM's custom all-reduce failed during CUDA-graph capture with "invalid argument" (it needs GPU peer-to-peer access, which the container most likely did not allow); all-reduce then runs over NCCL |
| `VLLM_ALLREDUCE_USE_FLASHINFER=0` | FlashInfer's all-reduce kernel is compiled at start-up and failed to build without `ninja` on the path |
| `VLLM_USE_DEEP_GEMM=0`, `VLLM_MOE_USE_DEEP_GEMM=0` | DeepGEMM compiles its FP8 kernels with `nvcc` ≥ 12.9; the image has 12.8. Cutlass (linear layers) and Triton (experts) kernels are used instead |
| `bin_on_path: true` (config) | puts the vLLM environment's `bin/` (with `ninja`) first on `PATH`, because FlashInfer still compiles its gated-delta-net prefill and sampling kernels at start-up |
| `--max-model-len 16000`, `--max-num-seqs 64`, `--limit-mm-per-prompt '{"image": 2}'` | longest prompt plus completion; 64 parallel requests; photo and diagram |
| `--host 127.0.0.1` | the server is reachable only from the same machine |

Effective engine settings reported in the log: activation type bfloat16, FP8 quantisation from the checkpoint, prefix
caching and chunked prefill on (8,192 batched tokens), engine seed 0, CUDA graphs in mode FULL_AND_PIECEWISE,
FlashAttention 3, Triton FP8 expert kernels, Cutlass block-scaled FP8 linear kernels.

**Required in addition for a run without network access** (not set in the reference run, which kept vLLM's usage
reporting on): the settings of section 4, among them `VLLM_NO_USAGE_STATS=1`, `DO_NOT_TRACK=1`, `VLLM_DO_NOT_TRACK=1`,
`VLLM_HOST_IP=127.0.0.1`, `GLOO_SOCKET_IFNAME=lo` and `NCCL_SOCKET_IFNAME=lo`. With tensor parallelism 4, the two socket
settings have not been tested.

### 10.6 Data and job

The runner needs a data folder (`K12_DATA`) with `train.csv`, `test.csv` (the root CSV files of the competition),
`competition_data/taxonomy_v1.csv`, `competition_data/images/test/` and `competition_data/images/questions/`. It checks the
row order of `test.csv` against a pinned digest before it starts (`vlm.common.load_table`). The parity probe also needs
`competition_data/images/train/` and the row list `data/derived/open_teacher/rows_parity64_397b.npy` (64 training rows,
seed 20261030; SHA-256 `7212015c96374a461b50a0206701eaff1e9dd179713f3799fc398fde05b1f97b`).

```bash
cd src
K12_DATA=/workspace/k12/data K12_MODELS=/workspace/models PYTHONPATH="$PWD:$PWD/.." \
  HF_HUB_DISABLE_PROGRESS_BARS=1 TOKENIZERS_PARALLELISM=false \
  python -u -m vlm.runner --config ../configs/vlm/q397p_rp.yaml  --out /workspace/out   # parity probe, 64 training rows
  python -u -m vlm.runner --config ../configs/vlm/q397te_rp.yaml --out /workspace/out   # test side, 3,498 rows
```

Job settings (`q397te_rp.yaml`, job `proto_te`): `kind: proto_gen`, `split: test`, `order_seed: 0`, `tag: proto`,
`image: {max_side: 1024, question_max_side: 448}`, `max_tokens: 2500`, `finish_tokens: 1500`, `note_chars: 16000`,
`concurrency: 64`, `guard_rows: 24`, `guard_min_valid: 0.5`, `request_timeout_s: 1200`, and the `server` block of
section 5 (`startup_s: 3600`). The parity configuration differs only in `split: train`, the row list and `must_pass`.
A variant for 8 × H100 with tensor parallelism 8 (`q397te8_rp.yaml`, `q397p8_rp.yaml`) is otherwise identical; it has
never been run.

### 10.7 Order of the steps

| Step | Network | Command or action | Pass condition |
|---|---|---|---|
| 0 | — | Check GPUs, driver, disk (≥ 600 GB free) and host memory | four cards of the H200 class as in the reference run, or see section 10.12 |
| 1 | allowed | Copy the model revision into `$K12_MODELS/Qwen3.5-397B-A17B-FP8/` with the call the runner itself uses: `python -c "from huggingface_hub import snapshot_download as d; d('Qwen/Qwen3.5-397B-A17B-FP8', revision='ea5b4f81096f3901c91dea97f81324302495781d', local_dir='/workspace/models/Qwen3.5-397B-A17B-FP8', max_workers=32)"` | size and SHA-256 of all 107 files equal [weights_manifest.json](weights_manifest.json) |
| 2 | allowed | `uv venv /workspace/vllm_env --python <base interpreter>`, then install from the wheelhouse of the lock: `uv pip install --offline --no-index --require-hashes --find-links <wheelhouse> --python /workspace/vllm_env/bin/python -r vllm.lock`; write any text into `/workspace/vllm_env/.k12_ready` | with this marker present, the runner uses the environment and installs nothing (`runner.ensure_vllm_env`). Requires the completed lock (section 10.12) |
| 3 | **off** | Export the settings of section 4 (`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `VLLM_NO_USAGE_STATS=1`, `DO_NOT_TRACK=1`, `VLLM_HOST_IP=127.0.0.1` and the others); run in a container without a network interface | — |
| 4 | off | Parity probe: runner with `q397p_rp.yaml`, then the parity check against the training-side answers of the hosted API (its script is not part of this repository; the build scripts read its stored verdict) | valid answers ≥ 97 % of the 64 rows; first-pass input tokens within 1 % of the reference count in ≥ 95 % of rows; same top code as the reference in ≥ 70 % of rows. Otherwise stop |
| 5 | off | Test side: runner with `q397te_rp.yaml` | job completes; 3,498 rows answered |
| 6 | off | `python -I eda/c110_q397_deploy.py test-status runs/q397te_rp/proto_te/proto` | 3,498 rows; valid answers ≥ 97 %; share of `NONE` at the top within the training value ± 0.06; model and settings equal those of the training side |
| 7 | off | Downstream: `python -I tools/make_q397m3_notes.py --test-from runs/q397te_rp/proto_te/proto`, then the M3 ranking `q397m3te_rp`, then the assembly | sections 5 and 9 |

The reference for steps 4 and 6 are the training-side answers of the same model through the hosted API
(`data/derived/open_teacher/qwen35-397b/labels.jsonl`). The runner itself starts and stops the server; every step prints
only aggregates.

### 10.8 Run time and cost of the reference run

| Phase | Time |
|---|---|
| Download of the revision (406.2 GB) | 1,030 s (about 0.39 GB/s) |
| Server start after the download | about 7.5 min: weights from the local disk 233 s, `torch.compile` 38 s, CUDA graphs, warm-up and kernel compilation (engine initialisation 144 s) |
| Parity probe, 64 training rows | 83.6 s (1.31 s per row at 64 parallel requests) |
| Test side, 3,498 rows | 1.49 s per row, about 87 min (one run of 90 min wall time) |
| Generation throughput | about 2,700–2,800 tokens/s |

The successful attempt (download, start, parity probe, test side) took about 2.05 h on 4 × H200 at 21.16 USD/h, about
43 USD. Earlier attempts on the same hardware class failed at the server start and cost about 46 USD; their causes are
the reasons for the switches of section 5. With the weights already on the disk, a repetition takes about 1.6 h.

### 10.9 Expected outputs

`<out>/q397te_rp/proto_te/`: `proto_labels.jsonl` (parsed answer per row and request), `proto_raw.jsonl` (thinking and
answer text per row and request), `proto_meta.json` (aggregates), `proto_server.log`, `summary.json`; the runner also
writes `pip_freeze_vllm.txt` when it installs the vLLM environment itself.

Aggregates of the reference run: 3,498 rows answered, 0 failed, 3,492 valid (99.8 %), share of `NONE` at the top 0.320
(training rows through the hosted API: 0.316). The answer files have 6,633 records each: 3,498 first requests and 3,135
completion requests, i.e. 90 % of the rows had no valid answer after the first request and were completed without
thinking (parity probe: 59 of 64 first requests ended at the token limit).

Frozen files of the reference run (read by the build scripts; never inspected row by row):

| File | Bytes | SHA-256 |
|---|---:|---|
| `artifacts/runs/q397te_rp/q397te_rp/proto_te/proto_raw.jsonl` | 29,573,905 | `824724105f1b4abb05147db26794bdd3b1b2c030725554a7126c4fba888a6493` |
| `artifacts/runs/q397te_rp/q397te_rp/proto_te/proto_labels.jsonl` | 1,503,024 | `82e29b9663f0dd2c6f32cb7e52676f4e8fa9e6e27c5b03a2d6c99e1720d27be0` |
| `artifacts/runs/q397te_rp/q397te_rp/proto_te/proto_meta.json` | 416 | `3794090406eda00c149145503ecb8146deda8124440b589a0cd9ef9272a9d902` |
| `artifacts/runs/q397te_rp/q397te_rp/proto_te/summary.json` | 637 | `91573a0318c4f0c87ebedf47cd7f79dcec66be0b938280c33a0c46ed5a928c57` |
| `data/derived/sft/q397m3_test_notes.jsonl` (notes for M3, derived from `proto_raw.jsonl`) | 25,640,297 | `4b29f057d50c7decfc3db73ddb37182d26b51326e3af05683d4665a56f20eb60` |

### 10.10 Fidelity check

The parity probe runs the same protocol on 64 random training rows for which the hosted-API measurement has a valid
answer, and compares the number of **input tokens of the first request per row** with the count reported by the API.
Equal counts show that system text, taxonomy, question, diagram and photo reach the model in the same form; the images
were sent to the API as JPEG (quality 75) at the same pixel size and locally as PNG, which gives the same number of image
tokens. Result of the reference run: 64 of 64 valid, input tokens **identical in 64 of 64 rows**, same top code in 78.1 %
of rows (rule: ≥ 70 %), MAP@3 0.695 locally against 0.716 through the API on these 64 rows (reported, not part of the rule).

### 10.11 What is deterministic, and how the solution deals with the rest

| Step | Deterministic? | Evidence |
|---|---|---|
| Prompt and image construction | yes | identical input-token counts in 64 of 64 rows against an independent implementation |
| Text generation | **no** | greedy decoding, but continuous batching of 64 parallel requests, chunked prefill, prefix caching and FP8 expert kernels make the result depend on the composition of the batches |
| Export of the notes for M3 | yes | byte-identical rebuilds; no time stamps in the file |
| M3 scores behind given notes | yes on the same hardware and number format | exact scoring behind a fixed text |
| Assembly of the submission | yes | byte-identical rebuild of both submissions on a CPU without network access |

**Measured agreement between draws.** No second local draw of the 397B channel exists. Measured are: the local FP8 path
against the hosted API on the 64 parity rows, same top code in 78.1 % of rows; for the 27B protocol channel, which uses
the same code path, two runs of the same protocol agree on the top code in 84.5 % of 860 training rows; for the 27B
reasoning channel, two greedy runs with identical prompts agree in 84 % of rows.

**How the solution handles it.** The answers of the reference run are frozen with their SHA-256 and are part of the
private package. The build scripts read the frozen files and reproduce both submitted files byte for byte; this mode
needs only a CPU. Running sections 10.5–10.7 again produces a new draw of every answer: statistically equivalent, not
identical. Its files have new checksums, which the build scripts reject by design, because the deployment files record
the checksums of the frozen test-side inputs (submission A: the field `extra_columns.test.files_sha256` of its
second-stage file). How such a draw is assembled into A and B:
[VERIFICATION.md §5.2](VERIFICATION.md#52-rescore-and-full-gpu).

### 10.12 Open points

These are not verified and are therefore not claimed above:
- **A run without network access** of sections 10.5–10.7 has not been performed. The reference run downloaded weights and
  packages at run time and kept vLLM's usage reporting on. The check would run steps 1–2 with network and steps 3–6 in a
  container without a network interface, with empty compiler caches, on 4 × H200.
- **Other hardware.** 8 × H100 (80 GB) with tensor parallelism 8 has a configuration but has never run. Four cards with
  96 GB (89.4 GiB each) cannot even hold the 93.35 GiB of weights per card at tensor parallelism 4 (arithmetic, not
  tested). No 4-bit or other smaller release of the model has been tested; it would give different texts.
- **Driver version** of the reference machine (not recorded).
- **Complete lock file** with hashes for all pins and an offline wheelhouse.
- **Rerun variance of the submitted files** when all generated texts are replaced at once: not measured directly;
  estimated at about ±0.003 MAP@3 (1 sd) on the private rows
  ([VERIFICATION.md §8.1](VERIFICATION.md#81-how-close-a-full-rerun-should-come)).
