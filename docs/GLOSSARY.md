# Glossary

*Version 0.6, 2026-11-21.*

This page explains the internal names that appear in the documentation, in commands and in file names. The other
documents use plain descriptions where they can and give a name where you need to type it or find it in a file. Many
file and folder names are kept as they were during development, because the code and the pinned checksums refer to them.

## Basic terms

| Name | What it is | Where you meet it |
|---|---|---|
| root code | The label: the taxonomy code of the misconception from which the rest of the working follows | `train.csv`, column `root_code` |
| `NONE` | The class for a working without a major error | all outputs |
| 76 classes | `NONE` plus the 75 taxonomy codes, in the order of `taxonomy_v1.csv` | `src/labels.py`, all probability matrices |
| `REL-01` | The taxonomy code for an upload that does not match the question; we put it first for blank photos | the blank-photo rule |
| pool, unlabelled set | The 50,000 photos of `unlabelled.csv` | teacher pseudo-labels, orientation network |
| teacher | Claude Haiku 5.5, which labelled the pool during development; its labels are frozen | `data/derived/teacher/` |
| general codes | `ANS-01`, `CAL-01`, `WOP-01`, `FRM-01` and `OTH-00`, the five codes that the host capped at 57 training rows each | reranker features |

## Submissions and lines

| Name | What it is | Where you meet it |
|---|---|---|
| A | Final submission `sub_primary_m3x5h.csv`, Kaggle reference 58353348 | VERIFICATION §1; `check_A.csv`, `new_A.csv` |
| B | Final submission `sub_primary_m3x4.csv`, Kaggle reference 58139357 | VERIFICATION §1; `check_B.csv`, `new_B.csv` |
| intermediate form | B plus the nine extra reranker columns, without the fifth M3 draw. We submitted it but did not select it; the build of A builds it on the way | `primary_cols`, `sub_primary_cols.csv` |
| primary line | The design that maximises cross-validated MAP@3. Both final submissions come from it | build scripts `primary_*`; METHOD §11 |
| safety line | A second design without teacher-derived input, API outputs or the 397B model; not selected. Its strictly inductive form processes each test row on its own | build scripts `safety_*`, `s030n`, `s036`; METHOD §11 |
| `best`, `safe` | Older names of the primary line and of the strictly inductive line, used as arguments of the older build scripts and of `tools/replay/gates.py --line` | `s029_m3x3.sh safe`, `s034_stage2_main.sh best` |

## Channels and draws

| Name | What it is | Where you meet it |
|---|---|---|
| channel | One model set-up that returns a probability for each of the 76 classes for every row | all documents |
| draw | One generated text per row from one run. Another run of the same set-up gives another draw, because text generation is not deterministic | "M3 draws", "reasoning draws" |
| exact scoring, ranking | The log-probability of every code as a continuation of a prompt, computed through a prefix tree over the code tokens; a ranking does this behind a generated text | `src/vlm/engine.py`; jobs named `*_rank` |
| `v1` | 9B scorer: Qwen3.5-9B with our LoRA adapter, scoring every code right after the prompt | `oof_v1.npy`; runs `ft_v1*_rp`, `v1te_a_t4`, `v1te_b_t4` |
| `v1r` | `v1` with rotated photos turned upright first by the orientation network | run `v1rot_t4` |
| `zs27q` | 27B zero-shot scorer: Qwen3.8-27B in 4-bit, without training, with same-question examples, the topic block and the task block (called question block in the configurations) | runs `zs27q_test0_rp`, `zs27q_test1_rp` |
| `zs27t` | The same scorer with the topic block only; safety line | runs `zs27t_all_rp`, `px27_t4zt0`, `px27_t4zt1` |
| `think18nb` | 27B reason-then-name channel of the primary line. The untrained 27B reasons for up to 1,800 tokens, then every code is scored behind its text. It takes draw g on the wide gate, the mean of g and n on the narrow gate, n on the band and `zs27q` elsewhere | runs `zs27v_te_rp`, `zs27v_n3_rp` |
| g, n | The two reasoning draws. g uses the prompt of `zs27q` with the task block; n uses the prompt of `zs27t`, without teacher content | jobs `think_te`, `think_nte` |
| `think18sb` | Reason-then-name channel of the safety line: draw n on that line's narrow gate and band, `zs27t` elsewhere | safety line |
| `pr27` | 27B protocol channel: the untrained 27B follows a fixed marking protocol, sees no labelled row and states three codes with probabilities | runs `pr27_rp`, `pr27te_rp` |
| `pr27x` | Exact ranking of all codes by the untrained 27B behind the frozen notes of the protocol answers | `oof_pr27x.npy`; runs `px27_rp`, `n1_rp` |
| `pr27xm` | Mean of `pr27` and `pr27x`; used by the safety line and the fallbacks | `oof_pr27xm.npy` |
| `m3px27` | M3's ranking of all codes behind the frozen protocol notes | runs `m3px27_oof_rp`, `m3px27te_rp` |
| `m3px27xm` | Mean of `pr27` and `m3px27`; the protocol channel of both final submissions | `oof_m3px27xm.npy`, `meta_m3px27xm.json` |
| M3 | Our fine-tuned 27B reasoner: Qwen3.8-27B with our LoRA adapter, trained on the gold labels to write notes, the root finding and the code. All codes are then ranked behind its text | `weights/m3_full/`; runs `m3_*` |
| d1, d2, d3, d4 | The M3 draws. d1 is greedy with up to 3,200 tokens and d2 greedy with up to 5,200. d3 and d4 are sampled (temperature 0.6, top-p 0.95, top-k 20) with seeds 3 and 4 | runs `m3_fullte_rp` (d1), `m3d2te_rp`, `m3d3te_rp`, `m3d4te_rp` |
| h | The fifth M3 draw: sampled with seed 5, with the photo at 2,048 px instead of 1,024 px, only on the rows of the character-height list; used in A | runs `m3hte_a_rp`, `m3hte_b_rp`, `m3ha_oof_rp`, `m3hb_oof_rp` |
| `m3x3_full`, `m3x4_full` | M3 channel as the mean of d1–d3 or of d1–d4. `m3x4_full` is the M3 channel of B | `oof_m3x3_full.npy`, `oof_m3x4_full.npy` |
| `m3x5h` | M3 channel of A: per row the mean of the draws that cover the row, d1–d4 everywhere and h on the listed rows | `oof_m3x5h.npy`, `meta_m3x5h.json` |
| `m3p5`, `m3x2_full_cv01` | Earlier M3 channels that the checks compare with: draw d1 alone, and the mean of two draws | `oof_m3p5.npy`, `oof_m3x2_full_cv01.npy` |
| `q397` | The codes that Qwen3.5-397B-A17B states itself under the marking protocol, without training | `oof_q397.npy` |
| `q397m3` | 397B notes, ranked by M3: M3 ranks all codes behind the first-pass notes of the 397B | runs `q397te_rp`, `q397m3te_rp`, `q397m3_oof_rp` |

## Context and combination

| Name | What it is | Where you meet it |
|---|---|---|
| same-question examples | Up to four labelled workings of other students for the same question, from the other folds in cross-validation | 9B and 27B zero-shot scorers, reasoning |
| topic block | Root codes of other questions of the same topic, with their frequency and a short note, from the other folds | zero-shot scorers, reasoning |
| task block | Per question, a summary of the teacher pseudo-labels on pool photos of the same question; primary line only | `zs27q`, draw g; `teacher_labels` in the configurations |
| marking protocol | Judging rules and the taxonomy only, without any labelled row or block | `pr27`, M3, 397B; `src/vlm/proto.py` |
| gates | Row lists that decide where reasoning texts are generated. wide: P(`NONE`) < 0.5 in the fast channel. narrow: wide, and the top class of the 9B and fast channel together below 0.6. band: outside the wide gate with P(`NONE`) < 0.95 | `data/derived/think_rows/`, `tools/replay/gates.py` |
| floor | A small uniform share mixed into a channel before the blend, so that a near-certain wrong channel cannot veto the right class | `--floor`, blend files |
| blend | Log-linear combination of the channels with weights fitted on LogLoss, plus an arithmetic share β (0.6 in the primary line) | `stack_<name>.json` |
| alignment | Class-marginal alignment: class weights that bring the mean predicted mass per class over all 3,498 test rows to the training frequency (classes with at least 3 training rows, strength 0.75). It uses no label. The inductive alignment of the strictly inductive form fits the weights on the training rows instead | `--align 0.75`, `--align-inductive 0.75` |
| second stage, reranker | A conditional logit that reorders the five best codes of each row after the alignment | `stage2_<name>.json`, `src/stage2.py` |
| extra columns | The nine reranker columns of A and of the intermediate form: eight from the frozen protocol answers of the 27B and the 397B, one from the taxonomy sections seen per topic | `src/stage2_sources.py`, field `extra_columns` |
| mechanical match | The final answer read by the protocol channel equals the answer key under a number parser | `src/answer_match.py`, field `mech27` |
| NONE guard | If the aligned blend puts `NONE` first, the reranker may not replace it | `protect_none` in `src/stage2.py` |
| note guard | Test-side check of 13 statistics of the frozen M3 test notes against the training draws; if one fails, the build switches to the fallback form | METHOD §9.2 |
| fallback form | The reranker without the note features | `stage2_*.json` |
| test guards | Checks of test-side aggregates, such as fault bands, the guard per extra column and the guard of draw h. They stop a build and never tune anything | build logs, `guards_*.json` |
| blank-photo rule | A photo that is blank gets `REL-01` at rank 1 | `data/derived/blank/`, `--blank-rows`, `tools/make_blank_list.py` |
| orientation network | Two small CNNs that predict the rotation of a photo; trained on the CPU from rotations that we apply ourselves | `weights/orient_cnn/`, `artifacts/orient/`, `tools/orient_net.py` |
| rotation lists `c50`, `c80` | `c50`: photos turned upright for the 9B run (confidence ≥ 0.5). `c80`: rows on which the upright reading replaces the plain one (confidence ≥ 0.8) | `data/derived/orient/`, `data/derived/orient_repro/` |
| `orient/`, `orient_repro/` | Rotation lists of the first, unseeded network (kept frozen) and of the released, seeded network | `data/derived/` |
| `ORIENT=repro`, `ORIENT=legacy` | Choice of the rotation list in the build scripts: `repro` (the default) uses the released network's list; `legacy` rebuilds earlier files with the first-version list | `reproduce.sh --full` |
| character-height list `h20` | Rows whose median character height in the photo at long side 1,024 px is at most 20 px | `data/derived/charsize/`, `tools/make_charsize_rows.py` |

## Evaluation terms

| Name | What it is | Where you meet it |
|---|---|---|
| OOF, out-of-fold | Predictions for a training row from a model that did not see the row's fold | `oof_<channel>.npy` |
| CV | Cross-validated MAP@3 on the 1,500 training rows with the five fixed folds | all results |
| per fold, as one set | CV forms: the alignment applied inside each 300-row fold, or to all 1,500 rows at once as on the test set | VERIFICATION §1, RESULTS |
| ½(F+M) | The mean of the per-fold and the as-one-set difference | RESULTS, VALIDATION |
| meta-split | A new stratified 5-fold split of the 1,500 OOF rows with another seed, on which the nested fit of blend and reranker is repeated | RESULTS, VALIDATION §5.3 |
| copy null point | A control: a component with weight 0, or an existing draw added again as if it were new, must reproduce the previous result | build logs, VALIDATION §5.3 |
| placebo | A control with shuffled features or labels that must give no gain | VALIDATION §5.3 |

## Modes, scripts and configurations

| Name | What it is | Where you meet it |
|---|---|---|
| `assemble` | Mode on the CPU: blend, alignment, reranker and file from the frozen channel outputs | VERIFICATION §5.1 |
| `rescore` | Mode on GPUs: every scoring and ranking step on the test photos, behind the frozen texts | VERIFICATION §5.2 |
| `full` | Mode on GPUs: everything, including new texts | VERIFICATION §5.2 |
| quick check | `bash reproduce.sh <package-dir>`: two calls of `tools/make_submission.py` that write A and B from the frozen outputs | VERIFICATION §5.1 |
| full proof, build chain | `bash reproduce.sh <package-dir> --full`: the build scripts that refit or verify every blend and second-stage file | VERIFICATION §5.1 |
| `check_A.csv`, `check_B.csv` | The files that the quick check writes | `artifacts/submissions/` |
| `new_A.csv`, `new_B.csv` | The files that `tools/gpu_inference.sh` writes from new outputs | `artifacts/submissions/` |
| `configs/vlm/<run>.yaml` | One configuration per model run; the model runner runs every job in it | `src/vlm/runner.py` |
| `configs/vlm/test_side/` | Test-side jobs of the reference configurations for `full`, without training jobs | `tools/gpu_inference.sh` |
| `configs/vlm/rescore/` | The same for `rescore`, without any job that generates text | `tools/gpu_inference.sh` |
| `sNNN_*.sh`, such as `s029`, `s034`, `s036` | Build scripts named after the number of one of our earlier Kaggle submissions, in our own count; they rebuild that file or a file of its chain | `tools/builds/` |
| `n` after the number: `s030n`, `s033n`, `s034n` | The same file rebuilt with the rotation list of the released network | `tools/builds/` |
| `primary_*`, `safety_*` | Build scripts, deployment files and outputs named by line and role: `primary_m3x5h` (A), `primary_m3x4` (B), `primary_cols` (intermediate form); `primary_m3px27`, `primary_final`, `primary_i1` are predecessors of B; `safety_*` belong to the safety line | `tools/builds/`, `artifacts/` |
| `eda/aNN_*`, `eda/cNNN_*`, `eda/i5_*` | Step scripts; the number is the step number of our development. `a` scripts build artifacts, `c` scripts evaluate and deploy, `i5` scripts compute image statistics. They are kept because the build scripts call them or check their stored outputs | `eda/` |

## Run and job names

| Name | What it is | Where you meet it |
|---|---|---|
| run | One configuration file and its output folder | `artifacts/runs/<run>/<run>/<job>/` |
| `_rp` | A run on a rented GPU machine: A40, RTX PRO 6000 or 4 × H200 | most run names |
| `_t4` | A run on Kaggle T4 × 2 | `v1te_a_t4`, `v1rot_t4`, `px27_t4zt0` |
| `te`, `test` | Test side | `m3d4te_rp`, `zs27q_test0_rp`, job `m3_test_rank` |
| `oof`, `f0` … `f4`, `f034` | Training side, out of fold, and fold numbers | `m3d4_oof_rp`, `m3_f034_rp`, job `m3_train_f1` |
| `_a`, `_b`, `test0`, `test1` | Two halves of the test rows or of a row list | `v1te_a_t4`, `m3hte_a_rp`, `zs27q_test0_rp` |
| `full` | Full fit on all 1,500 training rows, or the run that holds it | `m3_full`, `v1_full`, `m3_fullte_rp` |
| `pilot` | The first M3 run, on folds 1 and 2 | `m3_pilot_rp` |
| `q397p_rp` | Parity probe of the 397B path on 64 training rows | `artifacts/runs/q397p_rp/` |
| `q397_tinker` | Folder of the frozen 397B answers on the training rows from the Tinker API | `artifacts/runs/q397_tinker/` |
| `m3_5f`, `m3long_comb` | Combined fold folders of the M3 draws d1 and d2 | `artifacts/runs/` |
| job name parts | `_train`: training. `_merge`: merging the adapter into the base model. `_gen`: text generation. `_rank`: exact ranking behind a text. `smoke`, `check`: short pre-check jobs | inside each configuration |
| tags `bench`, `test` | Last part of a source path: out-of-fold training rows or test rows | arguments of `tools/make_submission.py` |

## Files and folders

| Name | What it is | Where you meet it |
|---|---|---|
| `oof_<channel>.npy` | 1,500 × 76 OOF matrix: rows in `train.csv` order, columns `NONE` plus the taxonomy | `artifacts/` |
| `meta_<channel>.json` | Sources, checksums and settings of an OOF matrix | `artifacts/` |
| `stack_<name>.json` | A blend file: channels, floors, weights, arithmetic share, last step | `artifacts/` |
| `stage2_<name>.json` | A second-stage deployment file: models, guards, input checksums | `artifacts/` |
| `test_<name><suffix>.npy` | 3,498 × 76 test probabilities of an assembled file | `artifacts/` |
| `sub_<name><suffix>.csv`, `.json` | A submission file and its record of sources, checksums and aggregates | `artifacts/submissions/` |
| long older names, such as `sub_v1r_zs27t_think18sb_pr27xm_m3x3_full_ar_ia_t4.csv` | Files of earlier builds whose names list the channels of the blend; the short tags at the end are build labels | `artifacts/submissions/`, `artifacts/stack_*.json` |
| `artifacts/runs/<run>/<run>/<job>/` | Frozen outputs of one job of a model run | private package |
| `artifacts/eda/private/<topic>/` | Stored reports of the step and build scripts. Some hold per-row values, so the folder is private | private package |
| `artifacts/repin_backup/` | Copies that `tools/repin_test_side.py --write` keeps before it re-records checksums | after a `full` run |
| `weights/` | Our adapters and the orientation network | private package |
| `PACKAGE_INDEX.tsv` | Every package file with its size, SHA-256, role and whether it is derived from test rows | private package |
| `MANIFEST.sha256` | Checksums of all package files, for `sha256sum -c` | private package |
| `README_PACKAGE.md` | Where each package folder goes | private package |
| package roles | `assemble`, `assemble-generated`, `rescore`, `training`, `check` and `weights`; VERIFICATION §9 explains them | `PACKAGE_INDEX.tsv` |
| `decisions/` | One recorded decision that a build script of the safety line checks by its checksum | repository |

Folders under `data/derived/` in the private package:

| Folder | Content |
|---|---|
| `blank/` | The blank-photo list |
| `charsize/` | The character-height lists of draw h and their meta file |
| `orient/`, `orient_repro/` | Rotation lists of the first and of the released orientation network |
| `think_rows/` | Gate lists of the reasoning draws |
| `sft/` | Frozen M3 texts and training notes of the draws d1–d3, the 397B notes in the M3 format, and the M3 training targets |
| `sft_k2/` | Frozen M3 test texts of draw d4 |
| `sft_m3px27/`, `sft_m3px27te/` | Protocol notes in the M3 format, training and test side |
| `sft_q397m3/`, `sft_q397m3te/` | 397B notes in the M3 format as the model runs read them, training and test side |
| `proto_notes/`, `proto_notes_test/` | Frozen protocol notes of the training rows (two draws) and of the test rows |
| `proto_rows/` | Row lists that split the training rows between the jobs of the protocol-answer run |
| `open_teacher/` | Frozen 397B answers on the training rows |
| `teacher/` | Frozen teacher pseudo-labels of the pool, `pool_a` and `pool_b` |
| `m3x3_full_rows.npy` | The training rows that all M3 draws cover (all 1,500) |
