# Code map

Reference document. To reproduce the submissions, start with [VERIFICATION.md](VERIFICATION.md).

*Status: 2026-11-21 (version 0.6). Every script and module of this repository with its role. How to run the steps is in
[TRAINING.md](TRAINING.md) and [INFERENCE.md](INFERENCE.md); the method is in [METHOD.md](METHOD.md); the internal names
are explained in [GLOSSARY.md](GLOSSARY.md).*

**Contents:**
1. [Conventions](#1-conventions)
2. [Libraries](#2-libraries)
3. [Model code](#3-model-code-srcvlm)
4. [Step scripts](#4-step-scripts-eda)
5. [Tools](#5-tools)
6. [Build scripts](#6-build-scripts-toolsbuilds)
7. [Run configurations](#7-run-configurations-configsvlm)
8. [Tests](#8-tests)
9. [Channel names and file formats](#9-channel-names-and-file-formats)
10. [Steps whose scripts are not part of this repository](#10-steps-whose-scripts-are-not-part-of-this-repository)

---

## 1. Conventions

- All commands run from the repository root. Python scripts are started with `python -I` (isolated mode: no user
  site-packages and no `PYTHON*` environment variables); the exceptions are the model runner, which is started as a
  module from `src/` ([TRAINING.md](TRAINING.md#2-environment)), and the tests (`python -m pytest`). The CPU steps pin
  `OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2`.
- `data/` holds the competition data and the private derived inputs (`data/derived/`), `artifacts/` the private model
  outputs, blend files and submissions. Both folders are not part of the repository ([DATA_AND_COMPLIANCE.md](DATA_AND_COMPLIANCE.md)).
- Step scripts in `eda/` carry a short prefix (`a…` for artifact builders, `c…` for evaluations and deployment paths). Build
  scripts and some artifacts carry historical build labels (`s029`, `s030n`, `s034`, `s035`, …); the checksums recorded in
  the frozen artifacts refer to these names, so they are kept. The build scripts added in versions 0.2 and 0.3, their
  deployment files, output files and report folders are named by role instead (`primary_i1`, `primary_final`, `safety_final`,
  `safety_nd`, `safety_ndr`, `primary_m3px27`; report folders `stage2_i1/`, `stage2_mm/`, `safety_set/`, `safety_nd/`,
  `safety_ndr/`, `m3px27_eval/` and `primary_m3px27/` under `artifacts/eda/private/`).
- Scripts print only aggregates and checksums for test rows; per-row outputs go to files.
- The build scripts of version 0.1 and earlier take `best` (primary line) or `safe` (strictly inductive line, today the
  strictly inductive form of the safety line) as argument, and `tools/replay/gates.py --line` accepts `best`, `safe` or
  `all`; these are historical names of the two lines.
- `decisions/` holds recorded decisions that a step script checks by a SHA-256 pinned in its code
  (`decisions/override_cv_condition_ii.json`, checked by `eda/c123_safety_set.py`).
- `env/` holds the pinned requirements of the four environments: `requirements-cpu.txt` (assembly and tests),
  `requirements-gpu.txt` (training, merging and scoring with transformers), `requirements-vllm.txt` (text generation) and
  `requirements-kaggle-t4.txt` (the wheels on top of the Kaggle GPU image); each header names the Python and CUDA version.

## 2. Libraries

| Module | Role |
|---|---|
| `kaggle_core/artifacts.py` | Writing and collecting OOF/test/meta artifacts |
| `kaggle_core/folds.py` | Fold creation (stratified, group, plain) and the check that all channels share the same folds |
| `kaggle_core/metrics.py` | Fast metrics, metric registry, class multipliers |
| `kaggle_core/submission.py` | Generic submission validation and writing |
| `src/labels.py` | Label space: `NONE` + the 75 taxonomy codes in file order (76 classes); codes are only checked against the class list |
| `src/metric.py` | MAP@3 (`map3`, `map3_from_proba`, `top3_indices`), bootstrap and paired bootstrap |
| `src/submission_io.py` | Building, strictly checking and writing a submission (format traps of this competition) |
| `src/transcription.py` | Reading the annotators' transcriptions and removing label leaks (training side only) |
| `src/stage2.py` | Second stage: candidate set, pair features (sets `main`, `fallback`, `main_guard`, `main_guard_i1`, `main_guard_i1_mm`, `fallback_i1`, `fallback_i1_nd`, `fallback_i1_ndr`, `fallback_i1_ndr_mmid`), row × candidate interactions (`i1_interactions`), the mechanical-match row feature (`attach_mech`), the regex note columns (`attach_r`), conditional logit, abstention gate, NONE guard (`protect_none`), notes statistics and their guard |
| `src/note_sim.py` | Example similarity of the M3 notes for the safety line: TF-IDF per draw fitted on the training notes only, similarity to notes of real assignments of a code from other questions (`demo_sim`), similarity guard |
| `src/note_regex.py` | Versioned extractor of the regex-exact note columns R (mentions in the M3 notes and in the protocol-channel note) and their guard |
| `src/stage2_sources.py` | Extra second-stage columns of the final submission A and the intermediate form: readers of the frozen protocol answers of our 27B and of the 397B (training and test side, parser cross-check, checksums), the nine column sources (`SOURCE_ORDER`, variants `VARIANTS`), their test guard per column and source (`GUARD_RULE`, `guard_stats`, `guard_check`) and the positive controls (`CONTROLS`) |
| `src/answer_match.py` | Mechanical comparison of a final answer with the answer key without `eval` (AST evaluator, Indonesian number format, LaTeX), reader of the frozen protocol answers, minor-code mask |
| `src/orient.py` | Orientation network: ink maps, training with fixed seeds, inference, rotation lists |
| `kaggle_core/__init__.py`, `src/vlm/__init__.py` | Package markers |

## 3. Model code (`src/vlm/`)

| Module | Role |
|---|---|
| `common.py` | Data root discovery (local, Kaggle, `K12_DATA`), table loading with row-order checksums, shared folds (hash-checked), teacher pool rows, image loading |
| `data.py` | `SampleBuilder`: one sample per row (images, taxonomy header, same-question examples from other folds, topic block, question block, candidates) |
| `prompts.py` | Prompt texts and training targets per task (`diag`, `full`, `cot`, `why`, `read`, `fac`, `ans`, `think`, `up`) |
| `engine.py` | Model loading, encoding, greedy generation, exact scoring of all codes through a prefix tree (`Engine.score`), `probs_from_logp` |
| `train.py` | LoRA fine-tuning with separately averaged loss groups (code, answer check, auxiliary targets, notes) and a cached prompt header |
| `think_vllm.py` | Local vLLM server (127.0.0.1 only) that generates reasoning texts; scoring happens afterwards on the in-house path |
| `proto.py` | Teacher protocol without labelled rows (`proto_gen`), answer parsing, ranking behind frozen notes (`proto_rank`) |
| `sft.py` | M3: training targets, merging LoRA into the base weights, generation (`sft_gen`), ranking in sft mode, combining fold runs |
| `runner.py` | Entry point of all model runs (`python -m vlm.runner --config … --out …`, started from `src/` as in [TRAINING.md](TRAINING.md#2-environment)); job kinds `bench`, `train`, `eval`, `predict`, `think_gen`, `proto_gen`, `sft_gen`, `rationale_gen`, `merge` |

## 4. Step scripts (`eda/`)

| Script | Role |
|---|---|
| `a2_probe_stack.py` | Blend fitting on LogLoss (`fit_mix`) and linear probes; imported by the blend and second-stage scripts |
| `a6_assemble_oof.py` | Assembles fold-wise or multi-draw measurements into one OOF matrix (`--mean`, `--fill`, row lists after `@`) |
| `a8_proto_oof.py` | OOF artifact of a protocol channel (named codes; `--rank` for a pure ranking channel; `--debias`) |
| `a9_m3_draws_oof.py` | OOF artifact of M3 as the mean of several draws; `--verify`, `--gen-status` (generation completeness of a test draw) |
| `c21_margin_align.py` | Class-marginal alignment (`fit_weights`, `apply_weights`); used transductively and inductively |
| `c95_rank_mean.py` | Mean channel `pr27xm`: named codes averaged with the exact ranking behind the same notes; `--status` of a ranking run |
| `c98b_pool_eval.py` | Ranking loader shared with `a9_m3_draws_oof.py` (`load_rank`); otherwise a historical evaluation |
| `c99_m3_deploy.py` | M3 in deployment: blend weights with the channel, blend files of both lines, training-regime check of the test side (`--variant`) |
| `c107_q397_full.py` | Shared recipe code of the primary line (context, base channels, cross-fit, cluster bootstrap); evaluation of the 397B channel |
| `c108_m3_three_draws.py` | Evaluation and deployment verdict of the three-draw M3 channel (the build scripts read its stored verdict) |
| `c110_q397_deploy.py` | 397B deployment path: OOF artifact from the frozen answers, blend files (`stack`, `--replace-q397`, `--arith-beta-select`), test-side status |
| `c115_q397m3_eval.py` | Evaluation of `q397m3` (M3 scoring behind the 397B notes) and its state labels |
| `c117_q397m3_deploy.py` | `q397m3` deployment path: OOF artifact from the fold shards, guard of the test side (settings, notes, log, training regime) |
| `c118_stage2.py` | Second stage of the primary line: nested OOF reproduction (`oof`), deployment files (`fit`, `--guard-rule widened`), null models (`null`), feature set `main_guard` (`guard-oof`, `guard-fit`) |
| `c120_s036_safe_stage2.py` | Second stage of the strictly inductive line: nested OOF reproduction (`oof`), deployment file (`fit`), null model (`null`) |
| `c121_stage2_i1.py` | Row × candidate interactions in the primary line: nested reproduction with placebos (`oof`), new meta-splits (`splits`), expected-value rule (`rule`), deployment file `stage2_primary_i1.json` (`fit`), reference rows of the count features (`test-ref`), test side through `make_submission` (`submit`) |
| `c123_safety_set.py` | Safety line with alignment over the test set: nested reproduction of variants a and b (`oof`), new meta-splits (`splits`, `splits-inductive`), class masses of the test side (`test-mass`), safety-set rule with the recorded override (`rule`), deployment file `stage2_safety_final.json` and probes (`fit`), null model (`null`) |
| `c124_stage2_mm.py` | Minor-code × mechanical-match column in the primary line: shares of `mech27` (`parser`), nested reproduction with placebos (`oof`), new meta-splits (`splits`), expected-value rule (`rule`), deployment file `stage2_primary_final.json` (`fit`), null model with the field `mech27` (`null`) |
| `c129_safety_nd.py` | Note mentions and example similarity in the second stage of the safety line: frozen training notes of the M3 draws (`export-notes`), nested reproduction with copy null point and placebos (`oof`), new meta-splits (`splits`), rule with the risk rule (`rule`), deployment file `stage2_safety_nd.json` (`fit`), null model (`null`), copy null point with the weights of the predecessor (`b-zero`) |
| `c130_safety_ndr.py` | Regex-exact note columns in the second stage of the safety line: re-export check of the frozen protocol notes (`export-notes`), nested reproduction (`oof`), full draw swap over four M3 draws (`swap`), new meta-splits (`splits`), rule with the risk rule (`rule`), deployment file `stage2_safety_ndr.json` (`fit`), `null`, `b-zero` |
| `c132_m3px27_eval.py` | Evaluation of the channel `m3px27` (M3 ranking behind the protocol notes) by its pre-registered rule with controls of the same role; channel artifact `oof_m3px27` (`--oof-save`, `--oof-verify`); the build script reads its stored verdict |
| `c135_primary_m3x4.py` | Final submission B: the four-draw M3 channel `m3x4_full` (`oof`), blend (`stack`), the second-stage chain with the swapped channel (`fit`), copy null point "d4 removed" (`copy`, `clean-copy`), guard of the d4 test side (`test-guard`), null model (`null`); evidence of the four-draw rule (`bank`, `final`, `placebo`, `rule`) |
| `c151_primary_cols.py` | Intermediate form: second stage with the extra columns of `src/stage2_sources.py` (`test-sources`, `fit`, `copy`, `null`), the paired measurement over meta-splits (`measure`), the test guard from a report of `make_submission` (`test-guard`) and its positive controls (`controls`) |
| `c166_primary_m3x5h.py` | Final submission A: digests of the files of draw h (`digest`), the five-draw channel `m3x5h` with the row list (`oof`), blend (`stack`), second stage without and with the extra columns (`fit-base`, `fit`), copy null point "empty list" (`copy`, `clean-copy`), test guard of draw h and its positive controls (`h-guard`, `h-controls`), null model (`null`) |
| `c133_primary_m3px27.py` | Deployment path of the primary candidate with the mean channel `m3px27xm`: channel artifact (`oof`), blend file (`stack`), the second-stage chain refitted with the swapped channel (`fit`), copy null point of the training side (`copy`, `clean-copy`), guard of the test side (`test-guard`), null model (`null`) |
| `risk_rule.py` | Risk rule of the safety-line second stages: empirical-Bayes prior from earlier measurements of the same class and the posterior with selection correction |
| `i5_01_image_stats.py`, `i5_common.py` | Ink maps and image statistics used as input of the orientation network |

## 5. Tools

| Tool | Role |
|---|---|
| `reproduce.sh` (repository root) | One command for the CPU check: `bash reproduce.sh <package-dir>` checks the private package with its `MANIFEST.sha256`, copies its `artifacts/` (writable), links its `data/derived/`, runs the two `make_submission` calls of the quick check and compares both files with their checksums; `--full` also runs the build chain ([VERIFICATION.md §5.1](VERIFICATION.md#51-assemble-cpu)) |
| `tools/gpu_inference.sh` | GPU modes: `tools/gpu_inference.sh rescore\|full <package-dir> [--out DIR] [--dry-run]` runs the test-side jobs in order, copies their outputs to where the CPU tools read them, and writes `new_A.csv` and `new_B.csv` with the two calls of the quick check; it prints only aggregates against `check_A.csv` and `check_B.csv`. `--dry-run` prints every command and checks that each configuration and input folder exists ([VERIFICATION.md §5.2](VERIFICATION.md#52-rescore-and-full-gpu)) |
| `tools/make_inference_configs.py` | Writes `configs/vlm/test_side/` and `configs/vlm/rescore/` from the reference configurations (§7); `--check` compares the files present |
| `tools/repin_test_side.py` | After new protocol or 397B answers on the test rows: `--check` reports which pinned test-side checksums in the deployment files of A and B and in `meta_m3px27xm.json` differ; `--write` re-records them and keeps the old files under `artifacts/repin_backup/`; nothing is refitted |
| `tools/make_blank_list.py` | Blank-photo list (a photo is blank if, in grey scale at 256 × 256, its mean is below 12 and its standard deviation below 8): `check` recomputes it and compares with `data/derived/blank/` (counts and checksums only); `write --out DIR` writes a new list |
| `tools/make_submission.py` | Assembly of a submission from frozen channel outputs: draws, floors, blend, alignment, second stage (`--stage2`), `REL-01` for blank photos, checks; writes the CSV, a JSON record and the test matrix |
| `tools/make_folds.py` | Writes or checks `artifacts/folds.npy` from `configs/folds.parquet` (recorded among the input checksums of the second stage) |
| `tools/orient_list.py` | Rotation list of the `v1r` test side: `spec`, `check` (rejection of the first-version list, coverage, rotation, provenance), `verify-sub`, `compare` |
| `tools/orient_net.py` | Command line of the orientation network (`maps`, `check-maps`, `train`, `infer`, `robust`, `compare`) |
| `tools/make_orient.py` | Rotation lists `<table>_<name>.json` / `_rows.npy` / `.meta.json` from orientation predictions |
| `tools/freeze_m3_texts.py` | Freezes M3 notes and findings so that a second machine scores the same texts |
| `tools/make_q397m3_notes.py` | Notes of the 397B channel in the format of the frozen M3 notes (input of the `q397m3` ranking) |
| `tools/make_m3px27_notes.py` | Notes of the protocol channel in the format of the frozen M3 notes (input of the `m3px27` ranking) |
| `tools/make_charsize_rows.py` | Row lists of the character-height rule (`--max-height 20 --name h20`: training and test rows whose median character height in the photo at long side 1,024 px is at most 20 px), from the stored image statistics of `eda/i5_01_image_stats.py`; label-free |
| `tools/make_m3_targets.py` | Training targets of M3 per fold and for the full fit (`--full`; `--check` rebuilds them in memory and compares them with the files and the adapter recipes), from `train.csv`, the frozen protocol notes and `oof_pr27`, checked against the stored list `artifacts/eda/private/m3_targets/sft_manifest.json` |
| `tools/export_proto_notes.py` | Export of the first-pass reasoning of the protocol answers into one note file per table (`data/derived/proto_notes/`, `proto_notes_test/`); recorded state labels |
| `tools/replay/gates.py` | Regenerates the gate lists of the reasoning channels (`data/derived/think_rows/`) and checks them against the frozen lists |
| `tools/teacher_api.py` | **Development only.** Teacher pseudo-labels for training and pool rows through the Anthropic API (never test rows; key from the environment; spending cap) |
| `tools/open_teacher.py` | **Development only.** Measures open models through the Tinker API on training rows (the 397B training side) |

## 6. Build scripts (`tools/builds/`)

One script per submitted or candidate file. Each script verifies its pinned inputs by checksum, rebuilds or verifies its
blend and second-stage files, runs `tools/make_submission.py` and checks the output. Expected checksums are in
[INFERENCE.md](INFERENCE.md#2-lines-and-expected-checksums); which inputs are pinned is described in
[VALIDATION.md](VALIDATION.md#6-checks-before-a-submission).

| Script | Output | Role |
|---|---|---|
| `orient_list.sh` | — | Sourced helper: choice of the rotation list (`ORIENT=repro`, default, or `ORIENT=legacy`) and its checksums |
| `s029_m3x3.sh safe` | `sub_s030n_safe.csv` (repro) or `sub_v1r_zs27t_think18sb_pr27xm_m3x3_full_ar_ia_t4.csv` (legacy) | Strictly inductive line without second stage |
| `s029_m3x3.sh best` | `sub_v1r_zs27q_think18nb_pr27xm_m3x3_full_w14_ar_ra_t4.csv` | Historical five-channel primary file (only with `ORIENT=legacy`) |
| `s029_q397.sh` | `sub_s029_*.csv` | Historical six-channel files with the 397B answer channel (first-version rotation list) |
| `s029_q397m3.sh best` | `sub_s032n_q397m3.csv` (repro) or `sub_s029_best_q397m3.csv` (legacy) | Primary line without second stage (six channels with `q397m3`) |
| `s032_stage2.sh best` | `sub_s033n_stage2.csv` (repro) or `sub_s032_stage2.csv` (legacy) | Primary line, reranker fallback form |
| `s034_stage2_main.sh best` | `sub_s034n_stage2_main.csv` (repro) or `sub_s034_stage2_main.csv` (legacy) | Primary line, reranker main form (widened note guard) |
| `s030n_safe.sh`, `s033n_stage2.sh`, `s034n_stage2_main.sh` | as above | Entry points that run the three scripts above with `ORIENT=repro` |
| `s035_stage2_guard.sh` | `sub_s035_stage2_guard.csv` | Primary line, reranker main form with the NONE guard and general-code identifiers (released rotation list) |
| `s036_safe_stage2.sh` | `sub_s036_safe_stage2.csv` | Strictly inductive line with the second stage without notes and the NONE guard (released rotation list) |
| `primary_i1.sh` | `sub_primary_i1.csv` | Primary line, I1 form: the guard form with row × candidate interactions (`PRIMARY_I1_SPLITS=1` recomputes the meta-splits) |
| `primary_final.sh` | `sub_primary_final.csv` | Primary line, final candidate: the I1 form with the minor-code × mechanical-match column (`PRIMARY_FINAL_PROBE=1`, `PRIMARY_FINAL_SPLITS=1`) |
| `safety_final.sh` | `sub_safety_final.csv` | Safety line, alignment over the test set, second stage with interactions and NONE guard; needs `OVERRIDE_CV_II=override_cv_condition_ii` (`SAFETY_FINAL_PROBE=1`, `SAFETY_FINAL_VARIANT=a\|b`, `SAFETY_FINAL_SPLITS=1`) |
| `safety_nd.sh` | `sub_safety_nd.csv` | Safety line: the second stage of `safety_final` with the note mentions and the example similarity (`SAFETY_ND_SPLITS=1` recomputes the meta-splits; `SAFETY_ND_NO_REGRESSION=1` skips step 0, for debugging only) |
| `safety_ndr.sh` | `sub_safety_ndr.csv` | Safety line, final candidate: `safety_nd` with the six regex-exact note columns (`SAFETY_NDR_RECOMPUTE=1` recomputes the draw swap and the meta-splits; `SAFETY_NDR_NO_REGRESSION=1`) |
| `primary_m3px27.sh` | `sub_primary_m3px27.csv` | Primary line, candidate of version 0.3: the final form of `primary_final` with the mean channel `m3px27xm` instead of `pr27xm` (`PRIMARY_M3PX27_REGRESSION=primary_final\|full\|0`) |
| `primary_m3x4.sh` | `sub_primary_m3x4.csv` | **Final submission B**: `primary_m3px27` with the four-draw M3 channel `m3x4_full`; step 0 runs `primary_m3px27.sh` |
| `primary_cols.sh` | `sub_primary_cols.csv` | Intermediate form: submission B with the nine extra second-stage columns; step 0 runs `primary_m3x4.sh` |
| `primary_m3x5h.sh` | `sub_primary_m3x5h.csv` | **Final submission A**: the intermediate form with the M3 channel `m3x5h` (fifth draw at 2,048 px on the listed rows); step 0 runs `primary_cols.sh` |

The scripts are chained: the main-form script first runs the fallback-form script, which first runs the script without
second stage; `s035` runs `s034n_stage2_main.sh` and `s036` runs `s030n_safe.sh`; `primary_final.sh` runs
`primary_i1.sh`, which runs `s035`; `safety_final.sh` runs `s036`; `safety_nd.sh` runs `safety_final.sh` (with the recorded
override) and `primary_final.sh`; `safety_ndr.sh` runs `safety_nd.sh`; `primary_m3px27.sh` runs `primary_final.sh` and
requires the files of `safety_nd.sh` and `safety_ndr.sh` (it checks their checksums). The released-list builds compare their
output with the legacy-list file of the same line, so the legacy-list files must be built first.

## 7. Run configurations (`configs/vlm/`)

| Channel / step | Training side | Test side |
|---|---|---|
| `v1` (Qwen3.5-9B LoRA) | `ft_sum1_rp` (fold 1), `ft_v1a_rp`, `ft_v1b_rp`, `ft_v1c_rp` (full fit) | `v1_test_rp` (A40), `v1te_a_t4`, `v1te_b_t4` (Kaggle T4, offline) |
| `v1r` (rotation-corrected `v1`) | `v1rot_t4` (`rot_f*`, `base_f*`) | `v1rot_t4` (`rot_test`) |
| `zs27q` (27B zero-shot, topic + task block) | `zs27q2_f1_rp`, `zs27q_oof_rp` | `zs27q_test0_rp`, `zs27q_test1_rp` |
| `zs27t` (27B zero-shot, topic block) | `zs27t_all_rp` | `zs27t_all_rp`, `px27_t4zt0`, `px27_t4zt1` (T4, offline) |
| reasoning channel, draw g | `zs27v_f2_rp`, `zs27v_cut_rp`, `zs27v_all_rp`, `zs27v_te_rp` | `zs27v_all_rp`, `zs27v_te_rp` |
| reasoning channel, draw n | `zs27v_n_rp`, `zs27v_n3_rp` | `zs27v_n3_rp` |
| `pr27` / `pr27xm` (teacher protocol) | `pr27_rp`, `px27_rp`; second draw of the answers `pr27s2_rp`, `pr27s3_rp` (sensitivity check of the safety line only) | `pr27te_rp`, `n1_rp` (`rank_test`) |
| `m3px27` / `m3px27xm` (M3 behind the protocol notes) | `m3px27_oof_rp` | `m3px27te_rp` |
| M3 | `m3_pilot_rp`, `m3_f034_rp`, `m3long_rp`, `cv01_m3d2_f034`, `m3d3_oof_rp`, `m3_full_rp`; fourth draw `m3d4_oof_rp`; draw h at 2,048 px `m3ha_oof_rp`, `m3hb_oof_rp` | `m3_fullte_rp` (d1; `m3d1te_rp`: the same test jobs with the delivered adapter, without training), `m3d2te_rp` (d2), `m3d3te_rp` (d3), `m3d4te_rp` (d4), `m3hte_a_rp`, `m3hte_b_rp` (h, listed rows) |
| `q397` / `q397m3` | `q397p_rp` (fidelity check), `q397m3_oof_rp`; never run: `q397p8_rp` | `q397te_rp`, `q397m3te_rp`; never run: `q397te8_rp` (8×H100 variant) |
| CPU smoke tests | `local_cpu_test`, `local_m3_test`, `local_proto_rank` | — |

**Inference-only copies.** `configs/vlm/test_side/` and `configs/vlm/rescore/` hold the test-side jobs of the reference
configurations, unchanged and under the same run and job names, so that their outputs land where the CPU tools read them.
`test_side/` (for `full`) leaves out the training jobs and the jobs on training rows, except short pre-check jobs.
`rescore/` also leaves out every job that generates text. `tools/make_inference_configs.py` writes both folders, and
`tests/test_inference_configs.py` checks that each job equals the job of the same name in the reference file. Do not run
`m3_fullte_rp.yaml` itself for inference: its training jobs retrain M3 and overwrite the delivered adapter.

Some scripts and comments name runs of variants that are not part of the delivered lines; their configurations are not
included: `zs27v_n2_rp` (an earlier run of reasoning draw n, whose gate lists `tools/replay/gates.py` still describes),
`m3p_f1_rp`, `m3p_f2_rp` and `m3p_pool` (M3 trained with pool rows, evaluated by `eda/c98b_pool_eval.py` and not
adopted), and `pool_ans_t4` (an early run on pool photos whose row order the teacher plan reuses).

## 8. Tests

`python -m pytest tests -q`. Tests that need private files skip themselves when the files are missing.

| Test file | Covers |
|---|---|
| `conftest.py` | Import paths (repository root, `src/`, `tools/`) |
| `test_metric.py`, `test_submission_io.py`, `test_transcription.py` | Metric, submission checks, transcription cleaning |
| `test_make_submission.py` | Inductive alignment, blend with arithmetic share, protocol channels |
| `test_make_folds.py` | `artifacts/folds.npy` from `configs/folds.parquet` |
| `test_stage2.py` | Second stage (features, fit, gate, notes guard, deployment files of the reranker fallback and main forms) |
| `test_stage2_guard.py` | Feature set `main_guard`, NONE guard, `guard-oof`/`guard-fit`, checksums of `s035_stage2_guard.sh` |
| `test_stage2_safe.py` | Second stage of the inductive line (`apply_stage2_safe`, command-line checks, `c120`, checksums of `s036_safe_stage2.sh`) |
| `test_stage2_i1.py` | Interaction set `main_guard_i1` (feature order, permutation placebo, zero-weight null equals the guard form), `c121` checks, checksums of `primary_i1.sh` |
| `test_stage2_mm.py` | Column `minor_x_mech27` and set `main_guard_i1_mm`, the field `mech27` of the deployment file, `c124` checks, checksums of `primary_final.sh` |
| `test_safety_set.py` | Safety line with set alignment (`apply_stage2_safe` with `align_mode` `set`, set `fallback_i1`), the recorded override and its pinned checksum, `c123` checks, checksums and switches of `safety_final.sh` |
| `test_safety_nd.py` | Feature set `fallback_i1_nd`, example similarity and its guard, positive controls of `apply_stage2_safe`, `c129` checks, checksums of `safety_nd.sh` |
| `test_safety_ndr.py` | Extractor of the regex note columns and their guard, feature set `fallback_i1_ndr`, controls of the protocol notes, rule logic of `c130`, checksums of `safety_ndr.sh` |
| `test_m3px27.py` | Notes converter `tools/make_m3px27_notes.py`, the evaluation `c132` (technical checks, early stop, rule, controls) |
| `test_primary_m3x4.py` | `c135`: four-draw channel, bank and rule computations, copy null point, `make_submission.stage2_draws_first`, pins of `primary_m3x4.sh` |
| `test_primary_cols.py` | `src/stage2_sources.py` (column definitions, parser cross-check, guard, controls), `c151` checks, the extra-column path of `make_submission`, pins of `primary_cols.sh` |
| `test_primary_m3x5h.py` | `c166`: five-draw channel with row list, guards and controls of draw h, the row-list draw of `make_submission`, pins of `primary_m3x5h.sh` |
| `test_primary_m3px27.py` | Note guard and comparison form of the mean channel in `make_submission`, `c118.build_stage2` with a given deployment form, `c133` checks, checksums of `primary_m3px27.sh` |
| `test_make_m3_targets.py` | Record format and loss rule of the M3 target builder, mapping to the adapter recipes; with the private files the byte-identical rebuild of the six target files |
| `test_inference_configs.py` | `configs/vlm/test_side/` and `configs/vlm/rescore/`: every job equals the job of the same name in the reference configuration, no training job, no generation job in `rescore/` |
| `test_repin_test_side.py` | `tools/repin_test_side.py` on a temporary copy with synthetic files: `--check` without change, `--write` after a change, backup written, nothing else changed |
| `test_make_blank_list.py` | `tools/make_blank_list.py` on a synthetic black and a normal image; the test on the competition photos skips without the data |
| `test_answer_match.py` | Mechanical answer comparison (number formats, fractions, LaTeX, units, no `eval`) |
| `test_orient_list.py` | Choice and checks of the rotation list (`orient_list.sh`, `tools/orient_list.py`) |
| `test_orient_repro.py` | Orientation network and rotation lists |
| `test_gates.py` | Gate rules and gate tool |
| `test_c95_rank_mean.py`, `test_c99_m3_deploy.py`, `test_m3_draws.py`, `test_c108.py` | Protocol mean channel and M3 deployment |
| `test_c107.py`, `test_c110_q397_deploy.py`, `test_q397m3.py`, `test_s029_q397m3.py` | 397B channels and their build path |
| `test_vlm.py`, `test_vlm_parsers.py`, `test_m3_sft.py`, `test_train_init_partial.py` | Model code without a model (parsers, targets, scoring logic, adapter transfer) |
| `test_c97_reason_parse.py`, `test_open_teacher_continue.py` | Development API clients without network |

## 9. Channel names and file formats

| Name | Channel | Lines |
|---|---|---|
| `v1`, `v1r` | 9B fine-tuned scorer; `v1r` reads rotated photos upright | both (`v1r`) |
| `zs27q`, `zs27t` | 27B zero-shot scorer with task block / topic block | primary / inductive |
| `think18nb`, `think18sb` | 27B reason-then-name (two draws / teacher-free draw) | primary / inductive |
| `pr27`, `pr27xm` | 27B protocol channel: named codes / mean with the exact ranking | both (`pr27xm`; in the primary line only in the fallbacks) |
| `m3px27`, `m3px27xm` | M3 ranking behind the protocol notes / mean of the named protocol codes with it | primary (`m3px27xm`, final candidate) |
| `m3x3_full` | M3, mean of three draws | both |
| `m3x4_full` | M3, mean of four draws (d1–d4) | primary (submission B) |
| `m3x5h` | M3, mean of the draws that cover a row: d1–d4, and draw h at 2,048 px on the rows with small script | primary (submission A) |
| `q397`, `q397m3` | 397B answers / M3 scoring behind the 397B notes | primary (`q397m3`) |

| File | Format |
|---|---|
| `artifacts/oof_<channel>.npy` | float64 `[1500, 76]`, rows in `train.csv` order, columns `NONE` + taxonomy |
| `artifacts/meta_<channel>.json` | Sources, checksums and settings of an OOF artifact |
| `artifacts/stack_<name>.json` | Blend file: channels, floors, weights θ, arithmetic share β, last step, cross-validated value |
| `artifacts/stage2_<name>.json` | Second-stage deployment file: models (features, weights, standardisation, gate), guards, input checksums |
| `artifacts/test_<name><suffix>.npy` | float64 `[3498, 76]` test probabilities of an assembled file |
| `artifacts/submissions/sub_<name><suffix>.csv` / `.json` | Submission and its record (sources, checksums, aggregates) |
| `artifacts/eda/private/<topic>/` | Reports of the step scripts and build scripts (aggregates only) |
| `data/derived/orient/`, `data/derived/orient_repro/` | Rotation lists: first-version network (frozen) / released network |

## 10. Steps whose scripts are not part of this repository

Some frozen inputs were produced by development helper scripts that are not included. Their outputs are delivered as
frozen files; the steps that consume them are in this repository. How each input is checked when it is read (pinned
checksum, recorded checksum, field checks, or only indirectly through the pinned output files) is described in
[VALIDATION.md](VALIDATION.md#6-checks-before-a-submission).

Some scripts write provenance or note fields into their output files (channel metadata, records of the assembly,
reports) that name development scripts which are not included. These texts are kept unchanged, so that the scripts
still write the files that were delivered.

| Step | Frozen output | Consumed by |
|---|---|---|
| Combination of fold runs of M3 draws d1 and d2 | `artifacts/runs/m3_5f/`, `artifacts/runs/m3long_comb/` | `eda/a9_m3_draws_oof.py` |
| 397B fidelity check on 64 training rows | `artifacts/eda/private/q397_deploy/c109_parity.json` | the build scripts of the primary line (stored verdict) |
| Posterior step of the M3 decision rule | `artifacts/eda/private/m3_draws_eval/c108_m3_three_draws.json` | the build scripts (stored verdict) |
| Bundle test of the extra columns and fidelity check of the column sources | measurement files under `artifacts/eda/private/` (reported in [VALIDATION.md](VALIDATION.md#8-final-submissions-version-04)) | not read by the build scripts of this repository; `c151` pins the resulting values |
| Evaluation of the five-draw rule (draw h) | evaluation files under `artifacts/eda/private/` | not read by the build scripts of this repository; `c166` pins the resulting values |
