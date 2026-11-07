# Changelog

This file records notable changes to the solution repository. Dates are UTC. The repository is a living document until
the final solution package is delivered.

## v0.2 — 2026-11-07

Third pre-final version: the final candidates of both lines.

### Added

- Primary line: row × candidate interactions in the second stage (feature set `main_guard_i1`, `i1_interactions` in
  `src/stage2.py`; `eda/c121_stage2_i1.py` with `oof`, `splits`, `rule`, `fit`, `test-ref` and `submit`; build script
  `tools/builds/primary_i1.sh`; tests `tests/test_stage2_i1.py`).
- Primary line, final candidate: the minor-code × mechanical-match column (feature set `main_guard_i1_mm`); the number
  parser without `eval` `src/answer_match.py`; `eda/c124_stage2_mm.py` with `parser`, `oof`, `splits`, `rule`, `fit` and
  `null`; build script `tools/builds/primary_final.sh`; tests `tests/test_answer_match.py`, `tests/test_stage2_mm.py`.
- Safety line, final candidate: the alignment over the test set with a second stage fitted anew in this form (feature sets
  `fallback` and `fallback_i1`; `eda/c123_safety_set.py` with `oof`, `splits`, `splits-inductive`, `test-mass`, `rule`,
  `fit`, `null`); build script `tools/builds/safety_final.sh`; tests `tests/test_safety_set.py`.
- `decisions/override_cv_condition_ii.json`: the recorded decision to build the safety candidate although the
  cross-validation part of one condition of its pre-registered rule is missed by one sixth in 9,000. Its SHA-256 is pinned
  in `eda/c123_safety_set.py`; the build script applies it only with `OVERRIDE_CV_II=override_cv_condition_ii`.

### Changed

- `src/stage2.py`: the feature sets `main_guard_i1`, `main_guard_i1_mm` and `fallback_i1`, and `attach_mech`; the
  existing feature sets and their files are unchanged, bit for bit.
- `tools/make_submission.py`: deployment files with `align_mode: set` (the safety line with alignment over the test set)
  run the second-stage path without notes and report the class masses; main models may name `main_guard_i1` and
  `main_guard_i1_mm`, the latter with the frozen test answers of the protocol channel checked by SHA-256. The submission
  files and test matrices of the earlier forms are rebuilt byte for byte; the new report fields are written only for the
  new forms, and the JSON records of files with the alignment over the test set no longer say "primary line only" in
  their description of the alignment.
- Since 2026-11-06 the alignment over the test set is admitted in both final submissions; the strictly inductive form of
  the safety line stays its fallback (`docs/DATA_AND_COMPLIANCE.md`, `docs/METHOD.md`).
- Documentation updated for this version: `README.md`, `docs/METHOD.md`, `docs/VALIDATION.md`, `docs/RESULTS.md`,
  `docs/TRAINING.md`, `docs/INFERENCE.md`, `docs/CODEMAP.md`, `docs/ROADMAP.md`, `docs/DATA_AND_COMPLIANCE.md`. Messages
  and help texts that said "primary line only" for the alignment over the test set now name both lines. Statements that
  no reranker feature looks at the final answer now name the exception of the final primary form.

### Fixed

- Abstention gate of the reranker: the fallback form reaches 0.8347, not 0.8351 (`docs/METHOD.md`, `docs/TRAINING.md`).
- The parity check of the 397B path compared the number of input tokens per row, not the inputs themselves
  (`docs/METHOD.md`, `docs/DATA_AND_COMPLIANCE.md`).
- The leave-one-out values marked for the inductive line in `docs/RESULTS.md` come from an inductive recipe with the
  397B answers added; this is now stated.
- Three remaining German words in a test name, an error message of `src/stage2.py` and a comment.

### Verification

- Two fresh copies of the repository with the private package, the data only as read-only links:
  - README quick start run verbatim: all 20 reference files byte-identical (`sub_primary_i1.csv`,
    `sub_primary_final.csv`, `sub_safety_final.csv` and their test matrices, the files and the test matrix of version
    0.1, `folds.npy`); the `make_submission` reference calls of `docs/INFERENCE.md` §5, run verbatim afterwards, give the
    same files.
  - Write path, with every file a build script can write deleted beforehand: the same 20 files byte-identical; the
    deployment files, meta-split files and probe files are rebuilt with their pinned checksums. Without
    `OVERRIDE_CV_II` the safety build stops with exit code 5; another value, or the override together with the probe
    switch, stops with exit code 2.
- Tests: 504 passed and 4 skipped with the private package; 442 passed and 66 skipped with the repository files only.

### State of the solution

| Line and form | Status | CV MAP@3 (per fold / as one set) | Public LB |
|---|---|---|---:|
| Primary line, final form (interactions and minor-code × mechanical-match column) | current candidate; submitted 2026-11-06 | 0.8409 / 0.8398 | 0.84100 |
| Primary line, I1 form | fallback; submitted 2026-11-06 | 0.8408 / 0.8387 | 0.84150 |
| Primary line, guard form | submitted 2026-11-01 | 0.8389 / 0.8377 | 0.84200 |
| Safety line, alignment over the test set, reranker with interactions | current candidate; submitted 2026-11-07 | — / 0.8217 | 0.83516 |
| Safety line, strictly inductive form with second stage | fallback; submitted 2026-11-01 | 0.8233 / — | 0.83183 |

### Known gaps

The following items are tracked in [docs/ROADMAP.md](docs/ROADMAP.md):
- the offline package: lock hashes, wheelhouse, offline runner, single entry point, manifests;
- smoke tests without network access for the model steps, including the 397B path;
- the remaining second-hardware checks of the M3 draws;
- an estimate of the rerun variance of the generated texts;
- the private copies of the frozen inputs and weights in the state of this version;
- the per-package licence table.

## v0.1 — 2026-11-01

Second pre-final version: new second-stage components in both lines and the released rotation list in all final builds.

### Added

- Primary line: feature set `main_guard` (identifiers of the five general codes) and the NONE guard in `src/stage2.py`;
  commands `guard-oof` and `guard-fit` in `eda/c118_stage2.py`; build script `tools/builds/s035_stage2_guard.sh`
  (reranker main form with NONE guard and general-code identifiers); tests `tests/test_stage2_guard.py`.
- Strictly inductive line: second stage with the features without notes and the NONE guard
  (`eda/c120_s036_safe_stage2.py` with `oof`, `fit` and `null`; inductive branch in `tools/make_submission.py` for
  deployment files with `align_mode: inductive`); build script `tools/builds/s036_safe_stage2.sh`; tests
  `tests/test_stage2_safe.py`.
- Released rotation list: `tools/orient_list.py` (`spec`, `check`, `verify-sub`, `compare`), the sourced helper
  `tools/builds/orient_list.sh` (`ORIENT=repro|legacy`), the entry points `tools/builds/s030n_safe.sh`,
  `s033n_stage2.sh` and `s034n_stage2_main.sh`; tests `tests/test_orient_list.py`.
- `tools/make_folds.py`: writes or checks `artifacts/folds.npy` from `configs/folds.parquet`; `--recompute` confirms
  the split from `train.csv`. Tests `tests/test_make_folds.py`.
- `docs/CODEMAP.md`: every script and module with its role.

### Changed

- The build scripts of the final lines use the list of the released orientation network by default (`ORIENT=repro`);
  `ORIENT=legacy` rebuilds the historical files byte for byte.
- `tools/make_submission.py` takes the feature set and the NONE guard from the model in the deployment file.
- Code and test identifiers use the terms of the documentation (for example `widened` for the widened rule of the note
  guard, `MAIN_FORM_FILE` in the tests). Only names, comments and help texts changed; all files are rebuilt byte for byte.
- Documentation updated for this version: status and results in `README.md`; `docs/METHOD.md` (NONE guard, general-code
  identifiers, widened note guard, second stage of the inductive line, released rotation list); `docs/VALIDATION.md`
  (checks of the second-stage components, risk rule of the inductive line and its sensitivity, checks before a
  submission); `docs/RESULTS.md`; `docs/TRAINING.md`; `docs/INFERENCE.md` (expected checksums of all lines, inputs per
  line from a trace of the files the build scripts open); `docs/ROADMAP.md`; `docs/DATA_AND_COMPLIANCE.md`;
  `THIRD_PARTY.md`.

### Fixed

- The documentation, comments and help texts no longer point to files that are not part of the repository; planned
  items are marked as planned, and steps whose scripts are not included are listed in `docs/CODEMAP.md`. The v0 entry
  below had listed environment lock files under "Added" in error; they were not part of v0, and the line has been
  removed.
- The checks of the build scripts are described as they are: which inputs are pinned by checksum and which are covered
  only through the pinned output files.
- The environment of the assembly lists every package it imports.
- The two historical build scripts call the assembly in isolated mode (`python -I`), like every other Python step of the
  build scripts; the files are rebuilt byte for byte.
- Corrected numbers: single-channel MAP@3 of the 397B answers (0.7432), number of `OTH-00` rows with the gold code at
  rank 1 (17 of 57), Anthropic API cost (about 43 USD) and the training-side effect of the released rotation list (a
  channel value, not a blend value).
- A test that needs the competition data now skips when the data are absent instead of failing.

### Verification

All candidate, fallback, intermediate and historical files were rebuilt byte for byte from the frozen inputs in fresh
copies of the repository. The tests pass both with the private artifacts and in the bare repository, where tests that
need private files skip themselves.

### State of the solution

| Line and form | Status | CV MAP@3 (per fold / as one set) | Public LB |
|---|---|---|---:|
| Primary line, reranker main form with NONE guard and general-code identifiers | current candidate; submitted 2026-11-01 | 0.8389 / 0.8377 | 0.84200 |
| Primary line, reranker main form | fallback; submitted 2026-11-01 | 0.8376 / 0.8346 | 0.84166 |
| Primary line, reranker fallback form | submitted 2026-10-31 (first-version rotation list) | 0.8356 / 0.8338 | 0.84033 |
| Strictly inductive line with second stage and NONE guard | current candidate; submitted 2026-11-01 | 0.8233 / — | 0.83183 |
| Strictly inductive line without second stage | fallback; submitted 2026-10-31 with the first-version rotation list | 0.8184 / — | 0.82966 |

### Known gaps

The following items are tracked in [docs/ROADMAP.md](docs/ROADMAP.md):
- the offline package: lock hashes, wheelhouse, offline runner, single entry point, manifests;
- smoke tests without network access, including the 397B path;
- the remaining second-hardware checks for the inductive candidate;
- an estimate of the rerun variance of the generated texts;
- the private copies of the frozen inputs and weights in the state of this version;
- the per-package licence table.

## v0 — 2026-11-01

First version of the solution repository (pre-final).

### Added

- Code for assembling a submission from frozen channel outputs (`tools/make_submission.py`, build scripts in
  `tools/builds/`), for model runs (`src/vlm/`, run configurations in `configs/vlm/`), for training targets, the
  orientation network, the second stage, OOF assembly, blend fitting and the checks run before a submission, and the
  tests.
- Documentation: `README.md`, `docs/METHOD.md`, `docs/VALIDATION.md`, `docs/RESULTS.md`, `docs/TRAINING.md`,
  `docs/INFERENCE.md`, `docs/DATA_AND_COMPLIANCE.md`, `docs/ROADMAP.md`, `THIRD_PARTY.md`; licence Apache 2.0.

### State of the solution

| Line | Status | CV MAP@3 (per fold / as one set) | Public LB |
|---|---|---|---:|
| Primary line, reranker fallback form | submitted 2026-10-31 | 0.8356 / 0.8338 | 0.84033 |
| Primary line, reranker main form | built and checked, not submitted | 0.8376 / 0.8346 | — |
| Strictly inductive line | submitted 2026-10-31 | 0.8184 / — | 0.82966 |

### Known gaps

The following items are tracked in [docs/ROADMAP.md](docs/ROADMAP.md):
- the switch to the rotation list of the released orientation network;
- a rebuild of the inductive line with the current code;
- the offline package: lock hashes, wheelhouse, offline runner, single entry point, manifests;
- smoke tests without network access;
- the remaining second-hardware checks;
- the per-package licence table.
