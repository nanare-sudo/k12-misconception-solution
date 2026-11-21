# Changelog

This file records the changes of each version of the solution repository. Dates are UTC. Version 0.6 is the state
delivered for the host's verification.

## v0.6 — 2026-11-21

Seventh version, the state delivered for the host's verification. It adds one command for the CPU check and the GPU
steps of inference from the delivered weights without retraining. The private input package is completed.

### Verified

- Two fresh clones of this version with the delivered zip, following README.md and docs/VERIFICATION.md: the quick check
  wrote `OK A 4505c159…797354e7` and `OK B 76da786d…a3565f9973` in 1 min 21 s and 1 min 14 s, and
  `bash reproduce.sh <package-dir> --full` passed in 1 h 33 min and 1 h 32 min. The second clone used the CPU build of
  torch. All 574 files of `MANIFEST.sha256` matched, and every package file was unchanged afterwards.
- `tools/gpu_inference.sh rescore` and `full` as dry runs: no input missing.
- `tools/make_blank_list.py check` (1 training row, 6 test rows), `tools/replay/gates.py check` (test rows with the A40
  outputs, and training rows), the rebuild of the 397B training notes, `tools/make_m3_targets.py --full --check` and
  `tools/repin_test_side.py --check` (12 of 12 fields unchanged): bit-identical or exit 0.
- Tests after `--full` in a fresh clone: 685 passed and 15 skipped without `K12_PACKAGE` and without the optional 0.8B
  model. With these and the photos in place, the five tests that need them pass as well, so 690 pass and 10 skip. After
  the quick check only, with all inputs: 662 passed and 38 skipped. None failed.

### Added

- `reproduce.sh`: one command for the CPU check. `bash reproduce.sh <package-dir>` checks the private package with its
  `MANIFEST.sha256`, copies its `artifacts/` (writable) and links its `data/derived/`, runs the two `make_submission`
  calls of the quick check and compares both files with their checksums (about 2 min). `--full` also runs the build
  chain (about 2 h).
- `tools/gpu_inference.sh rescore|full <package-dir> [--out DIR] [--dry-run]`: runs the GPU jobs on the test rows in
  order, copies their outputs to where the CPU tools read them, writes `new_A.csv` and `new_B.csv` with the two calls of
  the quick check and prints only aggregates. `--dry-run` prints every command and checks that each configuration and
  input folder exists. Its input check is `tools/gpu_inference_inputs.py`; tests `tests/test_gpu_inference_dry_run.py`.
- `configs/vlm/test_side/` and `configs/vlm/rescore/`: the jobs on the test rows of the reference configurations,
  unchanged and under the same run and job names; `rescore/` holds no job that generates text. Written by
  `tools/make_inference_configs.py` (`--check` compares) and checked by `tests/test_inference_configs.py`.
- `tools/repin_test_side.py` (`--check`, `--write`): re-records the pinned checksums of new protocol and 397B answers on
  the test rows in the deployment files of A and B and in `meta_m3px27xm.json`; it keeps the old files and refits
  nothing. Tests `tests/test_repin_test_side.py`.
- `tools/make_blank_list.py` (`check`, `write --out DIR`): the blank-photo rule, which was only delivered as a frozen
  list. Tests `tests/test_make_blank_list.py`.
- `docs/GLOSSARY.md`: the internal names of channels, runs and scripts, and the naming schemes of files and folders.

### Changed

- `docs/VERIFICATION.md`, `README.md` and `docs/WRITEUP.md` rewritten in plain English. VERIFICATION §5 starts with the
  quick check and gives the GPU modes `rescore` and `full` step by step. It also gives the download commands of the base
  models, the install of the CPU build of torch, the disk that the CPU check needs and how to run the tests. The
  write-up is shorter.
- Private input package: the metadata of our own files are cleaned (English descriptions, no internal references). The
  numbers in these files are unchanged. Where the code pins one of these files, it pins the new checksum, and
  `MANIFEST.sha256` covers all of them. Model outputs and run logs are unchanged.
- Private input package completed with the inputs of the training steps (first-version rotation lists of the training
  rows, row lists of the protocol-answer runs, the raw 397B answers on the training rows, the gate lists of the training
  rows, `artifacts/oof_v1.npy`) and with the inputs of the checks (the A40 test outputs from which the gate lists are
  computed, and the blank-photo list of the training rows). It holds 572 indexed files with 3.63 GB, 161 of them
  derived from test rows. The zip file has writable file permissions.
- Reference documents (METHOD, INFERENCE, TRAINING, VALIDATION, RESULTS, CODEMAP, PROVENANCE, DATA_AND_COMPLIANCE,
  MODEL_SIZES): version 0.6, storage locations, run status, the new tools, the reference CPU and its OpenBLAS kernel, the
  complete offline settings and the blank-photo rule. `docs/weights_manifest.json` lists the five M3 fold adapters and
  names the folders of the private package. ROADMAP shows the state of the delivery.

### Fixed

- GPU inference from the delivered weights. Followed as written, the earlier instructions retrained M3 and overwrote the
  delivered adapter (`m3_fullte_rp.yaml` starts with training jobs), stopped at a pre-check job of `v1rot_t4.yaml` whose
  inputs were not in the package, and regenerated every M3 text in `rescore`. The new configuration folders fix all three.
- Assembly after a `full` rerun. The deployment files pin the checksums of the frozen protocol and 397B answers on the
  test rows; the documentation said that a file from new answers could not be assembled. `tools/repin_test_side.py`
  re-records these checksums.
- Read-only copies. The files of the previous zip were read-only, so a copy made with `cp -r` stopped the build with a
  permission error. `reproduce.sh` makes its copy writable, and the new zip has writable permissions.
- CPU class. On a CPU on which OpenBLAS selects another kernel than on the reference CPU, the refitted weights differ by
  about 1e-8 and the build chain stops at its first refit check. The documentation names the reference CPU and kernel,
  `reproduce.sh --full` sets `OPENBLAS_CORETYPE=Haswell`, and the quick check refits nothing.
- Offline settings. The documented list lacked `HF_DATASETS_OFFLINE`, `HF_HUB_DISABLE_TELEMETRY`, `VLLM_DO_NOT_TRACK`,
  `VLLM_HOST_IP` and the socket interfaces of Gloo and NCCL; without `VLLM_HOST_IP`, vLLM makes a UDP connect to a public
  address. The removal of the image's `torchao` package is documented.
- Documentation. Storage locations named private Kaggle resources that the host cannot open; the reference documents now
  name the private input package. `docs/CODEMAP.md` listed `q397te8_rp` among the configurations that are not included.
  `docs/PROVENANCE.md` said that every M3 recipe records `teacher_derived`; the two pilot recipes do not.
  `docs/INFERENCE.md` said that M3 uses exactly three draws, and that `m3x3_full_rows.npy` is rewritten by
  `s029_m3x3.sh`; that script writes it only when the metadata file of the channel is absent.
- `docs/WRITEUP.md` names both commercial APIs and their frozen outputs, and gives the range of single M3 draws as
  0.750–0.777.
- Comments in six run configurations and in `src/vlm/sft.py` pointed to a section of `docs/CODEMAP.md` that no longer
  lists the M3 training targets and the protocol notes; they now name the builder and `docs/TRAINING.md`.

### Known gaps

The following items are tracked in [docs/ROADMAP.md](docs/ROADMAP.md#2-open-steps):
- runs without network access for the GPU model steps other than the 9B scorer, including the 397B path (step 2);
- hashes for every pin and an offline wheelhouse (step 4);
- the scripts that are not exported (step 6);
- a complete `rescore` or `full` run in one go, which has never been done; `tools/gpu_inference.sh` has run as a dry
  run and on its CPU steps, not yet on a GPU;
- the full proof on a CPU with AVX-512; the quick check does not depend on the CPU;
- the private-leaderboard scores (step 8).

## v0.5 — 2026-11-17

Sixth version: the first build of both final submissions with the code of this repository, the checksums it learned, the
private input package for the host's verification, pinned requirements, the M3 target builder and an inference-only
configuration of the first M3 draw.

### Verified

- **First build with the code of this repository.** The README quick start ran verbatim in a fresh copy of the repository
  whose `artifacts/` and `data/derived/` came from the private inputs (the frozen files of the internal store, and the private
  copies of the files that earlier versions of this repository write with English fields). Every build script exited with 0
  and every pinned file matched, among them both final submissions (`sub_primary_m3x5h.csv` `4505c159…797354e7`,
  `sub_primary_m3x4.csv` `76da786d…a3565f9973`), the intermediate form (`53ec2969…`), their three test matrices and every
  earlier file of the chain. No change of the build code was needed.
- **Pins of the regenerated JSON files.** The nine files that the version-0.4 code writes itself were printed as "FIRST
  BUILD" (each checksum equal to the file's), entered into the build scripts and modules and listed in
  [docs/INFERENCE.md](docs/INFERENCE.md#2-lines-and-expected-checksums). Compared field by field with the files from which the
  submitted files were built, none of them differs in any non-text value; they differ only in English text fields and renamed
  keys. The checksums of the channel metadata and blend file of submission B equal the values of the training-side rerun of
  version 0.4.
- **Private input package.** Every file that the first build opened was recorded with `strace`. The package holds exactly
  the files that the quick start read and did not write itself (464 files), the 11 pinned files written by the version-0.4
  code, the frozen inputs of the GPU model steps that the assembly does not read, the M3 training targets and our weights:
  545 files, 3.61 GB, 153 of them derived from test rows; each with its SHA-256 in the package manifest.
- **Second build.** A second fresh copy of this version, with the same package linked read-only, rebuilt every file byte for
  byte and passed every checksum check (the quick start took 1 h 48 min, the first 1 h 52 min, both on a machine shared
  with other jobs and with every file access traced). It read no file outside the package except the competition CSV files
  and the taxonomy, wrote nothing into the package, and made no network connection (its only network system calls were the
  IPv6 availability probe that `urllib3` makes on import: a socket bound to `::1`, port 0, never connected). Every package
  file of the roles `assemble` and `assemble-generated` was read.
- Tests: 641 passed and 11 skipped in the second copy with the private input package; 564 passed and 88 skipped with the
  repository files only.

### Added

- `tools/make_m3_targets.py`: the builder of the M3 training targets (folds 0–4 and the full fit), checked against the stored
  list of the rows on which the notes carry loss; `--check` rebuilds the six target files in memory and compares them with the
  files present and with `targets_sha256` of the six adapter recipes. It rebuilds all six files byte for byte. Tests
  `tests/test_make_m3_targets.py`.
- `env/requirements-cpu.txt`, `requirements-gpu.txt`, `requirements-vllm.txt`, `requirements-kaggle-t4.txt`: the exact package
  versions of the four environments as installed in the reference runs, each with its Python and CUDA version and install
  command (versions only; hashes and an offline wheelhouse are still open).
- `configs/vlm/m3d1te_rp.yaml`: draw d1 of M3 on the test rows with the delivered adapter of the full fit, without training
  (the test jobs of `m3_fullte_rp.yaml`); test in `tests/test_m3_sft.py`.
- `docs/VERIFICATION.md`: the host's checklist, item by item, including the expected deviation of a full rerun (§8.1:
  about ±0.003 MAP@3, 1 sd); `docs/WRITEUP.md`; `THIRD_PARTY.md` with links, licences and API cost.

### Changed

- Build scripts `tools/builds/primary_m3x4.sh`, `primary_cols.sh`, `primary_m3x5h.sh` and modules `eda/c151_primary_cols.py`,
  `eda/c166_primary_m3x5h.py`: the pins of the regenerated JSON files (20 places; the optional verdict file of the four-draw
  rule stays unpinned).
- `docs/weights_manifest.json`, `docs/PROVENANCE.md`: the five 9B fold adapters (training side and the check job of the 9B test
  configurations); the description checksum of the orientation network as read by this repository.
- Documentation: README (quick start with the private input package; status), INFERENCE §2–§7, TRAINING §2, §10.1, §10.2,
  PROVENANCE §0, §3, §8, §9, DATA_AND_COMPLIANCE, CODEMAP, ROADMAP (steps 1 and 3 done; 4 and 6 in part).

### Known gaps

The following items are tracked in [docs/ROADMAP.md](docs/ROADMAP.md#2-open-steps):
- smoke tests without network access for the GPU model steps of both final submissions, including the 397B path;
- hashes for every pin, an offline wheelhouse and a single entry point per line;
- a mode that re-records the checksums of newly computed test-side files;
- the scripts that are not yet exported (listed in [docs/PROVENANCE.md](docs/PROVENANCE.md#9-open-items-of-this-list)).

## v0.4 — 2026-11-15

Fifth version: the two final submissions, both from the primary line.

### Added

- **Primary submission B** (`primary_m3x4`): the version-0.3 primary candidate with a fourth M3 draw in the M3 channel
  (`m3x4_full`; second-stage features still from d1–d3, field `draws_first`). Deployment path `eda/c135_primary_m3x4.py`
  (`oof`, `stack`, `fit`, `copy`, `clean-copy`, `test-guard`, `null`; evidence of the four-draw rule `bank`, `final`,
  `placebo`, `rule`), build script `tools/builds/primary_m3x4.sh`, tests `tests/test_primary_m3x4.py`, test-side configuration
  `configs/vlm/m3d4te_rp.yaml`.
- **Intermediate form** (`primary_cols`; built on the way to submission A): nine extra second-stage columns from eight sources
  computed on the CPU from frozen protocol answers of our 27B and of the 397B, the blend channels, the taxonomy and the
  training labels of the reference rows (`src/stage2_sources.py`), with a test guard per column and source and positive
  controls. Deployment path `eda/c151_primary_cols.py` (`test-sources`, `fit`, `copy`, `measure`, `null`, `test-guard`,
  `controls`), build script `tools/builds/primary_cols.sh`, tests `tests/test_primary_cols.py`.
- **Primary submission A** (`primary_m3x5h`): the intermediate form with a fifth M3 draw that reads the photo at long side
  2,048 px, counted only on the rows whose median character height is at most 20 px (channel `m3x5h`; a "draw with row
  list"). Row-list tool `tools/make_charsize_rows.py`; deployment path `eda/c166_primary_m3x5h.py` (`digest`, `oof`, `stack`,
  `fit-base`, `fit`, `copy`, `clean-copy`, `h-guard`, `h-controls`, `null`, `measure`), build script
  `tools/builds/primary_m3x5h.sh`, tests `tests/test_primary_m3x5h.py`, configurations `configs/vlm/m3ha_oof_rp.yaml`,
  `m3hb_oof_rp.yaml`, `m3hte_a_rp.yaml`, `m3hte_b_rp.yaml`.
- `docs/PROVENANCE.md` (every frozen input of the two final submissions: producer, configuration, network status,
  determinism; the training chain and its open items), `docs/MODEL_SIZES.md` (parameters and bytes against the host's limits
  in two readings) and `docs/weights_manifest.json` (file list with checksums of the three base-model revisions and of our
  weights).
- `docs/INFERENCE.md` §10: recomputing the 397B channel on the test rows (hardware, environment, server switches, order of
  the steps for a run without network access, fidelity check, open points). Configurations `configs/vlm/q397te8_rp.yaml`,
  `q397p8_rp.yaml` (8×H100 variant, never run).

### Changed

- `tools/make_submission.py`: the complete internal delta since version 0.3 — draws with a row list (meta field
  `rank.row_list_draws`, `--rank-only-placeholder` for the stand-in and the copy null point), extra second-stage columns (field
  `extra_columns` of a deployment file, `--stage2-xcontrol`), `draws_first`, and two inert additions for candidates outside the
  final slots (a separate source of named codes, field `rank_mean.named`; the safety-line feature set `fallback_i1_ndr_mmid`
  with its mech27 guard and control `mech_flip`). Without these fields every earlier deployment file computes and reports
  byte-identically.
- `src/stage2.py`: the feature set `fallback_i1_ndr_mmid` (inert for the final submissions).
- Tests `tests/test_stage2_guard.py`, `test_stage2_i1.py`, `test_stage2_mm.py`, `test_stage2_safe.py`: four tests that read
  report files written by the build paths (not part of the frozen package) now skip when these files are absent, as the
  other package-dependent tests do.
- Documentation: README (final submissions, channels, results, quick start), METHOD §4.8 and §9.12 and §11 (final
  selection), VALIDATION §8, RESULTS, TRAINING §10.1 (the M3 training targets are **not** in the private package — version 0.3
  said they were), §10.3 and §16, INFERENCE §2/§3/§5/§6.1/§7/§9, DATA_AND_COMPLIANCE (host statements of 2026-10-26 and
  2026-11-10, sizes, both slots from the primary line), CODEMAP, ROADMAP, THIRD_PARTY.

### Verification of this version

- The new modules and build scripts are translations of the internal code that built the submitted files (internal
  commit of 2026-11-14; unchanged since). Their structure was compared with the internal code by masked syntax-tree
  comparison (strings masked, identifiers mapped by the rename table); every difference is a declared change (removed
  comparisons with internal measurement files that are not part of this repository, removed candidate outside the final
  slots, pins of regenerated JSON files emptied). Build scripts: heredocs compared as Python, shell lines normalised;
  configurations: parsed content equal.
- **Not run in this version:** a build of the new paths. The checksums of the JSON files that this code regenerates are
  learned on the first build in the package ([docs/INFERENCE.md](docs/INFERENCE.md#2-lines-and-expected-checksums)); the CSV
  and test-matrix checksums of both final submissions are pinned. Both final files were rebuilt byte for byte on a Kaggle
  CPU without network access with the internal code version.
- Training side of submission B rerun with the code of this repository (a scratch tree with only training-side files):
  the M3 channel file `oof_m3x4_full.npy` is byte-identical to its pinned checksum, the blend refit shows no numeric
  difference, and the fair-bank record of the original meta-split and the verdict of the four-draw rule equal the internal
  records. `src/stage2_sources.py` was checked against the internal module on synthetic inputs: columns, test guards, all
  seven controls and the checksums of the column matrices are identical.
- Tests: 605 passed and 42 skipped with a copy of the private package (no file written into the package); 560 passed and
  87 skipped with the repository files only.

### State of the solution

| Line and form | Status | CV MAP@3 (per fold / as one set) | Public LB |
|---|---|---|---:|
| **Primary submission A**: five M3 draws (the fifth at 2,048 px on the rows with small script) and nine extra second-stage columns | final submission; submitted 2026-11-14 | 0.8426 / 0.8404 | 0.84266 |
| Intermediate form: four M3 draws and the nine extra columns | step 0 of A; submitted 2026-11-13; not a final submission | 0.8441 / 0.8407 | 0.84283 |
| **Primary submission B**: four M3 draws | final submission; submitted 2026-11-08 | 0.8436 / 0.8380 | 0.84450 |

The gains that decided the forms were measured as paired differences over untouched meta-splits, not on the original
meta-split shown here ([docs/RESULTS.md](docs/RESULTS.md), [docs/VALIDATION.md](docs/VALIDATION.md#8-final-submissions-version-04)).

### Known gaps

The following items are tracked in [docs/ROADMAP.md](docs/ROADMAP.md#2-open-steps):
- the first build with the code of this repository (checksums of the regenerated JSON files);
- smoke tests without network access for the model steps of both final submissions, including the 397B path;
- the private copies of the frozen inputs, deployment files and M3 training targets in the state of this version;
- lock files, wheelhouse and a single entry point per line;
- a mode that re-records the checksums of newly computed test-side files;
- the scripts that are not yet exported (listed in [docs/PROVENANCE.md](docs/PROVENANCE.md#9-open-items-of-this-list));
- an estimate of the rerun variance of the generated texts.

## v0.3 — 2026-11-08

Fourth pre-final version: new final candidates of both lines.

### Added

- Safety line: note mentions and example similarity in the second stage (feature set `fallback_i1_nd`; `src/note_sim.py`;
  `eda/c129_safety_nd.py` with `export-notes`, `oof`, `splits`, `rule`, `fit`, `null` and `b-zero`; build script
  `tools/builds/safety_nd.sh`; tests `tests/test_safety_nd.py`).
- Safety line, final candidate: six regex-exact columns of the frozen M3 and protocol notes (feature set `fallback_i1_ndr`;
  the versioned extractor `src/note_regex.py`; `eda/c130_safety_ndr.py` with `export-notes`, `oof`, `swap`, `splits`, `rule`,
  `fit`, `null` and `b-zero`; build script `tools/builds/safety_ndr.sh`; tests `tests/test_safety_ndr.py`).
- Primary line, final candidate: the mean channel `m3px27xm`, in which M3 ranks all codes behind the frozen notes of the
  protocol channel instead of the untrained 27B (`tools/make_m3px27_notes.py`; run configurations
  `configs/vlm/m3px27_oof_rp.yaml`, `m3px27te_rp.yaml`; evaluation by its pre-registered rule `eda/c132_m3px27_eval.py`;
  deployment path `eda/c133_primary_m3px27.py` with `oof`, `stack`, `fit`, `copy`, `clean-copy`, `test-guard` and `null`;
  build script `tools/builds/primary_m3px27.sh`, whose steps 2–4 always run with `--save` (missing files are written,
  existing ones must be byte-identical, so that a package without the report files or the OOF matrix of the mean channel
  builds the same files); tests `tests/test_m3px27.py`, `tests/test_primary_m3px27.py`).
- `eda/risk_rule.py`: the empirical-Bayes posterior of the risk rule (prior from earlier measurements of the same class of
  component, selection correction), used by the rules of the two note-feature forms of the safety line.
- `tools/export_proto_notes.py`: the export of the first-pass notes of the protocol answers, which was listed as a step whose
  script was not included; it reproduces the frozen note files byte for byte.
- Run configurations of three runs whose outputs the new build scripts read for their checks: `configs/vlm/m3d4_oof_rp.yaml`
  (a fourth M3 draw on the training rows) and `configs/vlm/pr27s2_rp.yaml`, `pr27s3_rp.yaml` (a second draw of the protocol
  answers).

### Changed

- `src/stage2.py`: the feature sets `fallback_i1_nd` and `fallback_i1_ndr` and `attach_r`; the existing feature sets and
  their files are unchanged, bit for bit.
- `tools/make_submission.py`: deployment files of the safety line with the note feature sets read the frozen notes (checked by
  SHA-256), compute the example similarity and the regex columns, and stop without a file if a guard fails
  (`--stage2-control` runs the positive controls in memory); mean channels with the field `rank_mean.rank_compare: sft` are
  compared with every fold shard and checked by a note guard (`rank_mean.notes`). All earlier files are rebuilt byte for byte.
- `eda/c118_stage2.py`: `build_stage2` accepts a given deployment form of the training rows (used for the cross with β per fold
  of the new primary candidate); without it the function is unchanged.
- Documentation updated for this version: `README.md`, `docs/METHOD.md` (§4.7, §9.10, §9.11, §11), `docs/VALIDATION.md`,
  `docs/RESULTS.md`, `docs/TRAINING.md` (§9, §10.3, §12, §13.4–§13.6, §14), `docs/INFERENCE.md`, `docs/CODEMAP.md`,
  `docs/ROADMAP.md`, `docs/DATA_AND_COMPLIANCE.md`, `THIRD_PARTY.md`.
- `docs/TRAINING.md` §1: every training-side step of the channels of the final candidates with its section, configurations,
  inputs and hardware, and the provenance of every label or text not written by the annotators (training and pool rows
  only); §15: what a retraining gives and how far its results are expected to move.
- `docs/INFERENCE.md` §9 (new): running on a new test set — what stays, what is computed, the records that bind the final
  candidates to the competition test set, the behaviour of the guards and the calls of the files without second stage.

### Verification

- Two fresh copies of the repository with the private package, the data only as read-only links:
  - README quick start run verbatim: all 26 reference files byte-identical (`sub_primary_m3px27.csv`, `sub_safety_ndr.csv`,
    `sub_safety_nd.csv` and their test matrices, and the 20 files of versions 0.1 and 0.2); the two final candidates' build
    scripts, run once more, give the same files; the `make_submission` reference calls of `docs/INFERENCE.md` §5, run
    verbatim, give the same files, and the three calls without second stage of §9 give the files that the build scripts
    check.
  - Write path, with every file a build script can write deleted beforehand: the same 26 files byte-identical; the
    deployment files, meta-split files, the draw-swap file, the blend file and the mean channel are rebuilt with their
    pinned checksums. The primary build gives the same files again with only the metadata of its mean channel present,
    and with only its deployment files present.
  - Positive controls of the test side: copied, permuted and truncated M3 notes, shifted protocol notes and M3 notes in
    place of the protocol notes each stop the safety build without a file.
- The rebuilt protocol-note files and the notes of the M3 ranking behind them are byte-identical to the frozen files.
- Tests: 560 passed and 4 skipped with the private package; 487 passed and 77 skipped with the repository files only.

### State of the solution

| Line and form | Status | CV MAP@3 (per fold / as one set) | Public LB |
|---|---|---|---:|
| Primary line, final form with the M3 ranking behind the protocol notes | current candidate; submitted 2026-11-08 | 0.8426 / 0.8382 | 0.84116 |
| Primary line, final form with `pr27xm` | fallback; submitted 2026-11-06 | 0.8409 / 0.8398 | 0.84100 |
| Safety line, note mentions, example similarity and regex note columns | current candidate; submitted 2026-11-07 | 0.8316 / 0.8284 | 0.83716 |
| Safety line, note mentions and example similarity | fallback; submitted 2026-11-07 | 0.8311 / 0.8274 | 0.83750 |
| Safety line, alignment over the test set, reranker with interactions | submitted 2026-11-07 | — / 0.8217 | 0.83516 |
| Safety line, strictly inductive form with second stage | fallback under the strictest reading; submitted 2026-11-01 | 0.8233 / — | 0.83183 |

### Known gaps

The following items are tracked in [docs/ROADMAP.md](docs/ROADMAP.md):
- the offline package: lock hashes, wheelhouse, offline runner, single entry point, manifests;
- smoke tests without network access for the model steps, including the 397B path;
- the remaining second-hardware checks of the M3 draws;
- an estimate of the rerun variance of the generated texts;
- the private copies of the frozen inputs and weights in the state of this version;
- the per-package licence table.

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
