# Changelog

This file records notable changes to the solution repository. Dates are UTC. The repository is a living document until
the final solution package is delivered.

## v0 — 2026-11-01

First version of the solution repository (pre-final).

### Added

- Code for assembling a submission from frozen channel outputs (`tools/make_submission.py`, build scripts in
  `tools/builds/`), for model runs (`src/vlm/`, run configurations in `configs/vlm/`), for training targets, the
  orientation network, the second stage, OOF assembly, blend fitting and the checks run before a submission, and the
  tests.
- Environment locks for the CPU assembly, the GPU base environment, the vLLM environment and the Kaggle T4 path
  (`solution/env/`).
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
