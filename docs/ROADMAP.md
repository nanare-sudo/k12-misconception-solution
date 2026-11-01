# Roadmap

*Living document. Status: 2026-11-01. The competition closes on 2026-11-17 at 00:00 UTC.*

## 1. Where the solution stands

| Area | Status |
|---|---|
| Primary line | The reranker fallback form was submitted on 2026-10-31 (CV 0.8356 per fold / 0.8338 as one set). The main form is built and checked (CV 0.8376 / 0.8346) and is the candidate for the final primary-line file |
| Strictly inductive line | Submitted on 2026-10-31 (CV 0.8184 per fold) |
| Assembly | Deterministic on the CPU. Byte-identical rebuilds are verified for the recent primary-line files, including the submitted fallback form and the main-form candidate. A rebuild of the inductive line with the current code is pending |
| Trained weights | The 9B adapter, the M3 adapter, the orientation network and the blend and second-stage files are stored privately with checksums; the two adapters also have a second private copy |
| Frozen inputs | The frozen test-side inputs for `assemble` and `rescore` are stored privately with checksums |
| Environments | Exact version locks exist for the CPU assembly, the GPU base, the vLLM environment and the Kaggle T4 path. Hashes for every pin and the offline wheelhouse are not complete yet |
| Runs without network | The Kaggle T4 steps (9B scorer, `zs27t`) ran without network access. The text-generation and bf16 steps, which carry about two thirds of the blend weight, have not yet run without network access |
| Second hardware class | Verified for the 9B test side, `zs27t` and the M3 draw-d1 ranking. The draw-d2 ranking is half done; the draw-d3 and reasoning rankings are pending |

## 2. Remaining steps up to the final submission

| # | Step | Purpose | Status |
|---|---|---|---|
| 1 | Switch all build paths to the rotation list of the released orientation network, then rebuild and resubmit the final candidates | The delivered network must produce the submitted files; the new list differs on 24 of 3,498 test photos | open |
| 2 | Rebuild the inductive line with the current code and compare byte by byte | Proof that the current code reproduces the submitted file | open |
| 3 | Offline package: complete the lock hashes and build the wheelhouse; offline runner that never downloads and checks the base-model files against the weights manifest; single entry point (`tools/replay/assemble.py`, stage-wise `run_inference.sh` with the modes `assemble`, `rescore` and `full`); weights manifest; manifest of the frozen inputs of every final line; line definitions in `configs/lines/` | Reproduction without network access and without a human step | in progress |
| 4 | Dependency traces of the final build paths | Proof of which files each line reads, in particular that the inductive line reads no teacher-derived file | open |
| 5 | Smoke tests without network: (a) early environment test on one 96 GB GPU; (b) required: both lines on 16 training rows, every model step, empty caches; (c) the 397B path on 4×H200 | Evidence that every model step runs offline; (c) is required before a file with the 397B channel can be a final submission | open |
| 6 | Remaining second-hardware checks: M3 draw-d2 ranking (in progress), draw-d3 ranking, reasoning ranking | Evidence for statistically equivalent reruns | in progress |
| 7 | Inductive line with the reranker and a `NONE` guard | Candidate for the inductive final submission (planned CV gain +0.0049 ± 0.0022) | planned |
| 8 | Optional improvements under pre-registered rules: annotator-practice features for the reranker (running); a `NONE` guard and general-code indicators in the reranker (about +0.0015 expected) | Small further gains in the primary line | optional |
| 9 | Update the run-time and cost estimates for the final lines | Accurate figures in [INFERENCE.md](INFERENCE.md) | open |
| 10 | Documentation: licence table generated from the wheel metadata, licence file of the API-served 397B revision, final selection and scores | Complete solution package | open |
| 11 | Final selection: one file from each line, chosen by cross-validation and by the separation of techniques that rest on interpretation; never by public score | Two final submissions | by 2026-11-15 |

A test-time augmentation pilot for M3 (higher photo resolution for small handwriting) has been completed. It stayed
below its pre-registered threshold and behaved like an ordinary additional draw, so it is not pursued.

## 3. Timeline

| Date (2026) | Milestone |
|---|---|
| 11-05 | Code freeze. Afterwards only documentation and fixes, each confirmed by a byte-identical `assemble` |
| 11-10 | Last model change |
| 11-11 to 11-13 | Final files built from the package |
| 11-15 | Both final submissions selected |
| 11-17, 00:00 UTC | Competition deadline |
| After 11-17 | Private-leaderboard result; solution documentation for the host; publication of this repository when the rules allow it |

## 4. Open decisions

- **397B in a final submission.** Whether the final primary-line file includes the 397B channel depends on the smoke test
  of the 397B path without network access (step 5c).
- **Second final submission.** Either the inductive line as submitted, or the same line with the reranker and `NONE` guard
  (step 7). The choice depends on that variant's cross-validation and checks.
