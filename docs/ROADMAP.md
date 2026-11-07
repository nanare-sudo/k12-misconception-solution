# Roadmap

*Living document. Status: 2026-11-07 (version 0.2). The competition closes on 2026-11-17 at 00:00 UTC.*

## 1. Where the solution stands

| Area | Status |
|---|---|
| Primary line | Current candidate: the I1 form with the minor-code × mechanical-match column (CV 0.8409 per fold / 0.8398 as one set), submitted on 2026-11-06. Fallbacks: the I1 form (0.8408 / 0.8387), submitted on 2026-11-06, and the guard form of version 0.1 (0.8389 / 0.8377), submitted on 2026-11-01. Both additions met the expected-value rule ([VALIDATION.md](VALIDATION.md#52-pre-registered-decision-rules)) |
| Safety line | Current candidate: the channels of the strictly inductive line with the alignment over the test set and a second stage with interactions and NONE guard (CV as one set with gate 0.8217), submitted on 2026-11-07. The cross-validation part of one condition of its safety-set rule was missed by one sixth in 9,000 and overridden by a recorded decision ([decisions/override_cv_condition_ii.json](../decisions/override_cv_condition_ii.json)). Fallback: the strictly inductive line with second stage of version 0.1 (0.8233 per fold), submitted on 2026-11-01. That fallback meets the risk rule of the strictly inductive line under the rule's default assumption and fails it if the between-draw spread of the M3 texts is about 0.0017 or larger ([VALIDATION.md](VALIDATION.md#55-risk-rule-of-the-strictly-inductive-line)) |
| Assembly | Deterministic on the CPU. Byte-identical rebuilds are verified in fresh copies of this repository with the frozen inputs for the three files added in this version, their fallbacks, the intermediate files and the historical files with the first-version rotation list. The assembly of both current candidates and of their fallbacks also ran on a Kaggle CPU without network access, from a code package of the development version and the frozen inputs, and reproduced the submitted files byte for byte |
| Rotation list | All candidates and fallbacks use the list of the released orientation network; the first-version list is only used to rebuild the historical files |
| Trained weights | The 9B adapter, the M3 adapter, the orientation network and the blend and second-stage files are stored privately with checksums; the two adapters also have a second private copy. The deployment files added in this version still have to be added to the private copies |
| Frozen inputs | Unchanged from version 0.1. The new second stages add only derived files (deployment files, meta-splits, probe files), which the build scripts verify by pinned checksums |
| Environments | Pinned versions are documented ([INFERENCE.md](INFERENCE.md#4-environments)). Lock files with hashes for every pin and an offline wheelhouse are planned |
| Runs without network | The Kaggle T4 steps (9B scorer, `zs27t`, and the M3 rankings of the second-hardware checks) and the assembly on a Kaggle CPU ran without network access. The text-generation and bf16 steps, which carry about two thirds of the blend weight, have not yet run without network access |
| Second hardware class | Verified for the 9B test side, `zs27t`, the M3 draw-d1 ranking and the first half of the draw-d2 ranking (same top code in 97.8 % of 1,750 rows, NONE share at the top identical). The second half of draw d2 runs on Kaggle T4 since 2026-11-07; draw d3 follows, its last three parts after the weekly quota renews on 2026-11-14. The reasoning ranking is pending |

## 2. Remaining steps up to the final submission

| # | Step | Purpose | Status |
|---|---|---|---|
| 1 | Offline package: lock files with hashes and an offline wheelhouse; an offline runner that never downloads and checks the base-model files against a weights manifest; a single entry point per line that reads a line definition, verifies every input checksum and calls `tools/make_submission.py` without the chain of earlier builds; a manifest of the frozen inputs of every final line | Reproduction without network access and without a human step | in progress (assembly without network verified) |
| 2 | Smoke tests without network: (a) an early environment test on one 96 GB GPU, with two determinism probes (bit-identical 27B ranking on the same card, row-identical texts from deterministic generation); (b) required: both lines on a small set of training rows, every model step, empty caches; (c) the 397B path on 4×H200, together with the 397B training side recomputed on the same local FP8 path, which extends the parity check from 64 to 1,500 rows | Evidence that every model step runs offline; (c) is required before a file with the 397B channel can be a final submission | open |
| 3 | Second hardware class for the safety candidate: the remaining M3 rankings (second half of draw d2, draw d3) and the reasoning ranking on Kaggle T4, then the assembled file compared with the reference as aggregates | Evidence that the safety candidate is statistically equivalent on other hardware | in progress since 2026-11-07 |
| 4 | Rerun variance: estimate how much the assembled result moves when all generated texts are replaced by other draws at once, and report it with the final files | Expected deviation of a `full` rerun, since about two thirds of the blend weight rests on generated texts | open |
| 5 | Private copies of the frozen inputs and weights brought to the state of this version | A third party can run `assemble` from the private copies | open |
| 6 | Run-time and cost estimates for the final lines, including the 397B channel and the second stages | Accurate figures in [INFERENCE.md](INFERENCE.md) | open |
| 7 | Optional model changes until 2026-11-10, each under a pre-registered rule and only if it adds no new package component | Small further gains | three done in version 0.2 (second stages only); further ones optional |
| 8 | Documentation: per-package licence table from the wheel metadata, licence file of the API-served 397B revision, final selection and scores | Complete solution package | open |
| 9 | Final selection: one file from each line, chosen by cross-validation and by the separation of techniques that rest on interpretation, never by public score | Two final submissions | by 2026-11-15 |

## 3. Timeline

| Date (2026) | Milestone |
|---|---|
| 11-05 | Planned code freeze. The second-stage additions of version 0.2 followed on 11-06 and 11-07, each under a pre-registered rule and each confirmed by a byte-identical rebuild; since then only documentation and fixes |
| 11-07 | The weekly Kaggle GPU quota renews: second-hardware runs and package probes |
| 11-10 | Last model change |
| 11-11 to 11-13 | Final files built from the package |
| 11-14 | The weekly Kaggle GPU quota renews: last parts of the draw-d3 second-hardware check |
| 11-15 | Both final submissions selected |
| 11-17, 00:00 UTC | Competition deadline |
| After 11-17 | Private-leaderboard result; solution package for the host; publication of this repository when the rules allow it |

## 4. Open decisions

- **397B in the final primary file.** All three files of the primary line contain the 397B channel. The choice depends
  on the smoke test of the 397B path without network access (step 2c). If that test fails, the primary slot takes a file
  without the 397B channel, built from the existing channels; such a file would need a new build path with the released
  rotation list and its own cross-validation.
- **Safety slot.** Over 20 new meta-splits the candidate is ahead of the strictly inductive fallback (variant a
  +0.0007 ± 0.0002, variant b a further +0.0007 ± 0.0001). On the original meta-split, as one set, it stays below the
  fallback per fold and misses the cross-validation part of one condition of its rule by one sixth in 9,000, which a
  recorded decision overrides; it meets the test-mass condition (L1 0.028) and uses the alignment over the whole test
  set. The fallback learns nothing from the test rows, but its inductive alignment misses the test-side class masses
  (L1 0.087). The choice also takes the second-hardware evidence into account (step 3).

## 5. After the competition

- Add the private-leaderboard results to [README.md](../README.md) and [RESULTS.md](RESULTS.md).
- Prepare the solution package (code, weights, frozen inputs, instructions) for verification; the host reproduces the
  inference of the top ten of the final leaderboard.
- Publish this repository when the competition rules and the sponsor allow it. Trained weights and files derived from
  the competition data stay private unless the rules allow otherwise
  ([DATA_AND_COMPLIANCE.md](DATA_AND_COMPLIANCE.md#8-publication)).
