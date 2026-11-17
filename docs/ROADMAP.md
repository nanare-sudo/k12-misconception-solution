# Roadmap

*Living document. Status: 2026-11-17 (version 0.5 build branch). The competition closed on 2026-11-17 at 00:00 UTC.*

## 1. Where the solution stands

| Area | Status |
|---|---|
| Final submissions | Selected on 2026-11-15, both from the primary line. **Primary submission A** (`primary_m3x5h`): five M3 draws, the fifth with the photo at 2,048 px on the rows with small script, and nine extra second-stage columns from frozen protocol answers (CV 0.8426 per fold / 0.8404 as one set on the original meta-split; over 20 untouched meta-splits +0.0008 in ½(F+M) and +0.0011 as one set against the intermediate form). **Primary submission B** (`primary_m3x4`): four M3 draws (CV 0.8436 / 0.8380; fair bank +0.0011 over 20 meta-splits against the version-0.3 candidate). The intermediate form (`primary_cols`, B plus the columns) is built on the way to A; it is not a final submission. See [RESULTS.md](RESULTS.md) and [VALIDATION.md](VALIDATION.md#8-final-submissions-version-04) |
| Assembly | Deterministic on the CPU. From a code package of the development version and the frozen inputs, the assembly of both final submissions (and of every predecessor in their chain) ran on a Kaggle CPU without network access and reproduced the submitted files byte for byte. With the code of this repository the rebuild ran on 2026-11-17 in two fresh copies with the private input package: the README quick start reproduced both final submissions, the intermediate form and every predecessor in their chain byte for byte; the checksums of the JSON files that the version-0.4 code writes itself were learned on the first build and are pinned (step 1). The quick start took 1 h 52 min and 1 h 48 min on a machine shared with other jobs, with every file access traced |
| Code of the final submissions | Translated into this repository with the internal delta of the shared modules (`tools/make_submission.py`, `src/stage2.py`); structural equality with the internal code checked by masked syntax-tree comparison; tests pass without the competition data ([CHANGELOG.md](../CHANGELOG.md)) |
| Safety line | Documented as the earlier design; not one of the two final submissions. Its files and build paths of versions 0.1–0.3 are unchanged |
| Trained weights | The 9B adapter, the M3 adapter and the orientation network are stored privately with checksums ([weights_manifest.json](weights_manifest.json)) and are part of the private input package (`weights/`, with the five 9B fold adapters; the M3 fold adapters lie with their training runs); the base models are pinned by revision with every file's checksum |
| Frozen inputs | The private input package of this version holds every file that the README quick start reads (traced file by file), the files that the build scripts write and pin themselves, the frozen inputs of the GPU model steps that the assembly does not read, the M3 training targets and our weights: 545 files, 3.61 GB, 153 of them derived from test rows; each file with its SHA-256 in the package manifest. [PROVENANCE.md](PROVENANCE.md) describes the inputs of both final submissions |
| Sizes | 481.53 GB as deployed, 440.94 GB at 8 bit; 434.4 B parameters nominal in total ([MODEL_SIZES.md](MODEL_SIZES.md)) |
| Runs without network | The Kaggle T4 steps (9B scorer, `zs27t`, M3 rankings of the second-hardware checks) and the assembly on a Kaggle CPU ran without network access. **No GPU model step of the final submissions has run without network access yet**, in particular not the 397B path and the bf16 / vLLM steps (step 2) |

## 2. Open steps

| # | Step | Purpose | Status |
|---|---|---|---|
| 1 | **First build with the code of this repository** in a fresh copy with the private package: run `bash tools/builds/primary_m3x5h.sh` (it builds B and the intermediate form on the way), check that both CSV files and test matrices match the pinned checksums, enter the printed checksums of the regenerated JSON files into the build scripts and modules, and run the build a second time (byte-identical) | Evidence that this repository, not only the internal code, rebuilds both final files | **done** (2026-11-17; two fresh copies, byte-identical) |
| 2 | **Smoke tests without network** (G2, G5): (G2) every model step of both submissions on a small set of training rows on one 96 GB card, empty caches, the vLLM environment installed from a wheelhouse before the network is cut, base-model files checked against [weights_manifest.json](weights_manifest.json); (G5) the 397B path on 4×H200 as described in [INFERENCE.md](INFERENCE.md#107-order-of-the-steps): weights and wheelhouse with network, then the parity probe and a test-side check in a container without a network interface | Evidence that every model step runs offline | open; estimated cost G2 about 4–5 USD, G5 about 21–27 USD |
| 3 | Private copies brought to the state of this version: the regenerated JSON files of step 1, the blend and deployment files of versions 0.2–0.4, the M3 training targets with the list their builder checks against (and the builder), and the parity row list of the 397B check | A third party can run `assemble` and retrain from the private copies | **done** (2026-11-17): private input package with manifest; delivery by private link |
| 4 | Lock files with hashes for every pin and an offline wheelhouse; a single entry point per line that reads a line definition and calls `tools/make_submission.py` directly | Reproduction without a human step | in part: exact pins of every environment in `env/requirements-*.txt`; hashes, wheelhouse and the single entry point are open |
| 5 | A mode that re-records the checksums of newly computed test-side files (deployment files of the final submissions pin their frozen test-side sources), with a regression check that the frozen path stays byte-identical | Assembly from a `full` rerun or a new test set | open ([INFERENCE.md](INFERENCE.md#9-running-on-a-new-test-set)) |
| 6 | Scripts that exist but are not yet exported: M3 target builder, blank-photo list, combination of the fold runs of the first two M3 draws, 397B parity check, posterior step of the three-draw evaluation, the measurement scripts of the bundle test and of the five-draw rule | Complete training chain | in part: the M3 target builder is exported (`tools/make_m3_targets.py`); the rest is open ([PROVENANCE.md](PROVENANCE.md#9-open-items-of-this-list)) |
| 7 | Rerun variance: how much the assembled result moves when all generated texts are replaced by other draws at once | Expected deviation of a `full` rerun | done (2026-11-17): about ±0.003 MAP@3 (1 sd) on 2,498 rows; [VERIFICATION.md §8.1](VERIFICATION.md#81-how-close-a-full-rerun-should-come) |
| 8 | Documentation: per-package licence table from the wheel metadata, licence file of the API-served 397B revision, private-leaderboard scores | Complete solution package | open |

## 3. Timeline

| Date (2026) | Milestone |
|---|---|
| 11-08 | Fourth M3 draw (submission B) submitted |
| 11-13 | Extra second-stage columns (intermediate form) submitted |
| 11-14 | Fifth M3 draw with the large photo (submission A) submitted; netless CPU rebuild of both final files from the frozen inputs |
| 11-15 | Both final submissions selected; version 0.4 of this repository |
| 11-17, 00:00 UTC | Competition deadline |
| 11-17 | First and second build of both final submissions with the code of this repository (byte-identical); private input package; requirements files; M3 target builder |
| After 11-17 | Private-leaderboard result; steps 1–3 and the solution package for the host; publication of this repository when the rules allow it |

## 4. Open decisions

- **Smoke tests before or after the deadline.** Steps 1–3 do not change the submitted files; they are planned for the
  delivery of the solution package. Both final files contain the 397B channel, so both depend on step 2 (G5); no final
  file without the 397B channel is built.

## 5. After the competition

- Add the private-leaderboard results to [README.md](../README.md) and [RESULTS.md](RESULTS.md).
- Prepare the solution package (code, weights, frozen inputs, instructions) for verification; the host requests and
  verifies both the training and the inference pipeline of the winners.
- Publish this repository when the competition rules and the sponsor allow it. Trained weights and files derived from
  the competition data stay private unless the rules allow otherwise
  ([DATA_AND_COMPLIANCE.md](DATA_AND_COMPLIANCE.md#8-publication)).
