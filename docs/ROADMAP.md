# Roadmap

*Status: 2026-11-21 (version 0.6), the state delivered for the host's verification. The competition closed on
2026-11-17 at 00:00 UTC.*

## 1. Where the solution stands

| Area | Status |
|---|---|
| Final submissions | Selected on 2026-11-15, both from the primary line. **Primary submission A** (`primary_m3x5h`): five M3 draws, the fifth with the photo at 2,048 px on the rows with small script, and nine extra second-stage columns. **Primary submission B** (`primary_m3x4`): four M3 draws. Which file is which, with the Kaggle references: [VERIFICATION.md](VERIFICATION.md); CV values: [RESULTS.md](RESULTS.md) |
| Rebuild on the CPU | Both files are rebuilt byte for byte from the frozen inputs: on a Kaggle CPU without network access with the internal code version, and with the code of this repository in two fresh copies (2026-11-17) and once more from a fresh clone (2026-11-20). `bash reproduce.sh <package-dir>` runs the quick check, `--full` the build chain ([VERIFICATION.md §5.1](VERIFICATION.md#51-assemble-cpu)) |
| GPU steps | `tools/gpu_inference.sh rescore` or `full` runs the test-side jobs from the delivered weights and assembles A and B ([VERIFICATION.md §5.2](VERIFICATION.md#52-rescore-and-full-gpu)). A complete `rescore` or `full` run has never been done in one go |
| Runs without network | The CPU assembly and the 9B scorer on Kaggle T4 ran without network access. No other GPU model step of the final submissions has run without network access yet, in particular not the 397B path and the bf16 and vLLM steps (step 2) |
| Private input package | Delivered to the host by private link: frozen inputs, channel outputs, fitted files, training-side inputs, check inputs and our weights, every file with its SHA-256 in `MANIFEST.sha256` ([VERIFICATION.md §9](VERIFICATION.md#9-private-weights-and-inputs)) |
| Sizes | 481.53 GB as deployed, 440.94 GB at 8 bit; 434.4 B parameters nominal in total ([MODEL_SIZES.md](MODEL_SIZES.md)) |
| Private leaderboard | Published by the host after the verification |

## 2. Open steps

| # | Step | Status |
|---|---|---|
| 1 | First build with the code of this repository in a fresh copy with the private package; pins of the JSON files that this code writes | done (2026-11-17, two fresh copies, byte-identical; once more from a fresh clone on 2026-11-20) |
| 2 | **Runs without network access** for the GPU model steps: first every 27B and M3 step of both submissions on a few rows on one 96 GB card, with empty caches and the offline settings of [INFERENCE.md](INFERENCE.md#4-environments); then the 397B path on 4×H200 as described in [INFERENCE.md](INFERENCE.md#107-order-of-the-steps) | open |
| 3 | Private input package in the state of this version | done (2026-11-21): completed with the training-side row lists and the inputs of the checks; metadata in English, without internal references, checksums re-pinned |
| 4 | Lock files with hashes for every pin and an offline wheelhouse | open. The exact versions of every environment are pinned in `env/requirements-*.txt`, and `reproduce.sh` is the single command of the CPU check |
| 5 | A mode that re-records the checksums of newly computed test-side files | done (2026-11-21): `tools/repin_test_side.py`, [VERIFICATION.md §5.2](VERIFICATION.md#52-rescore-and-full-gpu) |
| 6 | Export of the scripts that produced frozen inputs | in part. Exported: the M3 target builder (`tools/make_m3_targets.py`) and the blank-photo list (`tools/make_blank_list.py`). Open: the combination of the fold runs of the first two M3 draws, the 397B parity check, the posterior step of the three-draw evaluation, and the measurement scripts of the bundle test and of the five-draw rule ([PROVENANCE.md](PROVENANCE.md#9-open-items-of-this-list)) |
| 7 | Rerun variance: how much the assembled result moves when all generated texts are replaced by other draws at once | done (2026-11-17): about ±0.003 MAP@3 (1 sd) on the 2,498 private rows ([VERIFICATION.md §8.1](VERIFICATION.md#81-how-close-a-full-rerun-should-come)) |
| 8 | Documentation | in part. Done: the licence of the API-served 397B (Hub repository Apache 2.0, licence file at main `8472618…`, checked 2026-11-20; the revision served by the provider was not recorded). Open: a licence table per package from the wheel metadata, and the private-leaderboard scores |

## 3. Timeline

| Date (2026) | Milestone |
|---|---|
| 11-08 | Fourth M3 draw (submission B) submitted |
| 11-13 | Extra second-stage columns (intermediate form) submitted |
| 11-14 | Fifth M3 draw with the large photo (submission A) submitted; netless CPU rebuild of both final files from the frozen inputs |
| 11-15 | Both final submissions selected; version 0.4 of this repository |
| 11-17, 00:00 UTC | Competition deadline |
| 11-17 | Version 0.5: first and second build of both final submissions with the code of this repository (byte-identical); private input package; the host invites the leading teams to send their solutions |
| 11-21 | Version 0.6: one-command CPU check, GPU inference without retraining, completed private package; delivered for the verification |
| After the verification | Private-leaderboard result |

## 4. Open decisions

None for the delivery. The runs of step 2 do not change the submitted files.

## 5. After the verification

- Add the private-leaderboard results to [README.md](../README.md) and [RESULTS.md](RESULTS.md).
- Answer the host's questions during the verification and fix any gap that the host finds.
- The code and the write-up may be published after the verification. Trained weights and files derived from the
  competition data stay private ([DATA_AND_COMPLIANCE.md](DATA_AND_COMPLIANCE.md#8-publication)).
