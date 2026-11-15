# Data and compliance

*Status: 2026-11-15 (version 0.4).*

This document states which data the solution uses and where each part is kept. It also explains why some parts are not in
this repository, how external APIs were used, and how the solution meets the competition's requirements for the final
submission.

## 1. Competition data

- **Source:** the competition page on Kaggle: `train.csv`, `test.csv`, `unlabelled.csv`, the taxonomy, the sample
  submission, and the photos of the training, test and unlabelled sets plus the question diagrams.
- **Licence:** subject to the competition rules. The data, and models trained on them, are used only for this
  competition.
- **Not redistributed.** No competition file, no verbatim excerpt (question text, transcription, explanation), no row
  identifier and no image is part of this repository. The only per-row item is the fold assignment by row position
  (`configs/folds.parquet`), which contains no identifier and no label.
- **Data protection.** The host screened the photos for personal information before release. We make no attempt to
  identify students.

## 2. What is not in this repository, and where it is

| Item | Why it is not here | Where it is kept |
|---|---|---|
| Competition data | Competition rules | Kaggle competition page |
| Per-row derived inputs: frozen reasoning texts, protocol answers and notes, M3 texts (test and training rows), 397B notes, notes in the M3 format, rotation, gate and blank-photo lists, training targets | Derived from competition data, row by row | Private Kaggle dataset `k12-derived` (frozen inputs for `assemble` and `rescore`); locally under `data/derived/` |
| Teacher pseudo-labels for the unlabelled pool | Derived from competition data; to be used only for this competition | Private (`data/derived/teacher/`), with checksums |
| OOF and test probability matrices, channel outputs, stored reports | Per-row predictions | Private Kaggle dataset `k12-derived` (the files the build scripts read); locally under `artifacts/` |
| Blend and second-stage files | Fitted weights, partly with row positions | Private Kaggle model `k12-weights`; locally under `artifacts/` |
| Trained weights: 9B adapter `v1_full`, M3 adapter `m3_full`, orientation network | Trained on competition data | Private Kaggle model `k12-weights` |
| Submission files | Per-row test predictions | Private; the earlier files that the build scripts read as references are part of `k12-derived` |
| Base models (Qwen3.5-9B, Qwen3.8-27B, Qwen3.5-397B-A17B-FP8) | Public and very large | Hugging Face at pinned revisions, with every file's size and checksum in [weights_manifest.json](weights_manifest.json); copied into the solution package for the host on request |
| API keys and credentials | Secrets | Never stored in the repository |

The code refers to these files by relative path; the build scripts and the metadata files record and check the SHA-256
of the pinned ones ([VALIDATION.md](VALIDATION.md#6-checks-before-a-submission)). All private files are available to the
competition host for verification. The private copies are being brought to the state of this repository version
([ROADMAP.md](ROADMAP.md)). The repository's `.gitignore` excludes `data/`, `artifacts/`, model weights and row-level
file formats.

## 3. Test data policy

- **No manual labelling and no manual transcription** of the test set, as the competition rules require.
- **No inspection of test outputs.** Test photos and per-row outputs derived from them (model texts, rankings,
  rotation predictions, predictions) are not looked at by a person. Scripts that touch the test side print only
  aggregates: counts, shares and checksums. All checks before a submission work on such aggregates
  ([VALIDATION.md](VALIDATION.md#6-checks-before-a-submission)).
- **No API on test rows.** Test rows were processed only on our own or rented machines with open-weight models.
- **No leaderboard probing.** No variant was chosen by its public score.

## 4. External APIs (development only)

| Service | Purpose | Rows sent | Part of the final inference |
|---|---|---|---|
| Anthropic Messages API: Claude Haiku 5.5, with Sonnet 5.5 for comparison measurements | Teacher pseudo-labels for the 50,000 unlabelled photos; measurements on training rows (teacher quality, a description experiment) | Unlabelled and training rows only | No. The frozen labels feed the per-question task block of the primary line |
| Tinker sampling API with open models (Qwen3.5-397B-A17B, Inkling-Small, Qwen3.8-27B) | Measurements of open models on training rows; the training-side outputs of the 397B channel, which are also the training side of six extra second-stage columns of submission A | Training rows only | No. The 397B test side runs on our own hardware with the same open model (FP8 release); a parity check on 64 training rows found the same number of input tokens in every row (the images were sent to the API as JPEG at quality 75) and the same top code in 78 % of rows. The provider's revision and number format are not recorded in our files ([PROVENANCE.md](PROVENANCE.md#42-397b-protocol-answers-on-the-training-rows)) |

**Host statements on API use** (competition forum):
- "No external APIs allowed in final solution. Solution should be reproducible without internet access though you are
  free to use any APIs for fine-tuning, enriching your models using unlabeled data or any other useful approaches"
- On paid API pseudo-labels for training and unlabelled rows: "Yes. You can do that as long as you are using them only
  for this competition." (2026-10-26) and "You can pseudo label unlabeled as well as labeled data provided in this
  competition." (2026-11-10)
- On additional files: "You can add any additional files to your submission but they can't be just predictions on test set
  directly from Claude or OpenAI." (2026-11-10)
- On the hosted API: "We fine-tuned the models mentioned in our blog using Thinking Machine's Tinker APIs. While this comes
  with a cost, participants may want to consider using it." (2026-10-12)
- On the verification of the winners: "We agree and  both training & inference pipeline for winners would be requested and
  verified." (2026-11-10) — the training chain and the origin of every pseudo-label are therefore documented in
  [TRAINING.md](TRAINING.md) and [PROVENANCE.md](PROVENANCE.md).

**Provider terms.** Both are business offerings. Inputs are not used to train or improve the provider's models, and the
provider claims no ownership of inputs or outputs. API outputs are frozen with checksums; training and assembly read only
the frozen files. Total API cost was about 0.1 kUSD (Anthropic about 43 USD, Tinker about 63 USD).

## 5. Final solution: offline, self-contained, within the limits

| Requirement | How the solution meets it |
|---|---|
| No external API and no internet access at inference time | All inference steps use open-weight models on local hardware; the text-generation server is bound to 127.0.0.1. Offline settings are listed in [INFERENCE.md](INFERENCE.md#4-environments). The offline package and a smoke test without network access for every model step are being completed ([ROADMAP.md](ROADMAP.md)) |
| No human step | `assemble` runs as single commands, with checks that stop on any deviation; `rescore` and `full` run configured jobs (a stage-wise runner is planned) |
| Self-contained: model weights and all labels made for the unlabelled data | Adapters and the orientation network are delivered with checksums; base models are pinned by revision; the teacher pseudo-labels are delivered frozen |
| Model size: "Any model of the same size or smaller than Nemotron 550B."; on the total: "We would consider any model up-to 512 GB weights in 8-bit or lower quants irrespective of VRAM required for inference" and (2026-11-10) "Combined model together need to be ~512GB" | The largest model is Qwen3.5-397B-A17B (397 B parameters nominal, 17 B active); all models together 434.4 B parameters nominal (440.9 B stored values). Weights as deployed 481.53 GB (each base model once, adapters as adapters), at 8 bit per stored value 440.94 GB; both below 512 GB ([MODEL_SIZES.md](MODEL_SIZES.md), with the counting rule and the counts that would exceed the limit) |
| Compute | The host sets no compute limit and reproduces inference for the top 10 of the final leaderboard. Reference hardware and run times are in [INFERENCE.md](INFERENCE.md) |
| The delivered solution must reproduce the submitted file | `assemble` reproduces each candidate file byte for byte from the frozen channel outputs (both final submissions were rebuilt on a CPU without network access from the frozen inputs; with the code of this repository the first build that pins its regenerated JSON files is still open, [ROADMAP.md](ROADMAP.md)); `rescore` and `full` regenerate the channel outputs ([INFERENCE.md](INFERENCE.md#1-three-modes)); no GPU model step has yet been run without network access. The rotation correction of the 9B channel uses the list of the released orientation network, which the delivered weights reproduce; the earlier list of a first network whose weights were not stored is used only to rebuild historical files |

## 6. The two lines and the techniques that rest on interpretation

**Final selection (2026-11-15): both final submissions come from the primary line** — primary submission A
(`primary_m3x5h`) and primary submission B (`primary_m3x4`). The safety line is no longer one of the two. The reason is that
the host's statements quoted in section 4 cover the techniques that separated the lines explicitly: pseudo-labels from a
paid API on unlabelled and labelled rows (2026-10-26, 2026-11-10), additional files other than test predictions of a
closed model (2026-11-10), the hosted API recommended by the host (2026-10-12), and models within the size limits of
section 5. Submission B is the fallback of submission A: it lacks the two components added last (the extra columns and the
fifth M3 draw), and its file was not changed by them. The transductive alignment over the test set is used in both. The
rest of this section records the earlier design.

Two final submissions may be selected. We use them to keep the techniques that depend on external APIs during development
out of one of the two lines, and to keep one file that holds under the strictest reading of the rules.

| Technique | Basis | Line |
|---|---|---|
| Transductive class-marginal alignment | It uses the set of test predictions: no label and no information about individual rows. It relies on the split being stratified by root code; the host said of the test share of `NONE`: "It should be similar". Asked about teams adapting to the test data that all participants have, the host answered: "We don't see any issue so far. Prize winner needs to submit final code, model weights and instructions to generate predictions from a model and won't be decided just based on submission file score during competition." | Primary line; since 2026-11-06 also the forms of the safety line with the alignment over the test set, including its final candidate |
| Teacher pseudo-labels of the unlabelled pool (task block) | API pseudo-labels on unlabelled rows are explicitly allowed for this competition | Primary line only |
| Blend weight of the 397B channel fitted on API outputs for training rows | API use on training rows is allowed; the test side is computed locally with the same open model (FP8 release) | Primary line only |

Until 2026-11-05 the transductive alignment was kept out of the second line as well. On 2026-11-06 the entrant admitted
it in both final submissions: the inductive alignment had missed the class masses of the test side (L1 0.087 from the
target, against 0.028 for the alignment over the test set), and the host's statement above raises no objection to it.

The **safety line** uses neither the teacher pseudo-labels, nor API-derived inputs, nor the 397B channel. Its final
candidate uses the alignment over the test set. The note features of its second stage read only frozen texts of our own
models: the M3 notes (M3 is trained on gold labels only) and the notes of the untrained 27B protocol channel; the reference
of the example similarity consists of training rows only, and test-set aggregates of these notes enter only guards that can
abort a build. Its **strictly inductive form**, the fallback of the safety slot, uses
none of the three techniques: every test row is processed independently with parameters fixed before inference (blend
weights, inductive class weights and the second stage are fitted on training rows only), and test-set aggregates enter
only guards that can abort a build but change no prediction, so that form holds under the strictest reading of the
rules. One technical detail: the build scripts of the safety line also open the OOF matrices of the two channels with the
teacher task block (`zs27q`, `think18nb`), because a shared loading function verifies the checksums of all base channels.
These matrices enter neither the blend, nor the second stage, nor the test side of the safety line.

The final safety candidate was built although the cross-validation part of one condition of its pre-registered rule was
missed by one sixth in 9,000; the decision is recorded with its scope in
[`decisions/override_cv_condition_ii.json`](../decisions/override_cv_condition_ii.json) and checked by checksum
([METHOD.md](METHOD.md#99-second-stage-of-the-safety-line-with-alignment-over-the-test-set)).

## 7. Licences

| Component | Licence |
|---|---|
| Our code and documentation | Apache License 2.0 ([LICENSE](../LICENSE)) |
| Qwen3.5-9B, Qwen3.8-27B, Qwen3.5-397B-A17B-FP8 (inference path) | Apache 2.0; licence files read at the pinned revisions |
| Qwen3.5-397B-A17B served through the Tinker API (training-side outputs only) | Apache 2.0 according to the model card; the licence file of the served revision is still to be confirmed |
| Teacher model (closed, through a commercial API) | Not part of the solution; used during development only, under the provider's commercial terms |
| Python packages and system libraries | See [THIRD_PARTY.md](../THIRD_PARTY.md) |

## 8. Publication

This repository stays private while the competition rules do not allow publication. It will be published when the rules
allow it or when the sponsor requests it. The trained weights and all files derived from the competition data remain
private unless the rules and the sponsor allow otherwise.
