# Data and compliance

*Status: 2026-11-01 (version 0).*

This document states which data the solution uses and where each part is kept. It also explains why some parts are not in
this repository, how external APIs were used, and how the solution meets the competition's requirements for the final
submission.

## 1. Competition data

- **Source:** the competition page on Kaggle: `train.csv`, `test.csv`, `unlabelled.csv`, the taxonomy, the sample
  submission, and the photos of the training, test and unlabelled sets plus the question diagrams.
- **Licence:** subject to the competition rules. The data, and models trained on them, are used only for this
  competition.
- **Not redistributed.** No competition file, no verbatim excerpt (question text, transcription, explanation), no row
  identifier and no image is part of this repository.
- **Data protection.** The host screened the photos for personal information before release. We make no attempt to
  identify students.

## 2. What is not in this repository, and where it is

| Item | Why it is not here | Where it is kept |
|---|---|---|
| Competition data | Competition rules | Kaggle competition page |
| Per-row derived inputs: frozen reasoning texts, protocol answers and notes, M3 texts, 397B notes, rotation, gate and blank-photo lists, training targets | Derived from competition data, row by row | Private Kaggle dataset `k12-derived` (frozen inputs for `assemble` and `rescore`); locally under `data/derived/` |
| Teacher pseudo-labels for the unlabelled pool | Derived from competition data; to be used only for this competition | Private (`data/derived/teacher/`), with checksums |
| OOF and test probability matrices, channel outputs | Per-row predictions | Private (`artifacts/`) |
| Blend and second-stage files | Fitted weights, partly with row positions | Private Kaggle model `k12-weights`; locally under `artifacts/` |
| Trained weights: 9B adapter `v1_full`, M3 adapter `m3_full`, orientation network | Trained on competition data | Private Kaggle model `k12-weights` |
| Submission files | Per-row test predictions | Private |
| Base models (Qwen3.5-9B, Qwen3.8-27B, Qwen3.5-397B-A17B-FP8) | Public and very large | Hugging Face at pinned revisions; file checksums in the weights manifest (**TODO**) |
| API keys and credentials | Secrets | Never stored in the repository |

The code refers to these files by relative path, and the manifests record their SHA-256. All private files are available
to the competition host for verification. The repository's `.gitignore` excludes `data/`, `artifacts/`, model weights and
row-level file formats.

## 3. Test data policy

- **No manual labelling and no manual transcription** of the test set, as the competition rules require.
- **No inspection of test outputs.** Test photos and per-row outputs derived from them (model texts, rankings,
  predictions) are not looked at by a person. Scripts that touch the test side print only aggregates: counts, shares and
  checksums. All checks before a submission work on such aggregates ([VALIDATION.md](VALIDATION.md#6-checks-before-a-submission)).
- **No API on test rows.** Test rows were processed only on our own or rented machines with open-weight models.
- **No leaderboard probing.** No variant was chosen by its public score.

## 4. External APIs (development only)

| Service | Purpose | Rows sent | Part of the final inference |
|---|---|---|---|
| Anthropic Messages API: Claude Haiku 5.5, with Sonnet 5.5 for comparison measurements | Teacher pseudo-labels for the 50,000 unlabelled photos; measurements of the teacher on training rows | Unlabelled and training rows only | No. The frozen labels feed the per-question task block of the primary line |
| Tinker sampling API with open models (Qwen3.5-397B-A17B, Inkling-Small, Qwen3.8-27B) | Measurements of open models on training rows; the training-side outputs of the 397B channel | Training rows only | No. The 397B test side runs on our own hardware with the same open weights; a parity check on 64 training rows confirmed identical model inputs |

**Host statements on API use** (competition forum):
- "No external APIs allowed in final solution. Solution should be reproducible without internet access though you are
  free to use any APIs for fine-tuning, enriching your models using unlabeled data or any other useful approaches"
- On paid API pseudo-labels for training and unlabelled rows: "Yes. You can do that as long as you are using them only
  for this competition."

**Provider terms.** Both are business offerings. Inputs are not used to train or improve the provider's models, and the
provider claims no ownership of inputs or outputs. API outputs are frozen with checksums; training and assembly read only
the frozen files. Total API cost was about 0.1 kUSD (Anthropic about 41 USD, Tinker about 63 USD).

## 5. Final solution: offline, self-contained, within the limits

| Requirement | How the solution meets it |
|---|---|
| No external API and no internet access at inference time | All inference steps use open-weight models on local hardware; the text-generation server is bound to 127.0.0.1. Offline settings are listed in [INFERENCE.md](INFERENCE.md#4-environments). The offline package and a smoke test without network access for every model step are being completed ([ROADMAP.md](ROADMAP.md)) |
| No human step | `assemble` runs as single commands, with checks that stop on any deviation; `rescore` and `full` run configured jobs (stage-wise runner: **TODO**) |
| Self-contained: model weights and all labels made for the unlabelled data | Adapters and the orientation network are delivered with checksums; base models are pinned by revision and file checksum; the teacher pseudo-labels are delivered frozen |
| Model size: "Any model of the same size or smaller than Nemotron 550B." | The largest model is Qwen3.5-397B-A17B (397 B parameters, 17 B active); it is used only in the primary line |
| Compute | The host sets no compute limit and reproduces inference for the top 10 of the final leaderboard. Reference hardware and run times are in [INFERENCE.md](INFERENCE.md) |
| The delivered solution must reproduce the submitted file | `assemble` reproduces each submitted file byte for byte from the frozen channel outputs; `rescore` and `full` regenerate the channel outputs ([INFERENCE.md](INFERENCE.md#1-three-modes)) |

## 6. The two lines and the techniques that rest on interpretation

Two final submissions may be selected. We use them to keep two kinds of technique out of one of the two lines:
techniques whose admissibility rests on our reading of the rules and the host's statements, and techniques that depend
on external APIs during development.

| Technique | Basis | Line |
|---|---|---|
| Transductive class-marginal alignment | It uses the set of test predictions: no label and no information about individual rows. It relies on the split being stratified by root code; the host said of the test share of `NONE`: "It should be similar" | Primary line only |
| Teacher pseudo-labels of the unlabelled pool (task block) | API pseudo-labels on unlabelled rows are explicitly allowed for this competition | Primary line only |
| Blend weight of the 397B channel fitted on API outputs for training rows | API use on training rows is allowed; the test side is computed locally with the same open weights | Primary line only |

The **strictly inductive line** uses none of these. Every test row is processed independently with parameters fixed
before inference. The line uses no teacher-derived input, no API-derived input and no transductive step, so it holds
under the strictest reading of the rules.

## 7. Licences

| Component | Licence |
|---|---|
| Our code and documentation | Apache License 2.0 ([LICENSE](../LICENSE)) |
| Qwen3.5-9B, Qwen3.8-27B, Qwen3.5-397B-A17B-FP8 (inference path) | Apache 2.0; licence files read at the pinned revisions |
| Qwen3.5-397B-A17B served through the Tinker API (training-side outputs only) | Apache 2.0 according to the model card (**TODO:** confirm the licence file of the served revision) |
| Teacher model (closed, through a commercial API) | Not part of the solution; used during development only, under the provider's commercial terms |
| Python packages and system libraries | See [THIRD_PARTY.md](../THIRD_PARTY.md) |

## 8. Publication

This repository stays private while the competition rules do not allow publication. It will be published when the rules
allow it or when the sponsor requests it. The trained weights and all files derived from the competition data remain
private unless the rules and the sponsor allow otherwise.
