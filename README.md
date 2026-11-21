# Misconception Detection in K-12 Student Math Workings: solution of team Leo

*Version 0.6, 2026-11-21: the state we deliver for the host's verification. We will add the private-leaderboard result
after the verification.*

This is our solution to a
[Kaggle competition](https://www.kaggle.com/competitions/misconception-detection-in-k-12-student-math-workings) hosted
by CoLearn Lab. Each test row is a photo of a student's handwritten maths working, with the question, an optional
diagram and the answer key. The model names the root misconception behind the working: one of 75 taxonomy codes, or
`NONE` if the working has no major error. Up to three codes per row are scored with MAP@3.

## The solution

Several vision-language models read the photo, the question and the answer key. We call each of these model set-ups a
channel. Every channel gives a probability for each of the 76 classes. Most channels compute the exact log-probability
of every code as the continuation of the prompt. The strongest channels first write a diagnosis of up to a few
thousand tokens and then name the code. Our best single model, M3, is Qwen3.8-27B fine-tuned with LoRA to work this
way. A blend combines the channels. A label-free alignment then brings the predicted class shares over the whole test
set to the class shares of the training labels. Last, a small reranker, our second stage, reorders the five best codes
of each row. All base models used at inference time are open-weight models under Apache 2.0. Our adapters and the
orientation network are in the private package. No inference step calls an API.

| Channel | Name in the code | Model | Trained by us | Out-of-fold MAP@3 |
|---|---|---|---|---:|
| 9B scorer, rotated photos turned upright | `v1r` | Qwen3.5-9B + LoRA | yes | 0.647 |
| Untrained 27B scorer with same-question examples | `zs27q` | Qwen3.8-27B, 4-bit | no | 0.681 |
| 27B reasoning channel | `think18nb` | Qwen3.8-27B | no | 0.767 |
| 27B marking protocol, ranked by M3 | `m3px27xm` | Qwen3.8-27B and M3 | M3 only | 0.755 |
| M3, mean of four draws (A: five) | `m3x4_full` (A: `m3x5h`) | Qwen3.8-27B + LoRA | yes | 0.804 (A: 0.810) |
| 397B notes, ranked by M3 | `q397m3` | Qwen3.5-397B-A17B (FP8) and M3 | M3 only | 0.744 |

A draw is one generated text per row; each new run gives a new draw. The method is in [docs/METHOD.md](docs/METHOD.md),
and [docs/GLOSSARY.md](docs/GLOSSARY.md) explains every internal name.

## Results

| Final submission | File | Kaggle reference | Public score | CV per fold / as one set |
|---|---|---|---:|---:|
| A | `sub_primary_m3x5h.csv` | 58353348 | 0.84266 | 0.8426 / 0.8404 |
| B | `sub_primary_m3x4.csv` | 58139357 | 0.84450 | 0.8436 / 0.8380 |

B combines the six channels above. A adds nine reranker columns, which are extra input features of the reranker. It
also adds a fifth M3 draw that reads the photo at 2,048 px, on the rows whose characters are at most 20 px high. CV is
out-of-fold MAP@3 on the 1,500 training rows with five stratified folds. "Per fold" applies the alignment inside each
fold of 300 rows; "as one set" applies it to all 1,500 rows at once, as on the test set. Each component of both
submissions was decided on cross-validation before its public score was known
([VERIFICATION §1](docs/VERIFICATION.md#1-which-submission-is-which), [RESULTS](docs/RESULTS.md)).

## Reproduce

1. Clone the repository, check out the tag `v0.6` and install the CPU environment with Python 3.13:

   ```bash
   git clone <repository URL> k12-solution && cd k12-solution
   git checkout v0.6
   python3.13 -m venv .venv && . .venv/bin/activate
   python -m pip install -r env/requirements-cpu.txt
   ```

   The pins include torch 2.11.0 with its CUDA libraries, about 3.5 GB to download; nothing runs on a GPU.
   [VERIFICATION §2](docs/VERIFICATION.md#2-environment) gives a smaller install with the CPU build of torch.

2. Put `train.csv` and `test.csv` from the root of the competition download under `data/`, and `taxonomy_v1.csv` and
   `sample_submission.csv` under `data/competition_data/`. Links are fine. Create the folder first with
   `mkdir -p data/competition_data`, or link the whole `competition_data/` folder of the download there. The CPU check
   reads no photo.

3. Download our private package (the link is in our email) and unpack it next to the clone: `unzip k12-host-package.zip`
   gives the folder `k12-host-package/`. Run the quick check from the repository root, with the path of that folder as
   `<package-dir>`, here `../k12-host-package`. It takes about 2 minutes:

   ```bash
   bash reproduce.sh ../k12-host-package
   ```

   It checks the four CSV files and the package against their checksums and sets up `artifacts/` and `data/derived/`. It
   then rebuilds both submission files from the frozen model outputs and compares them with the submitted files. It ends
   like this, with exit code 0; only the time differs:

   ```
   OK A 4505c15950167650a66cdc60eb9c153eebfb9ed6192277bd283045f3797354e7
   OK B 76da786dfdbe6a6ddcbcc93070f385e18870cb5cdaea895d92104aa3565f9973
   All checks passed (2m01s).
   ```

`bash reproduce.sh <package-dir> --full` also runs the full proof, the build chain of about 1.5 to 2.5 hours. It refits
or verifies every blend and second-stage file, compares it bit for bit with the delivered file and builds both
submissions again. The CPU check needs about 14 GB of free disk
([VERIFICATION §3](docs/VERIFICATION.md#3-hardware-and-run-times)).

On GPUs, `tools/gpu_inference.sh` runs the inference from our trained weights. In mode `rescore` it recomputes every
scoring and ranking step and reuses the generated texts that we froze. In mode `full` it also generates every text
anew. Both modes end by writing the two submission files
([VERIFICATION §5.2](docs/VERIFICATION.md#52-rescore-and-full-gpu)). The script has run as a dry run and on its CPU
steps. Its GPU jobs ran without network access on 8 test rows each on 2026-11-21; a run over all test rows
has not been done. Our inference on the test set ran on Kaggle T4 × 2, an A40 48 GB, an RTX PRO 6000
Blackwell 96 GB and, for the 397B model, 4 × H200. Training used the A40 and the RTX PRO 6000.

## Where to find each item of the host's checklist

| Item | Where |
|---|---|
| Team details | Team Leo, one member, Kaggle username `nanaresudo`; the contact address is in our email |
| Code | This repository at the tag `v0.6` |
| Environment: Python, requirements, CUDA | [VERIFICATION §2](docs/VERIFICATION.md#2-environment) and `env/requirements-*.txt` |
| Hardware, RAM and run times | [VERIFICATION §3](docs/VERIFICATION.md#3-hardware-and-run-times) |
| Where to place the competition data | [VERIFICATION §4](docs/VERIFICATION.md#4-where-to-place-the-data) |
| (a) Inference from the trained weights | [VERIFICATION §5](docs/VERIFICATION.md#5-a-inference-from-the-trained-weights-to-the-submission-file) |
| (b) Training from scratch | [VERIFICATION §6](docs/VERIFICATION.md#6-b-training-from-scratch) |
| Seeds; what cannot be rerun exactly; how close a rerun should come | [§7](docs/VERIFICATION.md#7-seeds), [§8](docs/VERIFICATION.md#8-what-cannot-be-rerun-exactly), [§8.1](docs/VERIFICATION.md#81-how-close-a-full-rerun-should-come) |
| Weights and large artefacts | [VERIFICATION §9](docs/VERIFICATION.md#9-private-weights-and-inputs) |
| Write-up | [docs/WRITEUP.md](docs/WRITEUP.md) |
| External models, data and APIs, with API cost | [THIRD_PARTY.md](THIRD_PARTY.md) |
| Which submission is which | [VERIFICATION §1](docs/VERIFICATION.md#1-which-submission-is-which) |
| Licence | [LICENSE](LICENSE) |

## Other documents

- [docs/METHOD.md](docs/METHOD.md): the method in full.
- [docs/VALIDATION.md](docs/VALIDATION.md): folds, uncertainty, decision rules and the checks before a submission.
- [docs/RESULTS.md](docs/RESULTS.md): progression, ablations, error analysis and compute.
- [docs/TRAINING.md](docs/TRAINING.md): every trained or fitted component, with commands and checks.
- [docs/INFERENCE.md](docs/INFERENCE.md): the three inference modes in detail, and the 397B path.
- [docs/DATA_AND_COMPLIANCE.md](docs/DATA_AND_COMPLIANCE.md): where data lives, API use, use of the test set.
- [docs/PROVENANCE.md](docs/PROVENANCE.md): origin of every frozen input of the two final submissions.
- [docs/MODEL_SIZES.md](docs/MODEL_SIZES.md): model sizes against the host's limits.
- [docs/weights_manifest.json](docs/weights_manifest.json): files of the base models and of our weights, with checksums.
- [docs/CODEMAP.md](docs/CODEMAP.md): every script and module with its role.
- [docs/GLOSSARY.md](docs/GLOSSARY.md): internal names of channels, runs, scripts, files and package folders.
- [docs/ROADMAP.md](docs/ROADMAP.md): open work.
- [CHANGELOG.md](CHANGELOG.md): versions of this repository.

## Repository layout

```
configs/       the fixed fold split (folds.parquet) and one YAML file per model run (vlm/)
src/           metric, labels, submission checks, second stage; vlm/ holds the model runner
tools/         assembly (make_submission.py), build scripts (builds/), GPU driver (gpu_inference.sh), helpers
eda/           step scripts that fit the blends and second stages; the build scripts call them
env/           pinned requirements of the four environments
tests/         unit and regression tests; how to run them: docs/VERIFICATION.md, section 5.1
kaggle_core/   small shared library;  decisions/  one recorded decision that a build script checks;  docs/
```

## Data and licence

The repository holds no competition data and nothing derived from it per row, except the fold assignment by row position
in `configs/folds.parquet`. Model outputs, pseudo-labels, notes and our weights are in the private package. Our code and
documentation are released under the [Apache License 2.0](LICENSE). Third-party models, packages and services keep their
own licences ([THIRD_PARTY.md](THIRD_PARTY.md)). We thank CoLearn Lab for hosting the competition and the Qwen team for
the open models that this solution is built on.
