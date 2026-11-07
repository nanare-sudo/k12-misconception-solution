# Misconception Detection in K-12 Student Math Workings: Solution

> **Status:** 2026-11-07 · version 0.2, pre-final. This is a living document. The final private-leaderboard result will
> be added after the competition closes (2026-11-17 00:00 UTC). Open work is listed in [docs/ROADMAP.md](docs/ROADMAP.md).

This repository contains the code and documentation of our solution to the Kaggle competition
[Misconception Detection in K-12 Student Math Workings](https://www.kaggle.com/competitions/misconception-detection-in-k-12-student-math-workings),
hosted by CoLearn Lab. Each test row is a photo of a student's handwritten working, together with the question and its answer key. The model
predicts the **root misconception** behind the working: one of 75 taxonomy codes, or `NONE` if the working has no major
error. Up to three codes per row are submitted in order of confidence, and the score is **MAP@3**.

## Summary

- **Several vision-language "channels" read the photo, the question and the answer key.** Each channel returns a full
  probability distribution over the 76 classes. Most channels get it by exact scoring of all 74 classes that occur in
  training, through a prefix tree over the code tokens, so a code gets a probability even if the model would not
  generate it.
- **The strongest channels reason before they name a code.** An open 27B model (Qwen3.8-27B) first writes a diagnosis in
  thinking mode; all codes are then scored behind that text. Our best single model, **M3**, is the same 27B fine-tuned with
  LoRA to write such a diagnosis, followed by the root finding and the code. The mean of three of its generations reaches
  **0.802 MAP@3** out of fold.
- **Channels differ on purpose in what they know.** Some see labelled workings of other students for the same question.
  Others see no labelled row at all and follow a fixed marking protocol. Their errors differ, and the blend gains from
  both kinds.
- **Combination:** a log-linear blend with an arithmetic share, probability floors and weights fitted on LogLoss. The last
  steps are a class-marginal alignment and a small conditional-logit reranker over the top five candidates. If the
  blend puts `NONE` first, the reranker may not replace it (NONE guard), because none of its features measures whether
  the final answer is correct. Row × candidate interactions let row properties such as the blend's uncertainty act
  through the candidates, and in the primary line one column marks minor codes in rows whose final answer matches the
  key mechanically.
- **Two lines.** The **primary line** maximises cross-validated MAP@3. The **safety line** uses no input derived from the
  development-time teacher model, no output of an external API and no 397B channel; its final candidate uses the
  alignment over the test set, and its strictly inductive form stays the fallback. The two final submissions are drawn
  from these two lines.
- **Reproducible on the CPU.** Every candidate file is rebuilt byte for byte on the CPU from frozen channel outputs and
  the stored blend and second-stage files; this is verified on fresh copies of the repository with the frozen inputs.
  The private copies of these inputs are being brought to the state of this version ([docs/ROADMAP.md](docs/ROADMAP.md),
  step 5). The rotation correction of the 9B channel uses the list of the released, seeded orientation network, whose
  weights are stored with the solution.

## How it works

```mermaid
flowchart TD
    IN["Photo of the working, question, diagram, answer key, grade, topic"]
    C1["9B fine-tuned scorer, rotation-corrected"]
    C2["27B zero-shot scorer, topic block plus task block in the primary line"]
    C3["27B reason-then-name"]
    C4["27B protocol channel"]
    C5["M3: fine-tuned 27B reasoner, mean of 3 draws"]
    C6["397B notes named by M3, primary line only"]
    FL["Probability floors"]
    BL["Log-linear blend with arithmetic share, weights fitted on LogLoss"]
    AL["Class-marginal alignment over the test set; inductive in the strictly inductive fallback"]
    RR["Conditional-logit reranker over the top 5 candidates, with row x candidate interactions; M3 note features in the primary line"]
    NG["NONE guard: a NONE top from the blend stays first"]
    BK["Blank photo: REL-01 at rank 1"]
    OUT["Top-3 codes per row"]
    IN --> C1
    IN --> C2
    IN --> C3
    IN --> C4
    IN --> C5
    IN --> C6
    C1 --> FL
    C2 --> FL
    C3 --> FL
    C4 --> FL
    C5 --> FL
    C6 --> FL
    FL --> BL
    BL --> AL
    AL --> RR
    RR --> NG
    NG --> BK
    BK --> OUT
```

| Channel | Code name | Model | Trained | OOF MAP@3 | Lines |
|---|---|---|---|---:|---|
| 9B fine-tuned scorer (rotation-corrected) | `v1r` | Qwen3.5-9B + LoRA | yes | 0.6474 | both |
| 27B zero-shot scorer with task block | `zs27q` | Qwen3.8-27B (4-bit) | no | 0.6814 | primary |
| 27B zero-shot scorer with topic block | `zs27t` | Qwen3.8-27B (4-bit) | no | 0.6730 | safety |
| 27B reason-then-name | `think18nb` | Qwen3.8-27B | no | 0.7672 | primary |
| 27B reason-then-name, teacher-free | `think18sb` | Qwen3.8-27B | no | 0.7516 | safety |
| 27B protocol channel | `pr27xm` | Qwen3.8-27B | no | 0.7459 | both |
| M3: fine-tuned 27B reasoner, mean of 3 draws | `m3x3_full` | Qwen3.8-27B + LoRA | yes | 0.8020 | both |
| 397B notes, named by M3 | `q397m3` | Qwen3.5-397B-A17B + M3 | M3 only | 0.7442 | primary |

All models in the inference path are open-weight models under the Apache 2.0 licence. A closed model served through a
commercial API acted only as a development-time teacher: it pseudo-labelled the unlabelled pool, never a test row.
Its frozen labels feed the per-question *task block* of the primary line only. The full method is described in
[docs/METHOD.md](docs/METHOD.md).

## Results

Cross-validated values are out-of-fold (OOF) MAP@3 on the 1,500 labelled rows (5 folds, stratified by root code).
*Per fold* applies the last step within each 300-row fold. *As one set* applies it to all 1,500 rows at once, which is how
the test set is processed by the final candidates of both lines. Definitions and uncertainty are in [docs/VALIDATION.md](docs/VALIDATION.md).

| Line and form | Channels | Last steps | CV per fold | CV as one set | Public LB |
|---|---|---|---:|---:|---:|
| **Primary line, final form: interactions and minor-code × mechanical-match column** (current primary candidate; submitted 2026-11-06) | 6 | alignment over the test set, reranker (64 features), NONE guard | 0.8409 | 0.8398 | 0.84100 |
| Primary line, I1 form: guard form with row × candidate interactions (submitted 2026-11-06) | 6 | alignment over the test set, reranker (63 features), NONE guard | 0.8408 | 0.8387 | 0.84150 |
| Primary line, guard form: reranker main form with NONE guard and general-code identifiers (submitted 2026-11-01) | 6 | alignment over the test set, reranker (48 features), NONE guard | 0.8389 | 0.8377 | 0.84200 |
| Primary line, reranker *main form* (submitted 2026-11-01) | 6 | alignment over the test set, reranker (43 features) | 0.8376 | 0.8346 | 0.84166 |
| **Safety line, alignment over the test set, reranker with interactions** (current safety candidate; submitted 2026-11-07) | 5 | alignment over the test set, reranker (55 features), NONE guard | — | 0.8217 | 0.83516 |
| Safety line, strictly inductive form with second stage (fallback; submitted 2026-11-01) | 5 | inductive alignment, reranker (40 features), NONE guard | 0.8233 | — | 0.83183 |
| Safety line, strictly inductive form without second stage (submitted 2026-10-31) | 5 | inductive alignment | 0.8184 | — | 0.82966 |
| Best single model: M3, mean of three draws (OOF) | 1 | none | 0.8020 | — | not submitted |

—: each form of the safety line is evaluated in its own deployment form: the strictly inductive form per fold (fixed
weights per row), the final candidate as one set (its value with the gate; per fold it reaches 0.8278). The single
model has no last step.

- **Paired differences** (standard error from a bootstrap over questions; ½(F+M) is the mean of the per-fold and the
  as-one-set difference):
  - minor-code × mechanical-match column over the I1 form: ½(F+M) +0.0006; over 20 new meta-splits +0.0005 ± 0.0001,
    positive in 19 of 20;
  - row × candidate interactions over the guard form: ½(F+M) +0.0014; over 20 new meta-splits +0.0011 ± 0.0002,
    positive in 18 of 20;
  - NONE guard and general-code identifiers over the main form: +0.0013 ± 0.0014 per fold, +0.0031 ± 0.0015 as one set;
  - safety line, alignment over the test set (variant a, without interactions) over the strictly inductive form: over 20
    new meta-splits +0.0007 ± 0.0002, positive in 16 of 20; the interactions add +0.0007 ± 0.0001. On the original
    meta-split the pre-registered CV condition was missed by one sixth in 9,000 and overridden by a recorded decision
    ([decisions/override_cv_condition_ii.json](decisions/override_cv_condition_ii.json)); the test-side class masses after
    the alignment lie at L1 0.028 from the target, against 0.087 with the inductive alignment;
  - second stage with NONE guard in the strictly inductive form: +0.0049 ± 0.0022 per fold. Its gain depends on the
    generated M3 texts; on the test side, whose M3 texts are independent draws, +0.002 to +0.003 is expected.
- All files since 2026-11-01 use the rotation list of the released orientation network; the two files of 2026-10-31 used
  the list of a first, unseeded run. Rebuilt with the released list, the reranker fallback form changes the top code of
  one test row and the strictly inductive form of none.
- Public leaderboard values are observations, not selection criteria. The public split has 1,000 of the 3,498 test rows,
  and differences of about ±0.003 to ±0.005 between consecutive submissions are within its paired noise. The private
  split (2,498 rows) decides the competition.
- The step-by-step progression, ablations and error analysis are in [docs/RESULTS.md](docs/RESULTS.md).
- Final scores: *to be added after the competition closes.*

## Repository layout

```
k12-solution/
├── README.md · LICENSE · THIRD_PARTY.md · CHANGELOG.md
├── docs/                  method, validation, results, training, inference, data & compliance, code map, roadmap
├── configs/
│   ├── folds.parquet      the fixed 5-fold split (row position → fold; hash-checked)
│   └── vlm/               one YAML file per model run (training, OOF and test-side jobs)
├── src/                   metric, label space, submission checks, transcription cleaning, second stage, answer matching,
│                          orientation net
│   └── vlm/               model runner: prompts, exact code scoring, LoRA training, vLLM text generation
├── decisions/             recorded decisions that a build script checks by their pinned checksum
├── kaggle_core/           small shared library: artifacts, folds, metrics, submission validation
├── eda/                   step scripts: OOF assembly, blend fitting, deployment checks, second stages
├── tools/
│   ├── make_submission.py assembly of a submission from frozen channel outputs
│   ├── builds/            one build script per submitted or candidate file, plus the rotation-list helper
│   ├── replay/gates.py    reconstruction and check of the gate lists of the reasoning channels
│   └── …                  fold array, rotation lists and their checks, frozen M3 texts, 397B notes; API clients (development only)
└── tests/                 unit and regression tests
```

Neither the competition data nor anything derived from it per row is part of this repository, except the fold
assignment by row position (`configs/folds.parquet`, no identifier and no label). The excluded files are
transcriptions, model texts, pseudo-labels, OOF matrices, frozen test outputs, submissions and trained weights; they are
kept privately and referenced by path and, where pinned, by SHA-256
(see [docs/DATA_AND_COMPLIANCE.md](docs/DATA_AND_COMPLIANCE.md)).
[docs/CODEMAP.md](docs/CODEMAP.md) maps every script to its role. Some file and script names carry historical build
labels such as `s029`, `s030n`, `s034` or `s035`; they are kept because the frozen artifacts and their checksums refer to
them. The build scripts added in version 0.2, their deployment files and output files are named by role:
`primary_i1`, `primary_final` and `safety_final`.

## Quick start

The solution can be reproduced at three depths:

| Mode | What is recomputed | Expected result | Hardware |
|---|---|---|---|
| `assemble` | Blend, alignment, reranker and output file, from frozen channel outputs | Byte-identical to the submitted file | CPU, minutes |
| `rescore` | Every scoring and ranking step on all test images, behind the frozen reasoning texts and gates | Identical on the reference hardware; statistically equivalent on other hardware | GPUs: about 10 h on a 96 GB card plus A40 and Kaggle T4 time, and 4×H200 for the 397B parity check (preliminary estimate from before the 397B channel and the second stages) |
| `full` | Everything from images and CSV files, including new reasoning texts | Statistically equivalent: each generated text is a new draw | GPUs: about 30 h on a 96 GB card plus A40 and Kaggle T4 time, and 4×H200 for the 397B notes (preliminary estimate from before the 397B channel and the second stages) |

**Assemble (CPU).** Put the competition data under `data/`. The private inputs, which are available to the competition
host for verification but are not public, go under `data/derived/` (frozen derived inputs such as gate lists, rotation
lists and notes) and under `artifacts/` (frozen channel outputs, OOF matrices, blend and second-stage files, the
description of the orientation network, and earlier submission files that the build scripts use as references); see
[docs/INFERENCE.md](docs/INFERENCE.md#3-data-layout). The private copies of these inputs are being brought to the state
of this version ([docs/ROADMAP.md](docs/ROADMAP.md), step 5). Then:

```bash
# Python 3.13 with the CPU versions listed in docs/INFERENCE.md (section 4)
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2
python -I tools/make_folds.py                              # artifacts/folds.npy from configs/folds.parquet
ORIENT=legacy bash tools/builds/s029_m3x3.sh safe          # earlier files with the first-version rotation list,
ORIENT=legacy bash tools/builds/s034_stage2_main.sh best   #   used as references by the two builds below
bash tools/builds/primary_final.sh                         # primary line, final candidate (runs primary_i1.sh, s035_… first)
OVERRIDE_CV_II=override_cv_condition_ii bash tools/builds/safety_final.sh   # safety line, final candidate (runs s036_… first)
sha256sum artifacts/submissions/sub_primary_final.csv artifacts/submissions/sub_safety_final.csv
```

The expected checksums of these files and of their fallbacks are in [docs/INFERENCE.md](docs/INFERENCE.md#2-lines-and-expected-checksums).
The build scripts use the rotation list of the released orientation network by default (`ORIENT=repro`); `ORIENT=legacy`
only rebuilds the earlier files. `OVERRIDE_CV_II=override_cv_condition_ii` applies the recorded decision
[decisions/override_cv_condition_ii.json](decisions/override_cv_condition_ii.json) (checked by its pinned checksum): it
overrides only the cross-validation part of one condition of the safety candidate's pre-registered rule; without it the
safety build stops after that rule with exit code 5. A single entry point per line that does not need the chain of earlier builds is planned
([docs/ROADMAP.md](docs/ROADMAP.md)). `rescore` and `full` run the model steps listed in [docs/INFERENCE.md](docs/INFERENCE.md).
Training of every component is described in [docs/TRAINING.md](docs/TRAINING.md). The unit and regression tests run with
`python -m pytest tests -q`; tests that need private files skip themselves.

## Status and next steps

- Both final candidates, their fallbacks and the historical files are rebuilt byte for byte from the frozen inputs in
  fresh copies of the repository. With the development version of the code, the assembly of both final candidates was
  also repeated on a Kaggle CPU without network access, byte for byte; the same check with this repository is planned.
- Open before the final selection (by 2026-11-15): smoke tests without network access for the model steps, including
  the 397B path, which decides whether the 397B channel stays in the primary file; the remaining second-hardware checks
  of the M3 draws; an estimate of the rerun variance of the generated texts; the private copies of the frozen inputs in
  the state of this version.
- Details and dates: [docs/ROADMAP.md](docs/ROADMAP.md).

## Documentation

| Document | Content |
|---|---|
| [docs/METHOD.md](docs/METHOD.md) | Data, channels, prompts and protocols, training, teacher pseudo-labels, orientation network and rotation list, blend, alignment, second stages with NONE guard, the two lines |
| [docs/VALIDATION.md](docs/VALIDATION.md) | Folds, measurement forms, paired uncertainty, decision rules, controls, checks of the second-stage components, risk rule of the inductive line, checks before a submission |
| [docs/RESULTS.md](docs/RESULTS.md) | Progression, submissions, single-channel values, ablations and null results, error analysis, compute |
| [docs/TRAINING.md](docs/TRAINING.md) | Every trained or fitted component: inputs, command, hardware, duration, expected checks |
| [docs/INFERENCE.md](docs/INFERENCE.md) | The modes `assemble`, `rescore` and `full`: commands, hardware, run times, expected checksums |
| [docs/DATA_AND_COMPLIANCE.md](docs/DATA_AND_COMPLIANCE.md) | What data lives where, API use, offline inference, test-data policy, licences |
| [docs/CODEMAP.md](docs/CODEMAP.md) | Index of scripts and modules, channel names, data formats |
| [docs/ROADMAP.md](docs/ROADMAP.md) | Current status and remaining steps up to the final submission |
| [THIRD_PARTY.md](THIRD_PARTY.md) | Models, services and packages with licences |
| [CHANGELOG.md](CHANGELOG.md) | Versions of this repository |

## License

Our own code and documentation are released under the [Apache License 2.0](LICENSE). Models, packages and services by
third parties keep their own licences; see [THIRD_PARTY.md](THIRD_PARTY.md). The competition data are subject to the
competition rules and are not distributed here.

## Acknowledgements

We thank CoLearn Lab for hosting the competition and providing the data, and the Qwen team for releasing the open models
this solution is built on.
