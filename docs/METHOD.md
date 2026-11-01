# Method

*Status: 2026-11-01 (version 0). Numbers are out-of-fold (OOF) MAP@3 on the 1,500 labelled training rows unless stated
otherwise. "±" is the standard error from a bootstrap over questions; see [VALIDATION.md](VALIDATION.md).*

**Contents:**
1. [Task and data](#1-task-and-data)
2. [Pipeline overview](#2-pipeline-overview)
3. [Shared building blocks](#3-shared-building-blocks)
4. [Channels](#4-channels)
5. [Teacher pseudo-labels for the unlabelled pool](#5-teacher-pseudo-labels-for-the-unlabelled-pool)
6. [Orientation network](#6-orientation-network)
7. [Blend](#7-blend)
8. [Class-marginal alignment](#8-class-marginal-alignment)
9. [Second stage: conditional-logit reranker](#9-second-stage-conditional-logit-reranker)
10. [Blank photos](#10-blank-photos)
11. [The two lines](#11-the-two-lines)
12. [Design principles](#12-design-principles)

---

## 1. Task and data

### 1.1 Task and metric

Each row is a photo of a handwritten solution by an Indonesian student (grades 4–12). The row also carries the question
text (Indonesian, with LaTeX), an optional question diagram, the answer key, the grade and the topic. **No transcription
is available at test time**, so reading the handwriting is part of the task. Training rows carry extra fields:
- a line-by-line transcription;
- whether the final answer is correct;
- a `clean` flag;
- the annotators' findings (step, span, category, severity, root flag, explanation).

The target `root_code` is the misconception from which the rest of the working follows. It is one of 75 taxonomy codes,
or `NONE` when the working has no major error. Minor slips that change neither the method nor the result count as
`NONE`.

The metric is **MAP@3**. Up to three codes per row are submitted in order of confidence. A hit at rank 1, 2 or 3 scores
1, 1/2 or 1/3. Codes outside the taxonomy are ignored. The implementation is in `src/metric.py`; it breaks ties
deterministically.

| Split | Rows |
|---|---:|
| Train (labelled) | 1,500 |
| Test | 3,498 (public leaderboard 1,000, private leaderboard 2,498) |
| Unlabelled pool (photo and question only) | 50,000 |
| Question diagrams | 119 |

### 1.2 Properties of the data that shaped the method

- **Output space.** There are 76 classes: `NONE` plus 75 codes. 74 of them occur as training labels (73 codes and
  `NONE`). `NOT-02` and `PRO-03` never occur as a root, and according to the host they cannot be correct in the test set.
  Every channel therefore gives zero probability to the two classes without a training example.
- **Class balance.** `NONE` covers 506 rows (33.7 %); the host states a similar share for the test set. Five *general*
  codes (`ANS-01`, `CAL-01`, `WOP-01`, `FRM-01`, `OTH-00`) were capped by the host at exactly 57 rows each. The tail is
  long: 13 codes have a single example, and 44 codes have at most 10.
- **Question bank.** There are 499 questions in 61 topics. 95.7 % of test rows have their question in the training set,
  but coverage is thin: the median is 3 training rows per question. 150 test rows belong to 37 questions without any
  training row. As a result, a row's code has often never been seen for its question. Rows of this kind account for
  about two thirds of the remaining loss ([RESULTS.md](RESULTS.md#6-error-analysis)).
- **Split.** According to the host, the split is disjoint by student and stratified by root code and grade. The data
  contain no student identifier, so the student-disjoint split cannot be reproduced locally.
- **Two sub-collections.** Rows 0–357 and 358–1,499 of `train.csv` differ in label mix (`NONE` 41.6 % vs 31.3 %) and in
  annotation style. The second part is ordered by category. Rows are therefore never processed in file order, and all
  runs use one fixed, stratified fold assignment.
- **Label noise** is announced by the host. For the same question and the same wrong final answer, two workings carry
  the same code in about three quarters of pairs (95 % CI 0.60–0.89).
- **Root versus first error.** In 92 % of error rows, the root is the earliest major finding in the annotators' list.
  The host's wording is "most of the time would be the earliest but not mandatory".
- **Transcription leaks.** 129 training transcriptions contain annotator verdicts. These are stripped
  (`src/transcription.py`) before a transcription is used anywhere, for example as a training target or in a
  same-question example.
- **Images.** The photos are JPEG files with a long side of 1,024 px and free aspect ratio; 6.7 % come at 2,048 or
  3,072 px. About one photo in eight lies on its side (11–13 %), and about 1 % is upside down. Handwritten characters
  are 19–20 px high at 1,024 px. 2.4–2.8 % of the images are screenshots. One training photo and six test photos are
  black.
- **Language.** Question texts are Indonesian; the annotators' notes are English.
- **Unlabelled pool.** The pool covers 284 questions with at most 350 photos per question. Diagram questions and upper
  grades are rare in it. According to the teacher pseudo-labels (§5), about 72 % of pool workings are `NONE`.

The competition overview reports reference accuracies: 73 % for the best off-the-shelf model, 45–59 % for open models
and 63–66 % after fine-tuning. These were measured as accuracy on given transcribed text, not as MAP@3 on photos, so they
cannot be compared with the numbers in this repository.

## 2. Pipeline overview

```
photo (long side 1,024 px) + question text / diagram (448 px) + answer key + grade + topic
        │
        ├── v1r       Qwen3.5-9B + LoRA, code-first scoring (rotated photos turned upright first)
        ├── zs27q/t   Qwen3.8-27B, no training, code-first scoring (topic block; zs27q adds the teacher task block)
        ├── think     Qwen3.8-27B, no training, reasons ≤ 1,800 tokens, then scores all codes behind its own text
        ├── pr27xm    Qwen3.8-27B, no training, marking protocol without labelled rows: notes → 3 codes + exact ranking
        ├── M3        Qwen3.8-27B + LoRA, reasons then names; mean of 3 draws
        └── q397m3    Qwen3.5-397B-A17B writes notes; M3 scores all codes behind them        [primary line only]
        │
        ▼  per channel: distribution over 76 classes → probability floor (mix with uniform)
   log-linear blend with arithmetic share β (weights fitted on LogLoss)
        ▼
   class-marginal alignment (primary: transductive over the full test set · inductive line: fixed weights)
        ▼
   [primary line] second stage: conditional-logit reranker over the top-5 candidates
        ▼
   rule: black/empty photo → REL-01 at rank 1      →  top-3 codes per row
```

The assembly is implemented in `tools/make_submission.py`; the steps follow the order shown above.

## 3. Shared building blocks

### 3.1 Inputs

- **Photo.** Resized to a long side of 1,024 px with the aspect ratio preserved. Photos are never squashed to a square,
  because the characters are small.
- **Question diagram.** If present, it is passed as a second image with a long side of 448 px.
- **Text.** Question text and answer key with normalised line endings, plus grade and topic.
- **Rotation (9B only).** Photos that the orientation network (§6) classifies as rotated with confidence ≥ 0.8 are turned
  upright before the 9B model reads them.

### 3.2 Exact scoring of all codes

Most channels do not just generate a code. They score every candidate:
1. The prompt ends in an answer prefix such as `Root:`. For reasoning channels, the prefix is the model's own text
   followed by `Root:`.
2. The engine (`src/vlm/engine.py`) computes the exact log-probability of each of the 74 classes with a training example
   as a continuation of that prefix.
3. Codes share token prefixes, so the candidates are organised as a prefix tree, and shared prefixes are computed once.
   The tree is checked against a reference implementation.
4. The log-probabilities are converted into a distribution over the 76 classes. The two classes without a training
   example get probability 0.

The result is a full distribution per row and channel. The blend, the alignment and the reranker need exactly this: they
work with probabilities for every class, including codes that a model would rarely generate.

### 3.3 Context given to the models

| Block | Content | Where used |
|---|---|---|
| Taxonomy | All 75 codes with name and description, plus a `NONE` line | All channels |
| Same-question examples | Up to 4 labelled workings of other students for the same question. Each shows the cleaned transcription, whether the final answer was correct, the root code and the annotators' short reason. One example per (code, answer correct) pair comes first; the rest fills up. In cross-validation they come only from the other folds; at test time from all 1,500 training rows | 9B scorer, 27B zero-shot scorers, reason-then-name |
| Topic block | Up to ten root codes from *other* questions of the same topic, each with its frequency and one short annotator note. A first line gives the number of error-free workings, so that the list is not read as "there is always an error". Built from other folds only | 27B zero-shot scorers, reason-then-name |
| Task block | A per-question summary of the teacher pseudo-labels on pool photos of the *same* question (§5): the share of error-free workings, then up to 8 codes with frequency, the most common student final answers for each code and one note. It contains no information from labelled training rows, because the teacher never saw one | Primary line only: `zs27q` and draw *g* of the reason-then-name channel |
| Marking protocol | Judging rules and the taxonomy only, **no labelled row and no block** | Protocol channel, M3, 397B |

### 3.4 Generating text, then scoring behind it

Text-generating channels write their reasoning with a local vLLM server. The server is bound to 127.0.0.1 and serves
64 concurrent requests. The exact scoring of §3.2 then runs on our own Hugging Face path, behind the generated text. This
split has two advantages. Texts can be frozen with checksums. Given a frozen text and the same hardware, the scoring is
deterministic, so the expensive and non-deterministic generation never has to be repeated just to rebuild a channel.

## 4. Channels

| Channel | Code name | Model | Trained | Context | OOF MAP@3 |
|---|---|---|---|---|---:|
| 9B fine-tuned scorer | `v1` / `v1r` | Qwen3.5-9B + LoRA | yes (gold labels) | taxonomy, ≤ 4 same-question examples | 0.6406 / 0.6474 |
| 27B zero-shot scorer, task block | `zs27q` | Qwen3.8-27B, 4-bit | no | examples, topic block, task block | 0.6814 |
| 27B zero-shot scorer, topic block | `zs27t` | Qwen3.8-27B, 4-bit | no | examples, topic block | 0.6730 |
| 27B reason-then-name | `think18nb` | Qwen3.8-27B, bf16 | no | as `zs27q` (draw g) / as `zs27t` (draw n) | 0.7672 |
| 27B reason-then-name, teacher-free | `think18sb` | Qwen3.8-27B, bf16 | no | as `zs27t` | 0.7516 |
| 27B protocol channel | `pr27xm` | Qwen3.8-27B, bf16 | no | marking protocol only | 0.7459 |
| M3: fine-tuned 27B reasoner | `m3x3_full` | Qwen3.8-27B + LoRA | yes (gold labels) | marking protocol only | 0.8020 |
| 397B notes, named by M3 | `q397m3` | Qwen3.5-397B-A17B + M3 | M3 only | marking protocol only | 0.7442 |

The question prior (per-question code frequencies, no image) reaches 0.5149 ± 0.011.

### 4.1 9B fine-tuned scorer (`v1`, `v1r`)

| Item | Setting |
|---|---|
| Base model | `Qwen/Qwen3.5-9B` at revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`, bf16 |
| Adapter | LoRA r 32, α 64, dropout 0.05 on layers 20–31 |
| Task format | The answer starts with `Root: <code>`. After the code come three auxiliary training targets: answer check ("Final answer correct: yes/no"), cleaned transcription and findings. At inference only the code position is scored |
| Prompt | Persona of an experienced mathematics teacher; the full taxonomy in a fixed prompt head; question, answer key, photo, diagram; up to 4 same-question examples from other folds (15 % of training samples are shown without examples) |
| Loss | The code loss is **summed** over the code tokens, not averaged over all target tokens. On one fold this gave +0.018 ± 0.010 over averaging and removed the over-prediction of `NONE`. Weights: code 1.0, answer check 0.5, auxiliary targets 0.5 |
| Optimisation | 3 epochs, learning rate 1.5e-4, gradient accumulation 8, seed 42 |
| Scoring | Exact scores of all 74 classes (§3.2) |
| Rotation correction (`v1r`) | Photos classified as rotated with confidence ≥ 0.8 (§6) are turned upright before reading. This affects 124 of the 1,500 training rows; on those rows MAP@3 rises by +0.108 ± 0.024 |
| Values | `v1` 0.6406 ± 0.0105, which is +0.125 ± 0.009 over the question prior. `v1r` 0.6474 |
| Test side | Kaggle T4×2 in fp16, without network access |

### 4.2 27B zero-shot scorers (`zs27q`, `zs27t`)

- **Model.** `Qwen/Qwen3.8-27B` at revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, loaded in 4-bit (nf4), no
  training.
- **Prompt.** Taxonomy, question, answer key, photo (1,024 px), diagram (448 px), up to 4 same-question examples and the
  topic block. `zs27q` adds the teacher task block.
- **Scoring.** Code-first, exact over all classes.
- **Values.** `zs27t` 0.6730, `zs27q` 0.6814. When these channels were introduced, the blend of the 9B scorer and `zs27q`
  reached 0.7109 with the task block and 0.6882 without it.
- **Run time.** 3.1–3.7 s per row on an A40 in 4-bit. `zs27t` also runs on a Kaggle T4 in 4-bit without network access
  (17–19 s per row).

### 4.3 27B reason-then-name (`think18nb`, `think18sb`)

The model first reasons about the working and only then names the code. This channel brought the largest single gain of
the project. When it was introduced, the blend of the 9B scorer and `zs27q` rose from 0.7109 by +0.051 ± 0.006
(z = 8.2), and to 0.7672 once the probability floor described below was added.

- **Model.** Qwen3.8-27B in bf16, no training. Thinking mode at the chat template's default effort, greedy decoding.
- **Procedure.** The model reasons for up to 1,800 thinking tokens. Then the thinking block is closed, `Root:` is forced,
  and all classes are scored exactly behind the model's own text (§3.2, §3.4).
- **Reasoning length.** At 700 / 1,200 / 1,800 / 2,500 tokens the channel reached 0.693 / 0.712 / 0.716 / 0.727 on one
  fold. Its contribution to the blend saturates at about 1,800 tokens, which is the setting used.
- **Two prompt variants (draws).** Draw **g** uses the `zs27q` prompt, which includes the task block. Draw **n** uses
  the `zs27t` prompt, which contains no teacher content.
- **Gates.** Reasoning is spent where it can change the decision. The rules are reconstructed exactly by
  `tools/replay/gates.py`. Here *q* denotes the fast channel of the primary line (`zs27q`) and *t* the fast channel of
  the inductive line (`zs27t`).

  | Gate | Rule | Share |
  |---|---|---|
  | wide | P(`NONE`) < 0.5 in the fast channel | 68–69 % of rows |
  | narrow | wide, and the top class of the blend "9B + fast channel" has probability < 0.6 | about 47 % |
  | band | outside the wide gate, but P(`NONE`) < 0.95 | 154 training rows |

- **Composition.** Rows outside all gates take the fast channel's distribution.
  - `think18nb` (primary line): draw g on the wide gate, mean(g, n) on the narrow gate, n on the band, `zs27q` elsewhere.
  - `think18sb` (inductive line): draw n on that line's narrow gate and band, `zs27t` elsewhere.
- **Blend settings.** The channel is usually near-certain, so in a product it would act as a veto. A probability floor
  of ε = 0.01 prevents this (+0.005 ± 0.002). In the primary line its blend weight is multiplied by 1.4; this factor is
  frozen.
- **Draws.** Every generated text is a draw. Two greedy runs with the identical prompt agree on the top code in only 84 %
  of rows, because the server batches requests differently. Averaging draws reduces this variance.
- **Values.** `think18nb` 0.7672, `think18sb` 0.7516.
- **Compute.** RTX PRO 6000 (96 GB). Text generation takes about 2.1 s per row, scoring 1.6–1.7 s per row. One test draw
  takes about 2.5 h and costs about 6.3 USD.

### 4.4 27B protocol channel (`pr27xm`)

The channels above all know something about the question: same-question examples, topic or task blocks, or fine-tuning
on rows of the same questions. Where a working shows an error type that has never occurred for its question, this
knowledge pulls the prediction away from the correct code. The protocol channel sees only the working itself, under a
fixed marking protocol.

- **Prompt** (`src/vlm/proto.py`). An instruction with judging rules: read the handwriting line by line, solve the
  question, compare with the answer key, and decide whether the final answer is correct. The root is the major error
  from which the rest follows (usually the earliest one); minor slips mean `NONE`. The instruction states the base rate:
  roughly one third of workings are `NONE`. This matches the training frequency and the host's published description.
  It also asks for a topic-specific code wherever one fits, and for the five general codes only when no specific code
  fits. After the instruction come the full taxonomy, the question, the answer key and the photo. **No labelled example
  and no topic or task block** are included. The channel therefore depends on no fold, can be evaluated honestly on all
  training rows, and is strictly inductive.
- **Generation.** The model thinks for up to 2,500 tokens. A second request without further thinking then completes the
  answer from the model's own notes (median about 340 tokens). The result is a JSON object with three codes and their
  probabilities. It also holds fields such as the final-answer check, the student's final answer, a transcription and
  the findings.
- **Channel definition.** `pr27xm` is the average of two distributions. The first takes the stated codes with their
  stated probabilities and spreads the rest uniformly. The second is the exact ranking of all 74 classes behind the
  frozen notes of the first pass, with the answer text removed and `Root:` scored. The weight is 0.5 on each part, and
  the floor is 0.05.
- **Class skew.** The channel names the general codes about seven times too often. The class-marginal alignment (§8)
  corrects this. Without the alignment the channel adds only +0.0021 ± 0.0040 to the primary line.
- **Values.** The stated codes alone reach about 0.745; `pr27xm` reaches 0.7459. Adding the protocol channel improved the
  primary line by +0.0102 ± 0.0040 and the inductive line by +0.0118 ± 0.0046, positive in all 5 folds. Moving from the
  stated codes alone to `pr27xm` added +0.0042 ± 0.0019 (primary) and +0.0060 ± 0.0026 (inductive).
- **Compute.** 3.3–3.5 s per row on an RTX PRO 6000; the test side costs about 8 USD.

### 4.5 M3: fine-tuned 27B reasoner (`m3x3_full`)

M3 is our best single model. It is the 27B model fine-tuned to reason in the marking protocol, write the root finding and
then name the code.

| Item | Setting |
|---|---|
| Base and adapter | Qwen3.8-27B in bf16 + LoRA r 32, α 64, dropout 0.05 on layers 40–63; vision tower frozen; gradient checkpointing |
| Prompt | As the protocol channel: rules and taxonomy as system text; question, diagram (448 px), answer key, photo (1,024 px). No labelled example, no block, no teacher label: M3 is trained on gold labels only |
| Training targets | Three parts per training row. (1) The row's own protocol-channel notes as reasoning. They carry loss only where the protocol channel ranked the gold code first; otherwise they are inserted as a closed thinking block without loss (at most 2,500 tokens). (2) `Finding:` followed by the annotators' explanation of the root finding (`none` for `NONE`). (3) `Root: <code>`. Loss weights: code 1.0, finding 1.0, notes 0.5, scaffold 0.5; the code loss is summed. Targets are built by `tools/make_sft_targets.py`, and each fold model trains only on rows from the other folds |
| Optimisation | 2 epochs, learning rate 1e-4, gradient accumulation 8, seed 42. 2,400 steps per fold (1,200 rows × 2 epochs); the full fit takes 3,000 steps on all 1,500 rows. 1.75–1.80 s per step on an RTX PRO 6000 (96 GB), peak memory about 72 GB |
| Inference | Merge the adapter into the base model. vLLM writes notes, finding and root. Then all 74 classes are ranked exactly behind the canonical text `notes </think> Finding: … Root:` |
| Draws | d1: greedy, up to 3,200 tokens. d2: greedy, up to 5,200 tokens. d3: sampled (temperature 0.6, top-p 0.95, top-k 20, seed 3), up to 5,200 tokens. The channel is the mean of the three distributions, with floor 0.05 |
| Values | d1 0.7621, d2 0.767, d3 0.7701. Mean of two draws 0.7891, mean of three 0.8020: +0.0129 ± 0.0041 over two, positive in 5 of 5 folds |
| Effect in the recipe | One draw: primary +0.0048 ± 0.0039, inductive +0.0061 ± 0.0043. Third draw over two: +0.0053 ± 0.0027 and +0.0064 ± 0.0025. A fourth draw adds nothing: recipe MAP@3 with K = 1 / 2 / 3 / 4 draws is 0.8189 / 0.8221 / 0.8252 / 0.8247 |
| Profile | Strongest where the code has never been seen for the question (+0.041 over the reason-then-name channel). Weaker where it has been seen (−0.047) |
| Cost | About 7–8 h and 17–20 USD per test draw on an RTX PRO 6000. The ranking also runs on a Kaggle T4 in 4-bit at 26.5 s per row |

### 4.6 397B notes, named by M3 (`q397m3`; primary line only)

- **397B in the protocol.** `Qwen3.5-397B-A17B` has 397 B parameters, 17 B of them active. It runs without training,
  with the same marking protocol as the 27B protocol channel and greedy decoding (2,500 thinking tokens plus at most
  1,500 completion tokens). The competition allows models up to 550 B parameters.
- **Training side.** Produced through the Tinker sampling API with the same open weights (images sent as JPEG, quality
  75). 1,496 of 1,500 answers are valid.
- **Test side.** Produced on our own hardware: FP8 weights (`Qwen/Qwen3.5-397B-A17B-FP8` at revision
  `ea5b4f81096f3901c91dea97f81324302495781d`), vLLM 0.31, 4×H200 with tensor parallelism 4. 1.49 s per row; 3,492 of
  3,498 answers are valid.
- **Parity check.** On 64 training rows, the API path and our own path produced identical input tokens in 64 of 64 rows,
  and the same top code in 78 % of rows.
- **Naming by M3.** M3 scores all 74 classes behind the 397B's first-pass reasoning notes, with the answer text removed.
  On the OOF side the fold adapters score behind the API-generated notes. On the test side the full-fit adapter scores
  behind the notes from our own run. M3 calibrates the 397B's candidates: the channel's LogLoss is 2.63, against 4.88
  for the 397B's own stated answers. MAP@3 is about the same (0.7442 against about 0.744).
- **Contribution.** As a sixth channel next to M3 with three draws, the 397B's own answers added +0.0040 ± 0.0029. In
  the earlier recipe with two M3 draws they had added +0.0082 ± 0.0029, so part of their value overlaps with M3.
  Replacing them with `q397m3` was not robust on its own across the three measurement forms: per fold +0.0034 ± 0.0028,
  without the last step −0.0033, as one set −0.0014. Together with the arithmetic share and the second stage, `q397m3`
  is ahead in 8 of 9 measured comparisons (as one set +0.0050 ± 0.0028).
- **Cost.** About 3.4 s per row for the M3 naming step; about 5.4 USD for the OOF side and about 10 USD for the test side.

## 5. Teacher pseudo-labels for the unlabelled pool

A closed commercial model labelled all 50,000 pool photos during development. The host explicitly allows this for
training and unlabelled rows. **No test row was ever sent to an API.** The frozen labels feed the per-question task block
of the primary line (§3.3) and nothing else.

| Item | Setting |
|---|---|
| Model and API | Claude Haiku 5.5 through the Anthropic Messages API in batch mode; reasoning effort `xhigh`; output limit 24,000 tokens |
| Prompt | Instruction and taxonomy **without annotator notes** (a 5,285-token header). Per working: grade, topic, question, diagram, answer key, photo. No labelled training row, no example, no task block |
| Output | Final answer correct (yes/no), line transcription, findings (step, code, severity), root code, top-3 codes, confidence |
| Coverage | All 50,000 pool workings: 10,489 + 39,511 requests gave 49,979 usable labels. Per question the median is 170 labels and 35 error rows; about 72 % of labels are `NONE` |
| Cost | 30.86 USD including a 284-row probe, about 0.0006 USD per working |
| Choice of setting | Measured on 120 training rows without notes: Haiku 5.5 at effort `high` 0.76, `xhigh` 0.81, `max` 0.79 (at seven times the cost); Sonnet 5.5 at `high` 0.76. With annotator notes in the prompt the score was 0.838 against 0.759 without. The notes act as hidden same-question examples, so the pool was labelled without them |
| Teacher on all 1,500 training rows | About 0.77 (measurement only), 0.643 on rows whose code was never seen for the question. This was below our blend of the time (0.785) |
| Value | The task block helped the zero-shot 27B by +0.008 ± 0.007 over all folds, by shifting probability mass: +0.080 where the code was never seen for the question, −0.042 where it was seen. After M3 joined, the teacher-derived share of the primary line was +0.004 ± 0.005 |
| Frozen files | `pool_a` (SHA-256 `44fd2e1b…`) and `pool_b` (`0f561932…`), kept privately |

Using the pool as *training data* did not help, so it is not part of the solution. A 9B student trained on the teacher
labels changed the blend by +0.0001 ± 0.0032 over 5 folds. M3 trained on teacher rationales as targets lost
−0.096 ± 0.017 as a channel: it copied the teacher's short style and lost its own long diagnosis.

## 6. Orientation network

The 9B scorer reads rotated photos poorly; the 27B channels do not. A small network therefore decides which photos are
turned upright before the 9B model reads them.

- **Model** (`src/orient.py`, `tools/orient_net.py`). Two self-supervised CNNs with 426,540 parameters each, working on
  64×64 ink maps of the photo. The labels are the rotations applied during training, so no manual annotation is needed.
- **Data.** Training photos plus a fixed sample of 10,000 pool photos (seed 20261010). Test maps are only predicted.
- **Recipe.** 12 epochs, 2 folds, 2 rounds; AdamW with a one-cycle schedule, label smoothing 0.05, random shifts of
  ±3 px. Confidently rotated templates (≥ 0.8) enter round 2 in their rotated form. Test-time augmentation averages over
  the 4 rotations in float64.
- **Use.** Photos with a non-upright prediction at confidence ≥ 0.8 are rotated before the 9B model reads them.
- **Effect.**
  - On the 124 confidently rotated training rows, the 9B scorer gains +0.108 ± 0.024 (z = 4.5). The channel goes from
    0.6406 to 0.6474, and the blend gains +0.001 to +0.002.
  - A wrong rotation costs about 0.11 on the affected row, which is why the confidence threshold is high.
  - The 27B channels gain nothing from rotation (reason-then-name: −0.027 ± 0.027).
- **Reproducibility.** The released network is seeded and its weights are stored (`orient_cnn.pt`, SHA-256
  `83cc1224…`); three training runs gave bit-identical weights.
  - The submissions made so far use the rotation list of an earlier, unseeded version of the network whose weights were
    not stored.
  - The released network changes that list on 24 of the 3,498 test photos. On the training side the change is worth
    −0.0003 ± 0.0011.
  - Before the final selection, all build paths switch to the list of the released network
    ([ROADMAP.md](ROADMAP.md)).

## 7. Blend

Each channel *k* provides a distribution *p_k* over the 76 classes (zero for the two classes without a training example).

1. **Probability floor.** *p′_k = (1 − ε_k)·p_k + ε_k·u*, where *u* is uniform over the classes with a training example.
   ε = 0.01 for the reason-then-name channel and 0.05 for the protocol channel, M3 and `q397m3`. The 9B and zero-shot
   scorers get no floor. Without floors, a near-certain wrong channel would veto the correct class in the product.
2. **Log-linear blend.** *z = Σ_k θ_k · log p′_k* and *P_LL = softmax(z)*. The weights θ ≥ 0 are not normalised (bounds
   [0, 4]), so their sum also acts as a temperature. They are fitted with L-BFGS-B on LogLoss over the OOF rows
   (`eda/a2_probe_stack.py`). In the primary line the weight of the reason-then-name channel is then multiplied by 1.4.
3. **Arithmetic share β.** *P = (1 − β)·P_LL + β · Σ_k (θ_k / Σθ) · p′_k*. In a pure product, one channel that is sure
   of a wrong class can override a correct minority vote; the arithmetic share keeps that vote alive.
   - Inductive line: β = 0.4 (+0.0038 ± 0.0023).
   - Primary line: first β = 0.2 (+0.0019 ± 0.0016). Since the second stage was added, β = 0.6, chosen as the minimum of
     LogLoss before the last step on the grid 0.1, 0.2, …, 0.8.

**Weights.** Deployed weights are fitted on all 1,500 OOF rows. In cross-validation the weights of each fold are fitted
on the other four folds.

| Channel | θ, primary line | θ, inductive line |
|---|---:|---:|
| 9B fine-tuned scorer (`v1r`) | 0.224 | 0.252 |
| 27B zero-shot scorer (`zs27q` / `zs27t`) | 0.079 | 0.046 |
| 27B reason-then-name (`think18nb` / `think18sb`) | 0.106 (after × 1.4) | 0.093 |
| 27B protocol channel (`pr27xm`) | 0.092 | 0.137 |
| M3 (`m3x3_full`) | 0.295 | 0.367 |
| 397B notes, named by M3 (`q397m3`) | 0.140 | not used |
| Share of Σθ on text-generating channels | 67.6 % | 66.7 % |

## 8. Class-marginal alignment

The split is stratified by root code, so the class frequencies of the test set should match those of the training set.
The host's answer about the test share of `NONE` was: "It should be similar". Blends of zero-shot channels, however, are
skewed towards some classes; the protocol channel, for example, overuses the general codes. The alignment corrects this
skew as the last step.

**Transductive alignment (primary line).**
- Class weights *w* are fitted iteratively so that the mean predicted mass over **all test rows** equals the training
  frequency. This is done for the 53 classes with at least 3 training rows; all other classes keep weight 1
  (`eda/c21_margin_align.py`).
- The weights are applied with strength 0.75 (*P·w^0.75*, then renormalised).
- The step uses only the set of predictions: no label and no information about individual rows. It is valid only on the
  complete test set, so the primary line must always be assembled from predictions for all 3,498 rows.
- When introduced, it gave +0.005 to +0.008 in cross-validation (z = 2.7–3.1). On the public leaderboard, two otherwise
  identical submissions differed by +0.0093.

**Inductive alignment (inductive line).**
- Same computation, but the class weights are fitted on the blend's OOF predictions for the training rows.
- The weights are fixed before inference and applied to each test row independently.
- Gain when introduced: +0.0109 ± 0.0031 (a control with shuffled class weights gave −0.0045).
- For the earlier blend without the protocol channel, the inductive version gave about zero. Its later gain comes from
  correcting the systematic class skew of the protocol channel, which is the same on training and test rows.

## 9. Second stage: conditional-logit reranker

After the alignment, a small linear model reorders the top five candidates of each row (`src/stage2.py`, fitted by
`eda/c118_stage2.py`). It is used in the primary line only.

- **Model.** A conditional logit (a softmax over the 5 candidates of a row) with one shared weight per feature and no
  class identity. Features are standardised; L2 regularisation uses a fixed λ = 0.1. The model is fitted with L-BFGS-B
  on all 1,500 OOF rows in deployment form.
- **Features.** 43 in the *main* form and 40 in the *fallback* form:
  - the base form after alignment (log p, rank, gap to the top, p);
  - the blend before alignment (log p, rank);
  - per channel log p and rank, except the 397B channels;
  - the number of channels with the candidate at the top and among the top 3;
  - per M3 draw the rank, the number of draws with the candidate at the top, and the spread of log p over the draws;
  - question and topic knowledge from reference rows (count, rows, share, novelty) and a global prior;
  - flags for "general code" and `NONE`;
  - row features: `NONE` mass, entropy, maximum, disagreement between draws and between channels, grade;
  - main form only: mentions of the candidate in the frozen M3 notes (count 0–3, last mention, note length).

  The 9 row-constant features get exactly zero weight in a conditional logit, so 34 (main) or 32 (fallback) features
  are effective.
- **Leakage control.** Knowledge features for a training row come only from the other folds; for test rows they come
  from all 1,500 training rows.
- **Abstention gate.** An inner cross-validation over blocks checks that the reranker beats the base order; otherwise it
  abstains. It did not abstain (inner 0.8351 against base 0.8286).
- **Order on the test set.** Blend → alignment over all 3,498 rows as one set → reranker → `REL-01` rule (§10).
- **Test-side guards.** Only aggregates are used; see [VALIDATION.md](VALIDATION.md#6-checks-before-a-submission).
  - (a) A notes guard compares 13 statistics of the frozen M3 test notes with the three training draws. If it fails, the
    fallback form, which does not use the notes, is applied automatically.
  - (b) The share of candidates new for their question is reported.
  - (c) The rate of changed top codes must lie within ½–2× the OOF rate (main form 4.3 %, fallback form 3.5 %);
    otherwise the build aborts.
- **Main versus fallback form.**
  - The file submitted on 2026-10-31 used the fallback form. Under the original band rule, one of the 13 note statistics
    lay just outside its band: the share of identical notes between draws d1 and d2 was 0.046 against a band ending at
    0.0396 (z = 1.74).
  - That band was narrower than the sampling noise of the training value itself. The corrected rule therefore also
    accepts share statistics that pass a two-sample binomial test with |z| ≤ 3. The rule can only widen the band.
  - Under the corrected rule all 13 statistics pass (largest |z| 1.60), and the main form is used.
- **Effect** (main form with β = 0.6, against the primary line before these two changes).
  - Per fold +0.0049, without the last step +0.0117, as one set +0.0092.
  - The reranker alone, on top of β = 0.6, adds +0.0060 ± 0.0027 as one set (57 rows better, 41 worse).
  - It moves candidates from general to specific codes (specific +0.027, general −0.019, never-seen codes +0.022). It
    costs a little on correct `NONE` tops (−0.0018 on 9 rows).
- **Controls.** All 10 new meta-splits were positive. A permutation placebo over 20 seeds averaged about 0 (maximum
  +0.0019). A label placebo leads to abstention. Weight zero reproduces the file without reranker byte for byte.

For the inductive line, the reranker gained +0.0027 to +0.0038 per fold, but only 2–3 of 5 folds were positive. It is
therefore not part of the inductive line so far; a variant with a `NONE` guard is planned ([ROADMAP.md](ROADMAP.md)).

## 10. Blank photos

A black or empty photo shows no working. The only such training row carries the meta code `REL-01` (upload does not
match the question), while the channels tend to describe an error-free working. Photos detected as blank
(`tools/make_blank_list.py`, a property of the single image) get `REL-01` at rank 1; the order of the other candidates
is kept. This applies to 1 training row and 6 test photos and is worth about +0.001.

## 11. The two lines

Some techniques rest on an interpretation of the rules; the transductive alignment is one. Others are explicitly allowed
but depend on external services during development: the teacher pseudo-labels and the API-generated 397B outputs on
training rows. Each such technique goes into at most one of the two final submissions. One line maximises
cross-validated MAP@3. The other is strictly inductive and teacher-free, so that it holds under the strictest reading of
the rules ([DATA_AND_COMPLIANCE.md](DATA_AND_COMPLIANCE.md#6-the-two-lines-and-the-techniques-that-rest-on-interpretation)).

| Property | Primary line | Strictly inductive line |
|---|---|---|
| Channels | 9B scorer, `zs27q`, `think18nb`, protocol channel, M3 (3 draws), `q397m3` | 9B scorer, `zs27t`, `think18sb`, protocol channel, M3 (3 draws) |
| Teacher-derived input | Task block in `zs27q` and in reasoning draw g | none |
| API-derived training-side outputs | 397B outputs on training rows (Tinker API); the test side runs locally | none |
| 397B | yes (`q397m3`) | no |
| Arithmetic share β | 0.6 | 0.4 |
| Last step | Transductive alignment over the full test set | Inductive alignment with fixed weights |
| Second stage | yes | no (variant planned) |
| Test-side hardware | 9B on Kaggle T4 (offline); 27B steps on RTX PRO 6000; `zs27q` on A40 (4-bit); 397B notes on 4×H200 | 9B and `zs27t` on Kaggle T4 (offline); 27B steps on RTX PRO 6000 |
| CV MAP@3 | Per fold 0.8376 / as one set 0.8346 (main form); 0.8356 / 0.8338 (fallback form) | Per fold 0.8184 |

In cross-validation the primary line is about 0.017 ahead of the inductive line; the row-level correlation of the two
lines is 0.91.

## 12. Design principles

The following principles guided the development; the evidence behind them is in [RESULTS.md](RESULTS.md).

- **Full distributions, not labels.** Every channel scores every class. This makes channels blendable, calibratable and
  rerankable.
- **Diversity of knowledge.** The largest gains after the first strong channel came from channels that *know different
  things*: question-specific context against none, a fine-tuned model against zero-shot models, and a much larger model.
  Most attempts to combine the same channels better gave nothing. The two exceptions, the arithmetic share and the
  reranker, were each worth a few thousandths.
- **Generate once, score many times.** Reasoning texts are frozen with checksums. Exact scoring behind a frozen text is
  deterministic on given hardware, so channels can be rebuilt without repeating the generation.
- **Average draws.** Every generated text is a draw. Averaging two or three draws of the same model was worth more than
  any single longer or more elaborate generation.
- **Judge a component in its deployed role.** A new channel is evaluated inside the complete recipe, including the last
  step. The protocol channel, for example, looks useless without the alignment and is valuable with it.
- **Decide on cross-validation, not on the public leaderboard.** Decision rules are fixed before the result is known,
  comparisons are paired, and controls (copy null points, placebos, planted signals) guard against self-deception
  ([VALIDATION.md](VALIDATION.md)).
