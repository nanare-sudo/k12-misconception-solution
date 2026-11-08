# Method

*Status: 2026-11-08 (version 0.3). Numbers are out-of-fold (OOF) MAP@3 on the 1,500 labelled training rows unless stated
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
  The blend therefore sets these two classes to zero (the stated-codes part of the protocol channel can name them), and
  they are never submitted.
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
        │   m3px27xm  the same 3 codes + the exact ranking of M3 behind the same notes        [primary line, final candidate]
        ├── M3        Qwen3.8-27B + LoRA, reasons then names; mean of 3 draws
        └── q397m3    Qwen3.5-397B-A17B writes notes; M3 scores all codes behind them        [primary line only]
        │
        ▼  per channel: distribution over 76 classes → probability floor (mix with uniform)
   log-linear blend with arithmetic share β (weights fitted on LogLoss)
        ▼
   class-marginal alignment (final candidates of both lines: over the full test set as one set ·
                             strictly inductive fallback of the safety line: fixed weights)
        ▼
   second stage: conditional-logit reranker over the top-5 candidates, then the NONE guard
   (primary: features incl. M3-note mentions, general-code identifiers, row × candidate interactions and the
    minor-code × mechanical-match column · safety line: the interactions, M3-note mentions, the similarity of the notes
    to notes of real assignments in other questions, and regex-exact columns of the M3 and protocol notes)
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
| Training targets | Three parts per training row. (1) The row's own protocol-channel notes as reasoning. They carry loss only where the protocol channel ranked the gold code first; otherwise they are inserted as a closed thinking block without loss (at most 2,500 tokens). (2) `Finding:` followed by the annotators' explanation of the root finding (`none` for `NONE`). (3) `Root: <code>`. Loss weights: code 1.0, finding 1.0, notes 0.5, scaffold 0.5; the code loss is summed. The target files are delivered frozen (their builder is not part of this repository; see [TRAINING.md](TRAINING.md#10-m3-fine-tuned-27b-reasoner-m3x3_full)), and each fold model trains only on rows from the other folds |
| Optimisation | 2 epochs, learning rate 1e-4, gradient accumulation 8, seed 42. 2,400 steps per fold (1,200 rows × 2 epochs); the full fit takes 3,000 steps on all 1,500 rows. 1.75–1.80 s per step on an RTX PRO 6000 (96 GB), peak memory about 72 GB |
| Inference | Merge the adapter into the base model. vLLM writes notes, finding and root. Then all 74 classes are ranked exactly behind the canonical text `notes </think> Finding: … Root:` |
| Draws | d1: greedy, up to 3,200 tokens. d2: greedy, up to 5,200 tokens. d3: sampled (temperature 0.6, top-p 0.95, top-k 20, seed 3), up to 5,200 tokens. The channel is the mean of the three distributions, with floor 0.05 |
| Values | d1 0.7621, d2 0.767, d3 0.7701. Mean of two draws 0.7891, mean of three 0.8020: +0.0129 ± 0.0041 over two, positive in 5 of 5 folds |
| Effect in the recipe | One draw: primary +0.0048 ± 0.0039, inductive +0.0061 ± 0.0043. Third draw over two: +0.0053 ± 0.0027 and +0.0064 ± 0.0025. A fourth draw adds nothing: recipe MAP@3 of the primary line with K = 1 / 2 / 3 / 4 draws is 0.8189 / 0.8221 / 0.8252 / 0.8247, for K = 1 and 2 as the mean over all subsets of d1–d3 |
| Profile | Strongest where the code has never been seen for the question (+0.041 over the reason-then-name channel). Weaker where it has been seen (−0.047) |
| Cost | About 7–8 h and 17–20 USD per test draw on an RTX PRO 6000. The ranking also runs on a Kaggle T4 in 4-bit at 26.5 s per row |

### 4.6 397B notes, named by M3 (`q397m3`; primary line only)

- **397B in the protocol.** `Qwen3.5-397B-A17B` has 397 B parameters, 17 B of them active. It runs without training,
  with the same marking protocol as the 27B protocol channel and greedy decoding (2,500 thinking tokens plus at most
  1,500 completion tokens). The competition allows models up to 550 B parameters.
- **Training side.** Produced through the Tinker sampling API with the same open model (images sent as JPEG, quality
  75); the test side uses its FP8 release. 1,496 of 1,500 answers are valid.
- **Test side.** Produced on our own hardware: FP8 weights (`Qwen/Qwen3.5-397B-A17B-FP8` at revision
  `ea5b4f81096f3901c91dea97f81324302495781d`), vLLM 0.31, 4×H200 with tensor parallelism 4. 1.49 s per row; 3,492 of
  3,498 answers are valid.
- **Parity check.** On 64 training rows, the API path and our own path produced identical input token counts in 64 of 64 rows,
  and the same top code in 78 % of rows.
- **Naming by M3.** M3 scores all 74 classes behind the 397B's first-pass reasoning notes, with the answer text removed.
  On the OOF side the fold adapters score behind the API-generated notes. On the test side the full-fit adapter scores
  behind the notes from our own run. M3 calibrates the 397B's candidates: the channel's LogLoss is 2.63, against 4.88
  for the 397B's own stated answers. MAP@3 is about the same (0.7442 against 0.7432).
- **Contribution.** As a sixth channel next to M3 with three draws, the 397B's own answers added +0.0040 ± 0.0029. In
  the earlier recipe with two M3 draws they had added +0.0082 ± 0.0029, so part of their value overlaps with M3.
  Replacing them with `q397m3` was not robust on its own across the three measurement forms: per fold +0.0034 ± 0.0028,
  without the last step −0.0033, as one set −0.0014. Together with the arithmetic share and the second stage, `q397m3`
  is ahead of the 397B answers in all three forms, by small and uncertain margins (per fold +0.0021, without the last
  step +0.0027, as one set +0.0050, each ± 0.0028); a cross-check over six base pairs gave +0.0011 to +0.0043 as one
  set.
- **Cost.** About 3.4 s per row for the M3 naming step; about 5.4 USD for the OOF side and about 10 USD for the test side.

### 4.7 M3 behind the protocol notes (`m3px27`, mean channel `m3px27xm`; final primary candidate)

- **Idea.** The mean channel `pr27xm` averages the codes named by the protocol channel with the exact ranking of the
  untrained 27B behind the frozen first-pass notes of the same answers. `m3px27` takes this ranking from M3 instead: the
  fine-tuned model reads the notes of the protocol channel as its own reasoning, writes the finding itself and names the
  code, as `q397m3` does behind the 397B notes (§4.6).
- **Notes.** `tools/make_m3px27_notes.py` converts the frozen first-pass notes (`data/derived/proto_notes/train_pr27.jsonl`,
  `data/derived/proto_notes_test/test_pr27.jsonl`) into the format of the frozen M3 notes (`data/derived/sft_m3px27/`,
  `sft_m3px27te/`): no finding is given, and the notes are character-identical with the input of the untrained ranking.
- **Runs.** Training side `m3px27_oof_rp`: per fold k the M3 adapter that has not seen fold k ranks all 74 classes behind the
  note of the row (five shards). Test side `m3px27te_rp`: the adapter of the full fit. The ranking settings are those of the
  M3 draws.
- **Channel.** `m3px27xm` = ½ (named codes of `pr27` + ranking of `m3px27`), floor 0.05, in place of `pr27xm`. The blend
  weights are refitted on LogLoss (β 0.6 as before; file `stack_primary_m3px27.json`), and the whole second-stage chain of
  the final primary form is refitted with the swapped channel (`eda/c133_primary_m3px27.py`).
- **Values.** `m3px27` alone 0.7453, against 0.7318 for the untrained ranking (+0.0136 ± 0.0063); `m3px27xm` 0.7551 against
  0.7459 for `pr27xm` (+0.0092 ± 0.0042). The two rankings agree on the top code in 85.3 % of the training rows.
- **Rule** (pre-registered, expected-value rule with controls of the same role): deploy only if ½(F+M) on the original
  meta-split lies above max(0, controls) and the mean over 20 new meta-splits is above 0 and above the mean of the
  controls. Controls: a copy of the untrained ranking (0 by construction), the mean of the untrained ranking and a fourth
  M3 draw, the fourth M3 draw, and a copy of the M3 channel `m3x3_full`; on the original meta-split they must meet their
  values fixed in advance to four decimals (positive control of the measurement). Result: ½(F+M) +0.00006 (per fold
  +0.0017, as one set −0.0016); over 20 new meta-splits +0.00013 ± 0.00018, against a control mean of −0.00045: deploy.
  Without the 397B channel (reported only) ½(F+M) +0.0019 and +0.0010 ± 0.0002 over 20 new meta-splits: most of the value
  of the change lies in the case that the 397B path cannot be repeated (§11).
- **Deployment form of the training rows.** In the nested evaluation the arithmetic share β is selected per fold; with
  `m3px27xm` fold 1 selects 0.7 instead of 0.6. The deployment features of the training rows therefore use the same cross
  with β per fold (`build_stage2` with a given deployment form); with 0.6 in every fold this form equals the earlier one bit
  for bit, which the copy null point checks. A fixed β of 0.6 would give another top code in 6 test rows.
- **CV** of the final primary form with `m3px27xm`: per fold 0.84256, without the last step 0.83456, as one set 0.83822 (with
  `pr27xm` 0.84089 / 0.83433 / 0.83978).
- **Copy null point.** With `m3px27` replaced by the untrained ranking, the blend file, the 15 row vectors of the chain, the
  five deployment files (up to provenance and channel names) and the test side (CSV and test matrix of the final form with
  `pr27xm`) are reproduced exactly.
- **Test side.** 3,498 of 3,498 rows ranked exactly; `NONE` at the top 0.346 (training rows 0.344; fault band 0.30–0.38);
  the same top code as the untrained test ranking in 82.3 % of the rows. Against the final form with `pr27xm` the top code
  changes in 52 test rows (1.5 %).
- **Cost.** About 4.7 USD for the training side and about 12 USD for the test side on an RTX PRO 6000.

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
| Choice of setting | Measured on 120 training rows without notes: Haiku 5.5 at effort `high` 0.76, `xhigh` 0.81, `max` 0.79 (at seven times the cost); Sonnet 5.5 at `high` 0.76. With annotator notes in the prompt Sonnet 5.5 scored 0.838 against 0.759 without. The notes act as hidden same-question examples, so the pool was labelled without them |
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
- **Use.** Photos with a non-upright prediction at confidence ≥ 0.5 are turned upright and read again by the 9B model
  (list `c50`); the upright reading replaces the plain one only on rows at confidence ≥ 0.8 (list `c80`).
- **Effect.**
  - On the 124 confidently rotated training rows, the 9B scorer gains +0.108 ± 0.024 (z = 4.5). The channel goes from
    0.6406 to 0.6474, and the blend gains +0.001 to +0.002.
  - A wrong rotation costs about 0.11 on the affected row, which is why the confidence threshold is high.
  - The 27B channels gain nothing from rotation (reason-then-name: −0.027 ± 0.027).
- **Reproducibility.** The released network is seeded and its weights are stored (`orient_cnn.pt`, SHA-256
  `83cc1224…`); three training runs gave bit-identical weights, and inference from the weights or directly from the
  images gives the same lists.

### 6.1 Released rotation list

The earlier submissions used the rotation list of a first, unseeded training run of the same network whose weights
were not stored. That list cannot be recomputed, so a file built from it cannot be regenerated from the delivered
package. The final candidates and their fallbacks therefore use the list of the released network.

- **Difference.** The released list marks 344 test rows instead of 350: 335 rows are shared with the same rotation, 15
  appear only in the old list and 9 only in the new one, so 24 of the 3,498 test photos are affected.
- **No new GPU computation.** All 344 rows had already been read upright by the existing rotation run with the same
  rotation. Switching only changes the row filter of the `v1r` source.
- **Effect on the files.** Against the files built with the first-version list, no top code changed for the inductive
  line and for the main form of the reranker, one changed for the fallback form; the ordered top three changed on 12 to
  19 rows (at most 0.54 %). The largest possible loss from these rows is 0.0012–0.0018; the expected effect is
  ±0.0005.
- **Training side.** The OOF matrix of `v1r` and the blend weights fitted on it stay on the first-version training list
  (124 rows; the released network marks 132). With the released list the channel scores 0.6471 instead of 0.6474 on the
  training rows (−0.0003 ± 0.0011), so the training side was not rebuilt.
- **Safeguards.** The build scripts select the list with `ORIENT=repro` (default) or `ORIENT=legacy` (only to rebuild
  the historical files). With `repro` they reject the first-version list by path and by checksum, check coverage,
  rotation and provenance before the assembly, and verify the `v1r` source in the written record afterwards
  (`tools/orient_list.py`, [VALIDATION.md](VALIDATION.md#6-checks-before-a-submission)).

## 7. Blend

Each channel *k* provides a distribution *p_k* over the 76 classes; the blend sets the two classes without a training
example to zero.

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

**Transductive alignment (primary line; since 2026-11-06 also the final candidate of the safety line, §9.9).**
- Class weights *w* are fitted iteratively so that the mean predicted mass over **all test rows** equals the training
  frequency. This is done for the 53 classes with at least 3 training rows; all other classes keep weight 1
  (`eda/c21_margin_align.py`).
- The weights are applied with strength 0.75 (*P·w^0.75*, then renormalised).
- The step uses only the set of predictions: no label and no information about individual rows. It is valid only on the
  complete test set, so the lines that use it must always be assembled from predictions for all 3,498 rows.
- When introduced, it gave +0.005 to +0.008 in cross-validation (z = 2.7–3.1). On the public leaderboard, two otherwise
  identical submissions differed by +0.0093.

**Inductive alignment (strictly inductive line, the fallback of the safety slot).**
- Same computation, but the class weights are fitted on the blend's OOF predictions for the training rows.
- The weights are fixed before inference and applied to each test row independently.
- Gain when introduced: +0.0109 ± 0.0031 (a control with shuffled class weights gave −0.0045).
- For the earlier blend without the protocol channel, the inductive version gave about zero. Its later gain comes from
  correcting the systematic class skew of the protocol channel, which is the same on training and test rows.
- On the test side the inductive weights miss the class masses: the mean test mass per class lies at L1 0.087 from the
  target, against about 0.02 for the OOF rows. This is why the final candidate of the safety line uses the alignment over
  the test set (§9.9).

## 9. Second stage: conditional-logit reranker

After the alignment, a small linear model reorders the top five candidates of each row (`src/stage2.py`). It is fitted
by `eda/c118_stage2.py` in the primary line (extended by `eda/c121_stage2_i1.py` and `eda/c124_stage2_mm.py`, and refitted
with the channel `m3px27xm` by `eda/c133_primary_m3px27.py`), by `eda/c120_s036_safe_stage2.py` in the strictly inductive line
and by `eda/c123_safety_set.py`, `eda/c129_safety_nd.py` and `eda/c130_safety_ndr.py` in the safety line with alignment over
the test set. The
blend has one weight per channel; the reranker can weigh the same information per candidate, for example how many
channels and M3 draws agree on a code and whether the code was seen for the question.

### 9.1 Model and features

- **Model.** A conditional logit (a softmax over the 5 candidates of a row) with one shared weight per feature and no
  class identity. Features are standardised; L2 regularisation uses a fixed λ = 0.1. The model is fitted with L-BFGS-B
  on all 1,500 OOF rows in the deployment form of its line.
- **Feature sets.**

  | Set | Features | Content | Used in |
  |---|---:|---|---|
  | `fallback` | 40 | base form after alignment (log p, rank, gap to the top, p); blend before alignment (log p, rank); per channel log p and rank, except the 397B channel; number of channels with the candidate at the top and among the top 3; per M3 draw the rank, the number of draws with the candidate at the top and the spread of log p over the draws; question and topic knowledge from reference rows (count, rows, share, novelty) and a global prior; flags "general code" and `NONE`; row features (`NONE` mass, entropy, maximum, disagreement between draws and between channels, grade) | primary line when the note guard fails; strictly inductive line |
  | `main` | 43 | `fallback` plus mentions of the candidate in the frozen M3 notes (count 0–3, last mention, note length) | primary line, main form |
  | `main_guard` | 48 | `main` plus identifiers of the five general codes (§9.4) | primary line, guard form |
  | `main_guard_i1` | 63 | `main_guard` plus 15 row × candidate interactions (§9.7) | primary line, I1 form |
  | `main_guard_i1_mm` | 64 | `main_guard_i1` plus the minor-code × mechanical-match column (§9.8) | primary line, final candidate |
  | `fallback_i1` | 55 | `fallback` plus the same 15 interactions | safety line with alignment over the test set (§9.9) |
  | `fallback_i1_nd` | 58 | `fallback_i1` plus two M3-note mentions and the example similarity (§9.10) | safety line, fallback of the final candidate |
  | `fallback_i1_ndr` | 64 | `fallback_i1_nd` plus six regex-exact columns of the M3 and protocol notes (§9.11) | safety line, final candidate |

  Row-constant features (9 in `main` and the sets built on it, 8 in `fallback` and `fallback_i1`) get exactly zero weight in
  a conditional logit, so 34, 39, 54, 55, 32 and 47 features are effective.
- **Leakage control.** Knowledge features for a training row come only from the other folds; for test rows they come
  from all 1,500 training rows.
- **Abstention gate.** An inner cross-validation over blocks checks that the reranker beats the base order; otherwise it
  abstains. It did not abstain in any deployed form: inner 0.8351 (main form) and 0.8347 (fallback form) against base
  0.8286, 0.8381 against 0.8286 for the main form with the NONE guard, and 0.8247 against 0.8184 in the inductive line;
  the values of the forms added in versions 0.2 and 0.3 are in [TRAINING.md](TRAINING.md#13-second-stages).
- **Order on the test set.** Blend → alignment → reranker → NONE guard (§9.3) → `REL-01` rule (§10). The reranker only
  reorders the first five candidates; the probability mass outside them is unchanged.

### 9.2 Test-side guards

Only aggregates are used; see [VALIDATION.md](VALIDATION.md#6-checks-before-a-submission).
- **(a) Note guard** (primary line, main forms). 13 statistics of the frozen M3 test notes are compared with the three
  training draws: mean number of codes mentioned, share of rows mentioning the top candidate, distribution of mention
  counts and share of word-identical notes per pair of draws. If any statistic fails, the fallback form, which does not
  use the notes, is applied automatically; there is no switch to override the guard.
  - *Band rule.* A statistic passes if it lies within the range of the training draws ± 10 % relative.
  - *Widened rule.* For share statistics the band was narrower than the sampling noise of the training value itself.
    The share of word-identical notes between the two greedy draws d1 and d2 is 0.036 on the 1,500 training rows, with a
    binomial standard error of about 0.005, while the band allowed only ±0.0036; the per-fold training values range from
    0.027 to 0.047. Under the widened rule a share statistic also passes if a two-sided two-sample binomial test against
    its own training value gives |z| ≤ 3, roughly a Bonferroni bound for 13 statistics at α = 0.05. The codes-per-row
    statistic keeps the band rule. The rule can only widen the band, and the positive controls still trigger the guard:
    swapped or copied note files, a codes-per-row value of 1.7 and a top-mention share of 0.40.
- **(b)** The share of candidates new for their question is reported.
- **(c)** The rate of changed top codes must lie within ½–2× the OOF rate; otherwise the build aborts. OOF rates: main
  form 4.3 %, fallback form 3.5 %, main form with NONE guard 3.7 %, inductive line 2.7 %.

### 9.3 NONE guard

- **Rule.** If the aligned blend has `NONE` at the top of a row and the reranker would put another class first, the row
  keeps its base order. The guard is a rule, not a feature. In the abstention gate the new order is evaluated after the
  guard, as it is applied.
- **Why.** In practice `NONE` answers the question "is the final answer correct?": the two agree in 96.7 % of training
  rows. When the guard was introduced, no reranker feature measured the correctness of the final answer, so the reranker
  could only judge `NONE` from agreement and rank features (the final form of version 0.2 adds such a feature only for
  minor-severity candidates, §9.8). On rows with a correct `NONE` top it lost 0.0018 (9 rows), and in a sample of the error
  analysis rows with gold `NONE` were the only group in which it lost more rows than it won (2 won, 6 lost). The guard
  leaves the `NONE` decision to the blend, whose channels read the final answer.
- **Scope.** Few rows are affected: as one set 7 of the 1,500 OOF rows and 8 of the 3,498 test rows in the primary line,
  3 OOF rows and 6 test rows in the inductive line.
- **Effect** in the primary line (mean of per fold and as one set): +0.0010, positive in 80 of 80 new meta-splits. As one
  set the gain lies almost entirely in rows with gold `NONE` whose `NONE` top the reranker had replaced.

### 9.4 General-code identifiers

- **Feature.** Five indicators, one per general code (`ANS-01`, `CAL-01`, `WOP-01`, `FRM-01`, `OTH-00`): 1 if the
  candidate is that code, otherwise 0. They are appended to the 43 main features (`main_guard`).
- **Why.** The reranker has no class identity. All five general codes have exactly 57 training rows, so the frequency
  features and the "general code" flag cannot tell them apart. The annotators, however, assign them very differently:
  comparable workings (same wrong final answer or same marked error) receive the same code in 0.83 of pairs for
  specific codes, but in 0.53 for general codes, from 0.74 for `ANS-01` down to 0.25 for `FRM-01` (small numbers).
  The identifiers let the reranker learn one offset per general code. The fitted standardised weights are negative for
  `ANS-01`, `CAL-01` and `WOP-01`, positive for `FRM-01` and about zero for `OTH-00`.
- **Effect.** On top of the NONE guard +0.0004, positive in 54 of 80 new meta-splits. Against random code identifiers
  that also receive the guard, the gain is not above the placebo 95 % quantile. The identifiers are a small positive
  expectation, not a robust effect; the robust part of the guard form is the NONE guard.

### 9.5 Forms in the primary line

| Form | Features | Note guard | NONE guard | CV per fold / as one set |
|---|---|---|---|---|
| Fallback form | `fallback` (40) | — | no | 0.8356 / 0.8338 |
| Main form | `main` (43) | widened rule; fallback form if it fails | no | 0.8376 / 0.8346 |
| Guard form: main form with NONE guard and general-code identifiers | `main_guard` (48) | widened rule; fallback form if it fails | yes | 0.8389 / 0.8377 |
| I1 form: guard form with row × candidate interactions (§9.7) | `main_guard_i1` (63) | widened rule; fallback form if it fails | yes | 0.8408 / 0.8387 |
| Final form: I1 form with the minor-code × mechanical-match column (§9.8) | `main_guard_i1_mm` (64) | widened rule; fallback form if it fails | yes | 0.8409 / 0.8398 |
| **Final form with `m3px27xm`**: the same chain refitted with the M3 ranking behind the protocol notes in the mean channel (§4.7) | `main_guard_i1_mm` (64) | widened rule; fallback form if it fails | yes | 0.8426 / 0.8382 |

- **History of the note guard.** The file submitted on 2026-10-31 used the fallback form: under the band rule one of the
  13 statistics, the share of word-identical notes between d1 and d2, was 0.046 on the test rows against a band ending
  at 0.0396. The widened rule was written after this guard had triggered. It rests only on training aggregates and
  sampling theory; no public score of the main form existed at that time. Under the widened rule all 13 statistics pass
  (largest |z| = 1.60, against the pair's own training value).
- **Effect of the reranker** (main form with β = 0.6, against the primary line with β = 0.2 and without reranker): per fold
  +0.0049, without the last step +0.0117, as one set +0.0092. The reranker alone, on top of β = 0.6, adds
  +0.0060 ± 0.0027 as one set (57 rows better, 41 worse). It moves candidates from general to specific codes (specific
  +0.027, general −0.019, never-seen codes +0.022).
- **Controls.** All 10 new meta-splits were positive for the main form. A permutation placebo over 20 seeds averaged
  about 0 (maximum +0.0019). A label placebo leads to abstention. Weight zero reproduces the file without reranker byte
  for byte. The checks of the NONE guard and the identifiers are in
  [VALIDATION.md](VALIDATION.md#54-validation-of-the-second-stage-components).

### 9.6 Second stage in the strictly inductive line

- **Form.** The same conditional logit (K = 5, λ = 0.1) applied after the *inductive* alignment, with the `fallback`
  feature set and the NONE guard. The line has no M3 note features (and therefore no note guard) and no 397B channel.
  Knowledge features come from the other folds for training rows and from all training rows for test rows.
- **Fit.** The line's alignment uses fixed weights per row, so the reranker is fitted in the per-fold form: for each
  outer fold *k*, every training block *j ≠ k* is scored with blend weights, inductive class weights and knowledge
  features taken from the folds other than *j* and *k*, so that no label of fold *k* enters. The deployment file is
  fitted on all 1,500 OOF rows in the same form. It names `align_mode: inductive`, so the assembly requires the inductive
  alignment and refuses the transductive one.
- **Inductive by construction.** All parameters are fixed before inference, and each test row is processed on its own.
  Test-set aggregates enter only the guards, which can abort a build but change no order. No teacher-derived input and
  no transductive step is involved.
- **Effect.** Per fold 0.8184 → 0.8233 (+0.0049 ± 0.0022). The largest positive weights fall on the log-probabilities
  of M3, of the base form, of the blend before the last step and of the reason-then-name channel, and on agreement
  between channels and between draws; the largest negative ones on the gap to the top, the ranks (of the base form and
  of draw d3) and the "general code" flag. The stage thus gives M3 and the reason-then-name channel more say, rewards
  agreement and moves mass from general to specific codes (gold specific +0.0101 ± 0.0037, gold general
  −0.0064 ± 0.0067). Almost all of the gain lies in the second sub-collection of the training data (+0.0063 ± 0.0025
  against +0.0005 ± 0.0046 in the first).
- **Dependence on the M3 draws.** The reranker reads the ranks of each M3 draw. Replacing one of the three draws by a
  fourth gives +0.0007 to +0.0049 over four draw sets; the deployed set is the most favourable one, and the only set with
  the same composition (two greedy draws and one sampled draw) gives +0.0007 ± 0.0020. On the test side, whose M3 texts
  are independent draws, the expected gain is therefore +0.002 to +0.003, not +0.0049. The decision rule and its
  sensitivity are in [VALIDATION.md](VALIDATION.md#55-risk-rule-of-the-strictly-inductive-line).

### 9.7 Row × candidate interactions (`main_guard_i1`)

- **Why.** A conditional logit compares the candidates of one row, so a feature that is constant within a row (entropy,
  top probability, disagreement, number of reference rows of the question) gets exactly zero weight: in `main_guard` nine
  of the 48 features had no effect. Such row properties can only matter through the candidates, for example "when the
  blend is uncertain, trust the base log-probability less" or "when the question has many reference rows, trust a general
  code less".
- **Feature.** 15 products of raw values, appended at the end of `main_guard` and standardised like all features: each of
  the five row features `row_ent`, `row_max`, `row_d_ntop`, `row_ch_ntop`, `task_rows` times each of the three candidate
  features `f_logp`, `is_general`, `is_none` (`i1_interactions` in `src/stage2.py`). The set was fixed before its evaluation. The
  first 48 features are those of the guard form, bit for bit; fallback model, note guard and NONE guard are unchanged
  (`eda/c121_stage2_i1.py`).
- **Effect** against the guard form, on the original meta-split: per fold +0.0019 ± 0.0011, as one set +0.0010 ± 0.0010,
  mean of the two forms ½(F+M) +0.0014, against a placebo mean of +0.0004 (15 placebos with the row factors permuted
  across rows within each fold; 1 of 15 at least as large). Over 20 new meta-splits ½(F+M) +0.0011 ± 0.0002, positive in
  90 %. Without the last step the form is −0.0020 (reported only). CV per fold 0.8408, as one set 0.8387.
- **Copy null points.** With the I1 columns set to zero the nested OOF row values of the guard form are reproduced bit for
  bit, and an I1 model with the 48 weights of the guard form and I1 weights 0 rebuilds the guard-form file and its test
  matrix byte for byte.

### 9.8 Minor-code × mechanical-match column (`main_guard_i1_mm`, final primary candidate)

- **Feature.** `minor_x_mech27` = 1[the candidate has the default severity "minor" in the taxonomy] × 1[the final answer
  read by the protocol channel matches the answer key mechanically]. The row part comes from the frozen answers of the
  protocol channel (field `student_final_answer`, last readable JSON answer per row; training side from the two passes of
  the training run, test side from the frozen test answers) and is compared with the answer key by a number parser
  without `eval` (`src/answer_match.py`: an AST evaluator for numbers, + − · / ** and `sqrt`, Indonesian number format,
  LaTeX fractions, units and words dropped; a match is an equal numeric value within 1e-6 or equal normalised text). It
  is appended at the end of `main_guard_i1`.
- **Why.** By the annotation protocol the root misconception is always a major finding, one that changes the solution
  path or the result. When the final answer is right, a minor slip is a plausible finding but hardly a root; the reranker
  had no way to see this combination. The weight is fitted, not fixed.
- **Effect** against the I1 form, on the original meta-split: per fold +0.0001 ± 0.0005, as one set +0.0011 ± 0.0006,
  ½(F+M) +0.0006 against a placebo mean of −0.0001 (15 placebos with the row values of `mech27` permuted within each
  fold; none as large). Over 20 new meta-splits +0.0005 ± 0.0001, positive in 19 of 20. CV per fold 0.8409, as one set
  0.8398. The read final answer matches the key mechanically in 27.3 % of the test rows (training rows 29.0 %); the
  column changes the top code of 6 of the 3,498 test rows against the I1 form.
- **Copy null point.** With the column weight set to zero and the 63 weights of the I1 form, the I1-form file and its test
  matrix are rebuilt byte for byte.

### 9.9 Second stage of the safety line with alignment over the test set

- **Why.** The inductive alignment of §8 brings the class masses of the training rows to their targets, but on the test
  side it misses them: the mean test mass per class lies at L1 0.087 from the target (0.112 before any alignment),
  against about 0.02 for the OOF rows. The alignment over the test set brings it to 0.028. Since 2026-11-06 the
  alignment over the whole test set is admitted in both final submissions (§11); the safety line keeps all its other
  restrictions.
- **Form.** The channels, floors and β of the strictly inductive line; margin alignment 0.75 for classes with at least 3
  training rows **as one set** (as in the primary line); then the conditional logit (K = 5, λ = 0.1) fitted anew in this
  form, with the NONE guard (`eda/c123_safety_set.py`). Variant a uses the `fallback` features (40); variant b adds the
  same 15 interactions as §9.7 (`fallback_i1`, 55). No notes, no teacher channel, no 397B.
- **Pre-registered rule.** (i) Copy null points exact; (ii) variant a counts if its CV as one set (with gate and guard)
  is not below the CV of the strictly inductive line with second stage in its own deployment form (per fold) minus 0.002,
  **and** the test-side class masses after the alignment lie at most L1 0.035 from the target; (iii) variant b instead of
  a only if ½(F+M) b − a > 0, above the placebo mean and above 0 in the mean over at least 10 new meta-splits.
- **Result.** (i) holds. (ii): the test-side class masses pass (L1 0.028), but the CV condition is missed by one sixth in
  9,000 (variant a as one set 7,391 against the threshold 7,392): the inner abstention gate abstains in one fold, which
  costs 22 sixths there, and it does not abstain when the deployment file is fitted on all 1,500 rows. Over 20 new
  meta-splits variant a lies +0.0007 ± 0.0002 above the strictly inductive line (16 of 20 positive, 20 of 20 above
  −0.002). (iii) selects variant b: +0.0007 ± 0.0001 over a in the mean of 20 new meta-splits; on the original meta-split
  variant b as one set with gate reaches 0.8217.
- **Override.** The entrant decided on 2026-11-06 to build and submit the candidate although the CV part of condition (ii)
  is missed. The decision is recorded in [`decisions/override_cv_condition_ii.json`](../decisions/override_cv_condition_ii.json),
  whose SHA-256 is pinned in `eda/c123_safety_set.py`. It overrides only the CV part of (ii): the test-mass part, (i),
  (iii), the copy null points, the guards and all checksums stay in force. Without the switch `OVERRIDE_CV_II` the build
  script ends with exit code 5 after the rule; with any other value it refuses to run.
- **Test side.** Variant b changes the top code of 70 of the 3,498 test rows against the strictly inductive line with
  second stage; 10 test rows are kept in their base order by the NONE guard; guard (c) 0.029 against the OOF share 0.026.
  The strictly inductive line with second stage stays the fallback of the safety slot under the strictest reading of the
  rules (§11).

### 9.10 Note mentions and example similarity in the safety line (`fallback_i1_nd`)

- **Features.** Three columns appended to `fallback_i1` (55, bit for bit): `m3_men` and `m3_last`, the mentions of the
  candidate in the frozen M3 notes d1–d3 as in the primary line, and `demo_sim`, the example similarity
  (`src/note_sim.py`): per candidate the mean of the three largest cosine similarities of the row's own M3 notes to the notes
  of training rows whose gold code is the candidate, never from the same question. Similarity: TF-IDF of word 1- and
  2-grams (sublinear, min_df 2), the last 4,000 characters of each note, averaged over d1–d3; vectoriser and IDF are fitted
  on the training notes only. The reference is nested: for a training row the training rows of the other folds, for a test
  row all 1,500 training rows. The training notes are delivered frozen (`data/derived/sft/m3_d{1,2,3}_train_frozen_notes.jsonl`;
  `eda/c129_safety_nd.py export-notes --verify` rebuilds them from the ranking runs).
- **Guards, abort only.** The widened note guard of the primary line (13 statistics of the M3 test notes), a similarity guard
  (mean top similarity of the test rows to the training rows against that of the training rows among each other; band ±10 %
  or |z| ≤ 3), guard (c) and a limit of L1 0.035 on the test-side class masses. A failing guard stops the build without a
  file; none switches to another form. Positive controls (notes copied from one draw, rows permuted, notes cut to 300
  characters) must trigger a guard.
- **Rule** (pre-registered): exact copy null points (the three columns at 0 reproduce `safety_final` bit for bit on the OOF
  rows; the weights of `safety_final` with 0 for the new columns reproduce its file byte for byte); ½(F+M) on the original
  meta-split > 0 and above the mean of 15 placebos (the three columns taken from another row of the same block); the mean
  over 20 new meta-splits > 0; the risk rule P(θ < −0.002) ≤ 0.05 with τ_d = 0
  ([VALIDATION.md](VALIDATION.md#55-risk-rule-of-the-strictly-inductive-line)).
- **Result.** ½(F+M) +0.0032 ± 0.0015 against `safety_final` (per fold +0.0033, positive in 4 of 5 folds; as one set
  +0.0031), placebo mean +0.0005 (maximum +0.0020); over 20 new meta-splits +0.0028 ± 0.0002, positive in 20 of 20 (placebo
  −0.0001); risk rule P = 0.0001 (posterior mean +0.0018; with τ_d 0.001: 0.0053). CV per fold 0.8311, as one set 0.8274;
  the gate does not abstain.
- **Test side.** Note guard 13 of 13; similarity guard 0.2176 against 0.2095, passed through the band (+3.9 %; z = +4.51 lies
  above 3: the test notes of the full-fit model are slightly more similar to the training notes than these are among each
  other); guard (c) 0.0374 (OOF 0.0367); class masses L1 0.0282. Another top code than `safety_final` in 70 test rows
  (2.0 %); the share of `NONE` at the top is unchanged (0.3548).

### 9.11 Regex-exact note columns (`fallback_i1_ndr`, final safety candidate)

- **Features.** Six columns appended to `fallback_i1_nd` (58, bit for bit), from the versioned extractor `src/note_regex.py`:
  from the M3 notes d1–d3 the mean log(1 + number of mentions), the number of draws in which the candidate is mentioned
  first, and the mean relative position of its last mention; from the frozen first-pass notes of the protocol channel (the
  same run as the channel `pr27xm`) whether the candidate is mentioned, whether it is the last code mentioned, and
  log(1 + number of mentions). Row-local and label-free. The protocol notes are delivered frozen and rebuilt from the raw
  answers by `eda/c130_safety_ndr.py export-notes --verify` (with `tools/export_proto_notes.py`).
- **Guards, abort only**: those of §9.10 plus a guard of the columns R: number of codes the protocol note mentions per row,
  share of rows in which it mentions the top of the base form, and the mean last position of the M3 mentions per draw (band
  ±10 % or |z| ≤ 3). Two further positive controls change only the protocol notes (shifted by one row; replaced by the M3
  notes d1); only the guard of the columns R detects them, as required.
- **Rule** (pre-registered): (1) in a full draw swap over the four sets of three draws from d1–d4 (d4: a fourth M3 draw on
  the training rows only) the gain over `safety_nd` is > 0 in the mean and in at least 3 of 4 sets; (2) ½(F+M) on the
  original meta-split > 0 and above the placebo mean (15 placebos); (3) the mean over 20 new meta-splits > 0; (4) the risk
  rule against `safety_nd` with τ_d = the spread of the four sets, P(θ < −0.002) ≤ 0.05 for the original observation and
  for the mean of the four sets.
- **Result.** (1) +0.00099 / +0.00026 / +0.00083 / +0.00015, mean +0.00056, 4 of 4 positive; (2) +0.00072 ± 0.00108 (per
  fold +0.00044, as one set +0.00100) against a placebo mean of −0.00012; (3) +0.00087 ± 0.00022 (85 % positive; placebo
  +0.00011); (4) P = 0.0007 and 0.0010 at τ_d = 0.00041. With a second draw of the protocol notes (sensitivity) the
  meta-splits give +0.00085 and P = 0.0006. CV per fold 0.8316, as one set 0.8284; the gate does not abstain.
- **Test side.** All guards held: note guard 13 of 13, similarity guard as in §9.10, guard of the columns R 5 of 5 within the
  band, guard (c) 0.0400 (OOF 0.0407), class masses L1 0.0282. Another top code than `safety_nd` in 38 test rows (1.1 %);
  the share of `NONE` at the top is unchanged.

## 10. Blank photos

A black or empty photo shows no working. The only such training row carries the meta code `REL-01` (upload does not
match the question), while the channels tend to describe an error-free working. Photos detected as blank (a property
of the single image; the list is delivered frozen, its script is not part of this repository) get `REL-01` at rank 1;
the order of the other candidates is kept. This applies to 1 training row and 6 test photos and is worth about +0.001.

## 11. The two lines

Some techniques rest on an interpretation of the rules; the transductive alignment is one. Others are explicitly allowed
but depend on external services during development: the teacher pseudo-labels and the API-generated 397B outputs on
training rows. Each technique of the second kind goes into at most one of the two final submissions. One line maximises
cross-validated MAP@3. The other, the **safety line**, is teacher-free: no input derived from the development-time
teacher model, no output of an external API and no 397B channel. Until 2026-11-05 it was also strictly inductive. Since
2026-11-06 the alignment over the whole test set is admitted in both final submissions: the host stated about adapting to
the test data shared with all participants that "We don't see any issue so far", and the inductive alignment missed the
test-side class masses (§9.9). The strictly inductive form of the safety line stays its fallback, so that one file holds
under the strictest reading of the rules
([DATA_AND_COMPLIANCE.md](DATA_AND_COMPLIANCE.md#6-the-two-lines-and-the-techniques-that-rest-on-interpretation)).

| Property | Primary line | Safety line |
|---|---|---|
| Channels | 9B scorer, `zs27q`, `think18nb`, protocol channel with the M3 ranking behind its notes (`m3px27xm`), M3 (3 draws), `q397m3` | 9B scorer, `zs27t`, `think18sb`, protocol channel (`pr27xm`), M3 (3 draws) |
| Teacher-derived input | Task block in `zs27q` and in reasoning draw g | none |
| API-derived training-side outputs | 397B outputs on training rows (Tinker API); the test side runs locally | none |
| 397B | yes (`q397m3`) | no |
| Arithmetic share β | 0.6 | 0.4 |
| Last step of the final candidate | Alignment over the full test set (as one set) | Alignment over the full test set (as one set) |
| Second stage of the final candidate | Reranker with M3-note features, general-code identifiers, row × candidate interactions and the minor-code × mechanical-match column, NONE guard; fallback features if the note guard fails | Reranker with the row × candidate interactions, M3-note mentions, example similarity and regex-exact columns of the M3 and protocol notes, NONE guard; a failing guard stops the build |
| Fallback | The same final form with `pr27xm` (version 0.2); then the I1 form | The form without the regex columns; then the form with interactions only; under the strictest reading the strictly inductive form: inductive alignment with fixed weights, second stage with the features without notes |
| Rotation list of `v1r` (test side) | Released orientation network | Released orientation network |
| Test-side hardware | 9B on Kaggle T4 (offline); 27B steps on RTX PRO 6000; `zs27q` on A40 (4-bit); 397B notes on 4×H200 | 9B and `zs27t` on Kaggle T4 (offline); 27B steps on RTX PRO 6000 |
| CV MAP@3 of the final candidate | Per fold 0.8426 / as one set 0.8382 | Per fold 0.8316 / as one set 0.8284 |
| CV MAP@3 of the fallback | Final form with `pr27xm` 0.8409 / 0.8398; I1 form 0.8408 / 0.8387 | Without the regex columns 0.8311 / 0.8274; interactions only 0.8217 (as one set, with gate); strictly inductive with second stage 0.8233 (per fold) |

The forms of the safety line are deployed, and therefore evaluated, differently: the forms with the alignment over the test
set as one set (variant b 0.8217 on the original meta-split; per fold, with the alignment inside each 300-row fold, 0.8278;
the note-feature forms were decided by the mean of both forms, §9.10 and §9.11), the strictly inductive form per fold
(0.8233). Paired over 20 new meta-splits, variant a as one set lies +0.0007 ± 0.0002 above the strictly inductive form,
variant b +0.0007 ± 0.0001 above variant a (§9.9), the note mentions and example similarity +0.0028 ± 0.0002 above variant b
and the regex columns +0.0009 ± 0.0002 above that. In cross-validation per fold the primary candidate is about 0.011 ahead of
the safety candidate and about 0.019 ahead of the strictly inductive form.
The row-level correlation of the two lines, measured between the reranker fallback form and the strictly inductive line
without second stage, is 0.91.

## 12. Design principles

The following principles guided the development; the evidence behind them is in [RESULTS.md](RESULTS.md).

- **Full distributions, not labels.** Every channel scores every class. This makes channels blendable, calibratable and
  rerankable.
- **Diversity of knowledge.** The largest gains after the first strong channel came from channels that *know different
  things*: question-specific context against none, a fine-tuned model against zero-shot models, and a much larger model.
  Most attempts to combine the same channels better gave nothing. The exceptions, the arithmetic share and the
  reranker with its NONE guard, were each worth a few thousandths.
- **Generate once, score many times.** Reasoning texts are frozen with checksums. Exact scoring behind a frozen text is
  deterministic on given hardware, so channels can be rebuilt without repeating the generation.
- **Average draws.** Every generated text is a draw. Averaging two or three draws of the same model was worth more than
  any single longer or more elaborate generation.
- **Judge a component in its deployed role.** A new channel is evaluated inside the complete recipe, including the last
  step. The protocol channel, for example, looks useless without the alignment and is valuable with it.
- **Decide on cross-validation, not on the public leaderboard.** Decision rules are fixed before the result is known,
  comparisons are paired, and controls (copy null points, placebos, planted signals) guard against self-deception
  ([VALIDATION.md](VALIDATION.md)).
- **Guards protect, they do not tune.** Test-side guards see aggregates only and can switch to a fallback form or abort
  a build. Their thresholds are fixed in advance; a correction may only widen them, and the positive controls must still
  trigger them. Where a component lacks the information for a decision, a fixed rule leaves that decision to the part
  that has it, as the NONE guard does for `NONE`.
- **Build only from what is delivered.** Every final file must be reproducible from the delivered weights and frozen
  inputs. The rotation list of a first network whose weights were not stored was replaced by the list of the released
  network, although the change brought no gain.
