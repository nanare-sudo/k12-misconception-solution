# Validation

*Status: 2026-11-01 (version 0).*

Every model decision in this project was made on cross-validation with the 1,500 labelled rows; the public leaderboard
was never used to choose between variants. This document describes the protocol: folds, leakage controls, the three
measurement forms, how uncertainty is estimated, how decisions are made, and which checks run before a submission file is
accepted.

## 1. Folds

| Item | Setting |
|---|---|
| Scheme | 5 folds, stratified by `root_code`, seed 42 (`kaggle_core.folds`) |
| Storage | Generated once and stored as `configs/folds.parquet` (row position → fold) |
| Integrity | Every run loads the same file and checks its hash (`07ec381cc58b2a80`, constant in `src/vlm/common.py`). This holds locally, on Kaggle and on rented GPUs. All channels therefore share identical folds |
| Input integrity | When a CSV table is loaded, its row order is checked against a fixed hash (`src/vlm/common.py`); a reordered or modified file stops the run |
| Test likeness | 90.7 % of validation rows have their question in the training part of their split; on the test set the share is 95.7 % |

**No leave-one-out**, neither as the outer split, nor as an inner selection criterion, nor in target encoding. Five
codes are capped at exactly 57 rows each, so removing one row changes the class frequencies in a way that reveals the
label. Leave-one-out would also break the stratification.

## 2. Leakage controls

- A fold model never sees its own fold.
- Same-question examples, topic blocks and all question or topic knowledge features come only from the other folds. At
  test time they come from all 1,500 training rows.
- The task block is built from teacher pseudo-labels on pool photos. The teacher never saw a labelled row, so the block
  carries no label information and needs no fold exclusion.
- The M3 training targets are built per row, and each fold model trains only on rows of the other folds.
- 129 training transcriptions contain annotator verdicts. These are removed (`src/transcription.py`) before a
  transcription is used anywhere.
- The protocol channels (27B protocol channel, M3, 397B) see no labelled row in their prompt and depend on no fold.
- The rows in `train.csv` are partly ordered by category. They are never processed in file order; generation jobs use
  a shuffled order.

## 3. Out-of-fold artifacts and metric

- Every channel writes an OOF matrix (1,500 × 76), a test matrix (3,498 × 76) and a metadata file with sources and
  checksums (`kaggle_core.artifacts`). A run writes these artifacts even when its score is poor, because OOF matrices are
  what the blend works with.
- MAP@3 is computed by `src/metric.py`, which breaks ties deterministically. LogLoss is the second criterion and the
  objective of the blend weights.

## 4. Three measurement forms

The last step of the primary line, the transductive alignment, depends on the set it is applied to. In cross-validation it
can be applied to each 300-row fold or to all 1,500 OOF rows at once; on the test set it runs over 3,498 rows. Results are
therefore reported in up to three forms:

| Form | Definition | Used for |
|---|---|---|
| **Per fold** | Blend weights fitted on the other four folds; the last step applied within each 300-row fold, with target frequencies from the other folds | Main value for both lines; deployment form of the inductive line, whose alignment uses fixed weights per row |
| **Without the last step** | Blend (and reranker) without the alignment | Diagnosis only |
| **As one set** | The alignment applied to all 1,500 OOF rows at once, as on the test set | Deployment form of the primary line |

For the primary line, a change is called robust only if it holds in all three forms. The `q397m3` channel is a warning
example. Without the second stage it was ahead per fold (+0.0034 ± 0.0028), but behind without the last step (−0.0033)
and as one set (−0.0014). It entered the primary line only after it was ahead in all forms together with the reranker
([METHOD.md](METHOD.md#46-397b-notes-named-by-m3-q397m3-primary-line-only)).

**Blend weights in cross-validation** are cross-fitted: the weights used for fold *k* are fitted on the other four folds.
Deployed weights are fitted on all 1,500 rows.

## 5. Uncertainty and decision rules

### 5.1 Paired comparisons

- Two variants are always compared on identical rows. We report the mean difference in MAP@3 and its standard error.
- The standard error comes from a **cluster bootstrap over `question_id`** with 10,000 draws. Rows of the same question
  are not independent, so a bootstrap over rows would understate the uncertainty. Measurements from the first week used a
  row bootstrap.
- A difference is read together with its standard error, the number of positive folds and the change in LogLoss. In the
  full recipe, paired standard errors are typically 0.002–0.004 ([RESULTS.md](RESULTS.md)). A gain of a few thousandths
  is therefore credible only if it is supported across folds and by LogLoss.
- On the public leaderboard (1,000 rows), the paired noise between consecutive submissions is about ±0.005. Public
  differences of that size are not interpreted.

### 5.2 Pre-registered decision rules

The decision rule for an experiment was written down before its result was known. It names the expected effect, the
threshold for adoption and the result that would stop the line of work.

- **Typical adoption rule for a new channel:**
  - gain ≥ +0.006 at z ≥ 1.5;
  - at least 4 of 5 folds non-negative;
  - a variant with fixed probability levels, instead of the model's stated probabilities, gains at least +0.003;
  - LogLoss improves with z ≥ 3.
- **Primary line:** a change is adopted if its expected gain is positive (empirical-Bayes posterior mean > 0).
- **Inductive line:** in addition, the posterior probability of a loss larger than 0.002 must be at most 5 %
  (P(θ < −0.002) ≤ 0.05).
- **Generated texts are draws.** Two single runs of a text-generating channel are never compared against each other.
  Variants are compared over several draws, or with the text held fixed and only the step behind it changed.

### 5.3 Controls

| Control | Purpose |
|---|---|
| Copy null point | An existing draw is added again as if it were a new member. A new channel must beat this |
| Label and permutation placebos | Shuffled labels or shuffled features must give no gain; a reranker fitted on placebo labels must abstain |
| Positive control | A signal of known size is planted. The measurement must detect it before a null result is accepted |
| Redundant control channels | Existing channels with a known value are re-added at the same position; they bound what "no new information" looks like |
| Weight zero | A component with weight 0 must reproduce the previous file byte for byte |

## 6. Checks before a submission

`tools/make_submission.py` and the build scripts in `tools/builds/` refuse to write a file if any of the following
checks fails. All test-side checks print **aggregates only**: counts, shares and checksums. No per-row test output is
displayed or inspected.

**Inputs**
- SHA-256 of every frozen input and of every blend and second-stage file.
- Blend files are recomputed from the OOF artifacts and compared character by character (`--verify`).
- The inductive class weights are recomputed from the OOF artifacts and compared with the stored checksum.

**Coverage and fault bands of the channels**
- Each test row is covered exactly once by each channel. Draws of the same channel are averaged only over the rows they
  cover.
- Protocol channel: at most 3 % of test rows may be without a valid answer.
- Exact-ranking channels: a ranking on at least 98 % of test rows, and a median candidate mass of at least 0.9.
- The share of `NONE` at the top of a ranking channel must lie in a fault band fixed in advance: 0.30–0.38, and
  0.27–0.35 for `q397m3`. The bands detect technical faults. They are never used to choose between variants.

**Consistency between training and test side**
- Each OOF draw of a channel needs exactly one test draw, in the same order.
- The generation settings of each test draw must match those of its OOF draw: maximum tokens, temperature, system text,
  image size and backend.
- For `q397m3`, the test ranking must have been computed behind the notes file whose checksum is stored in the channel
  metadata. That notes file must come from a recorded 397B test run (raw-output and metadata checksums) and name the
  expected model.

**Second stage (primary line)**
- The blend file referenced by the reranker file must match: checksum, channels and β.
- (a) Notes guard. 13 statistics of the frozen M3 test notes are compared with the three training draws: mean number of
  codes mentioned, share of rows mentioning the top candidate, distribution of mention counts, and share of identical
  notes per pair of draws.
  - A statistic passes if it lies within ±10 % of the range of the training draws. A share statistic also passes if a
    two-sample binomial test against its own training value gives |z| ≤ 3.
  - If any statistic fails, the fallback form, which does not use the notes, is applied automatically. There is no
    switch to override this guard.
- (b) The share of candidates new for their question, test against OOF, is reported.
- (c) The share of test rows whose top code changes must lie within ½–2× the OOF share; otherwise the build aborts.
- A reranker with weight zero reproduces the file without reranker byte for byte.

**Output**
- Format check (`src/submission_io.py`):
  - header `trace_id,root_code`;
  - every test `trace_id` exactly once, no foreign identifiers;
  - one to three codes per row, no duplicates within a row;
  - every code spelled exactly as in the taxonomy, or `NONE`;
  - no code without a training example.
- The SHA-256 of the written file is compared with the expected value of the line ([INFERENCE.md](INFERENCE.md)).

## 7. Known limitations

- **Partially nested blend CV.** Blend weights are cross-fitted, but the base channels' OOF predictions are not
  regenerated inside each meta fold. The CV of the blend is therefore an OOF-based, partially nested estimate, and the
  size of its optimism is unknown.
- **No fresh holdout.** All decisions were made on the same 1,500 labels.
- **Student-disjoint split.** The host's split is disjoint by student, but the data contain no student identifier. Our
  folds cannot reproduce this property.
- **CV versus public leaderboard.** Recent files scored higher on the public leaderboard than in cross-validation: the
  primary line by +0.005 to +0.007, the inductive line by about +0.011. A plausible cause is that test rows have more
  same-question reference rows: 1,500 instead of 1,200 training rows, and 95.7 % instead of 90.7 % seen questions. This
  is an observation, not a rule, and no decision relies on it.
