# Validation

Reference document. To reproduce the submissions, start with [VERIFICATION.md](VERIFICATION.md).

*Status: 2026-11-21 (version 0.6).*

Every model decision in this project was made on cross-validation with the 1,500 labelled rows; the public leaderboard
was never used to choose between variants. This document describes the protocol: folds, leakage controls, the three
measurement forms, how uncertainty is estimated, how decisions are made, how the components added last were checked,
and which checks run before a submission file is accepted.

## 1. Folds

| Item | Setting |
|---|---|
| Scheme | 5 folds, stratified by `root_code`, seed 42 (`kaggle_core.folds`) |
| Storage | Generated once and stored as `configs/folds.parquet` (row position → fold); `tools/make_folds.py` writes the array `artifacts/folds.npy` from it and can recompute the split from `train.csv` |
| Integrity | Every run loads the same file and checks its hash (`07ec381cc58b2a80`, constant in `src/vlm/common.py`). This holds locally, on Kaggle and on rented GPUs. All channels therefore share identical folds. The second stages record the checksum of `artifacts/folds.npy` among their inputs |
| Input integrity | When a CSV table is loaded, its row order is checked against a fixed hash (`src/vlm/common.py`); a reordered or modified file stops the run |
| Test likeness | 90.7 % of validation rows have their question in the training part of their split; on the test set the share is 95.7 % |

**No leave-one-out**, neither as the outer split, nor as an inner selection criterion, nor in target encoding. Five
codes are capped at exactly 57 rows each, so removing one row changes the class frequencies in a way that reveals the
label. Leave-one-out would also break the stratification.

## 2. Leakage controls

- A fold model never sees its own fold.
- Same-question examples, topic blocks and all question or topic knowledge features come only from the other folds. At
  test time they come from all 1,500 training rows.
- The knowledge features of the second stage follow the same rule. In a perturbation check, scrambling the labels of the
  evaluated fold left its features, candidate order and abstention gate bit-identical, while scrambling the labels of
  another fold changed them.
- The task block is built from teacher pseudo-labels on pool photos. The teacher never saw a labelled row, so the block
  carries no label information and needs no fold exclusion.
- The M3 training targets are built per row, and each fold model trains only on rows of the other folds.
- 129 training transcriptions contain annotator verdicts. These are removed (`src/transcription.py`) before a
  transcription is used anywhere.
- The protocol channels (27B protocol channel, M3, 397B) see no labelled row in their prompt, so their prompts depend on
  no fold. M3 itself is trained per fold (see above), and on the OOF side `q397m3` is scored with the fold adapters.
- The rows in `train.csv` are partly ordered by category. They are never processed in file order; generation jobs use
  a shuffled order.

## 3. Out-of-fold artifacts and metric

- Every channel writes an OOF matrix (1,500 × 76) and a metadata file with sources and checksums
  (`kaggle_core.artifacts`). Channels with their own test-side scores also write a test matrix (3,498 × 76). For channels
  whose test side is a ranking behind frozen texts, such as M3 and `q397m3`, the stored test matrix is empty, and
  `tools/make_submission.py` builds the test side from the frozen rankings.
- A run writes these artifacts even when its score is poor, because OOF matrices are what the blend works with.
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

- For the primary line, a change of the channels or of the blend counts as stable only if it holds in all three forms.
  The `q397m3` channel is a warning example. Without the second stage it was ahead per fold (+0.0034 ± 0.0028), but
  behind without the last step (−0.0033) and as one set (−0.0014). It entered the primary line only after it was ahead
  in all three forms together with the arithmetic share and the reranker, by small and uncertain margins (+0.0021 /
  +0.0027 / +0.0050, each ± 0.0028; [METHOD.md](METHOD.md#46-397b-notes-named-by-m3-q397m3-primary-line-only)).
- Components that act after the last step, the reranker and its guards, partly take over the role of the alignment, so
  the form without the last step overstates them. For these components the decision uses the per-fold form and the
  as-one-set form; the form without the last step is only reported.
- For the inductive line only the per-fold form is a deployment form; its other forms are reported for information.

**Blend weights in cross-validation** are cross-fitted: the weights used for fold *k* are fitted on the other four folds.
Deployed weights are fitted on all 1,500 rows.

## 5. Uncertainty and decision rules

### 5.1 Paired comparisons

- Two variants are always compared on identical rows. We report the mean difference in MAP@3 and its standard error.
- The standard error comes from a **cluster bootstrap over `question_id`** with 10,000 draws. Rows of the same question
  are not independent, so a bootstrap over rows would understate the uncertainty. Measurements from the first week used a
  row bootstrap.
- A difference is read together with its standard error, the number of positive folds and the change in LogLoss. In the
  full recipe, paired standard errors are typically 0.0014–0.004 ([RESULTS.md](RESULTS.md)). A gain of a few
  thousandths is therefore credible only if it is supported across folds, across new meta-splits and by LogLoss.
- On the public leaderboard (1,000 rows), the paired noise between consecutive submissions is about ±0.003 to ±0.005.
  Public differences of that size are not interpreted.

### 5.2 Pre-registered decision rules

The decision rule for an experiment was written down before its result was known. It names the expected effect, the
threshold for adoption and the result that would stop the line of work.

- **Typical adoption rule for a new channel:**
  - gain ≥ +0.006 at z ≥ 1.5;
  - at least 4 of 5 folds non-negative;
  - a variant with fixed probability levels, instead of the model's stated probabilities, gains at least +0.003;
  - LogLoss improves with z ≥ 3.
- **Primary line:** a change is adopted if its expected gain is positive (empirical-Bayes posterior mean > 0).
- **Second-stage components** (the NONE guard, the general-code identifiers and the second stage of the inductive line):
  - Δ > 0 against the current form both per fold and as one set (inductive line: per fold only, its deployment form);
    the other forms are only reported;
  - at least 20 new meta-splits with mean Δ > 0, at least 75 % of them positive;
  - Δ above the 95 % quantile of at least 200 placebo sets with the same number of columns;
  - an exact copy null point;
  - a test side built only from existing frozen outputs, with the released rotation list.
- **Inductive line:** in addition, the posterior probability of a loss larger than 0.002 must be at most 5 %
  (P(θ < −0.002) ≤ 0.05; §5.5).
- **Expected-value rule** (since 2026-11-06, for the primary slot; used for the row × candidate interactions and the
  minor-code × mechanical-match column): a component is deployed if (i) its copy null point is exact (OOF rows bit for
  bit, and the file with weight 0 byte for byte), (ii) the mean of the per-fold and the as-one-set difference, ½(F+M),
  is positive on the original meta-split and above the mean of at least 5 placebos, and (iii) the mean of ½(F+M) over at
  least 10 new meta-splits is positive. There are no fixed thresholds: the per-fold and the as-one-set form are weighted
  equally because the CV of the original meta-split was found to be about 0.002 above the mean over new meta-splits.
- **Safety-set rule** (2026-11-06, for the safety line with alignment over the test set): (i) exact copy null points;
  (ii) variant a counts if its CV as one set (gate and guard included) is not below the CV of the strictly inductive line
  with second stage in its deployment form minus 0.002, and if the test-side class masses after the alignment lie at most
  L1 0.035 from the target; (iii) variant b (with the interactions) instead of a only by the expected-value rule.
  The CV part of (ii) was missed by one sixth in 9,000 and overridden by a recorded decision (§5.4,
  [`decisions/override_cv_condition_ii.json`](../decisions/override_cv_condition_ii.json)).
- **Rules of the note-feature forms of the safety line** (2026-11-07; [METHOD.md](METHOD.md#910-note-mentions-and-example-similarity-in-the-safety-line-fallback_i1_nd)):
  exact copy null points; ½(F+M) on the original meta-split > 0 and above the mean of 15 placebos; the mean over 20 new
  meta-splits > 0; the risk rule of §5.5 with P(θ < −0.002) ≤ 0.05. The regex note columns must in addition be positive in
  the mean of a full draw swap over four M3 draw sets and in at least 3 of the 4 sets, and the risk rule uses the spread of
  the sets as τ_d and must hold for the original observation and for the mean of the sets. The test-side guards of both
  forms only abort a build; no size threshold applies.
- **Rule of the channel `m3px27`** (2026-11-07, for the primary slot): the expected-value rule with controls of the same
  role. Deploy only if ½(F+M) on the original meta-split lies above the maximum of 0 and four controls (a copy of the
  untrained ranking, the mean of the untrained ranking and a fourth M3 draw, the fourth M3 draw, a copy of the M3 channel)
  and the mean over 20 new meta-splits is above 0 and above the mean of the controls over the same splits; the controls must
  meet their values fixed in advance on the original meta-split to four decimals. Secondary roles of the channel would have
  needed a nested choice of role and were excluded; the recipe without the 397B channel is reported only.
- **Generated texts are draws.** Two single runs of a text-generating channel are never compared against each other.
  Variants are compared over several draws, or with the text held fixed and only the step behind it changed.

### 5.3 Controls

| Control | Purpose |
|---|---|
| Copy null point | An existing draw is added again as if it were a new member. A new channel must beat this |
| Weight zero | A component with weight 0 must reproduce the previous file byte for byte. For the second-stage components: identifier weights 0 and the NONE guard switched off reproduce the previous form bit for bit on the OOF rows and byte for byte on the test file |
| Label and permutation placebos | Shuffled labels or shuffled features must give no gain; a reranker fitted on placebo labels must abstain |
| Placebo sets of the same size | Random code identifiers, or feature columns permuted across rows within each fold, pass through the same chain as the real component (fit, guard, gate). The real component must lie above their 95 % quantile |
| New meta-splits | The whole nested computation is repeated on new stratified 5-fold meta-splits with other seeds. This measures the uncertainty from the meta-split; the channel OOF matrices stay those of the original folds and are not regenerated per meta-split |
| Positive control | A signal of known size is planted. The measurement must detect it before a null result is accepted |
| Redundant control channels | Existing channels with a known value are re-added at the same position; they bound what "no new information" looks like |

### 5.4 Validation of the second-stage components

**Primary line: NONE guard and general-code identifiers** ([METHOD.md](METHOD.md#93-none-guard)), Δ against the main
form of the reranker:

| Check | Result |
|---|---|
| Original meta-split | Per fold +0.0013 ± 0.0014 (4 of 5 folds positive); as one set +0.0031 ± 0.0015 (5 of 5); without the last step +0.0051 ± 0.0016 (reported only) |
| 80 new meta-splits | Per fold +0.0016, positive in 71 of 80; as one set +0.0014, positive in 73 of 80 |
| 300 placebo sets: identifiers of five random codes, without the guard | 95 % quantile +0.0010 per fold and +0.0012 as one set; the component lies above both |
| Stricter placebo: random identifiers with the same NONE guard | 95 % quantile +0.0016 per fold and +0.0031 as one set; the component does **not** lie above (10 % and 6 % of the placebo sets are at least as good) |
| Placebo with the same columns: the real identifier columns with candidate blocks permuted across rows within each fold, without the guard | 95 % quantile +0.0013 per fold and +0.0016 as one set; the component is level with it per fold and above it as one set |
| Parts | NONE guard alone +0.0010 (mean of both forms), positive in 80 of 80 meta-splits; identifiers on top of the guard +0.0004, positive in 54 of 80 |
| Copy null points | Identifiers 0 and guard off reproduce the main form bit for bit on the OOF rows. On the test side, the null model reproduces the file without second stage, and the main-form weights with identifier weights 0 and the guard off reproduce the main-form file, both byte for byte; with the guard on, exactly the guarded rows change |
| File swap | Notes of another draw in place of draw d1, with consistent provenance: the note guard triggers, and the fallback-form file results byte for byte |
| Test side (aggregates) | Other top code than the main-form file in 0.94 % of test rows (OOF 1.27 %); guard (c) 0.0254 within 0.0183–0.0733 |

The rule as written, with placebo sets without the guard, is met; the stricter reading is not, and against placebo sets
with the same columns the component is only level per fold. The stable part is the NONE guard, while the identifiers
add a small positive expectation. The expected gain over the main form is about +0.0013 to
+0.0016. This estimate is optimistic rather than conservative: the guard was found in an error analysis of the same
1,500 labels, and it acts on few rows.

**Strictly inductive line: second stage with NONE guard** ([METHOD.md](METHOD.md#96-second-stage-in-the-strictly-inductive-line)),
Δ against the inductive line without second stage:

| Check | Result |
|---|---|
| Deployment form, per fold | +0.0049 ± 0.0022 (z = 2.2; 3 of 5 folds positive); the abstention gate abstains in none of the 5 folds |
| Other forms (reported only) | Without the last step +0.0047; as one set +0.0016 |
| 40 new meta-splits | Mean +0.0055 (SD 0.0013), positive in 40 of 40; a second set of 40 seeds, computed with independent code, gives the same mean and 40 of 40 |
| Placebo sets: 28 of the 40 feature columns permuted across rows within each fold (two independent sets of 1,000) | 95 % quantile +0.0006 / +0.0007, maximum +0.0019 / +0.0018; no placebo reaches the real value |
| NONE guard alone | +0.0011 ± 0.0007, carried by 3 rows; resetting 3 random rows with a changed top code instead gives a 95 % quantile of +0.0007 |
| Copy null point | Weight zero reproduces the inductive line bit for bit on the OOF rows and the file without second stage byte for byte (CSV and test matrix) |
| Test side (aggregates) | Other top code than the file without second stage in 2.94 % of test rows (OOF 2.73 %); guard (c) 0.0294 within 0.0137–0.0547; no `NONE` top lost |
| Dependence on the M3 draws | With the fourth M3 draw replacing one of the three, the four draw sets give +0.0007 to +0.0049. The deployed set is the most favourable one; the only set with the same composition (two greedy draws, one sampled draw) gives +0.0007 ± 0.0020 |
| Risk rule | Met under the rule as written (τ_d = 0); fails from a between-draw spread of about 0.0017 (§5.5) |

On the test side, whose M3 texts are draws independent of the training draws, the expected gain is +0.002 to +0.003, not
the +0.0049 of the deployed draw set. Almost all of the measured gain lies in the second sub-collection of the training
data.

**Primary line: row × candidate interactions** ([METHOD.md](METHOD.md#97-row--candidate-interactions-main_guard_i1)), Δ against
the guard form, by the expected-value rule:

| Check | Result |
|---|---|
| Original meta-split | Per fold +0.0019 ± 0.0011; as one set +0.0010 ± 0.0010; ½(F+M) +0.0014; without the last step −0.0020 (reported only); in sixths of a row +17 / −18 / +9 (= an independent re-implementation of the same form) |
| 15 placebos (row factors permuted across rows within each fold) | ½(F+M) mean +0.0004; 1 of 15 at least as large as the real value |
| 20 new meta-splits | ½(F+M) +0.0011 ± 0.0002, positive in 18 of 20; above its own placebo in 20 of 20 |
| Copy null points | I1 columns 0 reproduce the guard form bit for bit on the OOF rows; on the test side the null model reproduces the file without second stage, and the 48 guard-form weights with I1 weights 0 reproduce the guard-form file and test matrix byte for byte |
| File swap | Notes of draw d2 in place of d1: the note guard triggers and the fallback-form file results byte for byte |
| Test side (aggregates) | Other top code than the guard-form file in 26 test rows |

**Primary line: minor-code × mechanical-match column** ([METHOD.md](METHOD.md#98-minor-code--mechanical-match-column-main_guard_i1_mm-final-primary-candidate)),
Δ against the I1 form, by the expected-value rule:

| Check | Result |
|---|---|
| Original meta-split | Per fold +0.0001 ± 0.0005; as one set +0.0011 ± 0.0006; ½(F+M) +0.0006; without the last step +0.0011 (reported only); in sixths +1 / +10 / +10 |
| 15 placebos (`mech27` permuted across rows within each fold) | ½(F+M) mean −0.0001; none at least as large as the real value |
| 20 new meta-splits | ½(F+M) +0.0005 ± 0.0001, positive in 19 of 20 |
| Copy null points | Column 0 reproduces the I1 form bit for bit on the OOF rows (also on the first new meta-split); on the test side the null model reproduces the file without second stage, and the 63 I1 weights with column weight 0 reproduce the I1-form file and test matrix byte for byte |
| Parser | The mechanical comparison is a re-implementation without `eval` of an independent reference parser used during development; both agree on all values read from the training and test answers and on all 4,998 answer keys |
| Test side (aggregates) | Other top code than the I1-form file in 6 test rows |

**Safety line with alignment over the test set** ([METHOD.md](METHOD.md#99-second-stage-of-the-safety-line-with-alignment-over-the-test-set)),
by the safety-set rule:

| Check | Result |
|---|---|
| (i) Copy null points | Weight 0 reproduces the base rows bit for bit (variant a), I1 columns 0 reproduce variant a bit for bit (variant b); on the test side weight 0 reproduces the file without second stage, and I1 weights 0 with the 40 weights of variant a reproduce the file of variant a (CSV and test matrix) |
| (ii) CV part | Variant a as one set (gate and guard) 7,391 of 9,000 sixths against the threshold 7,392 (strictly inductive form per fold 7,410 minus 0.002): **missed by one sixth**; the inner gate abstains in one fold (−22 sixths), which it does not do in the deployment fit on all 1,500 rows |
| (ii) Test-mass part | L1 of the mean test mass per class after the alignment 0.028 (limit 0.035); with the inductive alignment 0.087, which reproduces an independent check |
| (iii) Variant b over a | ½(F+M) on the original meta-split above the placebo mean; over 20 new meta-splits +0.0007 ± 0.0001 |
| Additional: variant a against the strictly inductive form over the same 20 new meta-splits | +0.0007 ± 0.0002; positive in 16 of 20, above −0.002 in 20 of 20 |
| Override | Only the CV part of (ii), by the decision recorded in `decisions/override_cv_condition_ii.json` (SHA-256 pinned in the code); without it the build ends with exit code 5 |
| Test side (aggregates) | Other top code than the strictly inductive form with second stage in 70 test rows (2.0 %); 10 rows kept by the NONE guard; guard (c) 0.029 against the OOF share 0.026; NONE mass after the alignment 0.3375 against the target 0.3373 |

**Safety line: note mentions and example similarity** ([METHOD.md](METHOD.md#910-note-mentions-and-example-similarity-in-the-safety-line-fallback_i1_nd)),
Δ against `safety_final` (variant b):

| Check | Result |
|---|---|
| Copy null points | The three columns at 0 reproduce variant b bit for bit on the OOF rows (row values, orders, NONE guard); on the test side the null model reproduces the file without second stage, and the weights of variant b with 0 for the new columns reproduce its file and test matrix byte for byte |
| Original meta-split | ½(F+M) +0.0032 ± 0.0015 (per fold +0.0033, 4 of 5 folds positive; as one set +0.0031) |
| 15 placebos (the three columns from another row of the same block) | Mean +0.0005, maximum +0.0020 |
| 20 new meta-splits | +0.0028 ± 0.0002, positive in 20 of 20; placebo −0.0001 |
| Risk rule (§5.5) | P(θ < −0.002) = 0.0001 at τ_d = 0 (posterior mean +0.0018); 0.0053 at τ_d = 0.001 (reported only) |
| Check before the rule (full draw swap) | Draws d1–d3 +0.0028, mean of the four sets of three +0.0018, all four positive |
| Positive controls on the test side | Copied, permuted and truncated test notes: a guard fires and no file is written; the similarity guard alone would not have detected the copy (it passes through the band) |
| Test side (aggregates) | Note guard 13 of 13; similarity 0.2176 against 0.2095 (band; z +4.51); guard (c) 0.0374 (OOF 0.0367); class masses L1 0.0282; another top code than variant b in 70 test rows |

**Safety line: regex-exact note columns** ([METHOD.md](METHOD.md#911-regex-exact-note-columns-fallback_i1_ndr-final-safety-candidate)),
Δ against the form with note mentions and example similarity:

| Check | Result |
|---|---|
| Copy null points | R columns 0 reproduce the predecessor bit for bit on the OOF rows; on the test side the null model reproduces the file without second stage, and the predecessor's weights with 0 for the R columns reproduce its file and test matrix byte for byte |
| (1) Full draw swap, four sets of three from d1–d4 | +0.00099 / +0.00026 / +0.00083 / +0.00015, mean +0.00056, 4 of 4 positive |
| (2) Original meta-split | ½(F+M) +0.00072 ± 0.00108 (per fold +0.00044, as one set +0.00100); placebo mean −0.00012 (maximum +0.00083) |
| (3) 20 new meta-splits | +0.00087 ± 0.00022, 85 % positive; placebo +0.00011 |
| (4) Risk rule, τ_d = spread of the four sets (0.00041) | P(θ < −0.002) = 0.0007 (original) and 0.0010 (mean of the sets); second draw of the protocol notes: 0.0006 |
| Positive controls on the test side | Copied, permuted, truncated notes, protocol notes shifted by one row or replaced by the M3 notes d1: a guard fires and no file is written; the last two are detected only by the guard of the columns R |
| Test side (aggregates) | All guards held (guard R 5 of 5 within the band); guard (c) 0.0400 (OOF 0.0407); another top code than the predecessor in 38 test rows |

**Primary line: M3 ranking behind the protocol notes** ([METHOD.md](METHOD.md#47-m3-behind-the-protocol-notes-m3px27-mean-channel-m3px27xm-final-primary-candidate)),
Δ against the final form with `pr27xm`, by the rule of §5.2:

| Check | Result |
|---|---|
| Original meta-split | ½(F+M) +0.00006 (per fold +15 sixths, as one set −14) |
| Controls of the same role (original / mean of 20 new meta-splits) | Copy of the untrained ranking 0 / 0 (bit-identical); mean of the untrained ranking and the fourth M3 draw −0.00094 / −0.00036; fourth M3 draw −0.00167 / −0.00045; copy of the M3 channel 0 / −0.00097; control mean −0.00045 |
| 20 new meta-splits | +0.00013 ± 0.00018, positive in 55 % |
| Recipe without the 397B channel (reported only) | Original +0.00194; 20 new meta-splits +0.00098 ± 0.00018, positive in 95 % |
| Copy null points | The untrained ranking in place of `m3px27` reproduces the blend file, the 15 row vectors of the chain, the five deployment files (up to provenance and channel names) and the test side (CSV and test matrix) exactly |
| Test side (aggregates) | 3,498 of 3,498 rows ranked exactly; NONE at the top 0.346 (training rows 0.344); same top code as the untrained test ranking in 82.3 % of rows (training rows 85.3 %); another top code than the final form with `pr27xm` in 52 test rows |

### 5.5 Risk rule of the strictly inductive line

The inductive slot requires, in addition to the checks above, that the posterior probability of a loss larger than 0.002
is at most 5 %: P(θ < −0.002) ≤ 0.05.

- **Posterior.** Empirical Bayes. The prior is a nonparametric maximum-likelihood estimate from 12 earlier measurements
  of the same class of component: steps that convert information the channels already provide, such as calibration
  steps, rerankers and verifiers on existing texts. 79 % of its mass lies at zero. The observation is the paired per-fold
  difference (+0.0049, standard error 0.0024, the larger of the question and the row bootstrap). A selection correction
  accounts for the form having been chosen among several measured variants (6 variants, correlation 0.5, spread 0.0015).
  The posterior mean is +0.0016; it depends strongly on the prior (−0.0004 to +0.0053 over the variants checked) and is
  not the decision quantity, which is P(θ < −0.002).
- **Between-draw variance τ_d.** The M3 texts are generated draws, and the component's gain depends on them (§5.4). A
  between-draw variance τ_d widens the posterior. The rule did not name τ_d, so its default τ_d = 0 applies; choosing
  another value after seeing the numbers would be a post-hoc choice. The mechanism supports a value near zero: the two
  sampled draws (d3 and the fourth draw) differ only in their seed, and no mechanism is known that shifts all rows in the
  same direction. The data cannot determine τ_d. The four draw sets give 0 by maximum likelihood (95 % profile interval
  up to 0.0053), 0.0014 by REML and 0.0021 by the method of moments; a test of homogeneity gives p = 0.22.
- **Sensitivity:**

  | τ_d | 0 | 0.0014 | 0.0017 | 0.0019 | 0.0021 |
  |---|---:|---:|---:|---:|---:|
  | P(θ < −0.002) | 0.0015 | 0.031 | 0.050 | 0.064 | 0.080 |

  The rule is met at τ_d = 0 and fails if the between-draw spread is about 0.0017 or larger. Averaged over a flat
  uncertainty of τ_d, P lies between 0.09 and 0.21, depending on the assumed upper bound.
- **Consequence.** Under the rule as written, the second stage with NONE guard became the candidate of the inductive
  line (version 0.1), with the line without second stage as its fallback. The choice of an inductive slot was to weigh
  this sensitivity against the expected gain of +0.002 to +0.003; in the final selection both slots went to the primary
  line. A further M3 draw on the training rows would narrow τ_d.
- **Use in the note-feature forms of the safety line.** The rules of §9.10 and §9.11 of [METHOD.md](METHOD.md) use the same
  posterior with the same prior (12 measurements of this class, `eda/risk_rule.py`; selection correction with 6 variants,
  correlation 0.5, spread 0.0015). For the note mentions and the example similarity τ_d = 0 decides (P = 0.0001; τ_d 0.001 is
  reported: 0.0053). For the regex note columns a fourth M3 draw was available: τ_d is the spread of the four draw sets
  (0.00041), and P is 0.0007 for the original observation and 0.0010 for the mean of the sets.

## 6. Checks before a submission

`tools/make_submission.py` and the build scripts in `tools/builds/` stop with an error if any of the following checks
fails; a file counts as built only if its build ends without error. All test-side checks print **aggregates only**:
counts, shares and checksums. No per-row test output is displayed or inspected.

**Inputs**
- Pinned or recorded SHA-256 checksums: the OOF matrices that enter the blend and the checks, the blend and second-stage
  files, the rotation lists, the frozen M3 test notes and the 397B notes, the earlier submission files used as
  references, and the fold array. The weights of the orientation network are checked against the checksum recorded in
  its description, if the weights are present.
- The frozen test-side score files of the channels (`artifacts/runs/`) and the blank-photo list are not pinned one by
  one. They are covered indirectly: a change that alters a delivered file is caught by the pinned output checksums and
  the byte-identical rebuild; a change that leaves every submission file unchanged has no effect on the result. The
  blank-photo list can also be recomputed from the photos with `python -I tools/make_blank_list.py check`.
- The stored verdicts of the M3 evaluation and of the 397B parity check are checked field by field, not by checksum.
- Blend files are recomputed from the OOF artifacts and compared character by character (`--verify`); the second-stage
  files are refitted and compared in the same way.
- The inductive class weights are recomputed from the OOF artifacts and compared with the stored checksum.

**Rotation list of the `v1r` test side**
- The build scripts select the list with `ORIENT` (`repro`, the default, or `legacy` to rebuild the historical files)
  and check its checksum.
- With `repro`, a `v1r` source on the first-version list is rejected, by path and by checksum.
- Before the assembly, `tools/orient_list.py check` verifies the shape of the list, the agreement of the row list and
  the JSON list, the provenance recorded in the description of the orientation network, that the rotation run covers
  every listed row, and that it used the same rotation.
- After the assembly, `tools/orient_list.py verify-sub` checks that the written record takes `v1r` from exactly this
  list and all other rows from the plain 9B reading.
- The released-list builds compare their output with the first-version-list file of the same line, as counts and shares.

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

**Second stage**
- The blend file referenced by the second-stage file must match: checksum, channels and β; K and λ must be 5 and 0.1.
- Feature set and NONE guard are taken from the model in the deployment file. A fallback model must never read the notes.
- (a) Note guard (primary line, main forms). 13 statistics of the frozen M3 test notes are compared with the three
  training draws: mean number of codes mentioned, share of rows mentioning the top candidate, distribution of mention
  counts, and share of identical notes per pair of draws.
  - A statistic passes if it lies within ±10 % of the range of the training draws. Under the widened rule, a share
    statistic also passes if a two-sample binomial test against its own training value gives |z| ≤ 3. For deployment
    files with the widened rule the assembly refuses other values for these two thresholds; the deployment file with the
    band rule (`stage2_s032.json`) is pinned by checksum in the build scripts.
  - If any statistic fails, the fallback form, which does not use the notes, is applied automatically. There is no
    switch to override this guard.
  - Positive controls (swapped or copied note files, shifted statistics) must trigger the guard.
- The NONE guard is applied before guard (c), as in the OOF computation.
- (b) The share of candidates new for their question, test against OOF, is reported.
- (c) The share of test rows whose top code changes must lie within ½–2× the OOF share; otherwise the build aborts.
- Inductive line: the deployment file names `align_mode: inductive`. The assembly then requires the inductive alignment
  with the strength and class threshold of the file, accepts only the feature set without notes, and checks that the M3
  test draws are those named in the file, in the same order. There is no note guard.
- Note-feature forms of the safety line (`fallback_i1_nd`, `fallback_i1_ndr`; only with `align_mode: set`): the frozen M3 test
  and training notes and, for the regex columns, the frozen protocol test notes are read with their checksums (deployment file
  = meta file = file); the widened note guard, the similarity guard, the guard of the columns R and guard (c) act **only as an
  abort**: a failing guard writes no file, and nothing switches to another form. The rule fields of the deployment file must
  equal the fixed values (band 10 %, |z| ≤ 3). `--stage2-control` runs the positive controls in memory (exit code 5 if a guard
  fires, 6 if none fires; never a file).
- Mean channel with an M3 ranking (`m3px27xm`, field `rank_mean.rank_compare: sft`): the test ranking is compared with every
  fold shard of the training side (only model, fold and note source may differ), and the note guard of the channel requires
  that every shard and the test ranking were computed behind the frozen note files named in the channel metadata, that the
  test notes come from the recorded protocol test run, and that they are character-identical with the input of the untrained
  ranking.
- A second stage with weight zero reproduces the file without it byte for byte.

**Output**
- Format check (`src/submission_io.py`):
  - header `trace_id,root_code`;
  - every test `trace_id` exactly once, no foreign identifiers;
  - one to three codes per row, no duplicates within a row;
  - every code spelled exactly as in the taxonomy, or `NONE`;
  - no code without a training example.
- Every candidate and fallback file is checked against its expected SHA-256 inside the build chain: the candidate
  scripts, the main-form script and `s029_m3x3.sh` with the released list pin their own output, and each script checks
  the earlier files it builds on. Run on their own, the other builds only print the checksum of their output
  (`s029_m3x3.sh` with `ORIENT=legacy`, `s029_q397m3.sh`, `s032_stage2.sh` including its entry point
  `s033n_stage2.sh`, and `s029_q397.sh`); a later script of the chain checks these files, and the expected values are in
  [INFERENCE.md](INFERENCE.md#2-lines-and-expected-checksums).
- An existing output CSV file is replaced only by a byte-identical rebuild, except by `s029_m3x3.sh` with
  `ORIENT=legacy` and `s029_q397.sh`, which overwrite their outputs. Test matrices written next to an unchanged CSV
  file are not compared; only the test matrix of the inductive file without second stage is pinned.
- If a build stops at the checksum of a file it has written in the same run, the file stays on disk but does not count
  as built. Delete it before running the build again: an existing file is only replaced by a byte-identical rebuild, so
  the next run would otherwise stop at the comparison with it.

## 7. Known limitations

- **Partially nested blend CV.** Blend weights are cross-fitted, but the base channels' OOF predictions are not
  regenerated inside each meta fold. The CV of the blend is therefore an OOF-based, partially nested estimate, and the
  size of its optimism is unknown.
- **No fresh holdout.** All decisions were made on the same 1,500 labels. The second-stage components added last were
  found and checked on these labels; the new meta-splits repeat the measurement on the same labels and add no
  independent evidence.
- **Student-disjoint split.** The host's split is disjoint by student, but the data contain no student identifier. Our
  folds cannot reproduce this property.
- **Dependence on generated texts.** About two thirds of the blend weight lies on channels that generate text, and
  components that read the M3 draws depend on which texts were generated. Their test-side gain is an expectation over
  draws.
- **CPU class.** The CPU fits of the blend and of the second stages are bit-identical on the reference CPU (Intel Core
  i5-8250U, OpenBLAS kernel `Haswell`) with two BLAS threads. With another OpenBLAS kernel the refitted weights differ by
  about 1e-8, and the build scripts stop at their first refit check. The two direct calls of the quick check refit nothing
  and reproduce both files byte for byte ([VERIFICATION.md §5.1](VERIFICATION.md#51-assemble-cpu)).
- **CV versus public leaderboard.** Recent files scored higher on the public leaderboard than in cross-validation: the
  three reranker files of the primary line by +0.004 to +0.007 against their as-one-set values (the file before the
  reranker by +0.011), the inductive line by +0.008 to +0.011 against its per-fold value. A plausible cause is that test
  rows have more same-question reference rows: 1,500 instead of 1,200 training rows, and 95.7 % instead of 90.7 % seen
  questions. This is an observation, not a rule, and no decision relies on it.

## 8. Final submissions (version 0.4)

The three steps from the primary candidate of version 0.3 to the two final submissions were each decided by a rule fixed
before the measurement, on meta-splits that no earlier decision had used, and checked on the build path by copy null
points, test-side guards and positive controls. All values are paired differences of MAP@3 on the 1,500 training rows,
computed exactly in sixths of a row and reported as ½(F+M) (the mean of the per-fold and the as-one-set form) and as the
as-one-set form M, which is how the test set is processed. A meta-split is a new stratified 5-fold split of the second-
stage evaluation (`eda/c121_stage2_i1.py split_folds`); the channel OOF matrices stay fixed.

### 8.1 Four M3 draws (submission B)

- **Design (fair bank).** A new draw is compared with the three draws in a frame in which all four play the same role:
  for every triple T_j of the four draws, R3_j = the line with the mean of T_j, R4_j = the line with the mean of all four,
  both with the second-stage features of T_j; G = mean_j R4_j − mean_j R3_j. The copy null point repeats the computation
  with one member of T_j used twice as the fourth member (G_copy, mean over the twelve duplicates).
- **Rule.** Deploy if mean G over the 20 meta-splits 1000–1019 is above 0 and above mean G_copy, and if on the original
  split R4_0 − R3_0 is above its copy null point; on the build path the copy null point of the test side (d4 removed) must
  rebuild `primary_m3px27` byte for byte.
- **Result.** G = +0.00109 ± 0.00015 (20 of 20 positive) against G_copy = −0.00036; final check on 20 further meta-splits
  2000–2019 (not used for the choice) +0.00137, 19 of 20 positive.
- **Code.** `eda/c135_primary_m3x4.py bank`, `final`, `placebo`, `rule`; the build path recomputes the deployment and the
  copy null points.

### 8.2 Extra second-stage columns (intermediate form, part of submission A)

- **Bundle test.** Candidate columns that were each too small to decide on their own were collected and tested together:
  the bundle against a placebo bundle (the same columns with their values permuted over the candidates within each row), over 20 meta-splits
  that the selection of the columns had not used. The bundles were fixed before the run. Rule: mean over the 20
  meta-splits above 0 in M and in ½(F+M), and real minus placebo above 0.
- **Result of the deployed bundle** (nine columns from eight sources, METHOD §9.12): +0.00169 ± 0.00015 over 4000–4019 (20
  of 20 positive), +0.00177 ± 0.00025 over 5000–5019 (placebo bundle −0.00044, real minus placebo +0.00222 ± 0.00022),
  +0.00178 ± 0.00025 over 6000–6019; original split +0.0016 (as one set +0.0027). The bundle with one more source (novelty
  of `OTH-00`) measured +0.0019 but failed the test guard (its task row count is not transferable) and was not deployed.
- **Test guard per column and source.** Every statistic of every column (means and shares per source, valid answers) on
  the test rows against the training rows: band ±15 % relative or |z| ≤ 3, valid answers ≥ 0.97 per source; rule fixed
  before the test side was read; only an abort, never a fallback. The 397B columns are the transfer risk (training side
  through the hosted API, test side from our own FP8 run with the same parser): all 43 statistics held, largest |z| 2.31.
- **Positive controls** (in memory, never a file): each applicable perturbation of the test sources must trigger the guard
  (`eda/c151_primary_cols.py controls`, exit code 6 if one does not).
- **Copy null points.** Columns set to zero in the fit reproduce rows and models of submission B; with zero weights the
  test side through the new code path rebuilds submission B byte for byte; the null model (all weights 0) leaves the file
  without second stage unchanged.

### 8.3 Fifth M3 draw with the large photo on the rows with small script (submission A)

- **Roles and rule.** Pre-registered: the best of the roles "fifth member of the M3 mean", "replacement of one 1,024-px
  member" and "own channel", each also restricted to the rows of the list "character height ≤ 20 px", is deployed only if
  over 20 untouched meta-splits (11000–11019) it lies above 0 in M and in ½(F+M) and above the same role filled with a
  1,024-px draw (copy null points: one of d1–d4 used twice).
- **Result.** Fifth member on the listed rows: +0.00083 ± 0.00017 in ½(F+M), +0.00108 ± 0.00030 in M (positive in 85 % and
  75 % of the meta-splits), against the intermediate form; on all rows +0.0007; as replacement or with half weight
  negative; as extra second-stage column −0.0018 ± 0.0003. Original split: ½(F+M) −0.0009, M −0.0002 (as one set 0.8404
  against 0.8407).
- **Caveat.** The best copy null point (d3 twice) reaches +0.0009: the gain is that of one more useful draw, not evidence
  that the larger photo is read better by the line. Submission B (without draw h and without columns) is the second final
  submission.
- **Test guard of draw h and positive controls** (METHOD §4.8): statistics of the listed test rows against the listed
  training rows (±10 % or |z| ≤ 3; answered ≥ 0.97), exact coverage of the list, settings of the OOF draw, adapter of the
  full fit; three perturbations must trigger it (`eda/c166_primary_m3x5h.py h-guard`, `h-controls`).
- **Copy null point "empty list".** With an empty row list the five-draw path reproduces the four-draw channel on the
  training side (blend, chain and columns of submission B and of the intermediate form) and rebuilds the intermediate
  form byte for byte on the test side. A stand-in run (d4 used as the fifth test draw on the list) checks that the path
  gives a formally valid file before the real test draw is read; its file is deleted.

### 8.4 Limits

All three steps were measured on the same 1,500 labels as everything before; the meta-splits measure the stability of a
decision, not new data. The bundle of §8.2 was assembled from single-column results on one set of meta-splits and
confirmed on three others. Public leaderboard values of the final submissions (A 0.84266, B 0.84450, intermediate form
0.84283) were observed after the decisions and are not evidence for or against them.
