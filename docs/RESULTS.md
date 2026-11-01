# Results

*Status: 2026-11-01 (version 0). Cross-validated (CV) values are out-of-fold MAP@3 on the 1,500 labelled rows; "±" is
the standard error of a paired difference from a bootstrap over questions. The measurement forms (per fold, without the
last step, as one set) are defined in [VALIDATION.md](VALIDATION.md#4-three-measurement-forms). Public leaderboard values
come from 1,000 test rows and are reported as observations only; the private leaderboard (2,498 rows) decides the
competition. Final scores will be added after the competition closes.*

**Contents:**
1. [Headline results](#1-headline-results)
2. [How the solution was built](#2-how-the-solution-was-built)
3. [Main submissions](#3-main-submissions)
4. [Single channels and their contribution](#4-single-channels-and-their-contribution)
5. [Ablations and null results](#5-ablations-and-null-results)
6. [Error analysis](#6-error-analysis)
7. [Compute and cost](#7-compute-and-cost)

---

## 1. Headline results

| Line | CV per fold | CV without last step | CV as one set | Public LB |
|---|---:|---:|---:|---:|
| Primary line, reranker main form (candidate) | 0.83756 | 0.83011 | 0.83456 | not submitted |
| Primary line, reranker fallback form (submitted 2026-10-31) | 0.83556 | 0.82989 | 0.83378 | 0.84033 |
| Primary line before the reranker (same channels, β = 0.2) | 0.83267 | 0.81844 | 0.82533 | 0.83616 |
| Strictly inductive line (submitted 2026-10-31) | 0.81844 | — | — | 0.82966 |
| Best single model: M3, mean of three draws | 0.8020 | — | — | — |

- The main form of the reranker gains +0.0020 ± 0.0014 per fold, +0.0002 ± 0.0019 without the last step and
  +0.0008 ± 0.0016 as one set over the fallback form.
- The primary line is about 0.017 ahead of the inductive line in CV; the row-level correlation of the two lines is 0.91.

## 2. How the solution was built

The competition opened on 2026-10-06. The first submission, a question prior without the image, followed on 2026-10-10.
The table shows each step that was adopted, with its CV effect at the time it was introduced. The values are measured in
the recipe of the respective line, per fold unless noted.

| Date | Step | Primary line | Inductive line |
|---|---|---|---|
| 10-10 | Question prior (per-question code frequencies, no image) | 0.5149 | — |
| 10-11 | Fine-tuned 9B scorer | 0.6406 (+0.125 ± 0.009 over the prior) | — |
| 10-11 | + 27B zero-shot scorer | 0.7109 (with task block) | 0.6911 (9B rotation-corrected + `zs27t`) |
| 10-12 | + 27B reason-then-name channel, with probability floor | 0.7672 (+0.056) | 0.7666 (+0.0754 ± 0.0068, gates incl. band) |
| 10-12 | + transductive class-marginal alignment | +0.005 to +0.008 | not used |
| 10-12 | Rotation correction for the 9B scorer; reasoning weight × 1.4 | 0.7851 (+0.0079 ± 0.0027) | — |
| 10-12 | Second reasoning draw, band gate, `REL-01` for blank photos | 0.7951 (+0.0100 ± 0.0043) | (included above) |
| 10-13 | + 27B protocol channel | 0.8053 (+0.0102 ± 0.0040) | 0.7783 (+0.0118 ± 0.0046) |
| 10-13 | Inductive alignment of the blend | — | 0.7892 (+0.0109 ± 0.0031) |
| 10-16 | Protocol channel with exact ranking behind its notes | 0.8096 (+0.0042 ± 0.0019) | 0.7952 (+0.0060 ± 0.0026) |
| 10-18 | + M3, one draw | 0.8143 (+0.0048 ± 0.0039) | 0.8013 (+0.0061 ± 0.0043) |
| 10-19 | M3 as the mean of two draws (600 rows) | 0.8169 → 0.8236 (+0.0067 ± 0.0039) | 0.7994 → 0.8075 (+0.0081 ± 0.0046) |
| 10-24/25 | Arithmetic share β in the blend | +0.0019 ± 0.0016 (β = 0.2) | +0.0038 ± 0.0023 (β = 0.4) |
| 10-26 | Blend weights fitted on all 1,500 rows | 0.81989 (+0.0007 ± 0.0018) | 0.81200 (+0.0003 ± 0.0023) |
| 10-31 | M3 as the mean of three draws | 0.82522 (+0.0053 ± 0.0027) | 0.81844 (+0.0064 ± 0.0025) |
| 10-31 | + 397B channel | 0.82922 (+0.0040 ± 0.0029) | not used |
| 10-31 | 397B answers replaced by 397B notes named by M3 | 0.83267 (without last step −0.0033, as one set −0.0014) | — |
| 10-31 | β = 0.6 and conditional-logit reranker (fallback form) | 0.83556; as one set 0.83378 | planned |
| 11-01 | Reranker main form | 0.83756; as one set 0.83456 | — |

Notes on the table:
- The second M3 draw existed only for two folds at first, so its values use 600 rows (folds 1 and 2). From 2026-10-26
  on, all values use all 1,500 rows. Values are comparable only within the same form; for example, 0.8236 (600 rows) and
  0.81989 (1,500 rows) cannot be compared.
- The reasoning channel was the largest single gain. After it, the gains came from channels that add *different
  knowledge*: the protocol channel without any labelled example, the fine-tuned reasoner M3 and the 397B model. The two
  combination steps that were adopted, the arithmetic share and the reranker, were each worth a few thousandths.
- From 2026-10-24 the test side of the 9B scorer (and of `zs27t` in the inductive line) ran on Kaggle T4 GPUs without
  network access. The blend was unchanged; the top code changed on 1–2 % of test rows.

## 3. Main submissions

Not every submission is listed. Dates are UTC.

| Date | Line | Change against the predecessor | CV (form) | Public LB |
|---|---|---|---|---:|
| 10-10 | — | Question prior, no image | 0.5149 ± 0.011 | 0.53083 |
| 10-11 | — | 9B fine-tuned scorer alone | 0.6406 ± 0.0105 | 0.66850 |
| 10-12 | primary | 9B scorer + `zs27q` + reason-then-name | 0.7672 ± 0.0095 | 0.76666 |
| 10-12 | primary | + transductive alignment | +0.008 ± 0.003 (paired) | 0.77600 |
| 10-12 | primary | Rotation-corrected 9B scorer, reasoning weight × 1.4, second reasoning draw, band gate, `REL-01` | 0.7951 (per fold) | 0.79750 |
| 10-12 | inductive | Rotation-corrected 9B + `zs27t` + teacher-free reasoning draw | 0.7666 (per fold) | 0.77766 |
| 10-13 | primary | + protocol channel | 0.8053 (per fold) | 0.80233 |
| 10-13 | inductive | + protocol channel, inductive alignment | 0.7892 (per fold) | 0.79816 |
| 10-16 | primary | Protocol channel with exact ranking | 0.8096 (per fold) | 0.80733 |
| 10-16 | inductive | Protocol channel with exact ranking | 0.7952 (per fold) | 0.80600 |
| 10-18 | primary | + M3 (one draw) | 0.8143 (per fold) | 0.82066 |
| 10-18 | inductive | + M3 (one draw) | 0.8013 (per fold) | 0.81850 |
| 10-19 | primary | M3 as the mean of two draws | 0.8236 (per fold, 600 rows) | 0.82116 |
| 10-19 | inductive | M3 as the mean of two draws | 0.8075 (per fold, 600 rows) | 0.82433 |
| 10-24 | single model | M3 alone (two draws, no last step) | 0.7969 (600 rows) | 0.78766 |
| 10-24 | primary / inductive | 9B (and `zs27t`) test side recomputed on Kaggle T4 without network; blend unchanged | as above | 0.82316 / 0.82450 |
| 10-24 | inductive | Arithmetic share β = 0.4 | 0.8131 (per fold, 600 rows) | 0.82600 |
| 10-25 | primary | Arithmetic share β = 0.2 | 0.8253 (per fold, 600 rows) | 0.82433 |
| 10-26 | primary | Blend weights from all 1,500 rows | 0.81989 (per fold) | 0.82883 |
| 10-31 | primary | M3 as the mean of three draws | 0.82522 (per fold) | 0.82766 |
| 10-31 | inductive | M3 as the mean of three draws, weights from 1,500 rows | 0.81844 (per fold) | 0.82966 |
| 10-31 | primary | + 397B channel | 0.82922 (per fold) | 0.83383 |
| 10-31 | primary | 397B answers → 397B notes named by M3 | 0.83267 (per fold); as one set −0.0014 against the previous file | 0.83616 |
| 10-31 | primary | β = 0.6 and reranker (fallback form) | 0.83556 (per fold) / 0.83378 (as one set) | 0.84033 |
| built | primary | Reranker main form | 0.83756 (per fold) / 0.83456 (as one set) | — |

Other calibration points: the 27B model fine-tuned code-first scored 0.71716 on the public leaderboard with the task block
and 0.69916 without it. The channels of the inductive line combined with the transductive alignment scored 0.82750.

Differences between consecutive submissions of about ±0.005 are within the paired noise of the public leaderboard.
Several steps with a clear CV gain show a smaller or even negative public difference, and the reverse also happens. No
variant was chosen by its public score.

## 4. Single channels and their contribution

**Single-channel OOF MAP@3** (1,500 rows, recomputed from the stored OOF matrices):

| Channel | OOF MAP@3 |
|---|---:|
| Question prior (no image) | 0.5149 ± 0.011 |
| 9B fine-tuned scorer `v1` / rotation-corrected `v1r` | 0.6406 ± 0.0105 / 0.6474 |
| 27B zero-shot scorer `zs27t` / `zs27q` | 0.6730 / 0.6814 |
| 27B reason-then-name `think18sb` / `think18nb` | 0.7516 / 0.7672 |
| 27B protocol channel: stated codes only / `pr27xm` | ≈ 0.745 / 0.7459 |
| 27B fine-tuned code-first with task block (not used) | 0.7280 |
| M3, draws d1 / d2 / d3 | 0.7621 / 0.767 / 0.7701 |
| M3, mean of two / three draws | 0.7891 / 0.8020 |
| 397B protocol answers / 397B notes named by M3 (`q397m3`) | ≈ 0.744 / 0.7442 |
| Teacher model on all 1,500 rows (measurement only, not a channel) | ≈ 0.77 |

**Leave-one-out contribution.** This is the loss when one channel is removed and the blend is refitted. It was measured
in an earlier version of the primary line, which had M3 with two draws plus the 397B answers and reached 0.82811. The
current recipe differs slightly. A small leave-one-out value means that the channel's information overlaps with other
channels, not that the channel is useless.

| Channel removed | Δ MAP@3 | Δ LogLoss |
|---|---:|---:|
| M3 (two draws) | +0.0128 ± 0.0040 (inductive line +0.0173 ± 0.0044) | — |
| 397B answers | +0.0082 ± 0.0029 | — |
| Reason-then-name `think18nb` | +0.0028 ± 0.0023 (inductive line, `think18sb`: +0.0023 ± 0.0022) | +0.017 ± 0.006 |
| 9B scorer `v1r` | +0.0018 ± 0.0030 (inductive line +0.0044 ± 0.0031) | +0.029 ± 0.008 |
| Protocol channel `pr27xm` | +0.0016 ± 0.0018 (only with the alignment) | — |
| Zero-shot `zs27q` | +0.0009 ± 0.0013 (Shapley share 14 %, i.e. overlap) | — |
| Zero-shot `zs27t` (inductive line) | +0.0000 ± 0.0012 | — |

## 5. Ablations and null results

A null result holds only for the role and form in which it was measured. The table states both.

| Idea | Role tested | Result |
|---|---|---|
| Teacher rationales as M3 training targets | M3 targets | Channel −0.096 ± 0.017, worse in every group; the model copies the short teacher style and loses its own long diagnosis |
| 9B student trained on teacher pool labels | Replacing or adding to the 9B scorer | Blend +0.0001 ± 0.0032 over 5 folds; two folds alone had shown +0.009 |
| 27B fine-tuned code-first with task block | Extra channel | +0.0008 ± 0.0033 next to the protocol channel |
| M3 with same-question examples in the prompt | Channel | −0.027 ± 0.014: gains on seen codes, loses on never-seen codes |
| M3 with LoRA on all layers and one more epoch | Channel | −0.011 ± 0.013 |
| Longer M3 generation (5,200 instead of 3,200 tokens), single draw | Channel | −0.0044 ± 0.0131; averaging draws is what helps |
| Fourth M3 draw | Blend member | No gain (K = 4: 0.8247 against K = 3: 0.8252) |
| Alternative M3 targets (rationalisations of the gold code; "read first" targets) | M3 targets | No clean targets / −0.029 on one fold |
| Test-time augmentation for M3 (higher photo resolution for small handwriting) | Extra draw | Pilot below its pre-registered threshold; behaves like an ordinary additional draw; not adopted |
| Teacher task block for the reason-then-name channel | Prompt | With the block 0.684, without it 0.674 (difference −0.011 ± 0.015): almost no effect, so the inductive line loses little by omitting it |
| Other model families as diagnosticians | Generator / extra channel | Gemma 4 31B −0.001 ± 0.006 as an extra channel; Inkling-Small (276B) −0.006 ± 0.005 |
| 397B with longer thinking (+5,000 tokens) | Channel / blend | The correct candidate appears more often but is rarely chosen; blend about +0.001 to +0.003 (projection) |
| Image statistics (brightness, sharpness, layout) as features | Extra features | −0.03; image-based gating or reranking no better than placebo |
| Specialist OCR/HTR readers (PaddleOCR-VL, PP-FormulaNet, Uni-MuMER) | Reader | Number F1 0.70, below the 0.89 of the 27B's own transcription; no reader replacement |
| Higher resolution or crops for reading | Reader input | No measurable gain for the tested reader |
| Trained text verifier (cross-encoder) | Extra channel | −0.0019 to +0.0012; pre-registered gates missed |
| Pooling predictions over test rows of the same question | Extra channel (transductive) | +0.0013 ± 0.0014 / −0.0004; a planted control of +0.005 was detected, so the measurement could see a real effect |
| Class transition matrix | Calibration | −0.011 to −0.014 |
| Combination and calibration variants: temperature per channel, Dirichlet/vector calibration, kNN, gating, row-wise weights, separate `NONE` model, LambdaMART, tree rankers | Blend / second stage | At least 13 null results; temperature per channel down to −0.0041. Tree models could not detect even a planted +0.005 signal on 1,200 rows |
| Residual head on hidden states | Channel | Worse than plain code scores in every form |
| `OTH-00` as a hedge at rank 3 | Post-processing rule | −0.0022 ± 0.0013 |
| 397B notes named by M3, without the reranker | Channel replacement | Not robust across the measurement forms ([VALIDATION.md](VALIDATION.md#4-three-measurement-forms)) |
| Rotation correction for the 27B channels | Input | No rotation penalty for these channels |

## 6. Error analysis

Basis: the primary line with the reranker main form, measured as one set (MAP@3 0.83456, loss 1 − MAP@3 = 0.1654).
Standard errors are bootstrapped over questions.

**By group**

| Group | Rows | MAP@3 | Share of loss |
|---|---:|---:|---:|
| Code never seen for this question (in the other folds) | 477 | 0.661 ± 0.017 | 65.1 % |
| Code seen for this question | 434 | 0.901 | 17.3 % |
| `NONE` | 506 | 0.953 | 9.6 % |
| Question without training rows in the other folds | 83 | 0.761 | 8.0 % |

**By error type** (gold class against the top prediction)

| Error type | Rows | Share of loss |
|---|---:|---:|
| General code → specific code | 93 | 25.8 % |
| Specific → general | 70 | 20.1 % |
| Specific ↔ specific, other section of the taxonomy | 66 | 18.1 % |
| Missed error (→ `NONE`) | 47 | 12.7 % |
| False alarm (`NONE` → error) | 35 | 9.6 % |
| General ↔ general | 24 | 7.9 % |
| Specific ↔ specific, same section | 20 | 5.8 % |

**Findings**

- **Rank of the gold class:** 1st in 76.3 % of rows, 2nd in 11.1 %, 3rd in 4.7 %, 4th–5th in 3.8 %, beyond 5th in
  4.0 %.
- **The final choice fails, not the candidate generation.**
  - 86 % of the loss sits in rows where at least one channel has the gold code among its top 3.
  - In only 19 rows (7.7 % of the loss) does no channel have it among its top 5.
  - Where the gold code is at rank 2–3, the M3 notes mention it in 92.9 % of the misses.
- **Naming practice.** In about half of the loss (53 %, 90 % range 44–61 %), the models locate the error correctly but
  name it as the wording of the taxonomy suggests, not as the annotators' conventions would. This comes from a forensic
  review of 71 cases; on 15 blindly re-judged cases the agreement was κ ≈ 0.88.
- **Oracle bounds** with the channels held fixed. These are frames, not achievable gains:
  - perfect order of the top 3: +0.087; of the top 5: +0.125;
  - knowing whether the gold code is general or specific: +0.050;
  - knowing the family (`NONE` / general / specific): +0.081;
  - perfect `NONE` separation: +0.028.
- **Hardest class.** `OTH-00`, the catch-all code: MAP@3 0.471, top 1 in 36 of 57 cases, 12.2 % of the loss.
- **Calibration** of the blend probabilities before the second stage. Underconfident overall: expected 0.802 against
  observed 0.829. Overconfident on never-seen codes: 0.695 against 0.648.
- **Label limits.**
  - The second sub-collection of the training data was labelled block by block by category, and the same error type
    received different codes in different blocks; the largest such pair is `EXP-04` / `EXP-07`.
  - Roughly 15–20 % of the loss looks irreducible with the available information. This is a rough estimate from
    overlapping cells, not a bound.
- **Open estimates.** The share of the loss caused by misreading the handwriting and by choosing the wrong root among
  several findings is uncertain. Reviewers estimated reading at about 17 % (range 10–24 %); one analysis found no
  evidence that reading is the missing information in the misses. The root choice was estimated at 4–12 %.

## 7. Compute and cost

| Resource | Use | Amount (order of magnitude) |
|---|---|---|
| Rented GPUs (RunPod Secure Cloud) | Training on A40 (48 GB) and RTX PRO 6000 (96 GB); 27B inference on RTX PRO 6000; 397B test side on 4×H200 | ≈ 400–450 USD; mostly RTX PRO 6000 (≈ 118 h) and A40 (≈ 95 h) |
| Anthropic API | Teacher pool labels (≈ 31 USD), teacher measurements on training rows, one rationale experiment | ≈ 41 USD |
| Tinker API | Open-model measurements on training rows, including the training side of the 397B channel | ≈ 63 USD |
| Kaggle (free GPU quota) | T4×2: offline inference of the 9B scorer and the 4-bit 27B, second-hardware checks | within the weekly free quota |
| Local machine | 8-core CPU, 15 GB RAM, no GPU: blending, second stage, analysis, assembly | — |
| **Total paid** | | **≈ 0.5 kUSD** |

Per-component run times are listed in [TRAINING.md](TRAINING.md) and [INFERENCE.md](INFERENCE.md).
