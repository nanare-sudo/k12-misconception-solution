# Results

*Status: 2026-11-15 (version 0.4). Cross-validated (CV) values are out-of-fold MAP@3 on the 1,500 labelled rows; "±" is
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

| Line and form | CV per fold | CV without last step | CV as one set | Public LB |
|---|---:|---:|---:|---:|
| **Primary submission A** (`primary_m3x5h`): five M3 draws, the fifth with the photo at 2,048 px on the rows with small script, and nine extra second-stage columns ⁵ | 0.84256 | — | 0.84044 | 0.84266 |
| Intermediate form (`primary_cols`): four M3 draws and the nine extra columns ⁵ | 0.84411 | — | 0.84067 | 0.84283 |
| **Primary submission B** (`primary_m3x4`): four M3 draws | 0.84356 | 0.83633 | 0.83800 | 0.84450 |
| Primary line, final form with the M3 ranking behind the protocol notes (`m3px27xm`) (primary candidate of version 0.3) | 0.84256 | 0.83456 | 0.83822 | 0.84116 |
| Primary line, final form with `pr27xm`: I1 form with the minor-code × mechanical-match column | 0.84089 | 0.83433 | 0.83978 | 0.84100 |
| Primary line, I1 form: guard form with row × candidate interactions | 0.84078 | 0.83322 | 0.83867 | 0.84150 |
| Primary line, guard form: reranker main form with NONE guard and general-code identifiers | 0.83889 | 0.83522 | 0.83767 | 0.84200 |
| Primary line, reranker main form | 0.83756 | 0.83011 | 0.83456 | 0.84166 |
| Primary line, reranker fallback form ¹ | 0.83556 | 0.82989 | 0.83378 | 0.84033 |
| Primary line before the reranker (same channels, β = 0.2) | 0.83267 | 0.81844 | 0.82533 | 0.83616 |
| **Safety line, reranker with note mentions, example similarity and regex note columns** (current safety candidate) ⁴ | 0.83156 | — | 0.82844 | 0.83716 |
| Safety line, reranker with note mentions and example similarity ⁴ | 0.83111 | — | 0.82744 | 0.83750 |
| Safety line, alignment over the test set, reranker with interactions (variant b) ³ | 0.82778 | 0.82411 | 0.82167 | 0.83516 |
| Safety line, alignment over the test set, reranker without interactions (variant a) ³ | 0.82556 | 0.82267 | 0.82122 | — |
| Safety line, strictly inductive form with second stage and NONE guard (fallback) | 0.82333 | — | — | 0.83183 |
| Safety line, strictly inductive form without second stage | 0.81844 | — | — | 0.82966 |
| Best single model: M3, mean of three draws | 0.8020 | — | — | — |

¹ With the reranker always applied, as deployed. With the nested abstention gate the fallback form scores 0.8341 per fold
and 0.8317 as one set.

—: the strictly inductive form is evaluated per fold, which matches its deployment with fixed weights per row; in the
other forms only the gain of its second stage is reported ([VALIDATION.md](VALIDATION.md#54-validation-of-the-second-stage-components)).
The single model has no last step.

³ With the nested abstention gate and the NONE guard, as the pre-registered rule of the safety line is written; its
deployment form is as one set. The deployment fit on all 1,500 rows does not abstain.

⁴ With the nested abstention gate and the NONE guard; the gate abstains in no fold, so the values equal the forced form. These
two forms were decided by the mean of the per-fold and the as-one-set difference; the form without the last step was not
computed. As one set with the gate the predecessor (variant b) reaches 0.82167 because its gate abstains in one fold; forced it
reaches 0.82433.

⁵ On the original meta-split; the steps were decided over 20 untouched meta-splits each (below and
[VALIDATION.md](VALIDATION.md#8-final-submissions-version-04)), on which their gains are positive although the original split
shows a small loss for the fifth draw. The single values of the original split differ by less than their noise.

- **Final submissions (paired, ½(F+M) unless stated).** Four M3 draws over three (submission B over the version-0.3
  candidate), fair-bank design over 20 meta-splits: +0.00109 ± 0.00015 (20 of 20 positive), copy null point −0.00036; on 20
  further meta-splits +0.00137. Nine extra columns over submission B: +0.00169 ± 0.00015, +0.00177 ± 0.00025 and
  +0.00178 ± 0.00025 on three disjoint sets of 20 meta-splits; original split +0.0016 (as one set +0.0027). Fifth draw with
  the large photo on the listed rows over the intermediate form: +0.00083 ± 0.00017, as one set +0.00108 ± 0.00030 over 20
  meta-splits; original split −0.0009 (as one set −0.0002); best copy null point (d3 twice) +0.0009.
- **M3 ranking behind the protocol notes** (`m3px27xm` instead of `pr27xm`) over the final form with `pr27xm`: ½(F+M) +0.00006
  (per fold +0.0017, as one set −0.0016); over 20 new meta-splits +0.00013 ± 0.00018, above the mean of four controls of the
  same role (−0.00045). Without the 397B channel: ½(F+M) +0.0019, over 20 new meta-splits +0.0010 ± 0.0002. The channel alone
  reaches 0.7551 against 0.7459 for `pr27xm`.
- **Regex-exact note columns** in the safety line over the form with note mentions and example similarity: ½(F+M) +0.00072
  ± 0.00108 against a placebo mean of −0.00012; full draw swap over four M3 draw sets +0.00056 (4 of 4 positive); over 20 new
  meta-splits +0.00087 ± 0.00022; risk rule P(θ < −0.002) = 0.0007 / 0.0010.
- **Note mentions and example similarity** in the safety line over variant b: ½(F+M) +0.0032 ± 0.0015 against a placebo mean
  of +0.0005; over 20 new meta-splits +0.0028 ± 0.0002, positive in 20 of 20; risk rule P(θ < −0.002) = 0.0001.
- **Minor-code × mechanical-match column** over the I1 form: per fold +0.0001 ± 0.0005, as one set +0.0011 ± 0.0006,
  ½(F+M) +0.0006 against a placebo mean of −0.0001; over 20 new meta-splits +0.0005 ± 0.0001, positive in 19 of 20.
- **Row × candidate interactions** over the guard form: per fold +0.0019 ± 0.0011, as one set +0.0010 ± 0.0010, ½(F+M)
  +0.0014 against a placebo mean of +0.0004; over 20 new meta-splits +0.0011 ± 0.0002, positive in 18 of 20.
- **Safety line, alignment over the test set:** over 20 new meta-splits variant a lies +0.0007 ± 0.0002 above the
  strictly inductive form with second stage (16 of 20 positive), and variant b +0.0007 ± 0.0001 above variant a. On the
  original meta-split the CV part of the pre-registered condition was missed by one sixth in 9,000 and overridden by a
  recorded decision ([METHOD.md](METHOD.md#99-second-stage-of-the-safety-line-with-alignment-over-the-test-set)).
  The test-side class masses after the alignment lie at L1 0.028 from the target, against 0.087 with the inductive
  alignment.
- **NONE guard and general-code identifiers** over the main form: +0.0013 ± 0.0014 per fold and +0.0031 ± 0.0015 as one
  set (+0.0051 ± 0.0016 without the last step, reported only). Over 80 new meta-splits: +0.0016 per fold and +0.0014 as
  one set. The robust part is the NONE guard (+0.0010, positive in 80 of 80 meta-splits); the identifiers add +0.0004.
- **Main form** over the fallback form: +0.0020 ± 0.0014 per fold, +0.0002 ± 0.0019 without the last step,
  +0.0008 ± 0.0016 as one set.
- **Second stage with NONE guard in the inductive line:** +0.0049 ± 0.0022 per fold, positive in 40 of 40 new
  meta-splits. The gain depends on the generated M3 texts; on the test side, whose M3 texts are independent draws,
  +0.002 to +0.003 is expected
  ([VALIDATION.md](VALIDATION.md#54-validation-of-the-second-stage-components)).
- Per fold, the primary candidate is about 0.011 ahead of the safety candidate and about 0.019 ahead of the strictly
  inductive form of the safety line.

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
| 10-24/25 | Arithmetic share β in the blend ² | +0.0019 ± 0.0016 (β = 0.2) | +0.0038 ± 0.0023 (β = 0.4) |
| 10-26 | Blend weights fitted on all 1,500 rows | 0.81989 (+0.0007 ± 0.0018) | 0.81200 (+0.0003 ± 0.0023) |
| 10-31 | M3 as the mean of three draws | 0.82522 (+0.0053 ± 0.0027) | 0.81844 (+0.0064 ± 0.0025) |
| 10-31 | + 397B channel | 0.82922 (+0.0040 ± 0.0029) | not used |
| 10-31 | 397B answers replaced by 397B notes named by M3 | 0.83267 (without last step −0.0033, as one set −0.0014) | — |
| 10-31 | β = 0.6 and conditional-logit reranker (fallback form) | 0.83556; as one set 0.83378 | see 11-01 |
| 11-01 | Reranker main form (widened note guard) | 0.83756; as one set 0.83456 | — |
| 11-01 | Test side of `v1r` on the rotation list of the released orientation network | expected effect ±0.0005; training side unchanged | expected effect ±0.0005 |
| 11-01 | NONE guard and general-code identifiers in the reranker | 0.83889; as one set 0.83767 (+0.0013 ± 0.0014 / +0.0031 ± 0.0015) | — |
| 11-01 | Second stage with the features without notes, and NONE guard | — | 0.82333 (+0.0049 ± 0.0022) |
| 11-06 | Row × candidate interactions in the reranker (expected-value rule) | 0.84078; as one set 0.83867 (½(F+M) +0.0014; 20 meta-splits +0.0011 ± 0.0002) | — |
| 11-06 | Minor-code × mechanical-match column in the reranker | 0.84089; as one set 0.83978 (½(F+M) +0.0006; 20 meta-splits +0.0005 ± 0.0001) | — |
| 11-06 | Alignment over the test set and second stage with interactions in the safety line | — | as one set 0.82167; over 20 meta-splits +0.0007 ± 0.0002 (a) and +0.0007 ± 0.0001 (b over a) |

² Measured on 1,500 rows with one M3 draw. On the 600 rows that had two M3 draws at the time, the gains were +0.0017
(β = 0.2) and +0.0056 (β = 0.4).

Notes on the table:
- The second M3 draw existed only for two folds at first, so its values use 600 rows (folds 1 and 2). From 2026-10-26
  on, all values use all 1,500 rows. Values are comparable only within the same set of rows; for example, 0.8236
  (600 rows) and 0.81989 (1,500 rows) cannot be compared.
- The reasoning channel was the largest single gain. After it, the gains came from channels that add *different
  knowledge*: the protocol channel without any labelled example, the fine-tuned reasoner M3 and the 397B model. The
  combination steps that were adopted, the arithmetic share, the reranker and its NONE guard, were each worth a few
  thousandths.
- From 2026-10-24 the test side of the 9B scorer (and of `zs27t` in the inductive line) ran on Kaggle T4 GPUs without
  network access. The blend was unchanged; the top code changed on 1–2 % of test rows.
- The switch to the released rotation list changes no CV value by construction: the training side of `v1r` and the blend
  weights stay as they were ([METHOD.md](METHOD.md#61-released-rotation-list)).

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
| 11-01 | primary | Reranker main form (widened note guard); released rotation list | 0.83756 (per fold) / 0.83456 (as one set) | 0.84166 |
| 11-01 | primary | + NONE guard and general-code identifiers | 0.83889 (per fold) / 0.83767 (as one set) | 0.84200 |
| 11-01 | inductive | Second stage with the features without notes, and NONE guard; released rotation list | 0.82333 (per fold) | 0.83183 |
| 11-06 | primary | + row × candidate interactions in the reranker | 0.84078 (per fold) / 0.83867 (as one set) | 0.84150 |
| 11-06 | primary | + minor-code × mechanical-match column | 0.84089 (per fold) / 0.83978 (as one set) | 0.84100 |
| 11-07 | safety | Alignment over the test set, second stage with interactions and NONE guard | 0.82167 (as one set, with gate) | 0.83516 |
| 11-07 | safety | + note mentions and example similarity in the second stage | 0.83111 (per fold) / 0.82744 (as one set) | 0.83750 |
| 11-07 | safety | + regex-exact columns of the M3 and protocol notes | 0.83156 (per fold) / 0.82844 (as one set) | 0.83716 |
| 11-08 | primary | M3 ranking behind the protocol notes in the mean channel; blend and second stage refitted | 0.84256 (per fold) / 0.83822 (as one set) | 0.84116 |
| 11-08 | primary (**final submission B**) | Fourth M3 draw in the M3 channel; blend and second stage refitted | 0.84356 (per fold) / 0.83800 (as one set); fair bank +0.0011 | 0.84450 |
| 11-13 | primary | + nine extra second-stage columns from frozen protocol answers (bundle test) | 0.84411 (per fold) / 0.84067 (as one set); +0.0017 to +0.0018 over three sets of meta-splits | 0.84283 |
| 11-14 | primary (**final submission A**) | Fifth M3 draw with the photo at 2,048 px on the rows with small script; blend and second stage refitted | 0.84256 (per fold) / 0.84044 (as one set); +0.0008 / +0.0011 over 20 meta-splits | 0.84266 |

Other calibration points: the 27B model fine-tuned code-first scored 0.71716 on the public leaderboard with the task block
and 0.69916 without it. The channels of the inductive line combined with the transductive alignment scored 0.82750.

Differences between consecutive submissions of about ±0.003 to ±0.005 are within the paired noise of the public
leaderboard. Several steps with a clear CV gain show a smaller or even negative public difference, and the reverse also
happens. The three files of 2026-11-01 differed from their predecessors by +0.0013, +0.0003 and +0.0022 on the public
leaderboard, against CV expectations of about +0.0010, +0.0013 to +0.0016 and +0.002 to +0.003. The three files of
2026-11-06/07 differed by −0.0005 (interactions; 26 test rows with another top code), −0.0005 (mechanical-match column;
6 rows) and +0.0033 (safety line with alignment over the test set; 70 rows), against CV expectations of about +0.001,
+0.0005 and +0.003 (the last from the better-calibrated test-side class masses). The three files of 2026-11-07/08 differed by
+0.0023 (note mentions and example similarity; 70 rows), −0.0003 (regex columns; 38 rows) and +0.0002 (M3 ranking behind the
protocol notes; 52 rows), against CV expectations of about +0.003, +0.0006 and +0.0001. The three files of 2026-11-08/14
differed by +0.0033 (fourth M3 draw; 74 rows), −0.0017 (extra columns; about 60 rows) and −0.0002 (fifth draw; 47 rows),
against CV expectations of about +0.001, +0.002 and +0.001. None of these differences is interpreted. No variant was chosen
by its public score; the two final submissions were chosen by cross-validation, with submission B kept as the second slot.

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
| 397B protocol answers / 397B notes named by M3 (`q397m3`) | 0.7432 / 0.7442 |
| Teacher model on all 1,500 rows (measurement only, not a channel) | ≈ 0.77 |

**Leave-one-out contribution.** This is the loss when one channel is removed and the blend is refitted. It was measured
in an earlier version of the primary line, which had M3 with two draws plus the 397B answers and reached 0.82811. The
values marked "inductive line" come from the inductive recipe of that time with the 397B answers added as a sixth
channel (0.82067), a variant that was never deployed. The current recipes differ slightly. A small leave-one-out value
means that the channel's information overlaps with other channels, not that the channel is useless.

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
| Test-time augmentation for M3: photo upscaled to a long side of 1,536 px on the 459 training rows with the smallest handwriting | Extra draw | +0.0062 ± 0.0158 against an ordinary draw on the same rows, below the pre-registered threshold (+0.03 at z ≥ 1.5); it behaves like an ordinary additional draw |
| Teacher task block for the reason-then-name channel | Prompt | With the block 0.684, without it 0.674 (difference −0.011 ± 0.015): almost no effect, so the inductive line loses little by omitting it |
| Other model families as diagnosticians | Generator / extra channel | Gemma 4 31B −0.001 ± 0.006 as an extra channel; Inkling-Small (276B) −0.006 ± 0.005 |
| 397B with longer thinking (+5,000 tokens) | Channel / blend | The correct candidate appears more often but is rarely chosen; blend about +0.001 to +0.003 (projection) |
| Image statistics (brightness, sharpness, layout) as features | Extra features | −0.03; image-based gating or reranking no better than placebo |
| Specialist OCR/HTR readers (PaddleOCR-VL, PP-FormulaNet, Uni-MuMER) | Reader | Number F1 0.70, below the 0.89 of the 27B's own transcription; no reader replacement |
| Higher resolution or crops for reading | Reader input | No measurable gain for the tested reader |
| Final-answer reader without the answer key (9B, no training) | Reranker feature next to the NONE guard | Passes its pre-registered first stage narrowly (missed errors 40 of 47, false alarms 22 of 35), but as a feature +0.0006 ± 0.0011 per fold and −0.0018 ± 0.0013 as one set, below the placebo 95 % quantile. Without the key the model reads the expected answer into wrong workings less often (3.7 % against 5.9 %), but on workings with a correct final answer it reads that answer correctly less often (−0.065 ± 0.016) |
| Trained text verifier (cross-encoder) | Extra channel | −0.0019 to +0.0012; pre-registered gates missed |
| Annotator-practice criteria (process versus object, stopped early, miscalculated, wrong operation, formula, no code fits), detected in the channel texts | Reranker features | Best set, selected among many tested feature sets, +0.0029 ± 0.0012 on the original meta-split (mean of the three forms), but +0.0008 over 40 new meta-splits and +0.0002 on top of the NONE guard and identifiers; half of the gain also appears with shuffled rows. The annotators' `NONE` and root rules are already followed by the channels |
| Description at the annotators' level before naming (oracle setting: gold transcription, 376 error rows of two folds) | Text in front of the 27B namer | M3 notes 0.723–0.730; M3 answering targeted questions 0.722; a stronger closed API model answering the same questions on training rows 0.706; the annotators' own explanation 0.795. Differences against the M3 notes: −0.000 ± 0.009, −0.017 ± 0.017 and +0.064 ± 0.017. The remaining gap is knowledge of the code in the annotators' wording, which a description from the photo does not reach |
| Psychology of error formation and labelling: error rules between the student's final answer and the key (sign, complement, reciprocal, powers of ten), repetition of the same error, strength of evidence and visible self-correction | Reranker features (oracles with gold transcriptions first) | Oracles +0.0002, +0.0004 and +0.0013, all below the pre-registered threshold of +0.004. Real error-rule signatures exist but are rare and fall on rows the line already solves; repeated errors carry general codes more often, not specific ones |
| Pooling predictions over test rows of the same question | Extra channel (transductive) | +0.0013 ± 0.0014 / −0.0004; a planted control of +0.005 was detected, so the measurement could see a real effect |
| Class transition matrix | Calibration | −0.011 to −0.014 |
| Combination and calibration variants: temperature per channel, Dirichlet/vector calibration, kNN, gating, row-wise weights, separate `NONE` model, LambdaMART, tree rankers | Blend / second stage | At least 13 null results; temperature per channel down to −0.0041. Tree models could not detect even a planted +0.005 signal on 1,200 rows |
| Residual head on hidden states | Channel | Worse than plain code scores in every form |
| `OTH-00` as a hedge at rank 3 | Post-processing rule | −0.0022 ± 0.0013 |
| A general code always at rank 2 | Post-processing rule | −0.0029 ± 0.0011 |
| 397B notes named by M3, without the reranker | Channel replacement | Not robust across the measurement forms ([VALIDATION.md](VALIDATION.md#4-three-measurement-forms)) |
| Rotation correction for the 27B channels | Input | No rotation penalty for these channels |

## 6. Error analysis

Basis: the primary line with the reranker main form, measured as one set (MAP@3 0.83456, loss 1 − MAP@3 = 0.1654).
Standard errors are bootstrapped over questions. The NONE guard and the identifiers of the current candidate change the
top code of 1.3 % of the OOF rows and do not alter the picture.

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

- **The final choice fails, not the candidate generation.** The gold class is at rank 1 in 76.3 % of rows, at rank 2 in
  11.1 %, at rank 3 in 4.7 % and further down in 7.9 %. 86 % of the loss sits in rows where at least one channel has the
  gold code among its top 3; in only 19 rows (7.7 % of the loss) does no channel have it among its top 5. Where the gold
  code is at rank 2–3, the M3 notes mention it in 92.9 % of the misses.
- **Never seen for the question (65 % of the loss).** Without an example of the code for its question, channels with
  question knowledge pull towards codes already seen for the question, and the blend is overconfident there: expected
  0.695 against observed 0.648 (overall it is underconfident, 0.802 against 0.829).
- **General versus specific (54 % of the loss, all directions).** In about half of the loss (53 %, 90 % range 44–61 %)
  the models locate the error correctly but name it as the wording of the taxonomy suggests, not as the annotators'
  conventions would; this comes from a forensic review of 71 cases, with κ ≈ 0.88 on 15 blindly re-judged cases. The
  annotators justify the choice consistently (general codes describe the process, specific codes the mathematical
  object), but assign general codes inconsistently: comparable workings receive the same code in 0.83 of pairs for
  specific codes and in 0.53 for general codes.
- **`NONE` versus error (22 % of the loss).** In practice `NONE` follows the correctness of the final answer, so these
  errors are mostly misjudged final answers. The reranker has no feature for this (the minor-code × mechanical-match
  column of the final primary form only concerns minor-severity candidates); the NONE guard therefore leaves `NONE`
  decisions to the blend.
- **Reading (about 17 % of the loss, range 10–24 %; the split into subtypes is uncertain).** Reading the expected
  answer from the key into a wrong working (4–6 %), layout such as fractions, stacked columns or division ladders
  (about 4 %), and misread characters (5–7 %). Neither upscaling the photo for small handwriting nor a final-answer
  reader without the key turned this into a gain (§5).
- **Hardest class.** `OTH-00`, the catch-all code: MAP@3 0.471, 12.2 % of the loss. The gold `OTH-00` is at rank 1 in
  17 of its 57 rows; the line puts `OTH-00` first in 36 rows.
- **Label limits.** The second sub-collection of the training data is ordered by category, one block per code.
  Comparable workings receive the same code in 0.83 of pairs for specific codes but only in 0.53 for general codes, and
  some specific pairs (for example `EXP-04` / `EXP-07`) are hard to separate; whether this depends on the labelling
  block cannot be determined, because the block coincides with the label. How much of the loss is irreducible with the
  available information is uncertain: an estimate from overlapping cells of this analysis gives roughly 15–20 %, an
  estimate from the consistency of the code assignment about 45 % (range 19–75 %); neither is a bound. The choice of
  the root among several findings accounts for an estimated 4–12 %.
- **Oracle bounds** with the channels held fixed. These are frames, not achievable gains: perfect order of the top 3
  +0.087, of the top 5 +0.125; knowing whether the gold code is general or specific +0.050; knowing the family (`NONE` /
  general / specific) +0.081; perfect `NONE` separation +0.028.

## 7. Compute and cost

| Resource | Use | Amount (order of magnitude) |
|---|---|---|
| Rented GPUs (RunPod Secure Cloud) | Training on A40 (48 GB) and RTX PRO 6000 (96 GB); 27B inference on RTX PRO 6000; 397B test side on 4×H200 | ≈ 400–450 USD; mostly RTX PRO 6000 (≈ 120 h) and A40 (≈ 95 h) |
| Anthropic API | Teacher pool labels (≈ 31 USD), teacher measurements on training rows, a rationale experiment and a description experiment on training rows | ≈ 43 USD |
| Tinker API | Open-model measurements on training rows, including the training side of the 397B channel | ≈ 63 USD |
| Kaggle (free GPU quota) | T4×2: offline inference of the 9B scorer and the 4-bit 27B, second-hardware checks, the key-free reader probe on training rows | within the weekly free quota |
| Local machine | 8-core CPU, 15 GB RAM, no GPU: blending, second stages, analysis, assembly | — |
| **Total paid** | | **≈ 0.5 kUSD** |

Per-component run times are listed in [TRAINING.md](TRAINING.md) and [INFERENCE.md](INFERENCE.md).
