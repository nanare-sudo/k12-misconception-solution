# Write-up: team Leo

*Misconception Detection in K-12 Student Math Workings (CoLearn Lab, Kaggle, 2026). Details: [METHOD.md](METHOD.md),
[RESULTS.md](RESULTS.md), [VALIDATION.md](VALIDATION.md); reproduction: [VERIFICATION.md](VERIFICATION.md).*

## Approach

Each row is a photo of a handwritten working with its question and answer key; the target is the root misconception (75
codes or `NONE`), scored by MAP@3. Several vision-language **channels** each return a full distribution over all classes,
mostly by exact scoring of every code behind a prompt, and a small stack combines them.

- **Channels.** A LoRA-tuned Qwen3.5-9B scorer; Qwen3.8-27B zero-shot scorers with same-question examples; the 27B
  *reasoning before naming* (a diagnosis first, then every code is scored behind it); a *protocol channel*, the
  27B under a fixed marking protocol with no labelled row in the prompt; **M3**, the 27B fine-tuned with LoRA on the gold
  labels to reason, state the root finding and name the code (best single model: 0.802 out of fold for the mean of three
  generations); and Qwen3.5-397B-A17B notes in the same protocol, named by M3.
- **Combination.** Probability floors; a log-linear blend with an arithmetic share, weights fitted on LogLoss; a
  class-marginal alignment over the whole test set (the split is stratified by root code); a conditional-logit reranker
  over the top five candidates that never replaces a `NONE` top of the blend; `REL-01` for blank photos.
- **Validation.** Five stratified folds, paired bootstrap over questions, decision rules fixed in advance, copy null
  points and placebos; the late steps were confirmed on 20 untouched re-splits. The public leaderboard was never used to
  choose.

## What worked

CV: out-of-fold MAP@3 on the 1,500 training rows, last step per fold (last three rows also as one set, as on the test
set). Public: 1,000 test rows; consecutive differences of ±0.003 to ±0.005 are within the noise.

| Step | CV | Public |
|---|---:|---:|
| Question prior, no image | 0.5149 | 0.53083 |
| Fine-tuned 9B scorer | 0.6406 | 0.66850 |
| + 27B zero-shot scorer, + 27B reasoning before naming | 0.7672 | 0.76666 |
| + class-marginal alignment over the test set | +0.005 to +0.008 | 0.77600 |
| + rotation correction, second reasoning draw, wider gate, `REL-01` rule | 0.7951 | 0.79750 |
| + 27B protocol channel | 0.8053 | 0.80233 |
| + exact ranking behind the protocol notes | 0.8096 | 0.80733 |
| + M3, one draw | 0.8143 | 0.82066 |
| M3 as the mean of three draws, blend fitted on all rows | 0.8252 | 0.82766 |
| + 397B notes named by M3 | 0.8327 | 0.83616 |
| + reranker with NONE guard and interactions; M3 behind the protocol notes | 0.8426 / 0.8382 | 0.84116 |
| **Submission B:** four M3 draws | 0.8436 / 0.8380 | 0.84450 |
| **Submission A:** B + nine reranker columns from frozen protocol answers + a fifth M3 draw at 2,048 px on small handwriting | 0.8426 / 0.8404 | 0.84266 |

The three final steps were decided over 20 untouched re-splits (paired): fourth M3 draw +0.0011 ± 0.0002, extra columns
+0.0017 ± 0.0002, fifth draw +0.0008 ± 0.0002. On the original split their single values differ by less than their noise.

- **Reasoning before naming** was the largest step (+0.051 ± 0.006). **Averaging generations** beat longer single
  generations: single M3 draws reach 0.762–0.770, the mean of two 0.789, of three 0.802.
- **Channels that know different things.** Same-question examples pull towards codes already seen for the question; the
  protocol channel and M3 see no labelled row and gain where the code is new (M3 +0.041 over the reasoning channel there).
- **Judging a component in its deployed role.** The protocol channel names general codes about seven times too often:
  without the alignment it adds +0.002 ± 0.004, with it +0.010 ± 0.004.

## What did not work

Null results hold only for the tested role.
- **Readers and larger photos.** Specialist OCR models read the numbers with F1 0.70, against 0.89 for the 27B. Photos at
  2,048 px lifted single channels by +0.02 to +0.03 but not the line (as a reranker column −0.0018); M3 retrained at
  2,048 px was no better on one fold (+0.004 ± 0.014).
- **Other models** in the same roles: Gemma 4 31B (−0.001 ± 0.006 as reasoning generator, +0.00001 ± 0.00022 in the
  protocol), Inkling-Small (−0.006 ± 0.005), gpt-oss-20b judging our frozen notes (−0.0003 ± 0.0002), GLM-4.1V-9B and
  Kimi-VL-A3B (channels 0.42 and 0.33 against 0.75).
- **M3 variants:** same-question examples in the prompt (−0.027 ± 0.014), LoRA on all layers (−0.011 ± 0.013), a novelty
  weight in the loss (−0.024 ± 0.014, one fold).
- **More combination:** at least 13 calibration and stacking variants (Dirichlet calibration, kNN, gating, LambdaMART,
  trees); a class transition matrix (−0.011 to −0.014); pooling over test rows of the same
  question (+0.0013 ± 0.0014, although a planted +0.005 was detected); annotation-practice features (+0.0008, half of it
  also with shuffled rows).

Most remaining loss lies on codes never seen for their question (65 %) and between general and specific codes, which the
annotators themselves assign inconsistently (comparable workings share the code in 0.53 of pairs for general codes, 0.83
for specific ones).

## How we used the unlabelled set

1. **Teacher pseudo-labels.** Claude Haiku 5.5 (Anthropic API, batch mode, effort `xhigh`) labelled all 50,000 pool
   workings from photo, question and answer key (49,979 usable labels, about 72 % `NONE`; about 31 USD). The prompt holds
   **no labelled row and no annotator note**: on 120 training rows the notes raised a teacher from 0.759 to 0.838, because
   they act as hidden same-question examples. No test row was ever sent to an API.
2. **Task block.** Per question, the pool labels are summarised (share of correct workings; up to eight codes with their
   frequency, typical wrong answers and a note) in the prompts of the 27B zero-shot scorer and one reasoning draw. It
   lifted the blend of the 9B and the zero-shot scorer from 0.688 to 0.711 by shifting mass (+0.080 on codes never seen
   for the question, −0.042 on seen ones). With M3 in the line, the teacher-derived part was worth +0.004 ± 0.005.
3. **Orientation network.** 10,000 pool photos, with the training photos, trained a self-supervised rotation classifier;
   turning photos upright raised the 9B scorer by +0.108 ± 0.024 on the 124 rotated training rows.
4. **Not adopted:** a 9B student trained on the pool labels (+0.0001 ± 0.0032 over five folds; two folds alone had shown
   +0.009); M3 trained on teacher rationales of pool rows (channel −0.096 ± 0.017: it copied the teacher's short style);
   propagating teacher codes from pool workings with the same question and read answer (−0.0001 ± 0.0001).

## Final submissions and reproducibility

Both final submissions come from one line and were chosen by cross-validation: **A** (2026-11-14, public 0.84266) and **B**
(2026-11-08, public 0.84450), which is A without its two last components. Both were rebuilt byte for byte on a CPU without
network access from the frozen model outputs. Every model that runs at inference time is open-weight (Apache 2.0); the
commercial API served only as a development-time teacher for the frozen pool labels. The largest model, Qwen3.5-397B-A17B,
runs in FP8 on 4×H200. What cannot be rerun exactly, and how close a rerun comes: [VERIFICATION.md](VERIFICATION.md).
