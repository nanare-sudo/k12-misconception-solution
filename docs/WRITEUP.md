# Write-up: team Leo

*Misconception Detection in K-12 Student Math Workings (CoLearn Lab, Kaggle, 2026). Version 0.6, 2026-11-21. Details:
[METHOD.md](METHOD.md), [RESULTS.md](RESULTS.md) and [VALIDATION.md](VALIDATION.md). Reproduction:
[VERIFICATION.md](VERIFICATION.md). Internal names: [GLOSSARY.md](GLOSSARY.md).*

## Approach

Each row is a photo of a handwritten working with its question and answer key. The target is the root misconception, one
of 75 codes or `NONE`, scored with MAP@3. Several vision-language models read the photo. Each model set-up, a channel,
returns a probability for every class, mostly by scoring every code exactly behind the prompt. A small stack combines
the channels.

- Channels: a Qwen3.5-9B scorer fine-tuned with LoRA. Qwen3.8-27B zero-shot scorers that see labelled workings of other
  students for the same question. The 27B reasoning before naming: it writes a diagnosis, then every code is scored
  behind that text. A protocol channel: the 27B under a fixed marking protocol, without any labelled row. M3, our best
  single model: the 27B fine-tuned with LoRA on the gold labels to reason, state the root finding and name the code. The
  mean of three of its generations reaches 0.802 out of fold. Finally, notes of Qwen3.5-397B-A17B in the same protocol,
  behind which M3 scores the codes.
- Combination: probability floors for the channels that generate text, so that one confident wrong channel cannot veto
  the right class. A log-linear blend with an arithmetic share, with weights fitted on LogLoss. A label-free
  class-marginal alignment over the whole test set, which relies on the split being stratified by root code. A
  conditional-logit reranker over the five best codes; it never replaces a `NONE` that the blend puts first. `REL-01`
  for blank photos.
- Validation: five stratified folds, a paired bootstrap over questions, decision rules fixed in advance, copy null
  points and placebos. The late steps were confirmed on 20 untouched meta-splits, which are new stratified splits of the
  out-of-fold rows. We did not use the public leaderboard to choose between variants.

## What worked

CV is out-of-fold MAP@3 on the 1,500 training rows, with the last step per fold. The last three rows also show it over
all rows as one set, as on the test set. Public scores come from 1,000 test rows; differences of ±0.003 to ±0.005
between consecutive submissions are within their noise.

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
| + 397B notes, scored by M3 | 0.8327 | 0.83616 |
| + reranker with NONE guard and interactions; M3 behind the protocol notes | 0.8426 / 0.8382 | 0.84116 |
| Submission B: four M3 draws | 0.8436 / 0.8380 | 0.84450 |
| Submission A: B + nine reranker columns + a fifth M3 draw at 2,048 px on the rows with small characters | 0.8426 / 0.8404 | 0.84266 |

The three final steps were decided over 20 untouched meta-splits, paired: the fourth M3 draw +0.0011 ± 0.0002, the extra
columns +0.0017 ± 0.0002 and the fifth draw +0.0008 ± 0.0002. On the original split their single values differ by less
than their noise.

- Reasoning before naming was the largest single step (+0.051 ± 0.006).
- Averaging generations beat longer single generations. Single M3 draws reach 0.750–0.777, the mean of two 0.789 and of
  three 0.802.
- Channels that know different things help each other. Same-question examples pull towards codes already seen for the
  question. The protocol channel and M3 see no labelled row and gain where the code is new; there, M3 is 0.041 ahead of
  the reasoning channel.
- A component has to be judged in its final role. The protocol channel names the five general codes about seven times
  too often. Without the alignment it adds +0.002 ± 0.004, with it +0.010 ± 0.004.

## What did not work

Each null result holds only for the role in which we tested it.

- Readers and larger photos: specialist OCR models read the numbers with an F1 of 0.70, against 0.89 for the 27B. Photos
  at 2,048 px lifted single channels by +0.02 to +0.03, but not the line (as a reranker column −0.0018). M3 retrained at
  2,048 px was no better on one fold (+0.004 ± 0.014).
- Other models in the same roles: Gemma 4 31B: −0.001 ± 0.006 as reasoning generator, +0.00001 ± 0.00022 in the
  protocol. Inkling-Small: −0.006 ± 0.005. gpt-oss-20b judging our frozen notes: −0.0003 ± 0.0002. GLM-4.1V-9B and
  Kimi-VL-A3B: channels at 0.42 and 0.33, against 0.75.
- M3 variants: examples of the same question in the prompt gave −0.027 ± 0.014, LoRA on all layers −0.011 ± 0.013, and a
  novelty weight in the loss −0.024 ± 0.014 on one fold.
- More combination: we tried at least 13 calibration and stacking variants, among them Dirichlet calibration, kNN,
  gating, LambdaMART and trees. A class transition matrix gave −0.011 to −0.014. Pooling over test rows of the same
  question gave +0.0013 ± 0.0014, although the same test found a planted +0.005. Annotation-practice features gave
  +0.0008, half of which also appeared with shuffled rows.

Most of the remaining loss lies on codes never seen for their question (65 %) and between general and specific codes.
The annotators themselves assign these inconsistently: comparable workings share the code in 0.53 of pairs for general
codes and in 0.83 for specific ones.

## How we used the unlabelled set

1. Teacher pseudo-labels. During development, Claude Haiku 5.5 (Anthropic API, batch mode, effort `xhigh`) labelled all
   50,000 pool workings from photo, question and answer key. This gave 49,979 usable labels, about 72 % of them `NONE`,
   for about 31 USD. The prompt held no labelled row and no annotator note. On 120 training rows the notes had raised a
   teacher from 0.759 to 0.838, because they act as hidden same-question examples. No test row was ever sent to an API.
2. Task block. Per question, a summary of the pool labels goes into two prompts of the primary line. It gives the share
   of correct workings and up to eight codes, each with its frequency, typical wrong answers and a note. It lifted the
   blend of the 9B and the zero-shot scorer from 0.688 to 0.711. On the 27B zero-shot channel it shifted mass: +0.080 on
   codes never seen for the question, −0.042 on codes already seen. With M3 in the line, the teacher-derived part was
   worth +0.004 ± 0.005.
3. Orientation network. A small rotation classifier decides which photos the 9B scorer reads upright. It learns from
   rotations that we apply to 10,000 pool photos and the training photos, so it needs no labels. On the 124 rotated
   training rows the 9B scorer gained +0.108 ± 0.024.
4. Not adopted. A 9B student trained on the pool labels: +0.0001 ± 0.0032 over five folds, although two folds alone had
   shown +0.009. M3 trained on teacher rationales of pool rows: the channel lost 0.096 ± 0.017, because it copied the
   teacher's short style. Passing teacher codes on from pool workings with the same question and read answer: −0.0001 ±
   0.0001.

## Final submissions and reproducibility

Both final submissions come from one line, and each of their steps was decided on cross-validation. A was submitted on
2026-11-14 (public 0.84266). B was submitted on 2026-11-08 (public 0.84450); it is A without its two last components.
`reproduce.sh` rebuilds both files byte for byte on a CPU from the frozen model outputs in about two minutes. We have
also rebuilt both on a Kaggle CPU with internet disabled. `tools/gpu_inference.sh` recomputes the test side on GPUs from
our trained weights.

Every model used at inference time is open-weight (Apache 2.0). Two commercial APIs were used during development only,
on training and unlabelled rows, never on a test row. The final submissions use two frozen sets of their outputs, both
delivered. The Claude Haiku 5.5 pool labels (Anthropic API) feed the task block. The 397B's answers on the training rows
(Tinker sampling API), and the notes made from them, enter the fitted weights and, in A, the training side of six
reranker columns. On the test side the largest model, Qwen3.5-397B-A17B, runs locally in FP8 on 4 × H200.

Text generation is not deterministic, so a full rerun gives a new draw of every text. We expect it within about ±0.003
MAP@3 (1 sd) of the submitted private score; [VERIFICATION.md §8](VERIFICATION.md#8-what-cannot-be-rerun-exactly) lists
what cannot be rerun exactly.
