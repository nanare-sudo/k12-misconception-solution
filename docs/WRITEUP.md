# Write-up: team Leo

*Misconception Detection in K-12 Student Math Workings (CoLearn Lab, Kaggle, 2026). Version 0.6, 2026-11-21. Details:
[METHOD.md](METHOD.md), [RESULTS.md](RESULTS.md) and [VALIDATION.md](VALIDATION.md). Reproduction:
[VERIFICATION.md](VERIFICATION.md). Internal names: [GLOSSARY.md](GLOSSARY.md).*

## Approach

Each row is a photo of a handwritten working with its question and answer key. The target is the root misconception,
one of 75 codes or `NONE`, scored with MAP@3. Several vision-language models read the photo. Each model set-up, a
channel, returns a probability for every class, mostly as the exact log-probability of every code after the prompt.

- Channels: a Qwen3.5-9B scorer fine-tuned with LoRA. Untrained Qwen3.8-27B scorers that see, in the prompt, up to four
  labelled workings of other students for the same question. The 27B reasoning channel: the untrained 27B writes a
  diagnosis, then every code is scored conditioned on that text. A protocol channel: the 27B under a fixed marking
  protocol, without any labelled row. M3, our best single model: the 27B fine-tuned with LoRA on the gold labels to
  reason, state the root finding and name the code. Last, notes of Qwen3.5-397B-A17B under the same protocol, on which
  M3 ranks the codes.
- Combination: probability floors for the channels that generate text, so that one confident wrong channel cannot veto
  the right class. A log-linear blend with weights fitted on LogLoss, mixed with the weighted arithmetic mean of the
  same channels at a share of 0.6. A label-free alignment of the class shares over the whole test set, which relies on
  the split being stratified by root code. A conditional-logit reranker over the five best codes, which never replaces a
  `NONE` that the blend puts first. `REL-01` for blank photos.
- Validation: five stratified folds, a paired bootstrap over questions and decision rules fixed in advance, with
  zero-weight checks, duplicate-draw controls and placebos. The late steps were confirmed on 20 untouched meta-splits,
  which are new stratified splits of the out-of-fold rows. No component was chosen by its public score.

## What worked

CV is out-of-fold MAP@3 on the 1,500 training rows, with the alignment applied inside each fold. The last three rows
also show it with the alignment over all rows as one set, as on the test set. Public scores come from 1,000 test rows;
differences of ±0.003 to ±0.005 between consecutive submissions are within their noise.

| Step | CV | Public |
|---|---:|---:|
| Question prior, no image | 0.5149 | 0.53083 |
| Fine-tuned 9B scorer | 0.6406 | 0.66850 |
| + untrained 27B scorer, + 27B reasoning channel | 0.7672 | 0.76666 |
| + class-marginal alignment over the test set | +0.005 to +0.008 (paired difference) | 0.77600 |
| + rotation correction, second reasoning draw, wider gate, `REL-01` rule | 0.7951 | 0.79750 |
| + 27B protocol channel | 0.8053 | 0.80233 |
| + exact ranking conditioned on the protocol notes | 0.8096 | 0.80733 |
| + M3, one draw | 0.8143 | 0.82066 |
| M3 as the mean of three draws; deployment weights fitted on all 1,500 rows | 0.8252 | 0.82766 |
| + 397B notes, ranked by M3 | 0.8327 | 0.83616 |
| + reranker with NONE guard and interactions; M3 ranking of the protocol notes | 0.8426 / 0.8382 | 0.84116 |
| Submission B: four M3 draws | 0.8436 / 0.8380 | 0.84450 |
| Submission A: B + nine reranker columns + a fifth M3 draw at 2,048 px on the rows with small characters | 0.8426 / 0.8404 | 0.84266 |

The three final steps were decided over 20 untouched meta-splits, paired: the fourth M3 draw +0.0011 ± 0.0002, the extra
columns +0.0017 ± 0.0002 and the fifth draw +0.0008 ± 0.0002. On the original split their single values differ by less
than their noise.

- Reasoning before naming was the largest single step (+0.051 ± 0.006).
- Averaging generations beat longer single generations. Single M3 draws reach 0.750–0.777, the mean of two 0.789 and of
  three 0.802.
- Channels that know different things help each other. The protocol channel and M3 see no labelled row and gain where
  the code is new for the question; there, M3 is 0.041 ahead of the reasoning channel.
- A component has to be judged in its final role. The protocol channel names the five general codes about seven times
  too often. Without the alignment it adds +0.002 ± 0.004, with it +0.010 ± 0.004.

## What did not work

Each null result holds only for the role in which we tested it.

- Readers and larger photos. Specialist OCR models read the numbers with an F1 of 0.70, against 0.89 for the 27B. Photos
  at 2,048 px lifted single channels by +0.02 to +0.03 but cost −0.0018 as a reranker column. They helped only as a
  fifth M3 draw on the rows with small characters (+0.0008, submission A).
- Other models in the same roles: Gemma 4 31B −0.001 ± 0.006 as reasoning generator, Inkling-Small −0.006 ± 0.005,
  gpt-oss-20b as a judge of our notes −0.0003 ± 0.0002. GLM-4.1V-9B and Kimi-VL-A3B reached 0.42 and 0.33 as channels,
  against 0.75.
- M3 variants as channels: examples of the same question in the prompt −0.027 ± 0.014, LoRA on all layers
  −0.011 ± 0.013 and, on one fold, a novelty weight in the loss −0.024 ± 0.014.
- More combination: at least 13 calibration and stacking variants, among them Dirichlet calibration, kNN, gating,
  LambdaMART and trees, gave nothing. Pooling over test rows of the same question gave +0.0013 ± 0.0014, although the
  same test found a planted +0.005.

Most of the remaining loss lies on codes never seen for their question (65 %) and between general and specific codes.
The annotators themselves assign these inconsistently: comparable workings share the code in 0.53 of pairs for general
codes and in 0.83 for specific ones.

## How we used the unlabelled set

1. Teacher pseudo-labels. During development, Claude Haiku 5.5 (Anthropic API, batch mode, effort `xhigh`) labelled all
   50,000 pool workings from photo, question and answer key. This gave 49,979 usable labels, about 72 % of them `NONE`,
   for about 31 USD. The prompt held no labelled row and no annotator note: on 120 training rows the notes had raised a
   teacher from 0.759 to 0.838, because they act as hidden same-question examples.
2. Task block. Per question, a summary of the pool labels goes into two prompts: the share of correct workings and up
   to eight codes with frequency, typical wrong answers and a note. It lifted the blend of the 9B scorer and the
   untrained 27B scorer from 0.688 to 0.711; with M3 in the pipeline, the teacher-derived part was worth +0.004 ± 0.005.
3. Orientation network. A small rotation classifier, trained on rotations that we apply to 10,000 pool photos and the
   training photos, decides which photos the 9B scorer reads upright. On the 124 rotated training rows the 9B scorer
   gained +0.108 ± 0.024.
4. Not adopted. A 9B student trained on the pool labels gave +0.0001 ± 0.0032 over five folds. M3 trained on teacher
   rationales of pool rows lost 0.096 ± 0.017 as a channel, because it copied the teacher's short style.

## Final submissions and reproducibility

Both final submissions come from one pipeline; B is A without its two last components. `reproduce.sh` rebuilds both
files byte for byte on a CPU from the frozen model outputs in about two minutes. A full rerun from our weights gives a
new draw of every generated text, which we expect within about ±0.003 MAP@3 (1 sd) of the submitted private score
([VERIFICATION.md](VERIFICATION.md)).

All base models used at inference time are open-weight models under Apache 2.0. Two commercial APIs were used during
development only, never on a test row, and their frozen outputs are delivered. The Claude Haiku 5.5 pool labels
(Anthropic API) feed the task block. The 397B answers on the training rows (Tinker sampling API) enter the fitted
weights and, in A, the training rows of six reranker columns.
