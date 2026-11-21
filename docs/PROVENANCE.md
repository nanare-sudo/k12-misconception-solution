# Provenance of every input of the two final submissions

Reference document. To reproduce the submissions, start with [VERIFICATION.md](VERIFICATION.md).

*Version 0.6. Lists every frozen file that the build of the two final submissions reads, where it came from, and whether
its production step can be repeated without network access. A second part lists the training chain: the inputs of the
trained components, which the assembly does not read but which a retraining needs.*

## 0. Scope and conventions

**The two final submissions** (both from the primary line; the build of A runs the build of the intermediate form, and that
runs the build of B, as their first steps):

| Name | Content | File (bytes, SHA-256) |
|---|---|---|
| **Primary submission B** (`primary_m3x4`) | Six channels (`v1r`, `zs27q`, `think18nb`, `m3px27xm`, `m3x4_full` = mean of four M3 draws, `q397m3`), log-linear blend with arithmetic share, class-marginal alignment over the test set, second stage with NONE guard and the minor-code × mechanical-match column, `REL-01` rule | 115,259 bytes, `76da786dfdbe6a6ddcbcc93070f385e18870cb5cdaea895d92104aa3565f9973` |
| Intermediate form (`primary_cols`; submitted, not a final slot) | Submission B with nine additional columns from eight sources at the end of the second stage's main model: five columns combining the protocol answers of our 27B and of the 397B, one 397B-only column, two columns from the 27B protocol answer's own final-answer check, one column from the taxonomy sections seen per topic | 115,268 bytes, `53ec296976834959caea500217aa8e01904fbdc8be40a0542f63a462e4251926` |
| **Primary submission A** (`primary_m3x5h`) | The intermediate form with one change: the M3 channel `m3x5h` is per row the mean of the draws that cover the row — d1–d4 on all rows, a fifth draw h (M3 with the photo at long side 2,048 px) on the rows whose median character height is at most 20 px (training 1,001 of 1,500 rows, test 2,459 of 3,498); blend and second stage refitted | 115,293 bytes, `4505c15950167650a66cdc60eb9c153eebfb9ed6192277bd283045f3797354e7` |

**What is listed.** The union of the two input sets that were traced on the build path (every file opened or checked
by the build scripts, recorded with size and SHA-256): **494 files, 2.66 GB** (submission A 490 files, submission B 405
files; 401 in common; the four files only in B's set are B's own deployment files, which A's build writes itself). 155 of
them are derived from test rows. Both files were rebuilt byte for byte from exactly these files on a CPU without network
access, with the internal code version from which this repository was translated. With the code of this repository they
were rebuilt in two fresh copies (2026-11-17) and once more from a fresh clone (2026-11-20), each from the private input
package ([VERIFICATION.md](VERIFICATION.md#51-assemble-cpu)). The base models are not files of the input sets; they are
listed in section 3 and in [weights_manifest.json](weights_manifest.json).

**Columns of the tables.**
- *Rows*: training rows (`train.csv`, 1,500), pool rows (`unlabelled.csv`, 50,000) or test rows (`test.csv`, 3,498).
- *Offline repeatable*: **verified** = the step ran without network access; **possible** = the step uses only
  open-weight models or CPU code and has an offline mode, but it has not been run without network access; **no** = needs
  a paid external API, therefore delivered frozen.
- *Deterministic*: **yes** = bit-identical on the same hardware class; **no** = a repetition gives a new draw (generated
  text) or a new model (GPU training).
- *Read by*: **assembly** (`tools/make_submission.py`), **check** (a check or refit of the build scripts that must
  reproduce a pinned value), **reference** (an earlier file compared with a recomputation), **rescore** (only needed to
  recompute a model step, not by the CPU assembly).

These 494 files are the input sets traced on our internal build; they include the competition CSV files and the M3
adapters. The private input package lists its own files in `PACKAGE_INDEX.tsv` (path, size, SHA-256, role, derived from
test rows or not; counts per role in [VERIFICATION.md §9](VERIFICATION.md#9-private-weights-and-inputs)). The build of both
submissions with the code of this repository reads the package files of the roles `assemble` and `assemble-generated`
and the competition CSV files. Paths below are relative to the repository root, where the package folders `artifacts/`
and `data/derived/` go. File names that carry internal build labels may appear under role names in this repository, with
new checksums where a text field changed.

## 1. Answers to the four control questions

**(1) Does an output of a closed model for a test row enter either submission? No.**
- The only outputs of a closed model are the teacher pseudo-labels (section 4.1): 49,979 labels, each keyed by its row
  position in `unlabelled.csv` (positions 0–49,999). The labelling tool accepts only the tables `train` and `unlabelled`
  and only the image folders `train`, `unlabelled` and `questions` (`tools/teacher_api.py`).
- They enter the submissions only as per-question summaries of **pool** rows (the "task block" in the prompts of two
  channels on the test rows) — pseudo-labels of the unlabelled data, which the host explicitly allows.
- Apart from `test.csv` itself, all test-derived inputs (154 files of the traced sets, and every test-derived file of the
  private package) were produced by our own runs of open-weight models (Qwen3.5-9B with our adapter,
  Qwen3.8-27B with and without our adapter, Qwen3.5-397B-A17B-FP8) on Kaggle notebooks or rented GPU machines, by our CPU
  steps (orientation network, blank-photo rule, character-height rule, export of notes, assembly), or are earlier
  submission files of the same kind. None of them was produced by an API.
- The open 397B model was also used through a hosted sampling API, but only for the 1,500 training rows (section 4.2).

**(2) Inputs whose producing script or configuration is not in the repository, or whose production had a manual
step.** No input was produced by hand. The following were produced by scripts that are not (yet) part of this
repository or were produced without a recorded configuration (details in section 9):
- the two combined run folders of the M3 draws d1 and d2 (combined locally from fold runs; no script exported);
- part of draw d2 (three folds) computed from a separate working copy of the code, whose model code is identical to the
  repository; its configuration was added afterwards;
- the zero-shot scorer's test runs (configuration in the repository; the code version was not recorded);
- the band rows of the reasoning gates (written by ad-hoc scripts after a recorded rule; the repository reconstructs the
  lists with `tools/replay/gates.py`, which checks the test-side lists bit for bit with the A40 outputs in the private
  package);
- the stored verdicts of the 397B parity check and of the posterior step of the M3 evaluation (scripts exist, not
  exported);
- the measurement files of the column fidelity check of the column sources (scripts exist, not exported; the build
  path of this repository does not read them);
- the evaluation files of the five-draw rule of submission A (script exists, not exported; the build path of this
  repository pins the resulting values instead of reading the files).

**(3) Training inputs that are not in the input sets** (the assembly starts behind the model outputs): the M3 training
targets, the inputs of the 9B training, the training photos and pool photos of the orientation network, the training-side
gate lists and rotation lists, and the image statistics of the character-height rule. The private package holds the M3
training targets, the training-side gate lists and rotation lists and the row lists of the protocol-answer runs; the
image statistics table is not included. See section 8.

**(4) Is every API-derived input covered by a statement of the host?** Yes; see section 4.

## 2. (a) Competition data

| Files | Rows | Producer | Read by | SHA-256 |
|---|---|---|---|---|
| `data/train.csv` | training | competition data (root CSV, English topic names) | assembly, checks | `38583e0fc020638fca6be5456df4a140d573c536eb07c6e1cacb02ce05e9175f` |
| `data/test.csv` | test | competition data | assembly | `703ec890aaa83f1e749e300ff40c376415cfe18a0b3bdb4ce14c81ecbb37e0ea` |
| `data/unlabelled.csv` | pool | competition data | rescore (task block of the teacher labels) | `94d0e67820fbf98e1926941b8ae71d2ee4e6b7fd299c95bf0d9f1bfd9e7420a8` |
| `data/competition_data/taxonomy_v1.csv` | — | competition data | assembly | `fd1b612149a1fae44bb93e14ad06738cc085ef88e8db53f2fc7550449f005523` |
| `data/competition_data/sample_submission.csv` | test | competition data | assembly (row order) | `ee7dd719d927427e375d6e2146bc7f970daf673fef9c991dc19b47b2b43de2aa` |

Photos and question diagrams (`competition_data/images/…`) are read by the model steps, not by the assembly. Licence:
competition rules (competition use only, no redistribution).

## 3. (b) Model weights

| Weights | Where | Producer | Licence | SHA-256 / revision | Read by |
|---|---|---|---|---|---|
| `Qwen/Qwen3.5-397B-A17B-FP8` | Hugging Face, pinned revision | Alibaba Cloud (Qwen) | Apache 2.0 | revision `ea5b4f81096f3901c91dea97f81324302495781d`; file list with SHA-256 in [weights_manifest.json](weights_manifest.json) | rescore (397B channel) |
| `Qwen/Qwen3.8-27B` | Hugging Face, pinned revision | Alibaba Cloud (Qwen) | Apache 2.0 | revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` | rescore (all 27B steps; base of M3) |
| `Qwen/Qwen3.5-9B` | Hugging Face, pinned revision | Alibaba Cloud (Qwen) | Apache 2.0 | revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a` | rescore (9B scorer) |
| M3 adapter, full fit (`m3_train_full/adapter/adapter_model.safetensors`) | private input package: `weights/m3_full/` (same file in `artifacts/runs/m3_fullte_rp/m3_fullte_rp/m3_train_full/`) | our LoRA training of the 27B on all 1,500 training rows (configuration `m3_fullte_rp`, job `m3_train_full`) | ours (Apache 2.0 for the code; weights trained on competition data, competition use only) | `85c34f5c396ec21ecb6de28356a631f4261264402b7a03283ad8fc168e334527` | check (checksum); rescore (all M3 test steps) |
| M3 fold adapters f0–f4 (`m3_train_f0` … `f4`) | private input package: `artifacts/runs/m3_pilot_rp/m3_pilot_rp/m3_train_f{1,2}/`, `artifacts/runs/m3_f034_rp/m3_f034_rp/m3_train_f{0,3,4}/` | LoRA training on the 1,200 rows of the other folds (`m3_pilot_rp` folds 1, 2; `m3_f034_rp` folds 0, 3, 4) | as above | f0 `b64aa0b7a7a89ddf85cf2f7c7de6b05336995a879920435f7a4f7c90e7dc1917`, f1 `3d48b2e7a81f3950e5c5d022fd0c7446499d011b100a428101eeb325bc7c8e65`, f2 `4b1b978d98b77b03c03fcc5166d4e61a30d14ce818307cfda607789bcb016532`, f3 `83bc2df70456c4ff330e29fb4c3682b4be69a4e932c017cece3ebb232e26d879`, f4 `4651477fdf50ebe49fdc015cf1bc19b5023f02b4753a95f361164b66de2ba86e` | check (checksums); training side only |
| 9B adapter `v1_full` | private input package: `weights/v1_full/` | LoRA training of the 9B on all 1,500 training rows (`ft_v1c_rp`) | as above | `321108733ee1ae9eafe14d98047eb592f2f8084df372f600aa46f4a0f0f1d9ca` | rescore (9B test runs); not read by the assembly |
| 9B fold adapters `v1_f0` … `v1_f4` | private input package (`weights/v1_folds/`) | LoRA training of the 9B on the 1,200 rows of the other folds (`ft_v1a_rp` folds 0, 2; `ft_sum1_rp` fold 1; `ft_v1b_rp` folds 3, 4) | as above | f0 `05f2aea57f07a2661e51deeb3972ecd959de83440d9579a7fbfbef7a902aaaf3`, f1 `a80afb910c4a89f54bb80a11723f51fae4373531f9e4d425cbbe007e9906783e`, f2 `bbf6040a79fdad94d6c336f113e8b3bfac926ffa78ff96c41d6fe9f445a967be`, f3 `0084a8c24538d6258b8a10566dfd74d14e545a35afb65314e9a701e07dd3446a`, f4 `7739fbcb08baf7875da13b75037a87597c702a1c9419b164754750c607145e58` | rescore of the training side (OOF of `v1`, the rotation run `v1rot_t4`) and the check job `check_f1` of the 9B test configurations (`v1_f1`); not read by the assembly |
| Orientation network (`artifacts/orient/orient_cnn.pt`, `.json`) | private input package: `weights/orient_cnn/` (same files in `artifacts/orient/`) | `tools/orient_net.py train --seed 0 --threads 2` on the CPU | as above | `83cc1224da0edbea0af640a8682ecb0b882f33b107cf89de9a7673ce19223abe` (weights), `f411f0d437208057b5cfafd29e96fcbf6cda76de95590abc468b1aff0c3ba365` (description as read by this repository; its four descriptive text fields are translated, every value equals the first stored description `5a5437d2d378fd74fcd5341cb59054b8b692886e6ee021f06c0a5e0c9b5931c1`) | check (the rotation list must be the network's output) |

Each M3 adapter has a training recipe (`sft_recipe.json`) that records the base model with its revision, the SHA-256 of
its training targets and the number of rows used. The recipes of folds 0, 3, 4 and of the full fit also record
`"pool": null` and `"teacher_derived": false`. The two pilot recipes (folds 1 and 2, `m3_pilot_rp`) were written before
these fields existed; their `targets_sha256` identifies the gold-only targets `data/derived/sft/m3_fold1_targets.jsonl`
and `m3_fold2_targets.jsonl`, which `python -I tools/make_m3_targets.py --full --check` rebuilds byte for byte. Our
internal traced input set flagged the full-fit adapter as test-derived only because the same run continued with the
first test draw. `PACKAGE_INDEX.tsv` flags it `no`: it was trained on the training rows only. GPU training is not
bit-reproducible; the delivered adapters are the ones that produced the submissions.

## 4. (c) Inputs produced through external APIs during development

### 4.1 Teacher pseudo-labels of the unlabelled pool

| | |
|---|---|
| Files | `data/derived/teacher/pool_a/labels.jsonl` (10,485 labels, `44fd2e1b9f849f5188e473a4bff1bd3f9aa1e3261a8099d4d47190bf60dc5587`), `data/derived/teacher/pool_b/labels.jsonl` (39,494 labels, `0f561932be41646526201bf89d8ff3f94f7428b127975c71dca8ac5dcbe06de2`), each with `manifest.json` (model, settings, SHA-256, share of `NONE`) |
| Rows | pool rows only: 49,979 distinct positions of `unlabelled.csv`. `pool_a`: 40 rows per question in a fixed random order (seed 7); `pool_b`: the rest |
| Producer | Claude Haiku 5.5 through the Anthropic Messages API (batch mode), thinking effort `xhigh`; prompt with the marking instructions, the taxonomy, the question, the answer key and the photo, without annotator notes, labelled rows or examples (`"notes": false`) |
| Script | `tools/teacher_api.py` (`plan` → `submit` → `collect`); cost about 31 USD |
| Offline repeatable | no (paid API); delivered frozen |
| Deterministic | no |
| Used for | the per-question task block of the zero-shot scorer `zs27q` and of reasoning draw g (`think18nb`), on training rows (OOF) and test rows: for each question, the codes the teacher found in pool rows of the same question, with counts and typical final answers |
| Read by | rescore (the assembly reads the frozen channel outputs) |

### 4.2 397B protocol answers on the training rows

| | |
|---|---|
| Files | `data/derived/open_teacher/qwen35-397b/labels.jsonl` (parsed answers, `6935614248a22da250054dd14b180fc5875832ddba7afc49c4b3da0337503280`), `…/raw.jsonl` (thinking and answer text, `7b6d6a119c8281804d09946dc743e729f1823492814a9cf8c574d432a4d7e942`); a byte-identical copy of the labels as run folder `artifacts/runs/q397_tinker/q397_tinker/proto_t/proto_labels.jsonl` |
| Rows | training rows only: all 1,500 (one first request per row, 1,341 completions); 1,496 rows with a valid answer |
| Producer | the open model `Qwen/Qwen3.5-397B-A17B` served by the Tinker sampling API (Thinking Machines; renderer `qwen3_5`, completion renderer `qwen3_5_disable_thinking`); the revision and number format used by the provider are not recorded in our files. Same protocol as the test side: thinking up to 2,500 tokens, completion without thinking up to 1,500 tokens, greedy; images as JPEG (quality 75) at the same pixel size |
| Script | `tools/open_teacher.py run` / `finish` (Tinker SDK in a separate environment); cost about 44 USD for the rows used |
| Offline repeatable | no through the API; the same protocol can be run locally with the FP8 release (as on the test side), which gives a new draw, not these answers |
| Deterministic | no |
| Used for | (i) the OOF matrix `oof_q397` (reference channel, checks); (ii) the notes of `data/derived/sft/q397m3_train_notes.jsonl`, behind which the M3 fold adapters score (OOF of `q397m3`, on which the blend weight of `q397m3` is fitted); (iii) in submission A, the training side of six second-stage columns; (iv) the reference of the 397B parity check |
| Read by | check (assembly reads only the frozen OOF matrices and the training notes) |

### 4.3 Host statements that cover these inputs (competition forum, account "CoLearn Lab")

| Thread | Date | Statement (verbatim) |
|---|---|---|
| 746450 | 2026-10-07 | "No external APIs allowed in final solution. Solution should be reproducible without internet access though you are free to use any APIs for fine-tuning, enriching your models using unlabeled data or any other useful approaches" |
| 747177 | 2026-10-08 | "Winning solution code should run without human intervention and need to be self contained including model weights, any manual or programmatic labeling done on unlabeled data we have shared." |
| 748667 | 2026-10-12 | "We fine-tuned the models mentioned in our blog using Thinking Machine's Tinker APIs. While this comes with a cost, participants may want to consider using it." |
| 751440 | 2026-10-26 | Asked whether images and question text from `unlabelled.csv` and `train.csv` may be sent to a paid commercial model API to create pseudo-labels that are included in the solution (no test image sent, solution offline): "Yes. You can do that as long as you are using them only for this competition." |
| 751440 | 2026-11-10 | "You can pseudo label unlabeled as well as labeled data provided in this competition." |
| 754757 | 2026-11-10 | "You can add any additional files to your submission but they can't be just predictions on test set directly from Claude or OpenAI." |
| 756885 | 2026-11-10 | "We agree and  both training & inference pipeline for winners would be requested and verified." |

Section 4.1 is pseudo-labelling of unlabelled data with a paid API (746450, 751440); section 4.2 is pseudo-labelling of
labelled data through the API the host recommends (748667, 751440 of 2026-11-10). No test row was sent to any API, and no
file of either section concerns a test row (754757). Both are frozen and part of the private package (747177).

## 5. (d) Our model outputs on the training rows

| Group | Files | Producer (model, hardware) | Configuration, job kind | Offline repeatable | Deterministic | Read by |
|---|---|---|---|---|---|---|
| OOF matrices and metadata | 24 files under `artifacts/oof_*.npy`, `artifacts/meta_*.json` | CPU assembly of the run outputs below (`eda/a6_assemble_oof.py`, `eda/a8_proto_oof.py`, `eda/c95_rank_mean.py`, `eda/a9_m3_draws_oof.py`, `eda/c117_q397m3_deploy.py`, `eda/c133_primary_m3px27.py`) | — | verified (rebuilt or checked by the build scripts offline) | yes | assembly (`oof_m3x4_full`, `oof_q397m3` and the metadata of the deployed channels), checks (the others) |
| M3 draws d1–d4 per fold | 127 files: `artifacts/runs/m3_pilot_rp`, `m3_f034_rp` (d1, generated after each fold's training), `m3_5f` (d1 combined), `m3long_rp` + `m3long_comb` and `cv01_m3d2_f034` (d2), `m3d3_oof_rp` (d3), `m3d4_oof_rp` (d4): notes, findings, rankings of all 74 codes, recipes | M3 fold adapters on the 27B in BF16; vLLM generation, exact ranking; RTX PRO 6000 96 GB | `configs/vlm/m3_pilot_rp.yaml`, `m3_f034_rp.yaml`, `m3long_rp.yaml`, `cv01_m3d2_f034.yaml`, `m3d3_oof_rp.yaml`, `m3d4_oof_rp.yaml`; kinds `train`, `merge`, `sft_gen`, `predict` | possible | generation no; ranking behind fixed text yes | assembly (rankings and metadata of the four draws), checks |
| Protocol channel of the 27B | 36 files: `artifacts/runs/pr27_rp` (answers, two jobs), `px27_rp` (ranking behind the notes), `m3px27_oof_rp` (M3 ranking behind the notes, five shards) | untrained 27B (BF16, vLLM) and M3 fold adapters; RTX PRO 6000 | `pr27_rp.yaml` (`proto_gen`), `px27_rp.yaml`, `m3px27_oof_rp.yaml` (`predict`, probe `proto_rank`) | possible | answers no; rankings yes | assembly (M3 rankings), checks; submission A: training side of its extra columns |
| M3 behind the 397B notes | 25 files: `artifacts/runs/q397m3_oof_rp` (five shards) | M3 fold adapters behind the notes of section 4.2; RTX PRO 6000 | `q397m3_oof_rp.yaml` | possible | yes (behind frozen notes) | assembly (metadata), checks |
| M3 draw h (photo at 2,048 px) per fold | 45 files: `artifacts/runs/m3ha_oof_rp` (folds 1, 2, 0), `m3hb_oof_rp` (folds 3, 4): generated notes `m3h_gen_f*/gen_notes.jsonl`, rankings `m3h_rank_f*/bench_proto_rank.npy` (f0 `6742b19c64a3ce3427cc4fd7055fa78df45e23b63f57be6b8413e26ce8da68b4`, f1 `c40782f2ded272ff4be329987c15ecd9fcb015acca170ae63492f057af266aef`, f2 `1b2961ef0ff0683a2502d973e2b969be27c3b0227b4eeb36b503ce3692d1686d`, f3 `189e0cd6855392b5064be52c9a03d9419854368e46ac57eedc82c9c1b9523b18`, f4 `108bb80dfedfeec00710bc666656718108ace6ad5bb7fbd44272d7161a4574c9`), recipes and merge summaries | the M3 fold adapters on the 27B in BF16, photo at long side 2,048 px in generation and ranking; vLLM generation (sampled with the settings of the model card, seed 5), exact ranking; RTX PRO 6000 96 GB | `configs/vlm/m3ha_oof_rp.yaml`, `m3hb_oof_rp.yaml` (`merge`, `sft_gen`, `bench` with probe `proto_rank`) | possible | generation no; ranking behind fixed text yes | assembly of submission A (rankings of the list rows; the build pins one digest over 42 files: the shard, generation and recipe files of the five folds and the two list files) |
| Row list "character height ≤ 20 px" (training side) | `data/derived/charsize/train_h20_rows.npy` (1,001 rows, `11afce9af96fca09032153b566ee81725f466ab927e782fcdad4324b55d18e77`), `charsize_h20.meta.json` (`17292d7b8a81108acaaa2401bf16e5b713f1347a22c2eb281da8a6fecfabec62`) | `tools/make_charsize_rows.py --max-height 20 --name h20` (CPU) on the stored per-image statistics of `eda/i5_01_image_stats.py` (median connected-component height in the photo at long side 1,024 px); the threshold is the lower two-thirds quantile of the training photos; label-free | — | possible (CPU) | yes | assembly and checks of submission A |
| 397B parity run | `artifacts/runs/q397p_rp/q397p_rp/proto_p/proto_raw.jsonl` (64 training rows, `d200e569ea9ea180f7568b2d0a86918033d7cf8c989d61c06a40a4c4edad721c`) | `Qwen3.5-397B-A17B-FP8` on 4 × H200, vLLM | `q397p_rp.yaml` (`proto_gen`) | possible | no | check (parity of the column sources; optional) |
| Frozen training-side notes and row lists | `data/derived/proto_notes/train_pr27.jsonl` (`d615b707778e1e7b685ff4a9b28c0b2b756b46aba7ce149c295f99527dea388d`), `data/derived/sft_m3px27/m3px27_train_notes.jsonl` (`907bdf35fa10ac06fb07eebef5f441d5a3eeae186193ae561ea451a457457816`), `data/derived/sft/q397m3_train_notes.jsonl` and its copy in `sft_q397m3/` (`5a8fb7358bf9a25bcd4f6b12b28c02b31e5ad345a3ee21f9d94bc182bf180d52`), `data/derived/m3x3_full_rows.npy` (`c83e7d6eb3bd6a2aa38bb7c9594f24e282d781b8f144c5f403c011b2004b6aed`), with meta files | `tools/export_proto_notes.py`, `tools/make_m3px27_notes.py`, `tools/make_q397m3_notes.py` (CPU, from the raw answers above and section 4.2) | — | verified (the protocol notes are rebuilt byte for byte by the build checks; the 397B notes by `tools/make_q397m3_notes.py` from section 4.2, which is not part of the build) | yes | assembly, checks |

The untrained 27B channels (`zs27q`, `think18nb`) and the 9B (`v1r`) enter through their OOF matrices only; their run
folders are not read by the build (section 8 lists them).

## 6. (e) Our model outputs on the test rows

No file of this section is inspected row by row; the build scripts print only aggregates and checksums.

| Channel | Files (key file and SHA-256) | Producer (model, number format, hardware) | Configuration, job kind | Offline repeatable | Deterministic | Read by |
|---|---|---|---|---|---|---|
| `v1`, `v1r` (9B scorer, plain and rotation-corrected) | 12 files in `artifacts/runs/v1te_a_t4`, `v1te_b_t4`, `v1rot_t4`; scores `test_diag.npy`: a `50d659f0f54f5be13088e569e55bef95177639f5ab6e309cdd1d4217fa3def16`, b `86c01c7f24b469254dfb93ea5be8448d8b68aa9783adfa5c81e950fc83240680`, rotated `754f6652440c1e48cade9a240074d899d8fddec39f808b1acc74360ad93b2ae1` | Qwen3.5-9B + adapter `v1_full`, FP16, Kaggle T4 × 2, Internet off | `v1te_a_t4.yaml`, `v1te_b_t4.yaml`, `v1rot_t4.yaml` (`predict`) | **verified** | yes (bit-identical on T4 × 2 in FP16) | assembly |
| `zs27q` (27B zero-shot scorer with topic and task block) | 8 files in `artifacts/runs/zs27q_test0_rp`, `zs27q_test1_rp`; scores: `fd62fd7a3d5eb9ea112f7dea7aeb59bfd0198e7029246500de4b7ae4ea928aa9`, `2bd8c912c45c7455d796c0bfb071fa4f12ba4e07e6f2a5b3aa5480e25e057601` | Qwen3.8-27B, 4-bit at load time, A40 48 GB | `zs27q_test0_rp.yaml`, `zs27q_test1_rp.yaml` (`predict`, probe `diag`); prompt with up to four labelled examples of the same question from the training rows, topic block, task block (section 4.1) | possible | scoring; repeatability on the same card not measured for this channel | assembly |
| `think18nb` (27B reasoning, draws g and n) | 12 files in `artifacts/runs/zs27v_te_rp` (draw g), `zs27v_n3_rp` (draw n): texts `gen_*/think_texts.jsonl` (g `f99f5433f8d7fac0a9c7ba7e93665bd40ed185cbe7bf49dc289c286d40220023`, n `98efbd65eec5388934e1acbf58a5a29790f821a3a6d42324176d9cf5e155f113`), scores `think_*/test_think.npy` (g `d0875a9c115fbb9114226530a5bf4c250e0c225e4f3f19c47241d535bd908e3b`, n `005e60a7a25bf7564bbe6ebddce07face163e4869398ca800d5db47896180ffc`) | untrained Qwen3.8-27B, BF16; texts by vLLM (thinking up to 1,800 tokens), exact scoring of all codes behind the text; RTX PRO 6000 | `zs27v_te_rp.yaml`, `zs27v_n3_rp.yaml` (`think_gen`, `predict`); g with task block, n without | possible | texts no; scores behind a text yes | assembly (scores), rescore (texts) |
| `m3px27xm` (protocol answers of the 27B and M3 behind their notes) | 19 files: `artifacts/runs/pr27te_rp/…/proto_te/` (answers `proto_raw.jsonl` `3e9b05623a0dbb7eb7bb9b211b662a93af3b6c85a096c70523a62dc1e35eae62`, `proto_labels.jsonl` `9abd513f8673eb0931a4138d3f91fb7844dcc68bce56e79b0e18ef5c963a8ba1`), notes `data/derived/proto_notes_test/test_pr27.jsonl` (`1b459a40278b814922bfedbff9867028f9e0e56e39ca54b6c7b471c42a833f3a`) and `data/derived/sft_m3px27te/m3px27_test_notes.jsonl` (`dddc7ce63816a60459ef7610484e141574ae47411f2ed1a5de932fc89ae02ab7`), M3 ranking `artifacts/runs/m3px27te_rp/…/test_proto_rank.npy` (`a8388edbc6cbe614f89373a9a01b2eae0c41ccd75996d3b83acd0fd8536b3b70`), untrained ranking `n1_rp` (checks only) | untrained 27B (BF16, vLLM) for the answers; M3 full-fit adapter merged into the 27B for the ranking; RTX PRO 6000 | `pr27te_rp.yaml` (`proto_gen`), `m3px27te_rp.yaml`, `n1_rp.yaml`; notes by `tools/export_proto_notes.py --allow-test` and `tools/make_m3px27_notes.py` (CPU) | possible (GPU steps); notes: verified (CPU, byte-identical) | answers no; notes yes; rankings yes | assembly; submission A also reads the answers as column sources |
| `m3x4_full` (B) / `m3x5h` (A) (M3, four draws d1–d4; A adds draw h on the list rows) | 45 files in `artifacts/runs/m3_fullte_rp` (d1), `m3d2te_rp`, `m3d3te_rp`, `m3d4te_rp` (generation `gen_notes.jsonl`, rankings `test_proto_rank.npy`: d1 `69f45474b330d9e71805c2f94033cec2743b358648996e0d180f9704b8369e73`, d2 `3304bedb5604e8548fb87ee057dd836a5d1be64790da137c1aaa3347f730e9ea`, d3 `d69460b944e8ea6e71fad3281fb9841acc314e08204df38b24153b42a21119f5`, d4 `fb86be66cab450aeacb999bf872a1043de06e685edb57c1ad971f501a1b4d5fb`); frozen texts `data/derived/sft/m3_d{1,2,3}_test_frozen_notes.jsonl`, `data/derived/sft_k2/m3_d4_test_frozen_notes.jsonl` | M3 full-fit adapter merged into the 27B (BF16); vLLM generation with the settings of the corresponding OOF draw (d1, d2 greedy; d3, d4 sampled with temperature 0.6, top-p 0.95, top-k 20, seeds 3 and 4), exact ranking; RTX PRO 6000 | `m3_fullte_rp.yaml`, `m3d2te_rp.yaml`, `m3d3te_rp.yaml`, `m3d4te_rp.yaml` (`merge`, `sft_gen`, `predict`); texts frozen by `tools/freeze_m3_texts.py` | possible | generation no; ranking behind the frozen texts yes | assembly (rankings, metadata); the second stage reads the frozen texts of d1–d3 |
| `m3x5h`, draw h (submission A only) | 18 files (16 test-derived) in `artifacts/runs/m3hte_a_rp`, `m3hte_b_rp` (two shards: every second row of the list): generated notes (`gen_notes.jsonl`: a `2147aaa36aec5746a4e0abaf3e2f5f91d61c19a658b2ec847e81e5ce0e17e11e`, b `67dc296a197ce608c07e4a5796f693e1091087a3bc8e280b237df0b2df916252`), rankings (`test_proto_rank.npy`: a `8955b4c6f79bcaadf2dbee26eac6dd38b0a608374620c10c9ae92707ee1fdf14`, b `b09e9c9f4b565729beb6acc652d74c7977294b1e07bbe7bb36097a8c7b1e3d76`), recipes, completeness files | M3 full-fit adapter merged into the 27B (BF16), photo at long side 2,048 px; generation with the settings of the OOF draw h (seed 5, sampled), exact ranking; RTX PRO 6000 | `m3hte_a_rp.yaml`, `m3hte_b_rp.yaml` (`merge`, `sft_gen`, `predict`), rows only from the test list | possible | generation no; ranking behind the texts yes | assembly of submission A (digest of all shard files pinned); the second stage does not read draw h |
| `q397m3` (397B notes, M3 behind them) | 15 files: answers `artifacts/runs/q397te_rp/…/proto_te/proto_raw.jsonl` (`824724105f1b4abb05147db26794bdd3b1b2c030725554a7126c4fba888a6493`) and `proto_labels.jsonl` (`82e29b9663f0dd2c6f32cb7e52676f4e8fa9e6e27c5b03a2d6c99e1720d27be0`); notes `data/derived/sft/q397m3_test_notes.jsonl` and copy (`4b29f057d50c7decfc3db73ddb37182d26b51326e3af05683d4665a56f20eb60`); M3 ranking `artifacts/runs/q397m3te_rp/…/test_proto_rank.npy` (`65eae425af6f4a4c4289517ab6d384fb28d1cfe8a1077d3f409b8a083a6cd398`) | `Qwen3.5-397B-A17B-FP8` on 4 × H200 (vLLM, tensor parallel 4) for the answers; M3 full-fit adapter on the 27B for the ranking (RTX PRO 6000) | `q397te_rp.yaml` (`proto_gen`), `q397m3te_rp.yaml`; notes by `tools/make_q397m3_notes.py --test-from …` (CPU) | possible (see [INFERENCE.md](INFERENCE.md#10-recomputing-the-397b-channel-on-the-test-rows)); notes verified | answers no; notes and ranking yes | assembly; submission B also reads the answers as column sources |
| Row lists | 15 files: test list of the character-height rule `data/derived/charsize/test_h20_rows.npy` (2,459 rows, `9b1a1837bc2cf399e2e221c49def4b4ae11d666b5346ecb1eddc076622c8340a`; submission A); blank photos `data/derived/blank/test_rows.npy` (`3ba899ac881a4ceb27c5204ab54dfd427bc889009670f8a81187af2b99f2db2d`), rotation list of the released network `data/derived/orient_repro/test_c80_rows.npy` (`aa6515c8bf17bf593c5c90aa90d9240e9992d84f650c5445a37a5f015dbbff1d`) and its JSON/meta files, the first-version rotation lists (checks), reasoning gates `data/derived/think_rows/test.npy` (`952e05f862f6c19ae6e42ac9b169438f4cd66f4a24825f5320756e0a89a5d8ef`), `test_tb_q.npy` (`1d4f2a25c49f75e479e0f0453952392be04c7769da0eb4bfc29c9685ba42fdff`), `test_n3.npy` and `band.meta.json` | `tools/make_charsize_rows.py` (CPU, stored image statistics, threshold from the training photos); brightness rule `tools/make_blank_list.py` (CPU); `tools/orient_net.py infer` (CPU); gates from the A40 versions of the fast channels (section 9) | — | possible (CPU). The build checks the rotation list against the checksum recorded by the stored network; the network and its lists were rebuilt bit for bit before. `tools/make_blank_list.py check` recomputes the blank-photo list and compares it with the frozen list; `tools/replay/gates.py` checks the test-side gate lists bit for bit with the A40 outputs in the private package | yes | assembly, checks |
| Earlier submissions | 9 files: seven earlier submission files and two test matrices (`artifacts/submissions/…`, `artifacts/test_…`) | earlier builds of the same scripts | — | verified (rebuilt in the same chain) | yes | reference (pinned checksums) |

## 7. (f) Fitted deployment files and stored reports

| Group | Files and SHA-256 | Producer | Offline repeatable | Deterministic | Read by |
|---|---|---|---|---|---|
| Fold array | `artifacts/folds.npy` (`fadf2771203c19d19e057d69a02413d9f01b26f85bbf76490046f2a6f8773b73`) | `tools/make_folds.py` from `configs/folds.parquet` | verified | yes | checks |
| Blend of submission B and of the intermediate form | blend file of the four-draw form (`a1aa2b5c42103f6085ecfec20877f34ee346023ed7b6c7149591f4bc9031cf64` as used for the submitted file): weights of the six channels fitted on LogLoss over all 1,500 OOF rows, reasoning-channel weight × 1.4, arithmetic share by its rule | the blend step of the four-draw build (CPU) | verified (recomputed and compared character for character) | yes | assembly |
| Blend of submission A | blend file of the five-draw form (`cf102415d86dd4598f81cd4fd0595c1616e69ee2d90b4416aa931740f0f5c80a` as used for the submitted file): the same recipe with `m3x5h` in place of `m3x4_full` | the blend step of the five-draw build (CPU) | verified | yes | assembly |
| Earlier blend files | five files (references of the chain) | earlier builds | verified | yes | reference |
| Second stage, submission B | deployment file of the four-draw form (`a799554f2c9dfc7544a6264fee46084b2cd84e8b51f2f2c6477f8e76819c85e8` as used) | the fit step of the four-draw build (conditional logit over the top five, interactions, minor-code × mechanical-match column, NONE guard; CPU) | verified (written or verified by the build) | yes | assembly |
| Second stage, intermediate form | deployment file with column sources (`06adc086cf664b5f6e3c2b718d889f99c3a15e6155c91abc39d448b1e499f8de` as used): the form of B plus the column definitions, the checksums of the training-side and test-side source files, the reference values of the test-side guard and the positive controls | the fit step of the column build (CPU; `src/stage2_sources.py`) | verified | yes | assembly |
| Second stage, submission A | deployment file of the five-draw form with the same column sources (`f878b92249cff8de6d2e4fada2bff0793d231ebc55657f34ea232fe276c64cb1` as used) and its base without columns (`e2ac13520576610a79ebafd6264a89a6883631da0eda3ad543632c4351115e93`) | the fit steps of the five-draw build (CPU) | verified | yes | assembly |
| Earlier second-stage files | five files (references of the chain) | earlier builds | verified | yes | reference |
| Stored reports | 29 files under `artifacts/eda/private/`: verdicts of the pre-registered rules (three-draw and four-draw M3 evaluation, 397B parity check with 64 rows, `m3px27` evaluation, `q397m3` evaluation, five-draw evaluation), row vectors of earlier checks, the measurement files of the column fidelity check (one of 125.7 MB with training-side source tables only) and the verdicts of the bundle test | evaluation scripts on the CPU (some not exported, section 9) | possible | yes | checks of the internal build (verdicts and pins); the build paths of this repository skip these comparisons and pin the resulting values |

The checksums of the blend and second-stage files above are those of the files the submitted files were built from
(private package). The build paths of this repository rewrite these JSON files with English descriptive fields and
renamed keys; their numbers are the same, their checksums differ ([INFERENCE.md](INFERENCE.md), section 2). One stored
report holds aggregates of the test side (a check of the mechanical answer comparison); no report holds per-row test
information. One file in this group is a parser script of an earlier independent review; a check of the
mechanical answer comparison loads it, if present, for an optional cross-check only.

## 8. Training chain

The assembly reads model outputs; the following steps produced the trained or fitted components and the OOF matrices.
Commands and checks are in [TRAINING.md](TRAINING.md); here only what a retraining needs and what is available.

| Step | Inputs | Script, configuration, seed | Hardware | Inputs available frozen in the private package? |
|---|---|---|---|---|
| Folds | `train.csv` | `kaggle_core.folds.make_folds` (5 folds, seed 42, stratified by root code) → `configs/folds.parquet` | CPU | yes (`configs/folds.parquet` in this repository) |
| 9B adapters `v1_f0` … `v1_f4`, `v1_full` | `train.csv`, training photos, diagrams, taxonomy, folds; targets built at run time; up to four same-question examples of the other folds | `ft_v1a_rp`, `ft_sum1_rp`, `ft_v1b_rp`, `ft_v1c_rp` (seed 42) | A40 | yes: needs only competition data |
| Orientation network | training photos and 10,000 pool photos (fixed sample, seed 20261010), labelled by the rotations applied in training | `tools/orient_net.py maps`, `train --seed 0 --threads 2` | CPU | yes: needs only competition data; bit-identical over three trainings |
| Teacher pool labels | pool rows | `tools/teacher_api.py` (paid API) | — | yes (frozen, section 4.1) |
| 397B answers on training rows | training rows | `tools/open_teacher.py` (hosted API) | — | yes (frozen, section 4.2) |
| OOF of the untrained 27B channels | training rows; same-question examples of the other folds; teacher labels (task block) | `zs27q_oof_rp`, `zs27q2_f1_rp`; reasoning draws g (`zs27v_all_rp`, `zs27v_f2_rp`, `zs27v_cut_rp`, `zs27v_te_rp`) and n (`zs27v_n_rp`, `zs27v_n3_rp`); protocol `pr27_rp`, `px27_rp` | A40; RTX PRO 6000 | the OOF matrices are frozen and in the private package; the run folders, including the reasoning texts of the training rows, are in our internal store and not in the package |
| M3 training targets | `train.csv` (gold code and the annotators' explanation of the root finding), folds, frozen protocol notes `train_pr27.jsonl`, `oof_pr27` (where the notes carry loss) | `tools/make_m3_targets.py --full` → `data/derived/sft/m3_fold{0..4}_targets.jsonl`, `m3_full_targets.jsonl` (full fit `b685a7b3da67822280ca51b8a1cf0c47e7ba41f0d6c561a33a7e7dfb855e35aa`; the recipe of every adapter pins its target file; `--check` rebuilds all six byte for byte) | CPU | yes: the six target files with their meta files and the stored list the builder checks against (`artifacts/eda/private/m3_targets/sft_manifest.json`) |
| M3 adapters | targets above, photos, diagrams | `m3_pilot_rp`, `m3_f034_rp`, `m3_fullte_rp` (LoRA r 32, α 64, layers 40–63, 2 epochs, lr 1e-4, seed 42) | RTX PRO 6000 | yes (adapters and targets) |
| M3 OOF draws, M3 behind notes | adapters; frozen notes | section 5 | RTX PRO 6000 | yes |
| Image statistics and the character-height list | training and test photos (statistics only; no photo is stored) | `eda/i5_01_image_stats.py` (per-image statistics, CPU), `tools/make_charsize_rows.py --max-height 20 --name h20` | CPU | the lists and their meta file yes; the statistics table is in the internal store (it can be recomputed from the photos) |
| OOF matrices, blend, second stages | frozen run outputs | section 5, 7 | CPU | yes; bit-identical |

What a retraining gives: the CPU steps reproduce bit for bit; GPU training and every generated text give new adapters,
new texts and therefore new checksums downstream, with scores within the measured spread ([TRAINING.md](TRAINING.md#15-reproducibility-of-training)).

## 9. Open items of this list

- **Scripts not exported** although they exist: the combination of the fold runs of draws d1 and d2, the 397B parity
  check, the posterior step of the M3 evaluation, the measurement scripts of the column fidelity check and of the bundle
  test of the column sources, the evaluation script of the five-draw rule.
- **Frozen texts of draw h on the test side:** the generated texts stay in the two run folders (`gen_notes.jsonl`, pinned
  through the digest of the shard files), which the build reads. Copies in the frozen-notes format of d1–d4
  (`data/derived/sft/m3_h_test_{a,b}_frozen_notes.jsonl`, written by `tools/freeze_m3_texts.py`) are part of the private input
  package for the model steps; the second stage does not read them.
- **Code version not recorded** for the two test runs of `zs27q`; **package list not recorded** for the text generation
  of reasoning draw g on the test rows (the machine was reused).
- **Reasoning gates:** the band part of the gate lists was written by ad-hoc scripts following the recorded rule.
  `tools/replay/gates.py` reconstructs the training-side lists bit for bit and checks the test-side lists bit for bit with
  the A40 outputs in the private package (role `check`).
- **Hosted 397B:** the revision and number format behind the hosted API are not documented by us. The Hub repository
  states Apache 2.0 (licence file at main `8472618…`, checked 2026-11-20); the revision served by the provider was not
  recorded.
- **Parity row list** `data/derived/open_teacher/rows_parity64_397b.npy` (needed to repeat the parity probe) and the outputs
  of the parity run are part of the private input package; the script of the parity check is not exported (above).
