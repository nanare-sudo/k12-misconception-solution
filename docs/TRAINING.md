# Training

*Status: 2026-11-01 (version 0). This document lists every component that is trained or fitted on the labelled data:
inputs, command, hardware, duration and the values to check. Model details are in [METHOD.md](METHOD.md); inference is
in [INFERENCE.md](INFERENCE.md).*

**The trained weights are delivered, not retrained.** The competition host reproduces inference only. GPU training is
not bit-reproducible (see §14), so a retrained model is a new model with a new test file. The adapters used for the
submissions are stored privately with their checksums ([DATA_AND_COMPLIANCE.md](DATA_AND_COMPLIANCE.md)).

## 1. Overview

```
train.csv + training photos + question diagrams + taxonomy
  │
  ├─ (§3)  folds ────────────────────────────────────────────────── configs/folds.parquet
  ├─ (§4)  9B LoRA: 5 fold adapters + full fit ──────────────────── v1_f0 … v1_f4, v1_full  → oof_v1
  ├─ (§5)  orientation network (CPU) → rotation lists ──────────── orient_cnn.pt            → oof_v1r
  ├─ (§6)  teacher pool labels (API, primary line only) ────────── pool_a, pool_b (frozen)
  ├─ (§7)  27B zero-shot scorers ────────────────────────────────── oof_zs27q, oof_zs27t
  ├─ (§8)  gates → reasoning draws g / n → scoring ──────────────── oof_think18nb, oof_think18sb
  ├─ (§9)  protocol answers → notes → exact ranking ─────────────── oof_pr27, oof_pr27xm
  ├─ (§10) M3 targets → M3 LoRA: 5 fold adapters + full fit ─────── m3_f0 … m3_f4, m3_full
  │         → OOF draws d1, d2, d3 ──────────────────────────────── oof_m3x3_full
  ├─ (§11) 397B training side (API) → notes → M3 scores behind them oof_q397m3          [primary line]
  ├─ (§12) blend weights (CPU) ──────────────────────────────────── stack_*.json
  └─ (§13) second stage (CPU) ───────────────────────────────────── stage2_*.json       [primary line]
```

| Component | Kind | Hardware | Deterministic |
|---|---|---|---|
| Folds, OOF assembly, blend weights, second stage, orientation network | fitting on CPU | CPU | yes, bit-identical |
| 9B adapter | LoRA fine-tuning | A40 48 GB | no (GPU) |
| M3 adapter | LoRA fine-tuning | RTX PRO 6000 96 GB | no (GPU) |
| OOF of the untrained channels and of M3 | inference: exact scoring behind generated texts | A40, RTX PRO 6000 | scoring yes on given hardware; text generation no |
| Teacher labels, 397B training side | paid APIs, frozen | — | no; delivered as frozen files |

## 2. Environment

| Environment | Use | Definition |
|---|---|---|
| GPU base | Training, adapter merge, exact scoring | Docker image `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`, Python 3.12. Pinned in `solution/env/base.lock`: torch 2.8.0+cu128, transformers 5.15.1, peft 0.21.2, accelerate 1.15.0, flash-linear-attention 0.5.2, triton 3.4.0, bitsandbytes 0.50.2 |
| vLLM | Text generation (reasoning, protocol answers, M3 draws) | Separate virtual environment, `solution/env/vllm.lock`: vllm 0.31.0, transformers 5.17.0, torch 2.13.0 |
| CPU | OOF assembly, blend, second stage, assembly | `solution/env/cpu.lock`: Python 3.13.5, numpy 2.4.2, scipy 1.17.1, pandas 2.3.3, scikit-learn 1.8.0 |

> **TODO.** The lock files pin exact versions; their hashes are completed by the offline wheelhouse build
> (`kernels/wheelhouse/`). Until then, install the pinned versions without hash checking.

**Base models** (fixed revisions; all Apache 2.0):

| Model | Revision |
|---|---|
| `Qwen/Qwen3.5-9B` | `c202236235762e1c871ad0ccb60c8ee5ba337b9a` |
| `Qwen/Qwen3.8-27B` | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` |
| `Qwen/Qwen3.5-397B-A17B` (training side, through the Tinker API) | served by the provider |

**Model runs.** Every GPU run is one YAML file in `configs/vlm/` and is started with the runner:

```bash
cd src
K12_DATA=../data K12_MODELS=/path/to/models PYTHONPATH="$PWD:$PWD/.." \
  python -u -m vlm.runner --config ../configs/vlm/<run>.yaml --out ../artifacts/runs/<run>
```

- `K12_DATA` points to the folder that holds `train.csv` and `competition_data/`. `K12_MODELS` holds the base models at
  their pinned revisions. With `HF_HUB_OFFLINE=1` the runner never downloads; without it, a missing model is fetched at
  the pinned revision.
- A run writes `<out>/<run>/<job>/…`. Its jobs (`train`, `merge`, `sft_gen`, `think_gen`, `proto_gen`, `bench`, …) run in
  the order of the file. Several configurations start with a short smoke job, for example 32 training steps, which must
  pass before the main job begins.
- Training does not fit on a Kaggle T4. The 9B adapter was trained on an A40 and M3 on an RTX PRO 6000.

## 3. Folds

| | |
|---|---|
| Command | `python -I tools/make_folds.py` *(TODO: in preparation; reproduces `kaggle_core.folds.make_folds` with 5 folds, seed 42, stratified by `root_code`)* |
| Output | `configs/folds.parquet` (row position → fold) |
| Check | Fold hash `07ec381cc58b2a80`. Every run verifies it |

## 4. 9B fine-tuned scorer (`v1`)

| | |
|---|---|
| Inputs | `train.csv`, training photos, question diagrams, taxonomy, folds. Training targets (code, answer check, cleaned transcription, findings) are built at run time from `train.csv`; there is no derived file |
| Configurations | `ft_v1a_rp` (folds 0 and 2), `ft_sum1_rp` (fold 1), `ft_v1b_rp` (folds 3 and 4), `ft_v1c_rp` (full fit on all 1,500 rows → adapter `v1_full`) |
| Recipe | LoRA r 32, α 64, dropout 0.05 on layers 20–31; 3 epochs; batch 8, gradient accumulation 8; learning rate 1.5e-4; summed code loss with weight 1.0, answer check 0.5, auxiliary targets 0.5; up to 4 same-question examples, dropped for 15 % of samples; photo 1,024 px, diagram 448 px; seed 42 |
| Hardware and time | A40 48 GB, bf16; 0.8–1.0 s per step, about 0.5 USD per fold. Full fit: 4,500 steps in 59 min |
| OOF assembly | `python -I eda/a6_assemble_oof.py v1 ft_v1a_rp/v1_f0/oof0 ft_sum1_rp/ft9b_sum_f1/oof1 ft_v1a_rp/v1_f2/oof2 ft_v1b_rp/v1_f3/oof3 ft_v1b_rp/v1_f4/oof4` |
| Checks | Each training job reports all planned steps done. `oof_v1` MAP@3 0.6406 ± 0.0105. `v1_full/adapter_model.safetensors`: 110,906,720 bytes, SHA-256 `321108733ee1ae9eafe14d98047eb592f2f8084df372f600aa46f4a0f0f1d9ca` |

The fold-1 run scored only the 8 most likely code prefixes exactly (`beams: 8`) instead of all 74. Its training recipe
is identical, and its OOF probabilities enter the blend in that form.

## 5. Orientation network and rotation-corrected 9B (`v1r`)

| | |
|---|---|
| Ink maps | `python -I tools/orient_net.py maps --tables train,test,unlabelled` (CPU) |
| Training | `python -I tools/orient_net.py train --seed 0 --threads 2 --weights artifacts/orient/orient_cnn.pt`. Training photos plus a fixed sample of 10,000 pool photos (seed 20261010); details in [METHOD.md](METHOD.md#6-orientation-network) |
| Rotation lists | `python -I tools/orient_net.py infer --from-images --tables train,test --out data/derived/orient_repro`. This writes a `c50` list (photos turned upright for the 9B run, with their rotation) and a `c80` list (rows on which the upright reading replaces the plain one), in the format of `tools/make_orient.py` |
| Rotated 9B reading | Configuration `v1rot_t4` (Kaggle T4×2, fp16): the fold adapters read the rotated training photos (jobs `rot_f0` … `rot_f4`) |
| OOF assembly | `python -I eda/a6_assemble_oof.py v1r --mean --fill v1 v1rot_t4/rot_f0/oof0+v1rot_t4/rot_f1/oof1+v1rot_t4/rot_f2/oof2+v1rot_t4/rot_f3/oof3+v1rot_t4/rot_f4/oof4@derived/orient/train_c80_rows.npy` |
| Hardware and time | CPU, minutes for the network. Rotated reading on Kaggle T4 |
| Checks | `orient_cnn.pt` SHA-256 `83cc1224da0edbea0af640a8682ecb0b882f33b107cf89de9a7673ce19223abe`, bit-identical over three training runs. 124 training rows in the `c80` list. `oof_v1r` MAP@3 0.6474 |

> **Note.** The submissions made so far use the rotation lists of an earlier, unseeded version of the network. The
> released network reproduces itself exactly but changes the test list on 24 of 3,498 photos. The switch to the
> released lists is part of the [roadmap](ROADMAP.md).

## 6. Teacher pool labels (primary line only)

| | |
|---|---|
| Command | `tools/teacher_api.py` (`plan` → `submit` → `collect`, Anthropic Messages API in batch mode). It reads only `unlabelled.csv` and `train.csv` and never the test table |
| Settings | Claude Haiku 5.5, effort `xhigh`, output limit 24,000 tokens; prompt without annotator notes, labelled rows or examples ([METHOD.md](METHOD.md#5-teacher-pseudo-labels-for-the-unlabelled-pool)) |
| Output | `data/derived/teacher/pool_a/` and `pool_b/`: 49,979 usable labels for the 50,000 pool workings |
| Cost | 30.86 USD |
| Checks | Frozen files with SHA-256 `44fd2e1b…` (`pool_a`) and `0f561932…` (`pool_b`) |

The labels cannot be regenerated without the paid API, so they are delivered as frozen files. The inductive line does not
use them.

## 7. 27B zero-shot scorers (`zs27q`, `zs27t`)

| | `zs27q` (primary line) | `zs27t` (inductive line) |
|---|---|---|
| Configurations | `zs27q_oof_rp`, `zs27q2_f1_rp` (fold 1) | `zs27t_all_rp` (all folds, also the test side used for the gates) |
| Hardware | A40 48 GB, 4-bit, 3.1–3.7 s per row | A40 48 GB, 4-bit |
| OOF assembly | `python -I eda/a6_assemble_oof.py zs27q zs27q_oof_rp/zs27q_oof4/bench zs27q2_f1_rp/zs27qa8_f1/bench` | `python -I eda/a6_assemble_oof.py zs27t zs27t_all_rp/zs27t_f1/bench zs27t_all_rp/zs27t_oof4/bench` |
| Check | MAP@3 0.6814 | MAP@3 0.6730 |

No training is involved. Same-question examples and topic blocks are drawn from the other folds.

## 8. Reasoning channels (`think18nb`, `think18sb`)

| | |
|---|---|
| Gates | `python -I tools/replay/gates.py write --out <new folder> --table train --line all`. This reconstructs the row lists of [METHOD.md](METHOD.md#43-27b-reason-then-name-think18nb-think18sb); the training-side lists are bit-identical to the frozen ones |
| Draw g (task-block prompt) | `zs27v_all_rp` (fold 1); `zs27v_f2_rp` and `zs27v_cut_rp` (fold 2, generated longer and cut to 1,800 tokens); `zs27v_te_rp` (job `think_tr`, remaining rows) |
| Draw n (teacher-free prompt) | `zs27v_n_rp` (folds 1 and 2); `zs27v_n3_rp` (job `think_ntr`, remaining rows) |
| Hardware and time | RTX PRO 6000 96 GB; vLLM text generation about 2.1 s per row, exact scoring 1.6–1.7 s per row |
| OOF assembly, primary | `python -I eda/a6_assemble_oof.py think18nb --mean --fill zs27q zs27v_all_rp/think_f1/bench/think+zs27v_cut_rp/cut1800_f2/bench/think+zs27v_te_rp/think_tr/bench/think@derived/think_rows/train.npy zs27v_n_rp/think_n/bench/think+zs27v_n3_rp/think_ntr/bench/think@derived/think_rows/train_tb_q.npy` |
| OOF assembly, inductive | `python -I eda/a6_assemble_oof.py think18sb --mean --fill zs27t zs27v_n_rp/think_n/bench/think+zs27v_n3_rp/think_ntr/bench/think@derived/think_rows/train_tb_t.npy` |
| Checks | `think18nb` MAP@3 0.7672; `think18sb` 0.7516 |

## 9. Protocol channel (`pr27`, `pr27xm`)

| | |
|---|---|
| Protocol answers | `pr27_rp` (jobs `proto_a`, `proto_b`; thinking up to 2,500 tokens, then a completion without thinking). RTX PRO 6000, 3.3–3.5 s per row, about 4.3 USD for the 1,500 training rows |
| Stated-codes channel | `python -I eda/a8_proto_oof.py pr27 runs/pr27_rp/proto_a/proto+runs/pr27_rp/proto_b/proto` |
| Frozen notes | `python -I tools/export_proto_notes.py --name train_pr27 --from pr27_rp/proto_a/proto --from pr27_rp/proto_b/proto` → `data/derived/proto_notes/train_pr27.jsonl` |
| Exact ranking behind the notes | `px27_rp` (job `rank_tr`), about 2.8 USD for the training side |
| Channel | `python -I eda/c95_rank_mean.py` (defaults: name `pr27xm`, the two answer jobs above and `runs/px27_rp/rank_tr/bench`) |
| Checks | `pr27` MAP@3 about 0.745, `pr27xm` 0.7459. Rebuilding the notes file and `oof_pr27` from the frozen answers is bit-identical |

## 10. M3: fine-tuned 27B reasoner (`m3x3_full`)

**10.1 Training targets.** `python -I tools/make_sft_targets.py --full` writes `data/derived/sft/m3_fold{0..4}_targets.jsonl`
(each with the 1,200 rows of the other folds) and `m3_full_targets.jsonl` (all 1,500 rows). The inputs are:
- `train.csv`, which gives the gold code and the annotators' explanation of the root finding;
- the folds;
- the frozen protocol notes `data/derived/proto_notes/train_pr27.jsonl`;
- `oof_pr27`, which decides where the notes carry loss.

Rebuilding all six files from the frozen inputs is bit-identical.

**10.2 Adapters and first draw.**

| | |
|---|---|
| Configurations | `m3_pilot_rp` (folds 1 and 2, preceded by a 32-step smoke job), `m3_f034_rp` (folds 0, 3 and 4), `m3_fullte_rp` (full fit → adapter `m3_full`, merge and test draw d1; `m3_full_rp` holds the same training jobs) |
| Recipe | LoRA r 32, α 64, dropout 0.05 on layers 40–63; vision tower frozen; 2 epochs; learning rate 1e-4; gradient accumulation 8; gradient checkpointing; summed code loss with weight 1.0, finding 1.0, notes 0.5, scaffold 0.5; notes capped at 2,500 tokens; seed 42 |
| Hardware and time | RTX PRO 6000 96 GB, bf16, peak about 72 GB. 1.75–1.80 s per step: 2,400 steps per fold (about 70 min) and 3,000 steps for the full fit (about 90 min). A pre-run aborts on cards that are too slow or too small |
| Per fold | Train → merge the adapter → vLLM writes notes, finding and root for the fold's 300 rows (draw d1: greedy, up to 3,200 tokens) → exact ranking of all 74 classes behind the canonical text |
| Checks | `m3_full/adapter_model.safetensors`: 300,456,960 bytes, SHA-256 `85c34f5c396ec21ecb6de28356a631f4261264402b7a03283ad8fc168e334527` |

**10.3 Further OOF draws.**

| Draw | Configurations | Settings |
|---|---|---|
| d1 | fold jobs of `m3_pilot_rp`, `m3_f034_rp`; combined by `eda/c95_m3_full_eval.py` into `runs/m3_5f` | greedy, up to 3,200 tokens |
| d2 | `m3long_rp` (folds 1, 2; combined by `eda/c98d_m3long_eval.py` into `runs/m3long_comb`), `cv01_m3d2_f034` (folds 0, 3, 4) | greedy, up to 5,200 tokens |
| d3 | `m3d3_oof_rp` (all folds) | sampled: temperature 0.6, top-p 0.95, top-k 20, seed 3; up to 5,200 tokens |

**10.4 Channel.**

```bash
python -I eda/a9_m3_draws_oof.py m3x3_full \
  --draw runs/m3_5f/m3_rank/bench \
  --draw runs/m3long_comb/m3l_rank/bench+runs/cv01_m3d2_f034/m3l_rank_f0/bench+runs/cv01_m3d2_f034/m3l_rank_f3/bench+runs/cv01_m3d2_f034/m3l_rank_f4/bench \
  --draw runs/m3d3_oof_rp/m3d3_rank_f0/bench+runs/m3d3_oof_rp/m3d3_rank_f1/bench+runs/m3d3_oof_rp/m3d3_rank_f2/bench+runs/m3d3_oof_rp/m3d3_rank_f3/bench+runs/m3d3_oof_rp/m3d3_rank_f4/bench \
  --rows-out derived/m3x3_full_rows.npy
python -I eda/c108_m3_three_draws.py     # evaluation of the three-draw channel in both lines; read by the build scripts
```

Checks: MAP@3 of d1 0.7621, d2 0.767, d3 0.7701; mean of three 0.8020. `python -I eda/a9_m3_draws_oof.py --verify m3x3_full`
recomputes the channel from the sources recorded in its metadata.

## 11. 397B training side and `q397m3` (primary line only)

| Step | Command | Notes |
|---|---|---|
| 397B protocol answers on training rows | `tools/open_teacher.py` (Tinker sampling API, `Qwen/Qwen3.5-397B-A17B`, training rows only) | 1,496 of 1,500 valid, about 0.03 USD per row. Frozen; it cannot be regenerated without the API |
| Copy into the run layout | `python -I eda/c110_q397_deploy.py source` | byte-identical copy of the frozen answers |
| 397B answer channel (reference) | `python -I eda/a8_proto_oof.py q397 runs/q397_tinker/proto_t/proto` | MAP@3 about 0.744 |
| Notes for M3 | `tools/make_q397m3_notes.py` | first-pass reasoning notes without the answer text |
| M3 scores behind the notes | configuration `q397m3_oof_rp` (fold adapters, one shard per fold) | about 3.4 s per row, about 5.4 USD |
| Channel | `python -I eda/c117_q397m3_deploy.py oof --save` | MAP@3 0.7442 |

## 12. Blend weights

| Line | Command | Output | SHA-256 |
|---|---|---|---|
| Inductive | `bash tools/builds/s029_m3x3.sh safe`. It calls `eda/c99_m3_deploy.py --channel m3x3_full --rows derived/m3x3_full_rows.npy --lines safe --save`, then adds β = 0.4 and the checksum of the inductive class weights | `artifacts/stack_v1r_zs27t_think18sb_pr27xm_m3x3_full_ar.json` | `7fb543ae5b7cb460a869411c1a97759d5594af66140d6922485b1b1ff421490c` |
| Primary | `python -I eda/c110_q397_deploy.py stack --line best --m3 m3x3_full --replace-q397 q397m3 --arith-beta-select --stem s032_stage2 --save` | `artifacts/stack_s032_stage2.json` | `d278bbedf7fb4d7564e584a55d45959b826748dfe937d8becf1e0818007281eb` |

Weights are fitted on LogLoss over all 1,500 OOF rows, on the CPU in seconds. The rebuild is character-identical, and
the build scripts verify it before every assembly.

## 13. Second stage (primary line only)

```bash
python -I eda/c118_stage2.py oof                                                   # nested OOF reproduction of the reranker
python -I eda/c118_stage2.py fit --stack s032_stage2 --save                        # → artifacts/stage2_s032.json (original band rule)
python -I eda/c118_stage2.py fit --stack s032_stage2 --guard-rule widened --save   # → artifacts/stage2_s034.json (corrected notes-guard rule)
```

| File | SHA-256 | Content |
|---|---|---|
| `artifacts/stage2_s032.json` | `c33174e3ab203b5c2559586eb6b02f67f1588aca4da6bde25078cfb318fed73e` | Weights and standardisation of both feature sets; notes guard with the original ±10 % band |
| `artifacts/stage2_s034.json` | `377d863bf08b07dda20bb37fab1af4a0c929e7fbbb8c3f221f1aed6e14f4397a` | Identical weights, feature lists, β, K, λ and gate; notes guard with the corrected rule (adds a binomial test with \|z\| ≤ 3 for share statistics) |

Checks: the abstention gate does not abstain (inner 0.8351 against base 0.8286). CV per fold / without the last step /
as one set: 0.83756 / 0.83011 / 0.83456 for the main form and 0.83556 / 0.82989 / 0.83378 for the fallback form.

## 14. Reproducibility of training

- **CPU steps are bit-identical:** folds, training targets, OOF assembly, blend weights, second stage and the orientation
  network.
- **GPU training is not bit-identical.** Deterministic algorithms are not enforced. Flash-linear-attention uses Triton
  kernels, sums run in bf16, and a different GPU changes the kernels. A retrained adapter follows the same recipe but is
  a new model. The submissions are therefore reproduced from the delivered adapters.
- **Generated texts are draws.** Reasoning texts, protocol answers and M3 generations are frozen with checksums; a new
  generation is a new draw.
- **API-derived inputs** (teacher labels, 397B training side) can be regenerated only with the paid APIs, so they are
  delivered as frozen files. The inductive line needs neither.
- **Estimated retraining cost** at the hardware listed above (rough figures):
  - 9B, five folds and full fit: about 6 A40 GPU hours, a few USD;
  - M3, five folds and full fit: about 8 GPU hours on an RTX PRO 6000, about 20 USD;
  - three new M3 OOF draws: about 10 further GPU hours.
