# Training

Reference document. To reproduce the submissions, start with [VERIFICATION.md](VERIFICATION.md).

*Status: 2026-11-21 (version 0.6). This document lists every component that is trained or fitted on the labelled data:
inputs, command, hardware, duration and the values to check. Model details are in [METHOD.md](METHOD.md); inference is
in [INFERENCE.md](INFERENCE.md); every script is listed in [CODEMAP.md](CODEMAP.md).*

**The trained weights are delivered, not retrained.** The host has said that both the training and the inference pipeline
of the winners will be requested and verified; this document therefore lists every training step with its inputs, and
[PROVENANCE.md](PROVENANCE.md) where each input came from. GPU training is not bit-reproducible (see §15), so a retrained
model is a new model with a new test file. The adapters used for the submissions are delivered in the private input
package with their checksums ([weights_manifest.json](weights_manifest.json)).

## 1. Overview

```
train.csv + training photos + question diagrams + taxonomy
  │
  ├─ (§3)  folds ────────────────────────────────────────────────── configs/folds.parquet, artifacts/folds.npy
  ├─ (§4)  9B LoRA: 5 fold adapters + full fit ──────────────────── v1_f0 … v1_f4, v1_full  → oof_v1
  ├─ (§5)  orientation network (CPU) → rotation lists ──────────── orient_cnn.pt            → oof_v1r (first-version list)
  ├─ (§6)  teacher pool labels (API, primary line only) ────────── pool_a, pool_b (frozen)
  ├─ (§7)  27B zero-shot scorers ────────────────────────────────── oof_zs27q, oof_zs27t
  ├─ (§8)  gates → reasoning draws g / n → scoring ──────────────── oof_think18nb, oof_think18sb
  ├─ (§9)  protocol answers → notes → exact ranking ─────────────── oof_pr27, oof_pr27xm
  │         → M3 ranking behind the same notes ──────────────────── oof_m3px27, oof_m3px27xm  [primary line]
  ├─ (§10) M3 targets → M3 LoRA: 5 fold adapters + full fit ─────── m3_f0 … m3_f4, m3_full
  │         → OOF draws d1, d2, d3 ──────────────────────────────── oof_m3x3_full
  │         → OOF draw d4 ───────────────────────────────────────── oof_m3x4_full              [final submissions]
  │         → OOF draw h (photo at 2,048 px) on the listed rows ─── oof_m3x5h                  [submission A]
  ├─ (§11) 397B training side (API) → notes → M3 scores behind them oof_q397m3          [primary line]
  ├─ (§12) blend weights (CPU) ──────────────────────────────────── stack_*.json
  ├─ (§13) second stages (CPU) ──────────────────────────────────── stage2_*.json
  └─ (§16) final submissions: blends, second stages, extra columns (CPU) ── stack_primary_*.json, stage2_primary_*.json
```

| Component | Kind | Hardware | Deterministic |
|---|---|---|---|
| Folds, OOF assembly, blend weights, second stages, orientation network | fitting on CPU | CPU | yes, bit-identical on the reference CPU ([VERIFICATION.md §5.1](VERIFICATION.md#51-assemble-cpu)) |
| 9B adapter | LoRA fine-tuning | A40 48 GB | no (GPU) |
| M3 adapter | LoRA fine-tuning | RTX PRO 6000 96 GB | no (GPU) |
| OOF of the untrained channels and of M3 | inference: exact scoring behind generated texts | A40, RTX PRO 6000 | scoring yes on given hardware; text generation no |
| Teacher labels, 397B training side | paid APIs, frozen | — | no; delivered as frozen files |

**Channels of the candidates of version 0.3.** Every training-side step of the channels that the three candidates of
version 0.3 use (`primary_m3px27`, `safety_ndr` and its fallback `safety_nd`), with the section that gives its command,
configuration and checks. Both final submissions use the channels of `primary_m3px27`, with M3 as `m3x4_full` in B and
`m3x5h` in A (next table and §16). All configurations are files in `configs/vlm/`, run with the runner of §2.

| Channel | Used by | Trained here | Section, configurations | Inputs besides `train.csv`, photos, diagrams and taxonomy | Hardware |
|---|---|---|---|---|---|
| `v1`, `v1r` | all three | 9B LoRA on the gold labels; orientation network | §4: `ft_v1a_rp`, `ft_sum1_rp`, `ft_v1b_rp`, `ft_v1c_rp`; §5: `tools/orient_net.py train`, `v1rot_t4` | same-question examples of the other folds; for the orientation network 10,000 pool photos, labelled by the rotations applied in training | A40; CPU; Kaggle T4 |
| `zs27q` | primary | no | §7: `zs27q_oof_rp`, `zs27q2_f1_rp` | examples and topic block of the other folds; task block from the frozen teacher labels of the pool (§6) | A40, 4-bit |
| `zs27t` | safety | no | §7: `zs27t_all_rp` | examples and topic block of the other folds | A40, 4-bit |
| `think18nb` | primary | no | §8: draws g and n | draw g as `zs27q` (with the task block), draw n as `zs27t` | RTX PRO 6000 |
| `think18sb` | safety | no | §8: draw n | as `zs27t` | RTX PRO 6000 |
| `pr27xm` | safety | no | §9: `pr27_rp`, `px27_rp` | marking protocol only, no labelled row | RTX PRO 6000 |
| `m3x3_full` | all three | M3 LoRA on the gold codes and the annotators' explanations | §10: `m3_pilot_rp`, `m3_f034_rp`, `m3_fullte_rp`; draws `m3long_rp`, `cv01_m3d2_f034`, `m3d3_oof_rp` | frozen protocol notes of the training rows (`train_pr27.jsonl`) and `oof_pr27` | RTX PRO 6000 |
| `m3px27xm` | primary | no (the M3 fold adapters of §10) | §9: `m3px27_oof_rp`; §13.6 | frozen protocol notes of the training rows | RTX PRO 6000 |
| `q397m3` | primary | no (the M3 fold adapters of §10) | §11: `q397m3_oof_rp` | frozen 397B protocol answers of the training rows | RTX PRO 6000 |
| Blends, second stages | all three | fits on the CPU | §12, §13.1, §13.3–§13.6 | OOF matrices; frozen M3 notes (draws d1–d3) and protocol notes of the training rows; for the rules also the fourth M3 draw `m3d4_oof_rp` and the second draw of the protocol answers `pr27s2_rp`, `pr27s3_rp` | CPU |

**Channels of the final submissions (version 0.4).** The channels of `primary_m3px27` with the M3 channel replaced:

| Channel | Used by | Trained here | Section, configurations | Inputs besides `train.csv`, photos, diagrams and taxonomy | Hardware |
|---|---|---|---|---|---|
| `m3x4_full` | submission B, intermediate form | as `m3x3_full` (same adapters) | §10.3: draws d1–d4 (`m3d4_oof_rp`); §16.1 | as `m3x3_full` | RTX PRO 6000 |
| `m3x5h` | submission A | as `m3x3_full` (same adapters) | §10.3: draws d1–d4 and h (`m3ha_oof_rp`, `m3hb_oof_rp`); §16.3 | as `m3x3_full`; row list of the character-height rule (`tools/make_charsize_rows.py`) | RTX PRO 6000; CPU |
| Extra second-stage columns | submission A, intermediate form | fits on the CPU | §16.2 | protocol answers of the 27B (`pr27_rp`) and of the 397B (hosted API, §11) on the training rows; training labels of the reference rows | CPU |
| Blends, second stages | both | fits on the CPU | §16 | as above | CPU |

**Provenance of labels and texts not written by the annotators.** Three kinds enter training or the training-side
context: the teacher labels (pool rows only, §6), the 397B protocol answers (training rows only, §11) and the texts of our
own model runs on training rows (the protocol notes, used as M3 inputs and targets, and the M3 draws d1–d4 and h, used
as channel members, second-stage features and in the rules). The 397B protocol answers of the training rows are, in
submission A, also the training side of six extra columns (§16.2). None of them was produced for or from a test row, and no test row was ever sent
to an API. Texts generated for test rows (reasoning, protocol answers and notes, M3 draws, 397B notes) are inference
outputs; no training step uses their content (the deployment fits only record checksums of some of these files). The
only step that uses the test rows as a set is the class-marginal alignment at
assembly ([METHOD.md](METHOD.md#8-class-marginal-alignment)); it uses no label.

## 2. Environment

| Environment | Use | Definition |
|---|---|---|
| GPU base | Training, adapter merge, exact scoring | Docker image `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`, Python 3.12; torch 2.8.0+cu128, transformers 5.15.1, peft 0.21.2, accelerate 1.15.0, flash-linear-attention 0.5.2, triton 3.4.0, bitsandbytes 0.50.2 |
| vLLM | Text generation (reasoning, protocol answers, M3 draws) | Separate virtual environment: vllm 0.31.0, transformers 5.17.0, torch 2.13.0 |
| CPU | OOF assembly, blend, second stages, assembly | Python 3.13.5, numpy 2.4.2, scipy 1.17.1, pandas 2.3.3, scikit-learn 1.8.0, pyarrow 23.0.1, PyYAML 6.0.3, Pillow 11.3.0, torch 2.11.0 and transformers 5.15.1 (imported by shared model helpers; used on the CPU); pytest 8.4.2 for the tests |

The complete pin set of every environment is in `env/requirements-gpu.txt`, `env/requirements-vllm.txt`,
`env/requirements-cpu.txt` and `env/requirements-kaggle-t4.txt` (exact versions as installed in the reference runs; each file
names its Python and CUDA version and how it is installed). Lock files with hashes for every pin and an offline wheelhouse
are not built ([ROADMAP.md](ROADMAP.md), step 4). Remove the `torchao` package of the GPU base image before installing the
pins (`python -m pip uninstall -y torchao`); the older version in the image conflicts with the pinned packages.

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
| Split | `configs/folds.parquet` (row position of `train.csv` → fold), part of this repository. It was created once with `kaggle_core.folds.make_folds` (5 folds, seed 42, stratified by `root_code`); `python -I tools/make_folds.py --recompute` recomputes it from `train.csv` and confirms that it is identical |
| Fold array | `python -I tools/make_folds.py` writes `artifacts/folds.npy` (int64, row order of `train.csv`); `--check` compares an existing file. The second stages record its SHA-256 among their inputs |
| Checks | Fold hash `07ec381cc58b2a80` (every run verifies it through `vlm.common.load_folds`); `artifacts/folds.npy` SHA-256 `fadf2771203c19d19e057d69a02413d9f01b26f85bbf76490046f2a6f8773b73` |

## 4. 9B fine-tuned scorer (`v1`)

| | |
|---|---|
| Inputs | `train.csv`, training photos, question diagrams, taxonomy, folds. Training targets (code, answer check, cleaned transcription, findings) are built at run time from `train.csv`; there is no derived file |
| Configurations | `ft_v1a_rp` (folds 0 and 2), `ft_sum1_rp` (fold 1), `ft_v1b_rp` (folds 3 and 4), `ft_v1c_rp` (full fit on all 1,500 rows → adapter `v1_full`) |
| Recipe | LoRA r 32, α 64, dropout 0.05 on layers 20–31; 3 epochs, one sample per step with gradient accumulation 8 (scoring batch 8); learning rate 1.5e-4; summed code loss with weight 1.0, answer check 0.5, auxiliary targets 0.5; up to 4 same-question examples, dropped for 15 % of samples; photo 1,024 px, diagram 448 px; seed 42 |
| Hardware and time | A40 48 GB, bf16; 0.8–1.0 s per step, about 0.5 USD per fold. Full fit: 4,500 steps in 59 min |
| OOF assembly | `python -I eda/a6_assemble_oof.py v1 ft_v1a_rp/v1_f0/oof0 ft_sum1_rp/ft9b_sum_f1/oof1 ft_v1a_rp/v1_f2/oof2 ft_v1b_rp/v1_f3/oof3 ft_v1b_rp/v1_f4/oof4` |
| Checks | Each training job reports all planned steps done. `oof_v1` MAP@3 0.6406 ± 0.0105. `v1_full/adapter_model.safetensors`: 110,906,720 bytes, SHA-256 `321108733ee1ae9eafe14d98047eb592f2f8084df372f600aa46f4a0f0f1d9ca` |

The fold-1 run scored only the 8 most likely code prefixes exactly (`beams: 8`) instead of all 74. Its training recipe
is identical, and its OOF probabilities enter the blend in that form.

## 5. Orientation network and rotation-corrected 9B (`v1r`)

| | |
|---|---|
| Ink maps | `python -I tools/orient_net.py maps --tables train,test,unlabelled` (CPU) |
| Training | `python -I tools/orient_net.py train --seed 0 --threads 2 --weights artifacts/orient/orient_cnn.pt`. Training photos plus a fixed sample of 10,000 pool photos (seed 20261010); details in [METHOD.md](METHOD.md#6-orientation-network). Writes the description `artifacts/orient/orient_cnn.json` next to the weights |
| Rotation lists | `python -I tools/orient_net.py infer --from-images --tables train,test --weights artifacts/orient/orient_cnn.pt --out data/derived/orient_repro`. This writes a `c50` list (photos turned upright for the 9B run, with their rotation) and a `c80` list (rows on which the upright reading replaces the plain one), in the format of `tools/make_orient.py`. Give `--weights` exactly in this form: the path is written into the meta file of each list, and `tools/orient_list.py check` requires it |
| Rotated 9B reading | Configuration `v1rot_t4` (Kaggle T4×2, fp16): the fold adapters read the rotated training photos (jobs `rot_f0` … `rot_f4`) and the rotated test photos (`rot_test`; for inference only: `configs/vlm/test_side/v1rot_t4.yaml`). Its pre-check job `smoke` and the fold jobs read the frozen first-version training lists `data/derived/orient/train_c50.json` and `train_c50_rows.npy` (private package), not the lists that `infer` writes to `orient_repro/`; the OOF assembly reads `data/derived/orient/train_c80_rows.npy`. The first version cannot be recomputed. To use the released network's lists instead, replace `orient/` by `orient_repro/` in `v1rot_t4.yaml` and in the OOF assembly (238/132 instead of 248/124 rows; `oof_v1r` 0.6471 instead of 0.6474) |
| OOF assembly | `python -I eda/a6_assemble_oof.py v1r --mean --fill v1 v1rot_t4/rot_f0/oof0+v1rot_t4/rot_f1/oof1+v1rot_t4/rot_f2/oof2+v1rot_t4/rot_f3/oof3+v1rot_t4/rot_f4/oof4@derived/orient/train_c80_rows.npy` |
| Hardware and time | CPU, minutes for the network. Rotated reading on Kaggle T4 |
| Checks | `orient_cnn.pt` SHA-256 `83cc1224da0edbea0af640a8682ecb0b882f33b107cf89de9a7673ce19223abe`, bit-identical over three training runs. Released lists: 132 training rows and 344 test rows in the `c80` lists; the test list has SHA-256 `aa6515c8bf17bf593c5c90aa90d9240e9992d84f650c5445a37a5f015dbbff1d`. `oof_v1r` MAP@3 0.6474 |

**Two rotation lists.** The earlier submissions and the training side of `v1r` use the lists of an unseeded first version
of the network (`data/derived/orient/`; 124 training rows and 350 test rows in the `c80` lists), whose weights were not
stored. The test side of the final candidates uses the released network's list (344 rows; it differs on 24 of 3,498 test
photos). All of its rows were already computed by the rotation run with the same rotation, so only the row filter of the
`v1r` source changes ([INFERENCE.md](INFERENCE.md#2-lines-and-expected-checksums)). The training side (`oof_v1r` and the
blend weights fitted on it) stays on the first-version list: with the released list the channel's training-side MAP@3
would be 0.6471 instead of 0.6474 (−0.0003 ± 0.0011), which is not distinguishable from zero.

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
| Gates | The private package holds the frozen training-side lists `data/derived/think_rows/train.npy`, `train_n3.npy`, `train_tb_q.npy` and `train_tb_t.npy` (rules: [METHOD.md](METHOD.md#43-27b-reason-then-name-think18nb-think18sb)). `python -I tools/replay/gates.py check --table train --line all` reconstructs them bit for bit. After a retraining of the 9B scorer (§4, §5) or of the zero-shot scorers (§7), rebuild them with `python -I tools/replay/gates.py write --out data/derived/think_rows_new --table train --line all` and copy the four files into `data/derived/think_rows/`: the reasoning configurations and the OOF assembly read that folder, and `write` never writes into it. `--line all` also reads `data/derived/orient/train_v50_rows.npy` |
| Draw g (task-block prompt) | `zs27v_all_rp` (fold 1); `zs27v_f2_rp` and `zs27v_cut_rp` (fold 2, generated longer and cut to 1,800 tokens); `zs27v_te_rp` (job `think_tr`, remaining rows) |
| Draw n (teacher-free prompt) | `zs27v_n_rp` (folds 1 and 2); `zs27v_n3_rp` (job `think_ntr`, remaining rows) |
| Hardware and time | RTX PRO 6000 96 GB; vLLM text generation about 2.1 s per row, exact scoring 1.6–1.7 s per row |
| OOF assembly, primary | `python -I eda/a6_assemble_oof.py think18nb --mean --fill zs27q zs27v_all_rp/think_f1/bench/think+zs27v_cut_rp/cut1800_f2/bench/think+zs27v_te_rp/think_tr/bench/think@derived/think_rows/train.npy zs27v_n_rp/think_n/bench/think+zs27v_n3_rp/think_ntr/bench/think@derived/think_rows/train_tb_q.npy` |
| OOF assembly, inductive | `python -I eda/a6_assemble_oof.py think18sb --mean --fill zs27t zs27v_n_rp/think_n/bench/think+zs27v_n3_rp/think_ntr/bench/think@derived/think_rows/train_tb_t.npy` |
| Checks | `think18nb` MAP@3 0.7672; `think18sb` 0.7516 |

## 9. Protocol channel (`pr27`, `pr27xm`)

| | |
|---|---|
| Protocol answers | `pr27_rp` (jobs `proto_a`, `proto_b`; thinking up to 2,500 tokens, then a completion without thinking). RTX PRO 6000, 3.3–3.5 s per row, about 4.3 USD for the 1,500 training rows. The row lists of the private package split the training rows into the two jobs that the later scripts read: `data/derived/proto_rows/train_sample.npy` (860 rows, job `proto_a`; SHA-256 `3eaeefefd8266385de47775af9262958224c17c9459fb4b5e288f37861c80a9a`) and `train_rest.npy` (640 rows, `proto_b`; `fc6a7c39dcf554b8c0b472ea6b7de6c01d005cc2fa12a4a215668c0aa74f9b75`); `train_api27.npy` (70 rows; `dc7f8813dbf087578696505855e813c57b5af6e0abfea088f34b764a1a5e9f16`) only makes 70 of them run first |
| Stated-codes channel | `python -I eda/a8_proto_oof.py pr27 runs/pr27_rp/proto_a/proto+runs/pr27_rp/proto_b/proto` |
| Frozen notes | `data/derived/proto_notes/train_pr27.jsonl`: the first-pass reasoning of the protocol answers, exported per row (`python tools/export_proto_notes.py --name train_pr27 --from pr27_rp/proto_a/proto --from pr27_rp/proto_b/proto`; the test notes with `--allow-test` into `data/derived/proto_notes_test/`). The files are delivered frozen with their checksums; `eda/c130_safety_ndr.py export-notes --verify` re-exports them from the raw answers byte for byte |
| Exact ranking behind the notes | `px27_rp` (job `rank_tr`), about 2.8 USD for the training side |
| Channel | `python -I eda/c95_rank_mean.py` (defaults: name `pr27xm`, the two answer jobs above and `runs/px27_rp/rank_tr/bench`) |
| Checks | `pr27` MAP@3 about 0.745, `pr27xm` 0.7459. Rebuilding `oof_pr27` from the frozen answers is bit-identical |
| Second draw of the answers | `pr27s2_rp` (`proto_a2`), `pr27s3_rp` (`proto_b2`): greedy, another processing order; exported as `data/derived/proto_notes/train_pr27d2.jsonl`. Used only as a sensitivity check of the rule of the regex note columns (§13.5) |
| M3 behind the notes (`m3px27`) | Notes in the format of the frozen M3 notes: `python -I tools/make_m3px27_notes.py --table both` → `data/derived/sft_m3px27/m3px27_train_notes.jsonl`, `data/derived/sft_m3px27te/m3px27_test_notes.jsonl` (character-identical with the input of the untrained ranking). Rankings: `m3px27_oof_rp` (five fold shards, per fold the M3 adapter without that fold; RTX PRO 6000, about 4.7 USD) and `m3px27te_rp` (full-fit adapter; about 12 USD). Channel artifact: `python -I eda/c132_m3px27_eval.py --oof-save` → `artifacts/oof_m3px27.npy` (SHA-256 `a0a9660f6f20099ff1019702a11d881eb4c85a3ce089a645ccbc0ac94428986f`) and `meta_m3px27.json`; evaluation by its rule: `--controls`, then the plain call (§13.6) |
| Mean channel `m3px27xm` | `python -I eda/c133_primary_m3px27.py oof --save` → `artifacts/oof_m3px27xm.npy` (SHA-256 `47e128eed7865deec8a5818cdbd514ed29246dec79a56f7da59d86ed3897cfe8`), `meta_m3px27xm.json`; MAP@3 0.7551. With the untrained ranking in place of `m3px27` the same path gives `oof_pr27xm.npy` bit for bit |

## 10. M3: fine-tuned 27B reasoner (`m3x3_full`)

**10.1 Training targets.** `data/derived/sft/m3_fold{0..4}_targets.jsonl` (each with the 1,200 rows of the other folds) and
`m3_full_targets.jsonl` (all 1,500 rows). They are built from:
- `train.csv`, which gives the gold code and the annotators' explanation of the root finding;
- the folds;
- the frozen protocol notes `data/derived/proto_notes/train_pr27.jsonl`;
- `oof_pr27`, which decides where the notes carry loss.

The builder is `tools/make_m3_targets.py`:

```bash
python -I tools/make_m3_targets.py --full            # writes the six target files (+ .meta.json) to data/derived/sft/ if absent
python -I tools/make_m3_targets.py --full --check    # writes nothing: rebuilds them in memory and compares them with the files
                                                     #   present and with targets_sha256 of the six adapter recipes
```

It recomputes from `oof_pr27` and the notes the rows on which the notes carry loss and stops unless they equal the
stored list `artifacts/eda/private/m3_targets/sft_manifest.json` (field `pr27_ok` per fold; SHA-256 pinned in the tool).
Token lengths in the meta files use the tokenizer of Qwen3.5-0.8B (`--tokenizer`, default `data/models/Qwen3.5-0.8B`;
`--check` needs no tokenizer) and the headers of the training photos; the meta files record their creation time and are
therefore not byte-reproducible, the target files are, and an existing target file is never overwritten by a different
one. The six target files, their meta files and the stored list are part of the private input package. Their checksums
are recorded in the training recipe of every adapter (`sft_recipe.json`: `targets_sha256`; full fit
`b685a7b3da67822280ca51b8a1cf0c47e7ba41f0d6c561a33a7e7dfb855e35aa`); a rebuild from the frozen inputs with the tool of
this repository is byte-identical for all six files. `src/vlm/sft.py` reads and checks them at training time.

**10.2 Adapters and first draw.**

| | |
|---|---|
| Configurations | `m3_pilot_rp` (folds 1 and 2, preceded by a 32-step smoke job), `m3_f034_rp` (folds 0, 3 and 4), `m3_fullte_rp` (full fit → adapter `m3_full`, merge and test draw d1; `m3_full_rp` holds the same training jobs; `m3d1te_rp` holds only the merge and test draw d1 with the delivered adapter) |
| Recipe | LoRA r 32, α 64, dropout 0.05 on layers 40–63; vision tower frozen; 2 epochs; learning rate 1e-4; gradient accumulation 8; gradient checkpointing; summed code loss with weight 1.0, finding 1.0, notes 0.5, scaffold 0.5; notes capped at 2,500 tokens; seed 42 |
| Hardware and time | RTX PRO 6000 96 GB, bf16, peak about 72 GB. 1.75–1.80 s per step: 2,400 steps per fold (about 70 min) and 3,000 steps for the full fit (about 90 min). A pre-run aborts on cards that are too slow or too small |
| Per fold | Train → merge the adapter → vLLM writes notes, finding and root for the fold's 300 rows (draw d1: greedy, up to 3,200 tokens) → exact ranking of all 74 classes behind the canonical text |
| Checks | `m3_full/adapter_model.safetensors`: 300,456,960 bytes, SHA-256 `85c34f5c396ec21ecb6de28356a631f4261264402b7a03283ad8fc168e334527` |

**10.3 Further OOF draws.**

| Draw | Configurations | Settings |
|---|---|---|
| d1 | fold jobs of `m3_pilot_rp`, `m3_f034_rp`; the five fold rankings are combined into the run folder `runs/m3_5f` | greedy, up to 3,200 tokens |
| d2 | `m3long_rp` (folds 1, 2; combined into `runs/m3long_comb`), `cv01_m3d2_f034` (folds 0, 3, 4) | greedy, up to 5,200 tokens |
| d3 | `m3d3_oof_rp` (all folds) | sampled: temperature 0.6, top-p 0.95, top-k 20, seed 3; up to 5,200 tokens |
| d4 | `m3d4_oof_rp` (all folds) | as d3 with seed 4; fourth member of the M3 channel of both final submissions (§16), earlier only in the draw swap of the rule of the regex note columns (§13.5) and as a control of the rule of `m3px27` (§13.6) |
| h (photo at 2,048 px) | `m3ha_oof_rp` (folds 1, 2, 0), `m3hb_oof_rp` (folds 3, 4) | as d4 with seed 5 and `image.max_side = image.min_side = 2048` in generation and ranking (`max_model_len` 20,000); the fold adapters were trained at 1,024 px. Fifth member of the M3 channel of submission A, only on the rows of the list "character height ≤ 20 px" (§16) |

The combined run folders `runs/m3_5f` and `runs/m3long_comb` are delivered frozen; the scripts that combined the fold
runs are not part of this repository. `eda/a9_m3_draws_oof.py` checks each draw (fold coverage, settings) when it reads
them.

**10.4 Channel.**

```bash
python -I eda/a9_m3_draws_oof.py m3x3_full \
  --draw runs/m3_5f/m3_rank/bench \
  --draw runs/m3long_comb/m3l_rank/bench+runs/cv01_m3d2_f034/m3l_rank_f0/bench+runs/cv01_m3d2_f034/m3l_rank_f3/bench+runs/cv01_m3d2_f034/m3l_rank_f4/bench \
  --draw runs/m3d3_oof_rp/m3d3_rank_f0/bench+runs/m3d3_oof_rp/m3d3_rank_f1/bench+runs/m3d3_oof_rp/m3d3_rank_f2/bench+runs/m3d3_oof_rp/m3d3_rank_f3/bench+runs/m3d3_oof_rp/m3d3_rank_f4/bench \
  --rows-out derived/m3x3_full_rows.npy
python -I eda/c108_m3_three_draws.py     # evaluation of the three-draw channel in both lines (see below)
```

Checks: MAP@3 of d1 0.7621, d2 0.767, d3 0.7701; mean of three 0.8020. `python -I eda/a9_m3_draws_oof.py --verify m3x3_full`
recomputes the channel from the sources recorded in its metadata. The build scripts read the stored verdict of
`c108_m3_three_draws.py` (`artifacts/eda/private/m3_draws_eval/c108_m3_three_draws.json`); its posterior step needs a
development helper that is not part of this repository, so the stored report is delivered frozen.

## 11. 397B training side and `q397m3` (primary line only)

| Step | Command | Notes |
|---|---|---|
| 397B protocol answers on training rows | `tools/open_teacher.py` (Tinker sampling API, `Qwen/Qwen3.5-397B-A17B`, training rows only) | 1,496 of 1,500 valid, about 0.03 USD per row. Frozen as `data/derived/open_teacher/qwen35-397b/labels.jsonl` (parsed answers) and `raw.jsonl` (raw texts); they cannot be regenerated without the API |
| Copy into the run layout | `python -I eda/c110_q397_deploy.py source` | byte-identical copy of the frozen answers |
| 397B answer channel (reference) | `python -I eda/a8_proto_oof.py q397 runs/q397_tinker/proto_t/proto` | MAP@3 0.7432 |
| Notes for M3 | `tools/make_q397m3_notes.py` | first-pass reasoning notes without the answer text, read from `raw.jsonl`. Check without the tokenizer and without touching the frozen files: `python -I tools/make_q397m3_notes.py --tokenizer '' --no-upload --out "$PWD/q397m3_train_notes_check.jsonl"`; its SHA-256 must be `5a8fb7358bf9a25bcd4f6b12b28c02b31e5ad345a3ee21f9d94bc182bf180d52`, the checksum of `data/derived/sft_q397m3/q397m3_train_notes.jsonl`. Without these switches the tool needs the tokenizer and overwrites the frozen files |
| M3 scores behind the notes | configuration `q397m3_oof_rp` (fold adapters, one shard per fold) | about 3.4 s per row, about 5.4 USD |
| Channel | `python -I eda/c117_q397m3_deploy.py oof --save` | MAP@3 0.7442 |

## 12. Blend weights

| Line | Command | Output | SHA-256 |
|---|---|---|---|
| Inductive | `bash tools/builds/s029_m3x3.sh safe`. It calls `eda/c99_m3_deploy.py --channel m3x3_full --rows derived/m3x3_full_rows.npy --lines safe --save`, then adds β = 0.4 and the checksum of the inductive class weights | `artifacts/stack_v1r_zs27t_think18sb_pr27xm_m3x3_full_ar.json` | `7fb543ae5b7cb460a869411c1a97759d5594af66140d6922485b1b1ff421490c` |
| Primary | `python -I eda/c110_q397_deploy.py stack --line best --m3 m3x3_full --replace-q397 q397m3 --arith-beta-select --stem s032_stage2 --save` | `artifacts/stack_s032_stage2.json` | `d278bbedf7fb4d7564e584a55d45959b826748dfe937d8becf1e0818007281eb` |
| Primary, final candidate (channel `m3px27xm` instead of `pr27xm`) | `python -I eda/c133_primary_m3px27.py stack --save` (θ on LogLoss over all 1,500 rows, β by the same rule: 0.6) | `artifacts/stack_primary_m3px27.json` | `ebdd585e4650922d59972b3a84eb85c44af4f2dc469dd723ef2100ab7d797a3c` |

Weights are fitted on LogLoss over all 1,500 OOF rows, on the CPU in seconds. The rebuild is character-identical, and
the build scripts verify it before every assembly. The blend files do not depend on the rotation list of the test side.

## 13. Second stages

### 13.1 Primary line

```bash
python -I eda/c118_stage2.py oof --stack s032_stage2                              # nested OOF reproduction of the reranker
python -I eda/c118_stage2.py fit --stack s032_stage2 --save                       # → artifacts/stage2_s032.json (original band rule)
python -I eda/c118_stage2.py fit --stack s032_stage2 --guard-rule widened --save  # → artifacts/stage2_s034.json (widened note guard)
python -I eda/c118_stage2.py guard-oof --stack s032_stage2                        # nested reproduction of main_guard (+ NONE guard)
python -I eda/c118_stage2.py guard-fit --stack s032_stage2 --save                 # → artifacts/stage2_s035.json
```

| File | SHA-256 | Content |
|---|---|---|
| `artifacts/stage2_s032.json` | `c33174e3ab203b5c2559586eb6b02f67f1588aca4da6bde25078cfb318fed73e` | Weights and standardisation of both feature sets; note guard with the original ±10 % band |
| `artifacts/stage2_s034.json` | `377d863bf08b07dda20bb37fab1af4a0c929e7fbbb8c3f221f1aed6e14f4397a` | Identical weights, feature lists, β, K, λ and gate; note guard with the widened rule (adds a binomial test with \|z\| ≤ 3 for share statistics) |
| `artifacts/stage2_s035.json` | `12f15243872681aeddad83c407e3e36955db18426e62331ee729869d474f4b86` | `stage2_s034.json` with only the main model replaced: feature set `main_guard` (the 43 main features + identifiers of the five general codes, 48) and the NONE guard; fallback model, note guard, β, K, λ unchanged |

Checks:
- Main and fallback forms: the abstention gate does not abstain (inner 0.8351 for the main form and 0.8347 for the fallback form,
  against base 0.8286). CV per fold / without the
  last step / as one set: 0.83756 / 0.83011 / 0.83456 for the main form and 0.83556 / 0.82989 / 0.83378 for the fallback
  form.
- `main_guard`: `guard-oof` reproduces the main form bit for bit and passes the copy null point (identifiers 0 and NONE
  guard off give the main form exactly). CV per fold / without the last step / as one set: 0.83889 / 0.83522 / 0.83767,
  against the main form +0.0013 ± 0.0014 / +0.0051 ± 0.0016 / +0.0031 ± 0.0015. The gate does not abstain (inner 0.8381
  against base 0.8286). `guard-fit` first recomputes `stage2_s034.json` character for character; a further null point
  requires the deployment features to equal the evaluated rows of `guard-oof`.

**Row × candidate interactions and the final primary form** (version 0.2; run by `tools/builds/primary_i1.sh` and
`tools/builds/primary_final.sh`):

```bash
python -I eda/c121_stage2_i1.py oof --stack s032_stage2                  # nested reproduction: guard form bit for bit, I1, placebos
python -I eda/c121_stage2_i1.py splits --stack s032_stage2 --n 20        # 20 new meta-splits (about 8 minutes)
python -I eda/c121_stage2_i1.py rule                                     # expected-value rule
python -I eda/c121_stage2_i1.py fit --stack s032_stage2 --save           # → artifacts/stage2_primary_i1.json
python -I eda/c124_stage2_mm.py parser                                   # mech27 on the training and test answers (shares only)
python -I eda/c124_stage2_mm.py oof                                      # nested reproduction: I1 form bit for bit, the column, placebos
python -I eda/c124_stage2_mm.py splits --n 20
python -I eda/c124_stage2_mm.py rule
python -I eda/c124_stage2_mm.py fit --save                               # → artifacts/stage2_primary_final.json
```

| File | SHA-256 | Content |
|---|---|---|
| `artifacts/stage2_primary_i1.json` | `f5e24adfd382d24b36bd88b57deb8904a9d5288863250d5d314ceb09e54b181d` | `stage2_s035.json` with only the main model replaced: feature set `main_guard_i1` (the 48 guard-form features + 15 interactions, 63) and the NONE guard; fallback model, note guard, β, K, λ unchanged |
| `artifacts/stage2_primary_final.json` | `40481bc4c83d0ce41b69ebfd10b0d1aaea0a8156c693f1ee7fc89ceb99674604` | `stage2_primary_i1.json` with only the main model replaced: feature set `main_guard_i1_mm` (64) and the NONE guard; adds the field `mech27` (file and SHA-256 of the frozen test answers of the protocol channel, training sources, minor codes) |

Checks:
- `main_guard_i1`: `c121 oof` reproduces the guard form bit for bit and passes the copy null point (I1 columns 0). CV per
  fold / without the last step / as one set: 0.84078 / 0.83322 / 0.83867; against the guard form in sixths +17 / −18 / +9. The gate does not
  abstain (inner 0.8389 against base 0.8286).
- `main_guard_i1_mm`: `c124 oof` reproduces the I1 form bit for bit and passes the copy null point (column 0). CV per fold
  / without the last step / as one set: 0.84089 / 0.83433 / 0.83978; against the I1 form in sixths +1 / +10 / +10. The gate does not abstain
  (inner 0.8391 against base 0.8286).
- Both `fit` commands first recompute their predecessor file character for character (`stage2_s035.json` and
  `stage2_primary_i1.json`), require the deployment features to equal the evaluated rows of their `oof` run, and write
  only if the expected-value rule is met (exit code 5 otherwise). The meta-split files
  (`artifacts/eda/private/stage2_i1/c121_i1_splits.json`, `artifacts/eda/private/stage2_mm/c124_mm_splits.json`) are
  written only if absent and pinned by the build scripts; `PRIMARY_I1_SPLITS=1` or `PRIMARY_FINAL_SPLITS=1` recomputes
  them and requires byte identity.

### 13.2 Strictly inductive line

```bash
python -I eda/c120_s036_safe_stage2.py oof          # nested OOF reproduction (per-fold form, inductive alignment)
python -I eda/c120_s036_safe_stage2.py fit --save   # → artifacts/stage2_s036_safe.json
```

| File | SHA-256 | Content |
|---|---|---|
| `artifacts/stage2_s036_safe.json` | `7acd67bc7602369b75838870f1e362a2b4065a7deb3dca515dc81a9dedaa145f` | Conditional logit over the first 5 candidates after the inductive alignment, feature set `fallback` (40 features, no notes), NONE guard, `align_mode: inductive` |

Checks: anchor of the inductive line per fold MAP@3 0.81844; deployment form with gate and NONE guard 0.82333
(+0.0049 ± 0.0022, bootstrap over questions); the gate does not abstain (inner 0.8247 against base 0.8184); the
copy null point (weight 0) reproduces the inductive line bit for bit. The fit uses the same per-fold form.

### 13.3 Safety line with alignment over the test set

Run by `OVERRIDE_CV_II=override_cv_condition_ii bash tools/builds/safety_final.sh`:

```bash
python -I eda/c123_safety_set.py oof                   # nested reproduction: variants a and b, copy null points, placebos
python -I eda/c123_safety_set.py splits --n 20         # 20 new meta-splits: b − a
python -I eda/c123_safety_set.py splits-inductive      # evaluation only, not run by the build script: a − strictly inductive form
python -I eda/c123_safety_set.py test-mass --name v1r_zs27t_think18sb_pr27xm_m3x3_full_ar --align-mode set -- <sources>
python -I eda/c123_safety_set.py test-mass --name v1r_zs27t_think18sb_pr27xm_m3x3_full_ar --align-mode inductive -- <sources>
python -I eda/c123_safety_set.py rule --test-report artifacts/eda/private/safety_set/c123_test_mass_set.json \
  --override-ii override_cv_condition_ii               # safety-set rule; without the switch: exit code 5
python -I eda/c123_safety_set.py fit --override-ii override_cv_condition_ii --save   # → artifacts/stage2_safety_final.json
python -I eda/c123_safety_set.py fit --probe --variant a --out artifacts/eda/private/safety_set/probe/stage2_safety_a_probe.json --save
```

`<sources>` are the test-side sources of the strictly inductive line without the alignment, REL-01 and second-stage
switches (as in the build script).

| File | SHA-256 | Content |
|---|---|---|
| `artifacts/stage2_safety_final.json` | `9b6dfbed3ac97a6083e45bab68a52b55118c3b0e3a35dd58371bad47c6e8c81a` | Conditional logit over the first 5 candidates after the alignment as one set, feature set `fallback_i1` (variant b, 55 features, no notes), NONE guard, `align_mode: set`, `align` 0.75, `align_min` 3; the field `override_ii` names the decision file |
| `artifacts/eda/private/safety_set/probe/stage2_safety_a_probe.json` | `49c37d1e5cb903e26adc5f7af4894ea3bb9ea32aa6ca1ae9f68a45c3b057b546` | The same fit for variant a (`fallback`, 40 features), marked as a probe: used only by the copy null point of variant b |

Checks: base as one set 7,377 of 9,000 sixths; variant a with gate and guard 7,430 / 7,404 / 7,391 per fold / without the
last step / as one set, variant b 7,450 / 7,417 / 7,395; copy null points a (weight 0 -> base) and b (I1 columns 0 -> a)
exact; test-side class masses after the alignment L1 0.028, with the inductive alignment 0.087. The deployment fit on
all 1,500 rows does not abstain (inner 0.8269 against base 0.8197).

### 13.4 Safety line: note mentions and example similarity

Run by `tools/builds/safety_nd.sh` (after its step 0, which builds `safety_final` and `primary_final`):

```bash
python -I eda/c129_safety_nd.py export-notes --verify   # the frozen M3 training notes d1–d3 = recomputation from the ranking runs
python -I eda/c129_safety_nd.py oof                     # nested reproduction: copy null point (ND columns 0 -> safety_final), 15 placebos
python -I eda/c129_safety_nd.py splits --n 20           # 20 new meta-splits (only if the file is absent; SAFETY_ND_SPLITS=1 recomputes it)
python -I eda/c129_safety_nd.py rule                    # pre-registered rule with the risk rule; exit code 5 if missed
python -I eda/c129_safety_nd.py fit --save              # → artifacts/stage2_safety_nd.json (--verify if the file exists)
```

| File | SHA-256 | Content |
|---|---|---|
| `artifacts/stage2_safety_nd.json` | `50217a6c48f16bf35afb97fd755611eba85c43fa81187d0d00a91dbb31cddbee` | Conditional logit after the alignment as one set, feature set `fallback_i1_nd` (58), NONE guard, `align_mode: set`; the frozen M3 test and training notes with their checksums, the references of the note and similarity guards (`guard_reference`, `sim_reference`), `guard_rule` with `abort_only` |
| `artifacts/eda/private/safety_nd/c129_splits.json` | `703703d46c913b6d5607c47183b734980cfb919d14283c2d83ac9e6c634e891e` | The 20 new meta-splits (pinned; recomputed only if absent) |

Checks: the copy null point holds bit for bit (the three columns at 0 reproduce the rows of `safety_final`); ½(F+M) +0.00322
± 0.00150 against `safety_final` (per fold 7,480 sixths, as one set 7,447), placebo mean +0.00048; over 20 new meta-splits
+0.00283 ± 0.00024; risk rule P(θ < −0.002) 0.0001. The deployment fit on all 1,500 rows does not abstain; its features
equal the evaluated rows of the nested run (`eval_features_sha256`).

### 13.5 Safety line: regex-exact note columns

Run by `tools/builds/safety_ndr.sh` (after its step 0, `safety_nd.sh`):

```bash
python -I eda/c129_safety_nd.py export-notes --verify   # M3 training notes, as in §13.4
python -I eda/c130_safety_ndr.py export-notes --verify  # the three frozen protocol note files = re-export from the raw answers
python -I eda/c130_safety_ndr.py oof                    # nested reproduction: R columns 0 -> safety_nd bit for bit, 15 placebos
python -I eda/c130_safety_ndr.py swap                   # full draw swap over the four sets of three from d1–d4 (only if absent)
python -I eda/c130_safety_ndr.py splits --n 20          # 20 new meta-splits (only if absent; SAFETY_NDR_RECOMPUTE=1 recomputes both)
python -I eda/c130_safety_ndr.py rule                   # pre-registered rule (1)–(4) with the risk rule; exit code 5 if missed
python -I eda/c130_safety_ndr.py fit --save             # → artifacts/stage2_safety_ndr.json (--verify if the file exists)
```

| File | SHA-256 | Content |
|---|---|---|
| `artifacts/stage2_safety_ndr.json` | `bfd8f197fccb3ad9cc623fe2d45930151ac0debf40be3eb5cbafb953c79b2927` | As `stage2_safety_nd.json` with the feature set `fallback_i1_ndr` (64), the frozen protocol test notes (`r_notes`: extractor version, columns, file and SHA-256) and the reference of the guard of the columns R (`r_reference`) |
| `artifacts/eda/private/safety_ndr/c130_swap.json`, `c130_splits.json` | `fd47ced958c232047fa4516c7b15a00fb583bc0181e63da2ad82c8be17dcc6e2`, `db40c8b8294963f3303c5e700b220868b9338b157c91f5c97b09d2281f1fb9ec` | Draw swap and 20 new meta-splits (pinned; recomputed only if absent) |

Checks: R columns 0 reproduce `safety_nd` bit for bit, and the first 58 columns equal those of `c129`; ½(F+M) +0.00072 ±
0.00108 against `safety_nd` (per fold 7,484 sixths, as one set 7,456), placebo mean −0.00012; draw swap +0.00099, +0.00026,
+0.00083, +0.00015 (mean +0.00056); 20 new meta-splits +0.00087 ± 0.00022; risk rule P(θ < −0.002) 0.0007 (original) and 0.0010
(mean of the sets). The deployment fit does not abstain.

### 13.6 Primary line: mean channel with the M3 ranking behind the protocol notes

The rule of the channel was evaluated once, before the test side was computed:

```bash
python -I eda/c132_m3px27_eval.py --oof-save    # channel artifact oof_m3px27 from the five shards (fold fidelity, settings, notes)
python -I eda/c132_m3px27_eval.py --controls    # controls of the same role on the original and 20 new meta-splits (cache)
python -I eda/c132_m3px27_eval.py               # rule -> artifacts/eda/private/m3px27_eval/c132_m3px27_eval.json (verdict DEPLOY)
```

The build script `tools/builds/primary_m3px27.sh` reads the stored verdict and its row file (pinned) and recomputes the rest;
steps 2–4 always run with `--save`, which writes missing files and requires existing ones to be byte-identical (a package
without the report files or the OOF matrix builds the same files):

```bash
python -I eda/c132_m3px27_eval.py --oof-verify  # oof_m3px27 = recomputation from the five shards
python -I eda/c133_primary_m3px27.py oof --save    # mean channel m3px27xm → artifacts/oof_m3px27xm.npy, meta_m3px27xm.json
python -I eda/c133_primary_m3px27.py stack --save  # → artifacts/stack_primary_m3px27.json
python -I eda/c133_primary_m3px27.py fit --save    # the chain of the final primary form with m3px27xm → artifacts/stage2_primary_m3px27.json
python -I eda/c133_primary_m3px27.py copy --keep   # copy null point of the training side (m3px27 := untrained ranking)
python -I eda/c133_primary_m3px27.py test-guard    # completeness, NONE band, settings, note guard and training regime of the test ranking
```

| File | SHA-256 | Content |
|---|---|---|
| `artifacts/stage2_primary_m3px27.json` | `9321d597b175d672fe5b31bf284ce984fbce149cb2e2cd8941a31aa8070d2500` | The fields of `stage2_primary_final.json` with the channel `m3px27xm`: main model `main_guard_i1_mm` (64) with NONE guard, fallback model (40), widened note guard, β 0.6, K 5, λ 0.1, `mech27`; plus `m3px27_build` (verdict, evaluated rows, copy null points, deployment form with β per fold) |
| `artifacts/eda/private/m3px27_eval/c132_m3px27_eval.json` and `…_rows.npz` | `b3b9c7b87de3d7d25f017369c19e68aec219516a1dd75f9dfcf9455a7bc30ce0`, `cf4af62a74cd3d315d7a0e0c3eefe8f3b8151f9db3d36cd4d2855ee40b5f0fc5` | Stored verdict of the rule and its row vectors (pinned) |

Checks: the chain gives per fold / without the last step / as one set 0.84256 / 0.83456 / 0.83822 for the final form with
`m3px27xm` (with `pr27xm` 0.84089 / 0.83433 / 0.83978); its row vectors equal those of the evaluation bit for bit; with the
untrained ranking in place of `m3px27` the blend file, the 15 row vectors and the five deployment files of the chain are
reproduced (up to provenance and channel names). The deployment form of the training rows is the cross with β per fold
(fold 1 selects 0.7, the others 0.6; METHOD §4.7); the fixed-β variant is written as a sensitivity file only.

All ten deployment files are rebuilt character for character by the build scripts (`--verify`) or written by them on
the first run (`--save`), and pinned there by checksum.

## 14. Scripts per channel

Every script that a training, OOF or test run of the deployed channels needs, with its role. Scripts not in this repository
are marked; their outputs are frozen inputs ([CODEMAP.md](CODEMAP.md#10-steps-whose-scripts-are-not-part-of-this-repository)).

| Channel | Training and OOF side | Test side |
|---|---|---|
| all | `tools/make_folds.py` (fold array), `src/vlm/runner.py` (all model runs) | `tools/make_submission.py` (assembly), `tools/builds/*.sh` (build scripts), `tools/make_blank_list.py` (blank-photo list; `check` compares with the frozen list) |
| `v1` | runner (`train`, `bench`), `eda/a6_assemble_oof.py` | runner (`bench` on the test table) |
| `v1r` | `tools/orient_net.py` (maps, training), `tools/make_orient.py`, runner (`rot_f*`), `eda/a6_assemble_oof.py` | `tools/orient_net.py infer`, `tools/orient_list.py` (choice and checks), runner (`rot_test`) |
| `zs27q`, `zs27t` | runner (`bench`), `eda/a6_assemble_oof.py` | runner (`bench`) |
| `think18nb`, `think18sb` | `tools/replay/gates.py`, runner (`think_gen` + scoring), `eda/a6_assemble_oof.py` | `tools/replay/gates.py check`, runner |
| `pr27xm` | runner (`proto_gen`, `proto_rank`), `eda/a8_proto_oof.py`, `eda/c95_rank_mean.py`, `tools/export_proto_notes.py` | runner, `tools/export_proto_notes.py --allow-test` |
| `m3px27xm` | `tools/make_m3px27_notes.py`, runner (M3 ranking behind the notes), `eda/c132_m3px27_eval.py`, `eda/c133_primary_m3px27.py` | runner, `tools/make_m3px27_notes.py`, `eda/c133_primary_m3px27.py test-guard` |
| `m3x3_full` | runner (`train`, `merge`, `sft_gen`, ranking), M3 targets (`tools/make_m3_targets.py`), combination of fold runs (frozen, script not included), `eda/a9_m3_draws_oof.py`, `eda/c108_m3_three_draws.py` | runner, `tools/freeze_m3_texts.py`, `eda/a9_m3_draws_oof.py --gen-status`, `eda/c99_m3_deploy.py --variant` |
| `q397m3` | `tools/open_teacher.py` (development only), `eda/c110_q397_deploy.py`, `tools/make_q397m3_notes.py`, runner, `eda/c117_q397m3_deploy.py`, `eda/c115_q397m3_eval.py` | runner, `tools/make_q397m3_notes.py`, `eda/c117_q397m3_deploy.py test-guard`, `eda/c110_q397_deploy.py test-status`; parity check (stored verdict, script not included) |
| Blends | `eda/c99_m3_deploy.py`, `eda/c110_q397_deploy.py`, `eda/c107_q397_full.py`, `eda/a2_probe_stack.py`, `eda/c21_margin_align.py` | `tools/make_submission.py` |
| Second stages | `eda/c118_stage2.py`, `eda/c120_s036_safe_stage2.py`, `eda/c121_stage2_i1.py`, `eda/c123_safety_set.py`, `eda/c124_stage2_mm.py`, `eda/c129_safety_nd.py`, `eda/c130_safety_ndr.py`, `eda/c133_primary_m3px27.py`, `eda/risk_rule.py`, `src/stage2.py`, `src/answer_match.py`, `src/note_sim.py`, `src/note_regex.py` | `tools/make_submission.py --stage2`, `eda/c121_stage2_i1.py submit` |

## 15. Reproducibility of training

- **CPU steps are bit-identical:** folds, OOF assembly, blend weights, second stages and the orientation network.
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
- **What a retraining gives.** The same commands, configurations and seeds give new adapters, new OOF matrices and new
  test outputs, so every downstream checksum changes: the blend weights and second stages are refitted on the CPU from the
  new OOF matrices (deterministic for given inputs), the pre-registered rules of §13 have to be evaluated again, and the
  pins of the build scripts no longer apply. Scores are expected to move within the spread measured on this data, not to
  repeat:
  - a new draw of a generated text: two greedy runs with the same prompt agree on the top code in 84 % of rows; the three
    M3 OOF draws (with different generation settings) reach 0.7621, 0.767 and 0.7701 as single channels; a rerun of the
    protocol answers changed the blend's top code on 8–9 % of rows of an error-heavy sample without a change in MAP@3
    ([INFERENCE.md](INFERENCE.md#63-reference-hardware-and-expected-agreement));
  - other hardware or number format: between an A40 in bf16 and a T4 in fp16 the 9B scorer's top code changes on about 9 %
    of rows and the blend's on 1–2 %;
  - the uncertainty of a single cross-validated value is about ±0.01 MAP@3 (`v1` 0.6406 ± 0.0105), so two complete
    retrainings are compared with the paired bootstrap of [VALIDATION.md](VALIDATION.md#51-paired-comparisons), not by
    their difference alone.

  A complete retraining of the adapters has not been measured. The spread of a rerun with all generated texts replaced at
  once is estimated at about ±0.003 MAP@3 (1 sd) on the private rows
  ([VERIFICATION.md §8.1](VERIFICATION.md#81-how-close-a-full-rerun-should-come)).

## 16. Final submissions (version 0.4)

Both final submissions keep the channels, the training side and the recipe of the primary candidate of version 0.3
(`primary_m3px27`, §13.6) and change only the M3 channel and the second stage. Everything below runs on the CPU from frozen
outputs; the only new GPU work were the draws d4 (OOF and test) and h (OOF and test), with the configurations of §10.3 and
[INFERENCE.md](INFERENCE.md#61-test-side-steps).

**16.1 Submission B: four M3 draws (`eda/c135_primary_m3x4.py`, build `tools/builds/primary_m3x4.sh`).** The channel
`m3x4_full` is the mean of the four OOF draws d1, d2, d3, d4 in this fixed order (`eda/a9_m3_draws_oof.py` rule; the first
three are `m3x3_full` bit for bit); it replaces `m3x3_full` in the recipe of `primary_m3px27`. The second-stage features
still read d1–d3 (field `draws_first: 3`), as the test side does.

```bash
python -I eda/c135_primary_m3x4.py oof --save      # artifacts/oof_m3x4_full.npy + meta (the first three draws = m3x3_full)
python -I eda/c135_primary_m3x4.py stack --save    # blend: θ on LogLoss over all 1,500 rows, β by the rule of the primary line
python -I eda/c135_primary_m3x4.py fit --save      # the second-stage chain of the primary line with the swapped channel
python -I eda/c135_primary_m3x4.py copy --keep     # copy null point "d4 removed": blend, chain and files = primary_m3px27
python -I eda/c135_primary_m3x4.py test-guard      # completeness and settings of the d4 test ranking (aggregates only)
# evidence of the four-draw rule (pre-registered; computed once, the build path does not need it):
python -I eda/c135_primary_m3x4.py bank --seeds orig,$(seq -s, 1000 1019) && python -I eda/c135_primary_m3x4.py rule
```

**16.2 Intermediate form: extra second-stage columns (`src/stage2_sources.py`, `eda/c151_primary_cols.py`, build
`tools/builds/primary_cols.sh`).** Submission B plus nine columns from eight sources at the end of the main model of the
second stage; the blend is unchanged. The columns are computed on the CPU from frozen files:
- the protocol answers of our 27B (training rows `artifacts/runs/pr27_rp`, test rows `pr27te_rp`) and of the 397B (training
  rows through the hosted sampling API, frozen as `artifacts/runs/q397_tinker/…/proto_labels.jsonl` (parsed answers) and
  `data/derived/open_teacher/qwen35-397b/raw.jsonl` (raw texts), from which `tools/make_q397m3_notes.py` builds the
  training notes `data/derived/sft_q397m3/q397m3_train_notes.jsonl` byte for byte (§11); test rows by our own FP8 run
  `q397te_rp`, parsed with the parser of the training side);
- the blend channels and the M3 draws of the line, the taxonomy (sections, default severity) and, for the label-dependent
  columns, the training labels of the reference rows only (the other folds for a training row; all 1,500 rows for a test
  row).

```bash
python -I eda/c151_primary_cols.py test-sources    # completeness, parser cross-check and checksums of the test-side sources
python -I eda/c151_primary_cols.py fit --save --cols qfp   # second stage with the nine columns -> artifacts/stage2_primary_cols.json
python -I eda/c151_primary_cols.py copy --cols qfp # copy null point: columns := 0 -> rows and models of submission B
python -I eda/c151_primary_cols.py measure --cols qfp --seeds orig,4000-4019   # paired measurement over meta-splits (evidence)
```

The deployment file records the definitions of the columns, the checksums of every source file (training and test side),
the checksums of the training-side column matrices, the reference values and the rule of the test guard per column and
source (band ±15 % relative or |z| ≤ 3; valid answers per source ≥ 0.97) and the positive controls.

**16.3 Submission A: fifth M3 draw on the rows with small script (`eda/c166_primary_m3x5h.py`, build
`tools/builds/primary_m3x5h.sh`).** The channel `m3x5h` is per row the mean of the draws that cover the row: d1–d4 on all
rows and draw h on the rows of the list "character height ≤ 20 px" (training 1,001 of 1,500 rows, test 2,459 of 3,498;
`tools/make_charsize_rows.py --max-height 20 --name h20`: the median height of the connected components in the photo at
long side 1,024 px, a label-free statistic of the image; the threshold is the lower two-thirds quantile of the training
photos). Outside the list the channel equals `m3x4_full` bit for bit. Blend and second stage are refitted as for
submission B, and the nine columns of the intermediate form are fitted again on top.

```bash
python -I eda/c166_primary_m3x5h.py digest --side train   # digest over the 42 files of draw h and the lists (pinned)
python -I eda/c166_primary_m3x5h.py oof --save       # artifacts/oof_m3x5h.npy + meta (field rank.row_list_draws)
python -I eda/c166_primary_m3x5h.py stack --save     # blend with m3x5h in place of m3x4_full
python -I eda/c166_primary_m3x5h.py fit-base --save  # chain of submission B with m3x5h -> artifacts/stage2_primary_m3x5h_base.json
python -I eda/c166_primary_m3x5h.py fit --save       # plus the nine columns -> artifacts/stage2_primary_m3x5h.json
python -I eda/c166_primary_m3x5h.py copy --keep      # copy null point "empty list": blend, chain and columns = submission B / intermediate form
python -I eda/c166_primary_m3x5h.py h-guard --spec <test shards>   # test guard of draw h (aggregates only)
python -I eda/c166_primary_m3x5h.py measure --seeds orig,11000-11019   # evidence: the five-draw rule over the 20 untouched meta-splits
```

The build script runs the measurement only on request (`PRIMARY_M3X5H_MEASURE=short` for the original split and two
meta-splits, `PRIMARY_M3X5H_MEASURE=full` for all 20); the deployment does not depend on it.

**16.4 Checks of the training side.** In each build the nested recomputation of the second stage must reproduce pinned
values (in sixths of a row; for example the MM rows of submission A 7,584 / 7,542 and with columns 7,583 / 7,564 per fold /
as one set), the copy null points must reproduce the predecessor's files, and with weight 0 the new columns must leave the
file without second stage unchanged. The CV evidence of the three steps is in [VALIDATION.md](VALIDATION.md#8-final-submissions-version-04)
and [RESULTS.md](RESULTS.md).
