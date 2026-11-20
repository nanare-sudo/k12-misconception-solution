# Third-party components

*Version 0.6, 2026-11-21.*

Our own code and documentation are licensed under the Apache License 2.0 ([LICENSE](LICENSE)). The components below
belong to third parties and keep their own licences. We read the model licences from the licence files at the pinned
revisions, unless a table says "according to the model card". Package licences follow the projects' published metadata.

In short:
- External datasets: none. All data come from the competition: the training, test and unlabelled sets, the taxonomy and
  the question diagrams
  ([competition data](https://www.kaggle.com/competitions/misconception-detection-in-k-12-student-math-workings/data);
  the competition rules apply).
- Pre-trained models at inference time: three open-weight Qwen models under Apache 2.0 (section 1.1), plus our own
  adapters and orientation network.
- Commercial model APIs: the Anthropic Messages API with Claude Haiku 5.5 and Claude Sonnet 5.5, and the Tinker sampling
  API for open models. We used them only during development, and only on training and unlabelled rows (section 2.1).
- API cost of one inference run on the test set: 0 USD, because the test inference calls no API. API cost of the whole
  development: 108.81 USD.

## 1. Models

### 1.1 Models of the solution

| Model | Source and revision | Role in the solution | Licence | In the final inference |
|---|---|---|---|---|
| Qwen3.5-9B | [`Qwen/Qwen3.5-9B`](https://huggingface.co/Qwen/Qwen3.5-9B/tree/c202236235762e1c871ad0ccb60c8ee5ba337b9a) @ `c202236235762e1c871ad0ccb60c8ee5ba337b9a` | Base model of the 9B scorer (our LoRA adapter `v1_full`) | Apache 2.0 | yes |
| Qwen3.8-27B | [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B/tree/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0) @ `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` | Zero-shot scorers, reasoning channel, protocol channel; base model of M3 (our LoRA adapter `m3_full`) | Apache 2.0 | yes |
| Qwen3.5-397B-A17B-FP8 | [`Qwen/Qwen3.5-397B-A17B-FP8`](https://huggingface.co/Qwen/Qwen3.5-397B-A17B-FP8/tree/ea5b4f81096f3901c91dea97f81324302495781d) @ `ea5b4f81096f3901c91dea97f81324302495781d` | 397B notes and protocol answers on the test side (both final submissions; in A also a source of extra second-stage columns) | Apache 2.0 | yes |
| Qwen3.5-397B-A17B | [`Qwen/Qwen3.5-397B-A17B`](https://huggingface.co/Qwen/Qwen3.5-397B-A17B), served by the Tinker sampling API; the provider's revision and number format were not recorded | 397B outputs on the training rows: blend weight of the 397B channel; in A the training side of six extra second-stage columns | Apache 2.0: the Hub repository and its licence file at the main revision `8472618…` (checked 2026-11-20); the revision that the provider served was not recorded | no; training side only, as frozen outputs |
| Qwen3.5-0.8B | [`Qwen/Qwen3.5-0.8B`](https://huggingface.co/Qwen/Qwen3.5-0.8B/tree/2fc06364715b967f1860aea9cf38778875588b17) @ `2fc06364715b967f1860aea9cf38778875588b17` | Tokenizer of the training-side builders (M3 targets, 397B notes); CPU smoke tests of the runner and tests that need a real model | Apache 2.0 | no |
| Claude Haiku 5.5 (Anthropic) | [Anthropic Messages API](https://docs.anthropic.com/), model ID `claude-haiku-5-5` | Teacher during development: pseudo-labels for the unlabelled pool, frozen | Provider's commercial terms | no; only its frozen pool labels |

Every file of the three base models of the inference path is listed with its size and checksum in
[docs/weights_manifest.json](docs/weights_manifest.json).

On Kaggle, the 27B weights were read from a public Kaggle mirror,
[`ugvfpdcuwfnh/qwen3-8-27b`](https://www.kaggle.com/models/ugvfpdcuwfnh/qwen3-8-27b) (variant
`transformers/transformers-bf16`, version 1). Its files are hash-identical to the pinned Hugging Face revision; we
checked all 32 files. The 9B weights were downloaded at the pinned revision by a separate notebook of our account and
attached as its output. Both were only means of transport: the solution refers to the Hugging Face revisions and their
file checksums.

### 1.2 Models evaluated during development, not part of the solution

None of these models produced anything that the final submissions use.

| Model | Source and revision | Use | Licence |
|---|---|---|---|
| Qwen3.5-4B | [`Qwen/Qwen3.5-4B`](https://huggingface.co/Qwen/Qwen3.5-4B) @ `851bf6e806ef` | Early channel, namer and verifier experiments | Apache 2.0 |
| Gemma 4 31B | [`google/gemma-4-31B-it`](https://huggingface.co/google/gemma-4-31B-it) @ `842da3794eaa` | Reasoning text generator (412 training rows) | Apache 2.0 (Gemma 4 licence page) |
| Gemma 4 31B, 4-bit release | [`google/gemma-4-31B-it-qat-w4a16-ct`](https://huggingface.co/google/gemma-4-31B-it-qat-w4a16-ct) @ `52f3f65bc7a0` | Protocol channel on the 1,500 training rows | Apache 2.0 (Gemma 4 licence page) |
| gpt-oss-20b | [`openai/gpt-oss-20b`](https://huggingface.co/openai/gpt-oss-20b/tree/6cee5e81ee83917806bbde320786a8fb61efebee) @ `6cee5e81ee83917806bbde320786a8fb61efebee` | Text-only judge over our frozen notes, training rows | Apache 2.0 |
| GLM-4.1V-9B-Thinking | [`zai-org/GLM-4.1V-9B-Thinking`](https://huggingface.co/zai-org/GLM-4.1V-9B-Thinking) @ `3c1471e51dc8` | Protocol channel on training and test rows, on our own hardware | MIT according to the model card |
| Kimi-VL-A3B-Thinking-2506 | [`moonshotai/Kimi-VL-A3B-Thinking-2506`](https://huggingface.co/moonshotai/Kimi-VL-A3B-Thinking-2506) @ `aa1730989e75` | Protocol channel on training and test rows, on our own hardware | MIT according to the model card |
| InternVL3.5-8B | [`OpenGVLab/InternVL3_5-8B-HF`](https://huggingface.co/OpenGVLab/InternVL3_5-8B-HF/tree/741a7d03020411e666c6109218ab71e08151ef86) @ `741a7d03020411e666c6109218ab71e08151ef86` | Small image-based verifier experiment | Apache 2.0 according to the model card |
| PaddleOCR-VL 1.6 | [`PaddlePaddle/PaddleOCR-VL-1.6`](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6/tree/c5630abae1d940eafe0697512a0325494b02ab42) @ `c5630abae1d940eafe0697512a0325494b02ab42`, with PaddleOCR 3.7.0, PaddlePaddle 3.2.1 and PP-DocLayoutV3 | Reader experiments on training rows | Apache 2.0 (PP-DocLayoutV3 according to the model card) |
| PP-FormulaNet_plus-M | [`PaddlePaddle/PP-FormulaNet_plus-M`](https://huggingface.co/PaddlePaddle/PP-FormulaNet_plus-M/tree/712e6e2e4c313b1ea163be5c350127b82662c58d) @ `712e6e2e4c313b1ea163be5c350127b82662c58d` | Formula reader experiment on training rows | Apache 2.0 according to the model card |
| Uni-MuMER-Qwen3.5-2B | [`phxember/Uni-MuMER-Qwen3.5-2B`](https://huggingface.co/phxember/Uni-MuMER-Qwen3.5-2B/tree/40a6288292057f1c162b3b0eaccd362036dbd495) @ `40a6288292057f1c162b3b0eaccd362036dbd495` | Handwritten-formula reader experiment on training rows | Apache 2.0 according to the model card |
| all-MiniLM-L6-v2 | [`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/tree/1110a243fdf4706b3f48f1d95db1a4f5529b4d41) @ `1110a243fdf4706b3f48f1d95db1a4f5529b4d41` | Base of a trained text verifier (cross-encoder), training rows | Apache 2.0 |
| BGE-small-en-v1.5 | [`BAAI/bge-small-en-v1.5`](https://huggingface.co/BAAI/bge-small-en-v1.5/tree/5c38ec7c405ec4b44b94cc5a9bb96e735b38267a) @ `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a` | Text embeddings of training findings, for analysis | MIT according to the model card |
| Qwen2.5-VL-3B-Instruct | [`Qwen/Qwen2.5-VL-3B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/tree/66285546d2b821cf421d4f5eb2576359d3770cd3) @ `66285546d2b821cf421d4f5eb2576359d3770cd3` | Synthetic hardware test on a TPU only; no competition data | Qwen Research License (non-commercial; evaluation only) |
| Inkling-Small (Thinking Machines) | [`thinkingmachines/Inkling-Small`](https://huggingface.co/thinkingmachines/Inkling-Small), through the Tinker sampling API | Protocol channel measurement on training rows | Apache 2.0 according to the model card |
| Qwen3.8-27B through the Tinker sampling API | as in section 1.1 | Control measurement of our own base model in the same protocol, training rows | Apache 2.0 |
| Claude Sonnet 5.5 (Anthropic) | [Anthropic Messages API](https://docs.anthropic.com/), model ID `claude-sonnet-5-5` | Teacher comparison and a description experiment on training rows | Provider's commercial terms |
| Claude Opus 5.5 (Anthropic) | [Anthropic Messages API](https://docs.anthropic.com/), model ID `claude-opus-5-5` | One connectivity check of the API key, without competition data | Provider's commercial terms |

## 2. Commercial model APIs and other services

### 2.1 Model APIs, during development only

No test row, no test photo and nothing derived from a test row was ever sent to a model API. The test inference calls no
API. It reads the frozen API outputs named in the column "Used by the final submissions".

| API | Model and version (model ID as requested) | Period | Purpose | Rows sent | Used by the final submissions | Cost |
|---|---|---|---|---|---|---:|
| [Anthropic Messages API](https://docs.anthropic.com/), batch mode | Claude Haiku 5.5, `claude-haiku-5-5`, effort `xhigh` | 2026-10-11 to 2026-11-07 | Pseudo-labels of all 50,000 unlabelled workings (49,979 usable, about 31 USD); teacher measurements on training rows; two experiments that were not adopted: teacher rationales on unlabelled rows as fine-tuning targets, and a per-question table of the wrong final answer that each misconception would produce (question text, answer key and diagram of training and unlabelled questions only, no student working) | unlabelled and training rows | the frozen pool pseudo-labels, through the per-question task block of two channels | 40.85 USD |
| Anthropic Messages API | Claude Sonnet 5.5, `claude-sonnet-5-5` | 2026-10-11 to 2026-11-01 | Teacher comparison on 120 training rows; a description experiment on training rows | training rows | no | 4.76 USD |
| Anthropic Messages API | Claude Opus 5.5, `claude-opus-5-5` | 2026-10-11 | One connectivity check | none, no competition data | no | 0.00 USD |
| [Tinker sampling API](https://thinkingmachines.ai/tinker/) (Thinking Machines) | Qwen3.5-397B-A17B (renderer `qwen3_5`; the provider's revision and number format were not recorded) | 2026-10-13 to 2026-10-31 | Protocol answers on all 1,500 training rows (frozen), after an earlier measurement on a stratified sample | training rows | the frozen training-side answers: blend weight of the 397B channel; in A the training side of six extra columns | 55.89 USD |
| Tinker sampling API | Inkling-Small | 2026-10-13 | Measurement on training rows | training rows | no | 5.96 USD |
| Tinker sampling API | Qwen3.8-27B | 2026-10-13 | Control measurement on training rows | training rows | no | 1.35 USD |
| Total | | | | | | 108.81 USD (Anthropic 45.61, Tinker 63.20) |

The costs are the sums of our cost ledgers over all calls, including experiments that were not adopted.

Cost of one inference run on the test set: 0 USD for model APIs. The open-weight models of section 1.1 run on our own or
rented GPUs; the rough GPU time and cost of a test-side run are in
[docs/VERIFICATION.md](docs/VERIFICATION.md#3-hardware-and-run-times). Re-creating the frozen API outputs is not part of
a test inference run. It would cost about 31 USD for the pool pseudo-labels and about 44 USD for the 397B answers on the
1,500 training rows. It would also give new draws, not the delivered files.

### 2.2 Compute and other services

| Service | Use | Test rows sent |
|---|---|---|
| [RunPod](https://www.runpod.io/) Secure Cloud | Rented GPUs for training and inference: mainly A40, RTX PRO 6000 and 4 × H200, and an A100 for checks on a second hardware class. The instances were deleted after each run | yes, for inference with open-weight models |
| [Kaggle Notebooks](https://www.kaggle.com/code) | T4 × 2 GPUs for inference without network access (9B scorer, 4-bit 27B), checks on a second hardware class, a reader probe on training rows, and the CPU rebuild of the final submissions without network access | yes, for inference with open-weight models |
| [Hugging Face Hub](https://huggingface.co/) | Download of the open-weight models at pinned revisions | no |

None of these services is needed to run `assemble`. The APIs of section 2.1 are not part of the final inference.

## 3. Python packages

The main packages and their pinned versions are listed below. The complete pinned lists, without hashes, are in
`env/requirements-*.txt` ([docs/INFERENCE.md §4](docs/INFERENCE.md#4-environments)). A licence table for every package
of the four environments is not included.

| Package | Versions (environment) | Licence |
|---|---|---|
| PyTorch (`torch`) | 2.8.0+cu128 (GPU base), 2.13.0 (vLLM), 2.11.0 (CPU assembly; Kaggle image) | BSD-3-Clause |
| torchvision | 0.23.0+cu128 (GPU base), 0.28.0 (vLLM), 0.26.0 (CPU) | BSD-3-Clause |
| Transformers | 5.15.1 (GPU base, CPU, Kaggle T4), 5.17.0 (vLLM) | Apache-2.0 |
| PEFT | 0.21.2 | Apache-2.0 |
| Accelerate | 1.15.0 | Apache-2.0 |
| bitsandbytes | 0.50.2 (GPU base, Kaggle T4) | MIT |
| flash-linear-attention, fla-core | 0.5.2 (GPU base) | MIT |
| Triton | 3.4.0 (GPU base), 3.7.1 (vLLM), 3.6.0 (CPU) | MIT |
| vLLM | 0.31.0 (vLLM) | Apache-2.0 |
| OpenAI Python client (`openai`, a dependency of vLLM; the jobs do not use it) | 3.41.0 in the lock file; 3.42.0 resolved in the 397B test run | Apache-2.0 |
| FlashInfer (`flashinfer-python`) | 0.7.0.post1 (vLLM) | Apache-2.0 |
| Tokenizers | 0.22.2 (GPU base, CPU), 0.23.3 (vLLM) | Apache-2.0 |
| Safetensors | 0.8.0 | Apache-2.0 |
| Hugging Face Hub (`huggingface-hub`) | 1.33.0 (GPU base, vLLM), 1.9.0 (CPU) | Apache-2.0 |
| NumPy | 2.1.2 (GPU base), 2.3.5 (vLLM), 2.4.2 (CPU) | BSD-3-Clause |
| SciPy | 1.17.1 (CPU) | BSD-3-Clause |
| pandas | 3.1.0 (GPU base), 2.3.3 (CPU) | BSD-3-Clause |
| scikit-learn | 1.8.0 (CPU; the TF-IDF vectoriser of the example similarity in the safety line, `src/note_sim.py`) | BSD-3-Clause |
| Pillow | 11.0.0 (GPU base), 13.0.0 (vLLM), 11.3.0 (CPU) | MIT-CMU (HPND) |
| PyArrow | 26.0.0 (GPU base), 23.0.1 (CPU) | Apache-2.0 |
| PyYAML | 6.0.3 | MIT |
| OpenCV (`opencv-python-headless`) | 5.0.0.93 (vLLM); 4.13.0.92 on the CPU, only to recompute the image statistics and the rotation list from the photos (docs/VERIFICATION.md §5.2) | Apache-2.0 |
| uv | 0.13.0 (environment set-up) | MIT or Apache-2.0 |
| pytest | 8.4.2 (tests only) | MIT |

## 4. System components

| Component | Use | Licence |
|---|---|---|
| NVIDIA CUDA runtime libraries (`nvidia-*` wheels) | GPU computation; installed with PyTorch and vLLM | NVIDIA proprietary licence (redistributable runtime components) |
| Docker image `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` | Base system of the GPU training and inference environment (Ubuntu 24.04, Python 3.12) | Image of the provider; contains PyTorch (BSD-3-Clause) and the CUDA runtime (NVIDIA licence) |
| Kaggle GPU image | Base system of the Kaggle T4 runs (Python 3.13, torch 2.11.0+cu128) | Provided by Kaggle |

## 5. Development tools, not part of the solution

| Tool | Use | Licence |
|---|---|---|
| Tinker Python SDK (`tinker` 0.33.1, `tinker_cookbook` 0.5.7) | Client of the Tinker sampling API (section 2.1), in a separate environment | Apache-2.0 |
| RunPod Python SDK (`runpod` 1.12.0) | Starting and deleting rented GPU instances | MIT |
| Kaggle CLI 2.2.4 | Data download, notebook runs, submissions | Apache-2.0 |
| AI coding assistants (Anthropic Claude Code, OpenAI Codex) | Writing code and documentation, running analyses | Provider terms |

The coding assistants are not part of the solution. None of their outputs is a label, pseudo-label, transcription or
prediction that the solution uses, and no test photo or per-row test output was shown to them.
