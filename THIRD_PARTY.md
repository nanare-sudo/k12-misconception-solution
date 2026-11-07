# Third-party components

*Status: 2026-11-01 (version 0.1).*

Our own code and documentation are licensed under the Apache License 2.0 ([LICENSE](LICENSE)). The components below
belong to third parties and keep their own licences. Model licences were read from the licence files at the pinned
revisions unless stated otherwise. Package licences follow the projects' published metadata.

> **Planned.** A complete per-package licence table will be generated from the wheel metadata of the offline wheelhouse.

## 1. Models

| Model | Source and revision | Role in the solution | Licence | In the final inference |
|---|---|---|---|---|
| Qwen3.5-9B | `Qwen/Qwen3.5-9B` @ `c202236235762e1c871ad0ccb60c8ee5ba337b9a` | Base model of the 9B fine-tuned scorer (our LoRA adapter `v1_full`) | Apache 2.0 | yes |
| Qwen3.8-27B | `Qwen/Qwen3.8-27B` @ `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` | Zero-shot scorers, reason-then-name channel, protocol channel; base model of M3 (our LoRA adapter `m3_full`) | Apache 2.0 | yes |
| Qwen3.5-397B-A17B-FP8 | `Qwen/Qwen3.5-397B-A17B-FP8` @ `ea5b4f81096f3901c91dea97f81324302495781d` | 397B notes on the test side (primary line only) | Apache 2.0 | yes (primary line) |
| Qwen3.5-397B-A17B | Served by the Tinker sampling API | 397B outputs on training rows (primary line only) | Apache 2.0 according to the model card; the licence file of the served revision is still to be confirmed | no |
| Qwen3.5-0.8B | `Qwen/Qwen3.5-0.8B` @ `2fc06364715b967f1860aea9cf38778875588b17` | CPU smoke tests of the runner and tests that need a real model | Apache 2.0 | no |
| Claude Haiku 5.5 (Anthropic) | Anthropic Messages API | Development-time teacher: pseudo-labels for the unlabelled pool (frozen) | Provider's commercial terms | no |

On Kaggle, the 27B weights were read from a public Kaggle mirror whose files are hash-identical to the pinned Hugging
Face revision. The mirror is only a means of transport; the solution references the Hugging Face revision and its file
checksums.

**Evaluated during development, not part of the solution:** Gemma 4 31B (Apache 2.0), Inkling-Small (through the
Tinker API; Apache 2.0 according to the model card), Claude Sonnet 5.5 (teacher comparison and a description
experiment on training rows, through the Anthropic API), PaddleOCR-VL 1.6, PP-FormulaNet_plus-M and
Uni-MuMER-Qwen3.5-2B (reader experiments; Apache 2.0 according to their licence files or model cards).

## 2. Services used during development

| Service | Use | Test rows sent |
|---|---|---|
| Anthropic Messages API | Teacher pseudo-labels for the unlabelled pool; teacher measurements on training rows | never |
| Tinker sampling API (Thinking Machines) | Measurements of open models on training rows; 397B outputs on training rows | never |
| RunPod Secure Cloud | Rented GPUs (mainly A40, RTX PRO 6000 and 4×H200) for training and inference; instances deleted after each run | yes, for inference with open-weight models |
| Kaggle Notebooks | T4×2 GPUs for offline inference (9B scorer, 4-bit 27B), checks on a second hardware class and a reader probe on training rows | yes, for inference with open-weight models |

None of these services is needed to run `assemble`. The APIs are not part of the final inference.

## 3. Python packages

Lock files with hashes for every package are planned ([docs/ROADMAP.md](docs/ROADMAP.md)). The main packages and their
pinned versions (see also [docs/INFERENCE.md](docs/INFERENCE.md#4-environments)):

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
| FlashInfer (`flashinfer-python`) | 0.7.0.post1 (vLLM) | Apache-2.0 |
| Tokenizers | 0.22.2 (GPU base, CPU), 0.23.3 (vLLM) | Apache-2.0 |
| Safetensors | 0.8.0 | Apache-2.0 |
| Hugging Face Hub (`huggingface-hub`) | 1.33.0 (GPU base, vLLM), 1.9.0 (CPU) | Apache-2.0 |
| NumPy | 2.1.2 (GPU base), 2.3.5 (vLLM), 2.4.2 (CPU) | BSD-3-Clause |
| SciPy | 1.17.1 (CPU) | BSD-3-Clause |
| pandas | 3.1.0 (GPU base), 2.3.3 (CPU) | BSD-3-Clause |
| scikit-learn | 1.8.0 (CPU) | BSD-3-Clause |
| Pillow | 11.0.0 (GPU base), 13.0.0 (vLLM), 11.3.0 (CPU) | MIT-CMU (HPND) |
| PyArrow | 26.0.0 (GPU base), 23.0.1 (CPU) | Apache-2.0 |
| PyYAML | 6.0.3 | MIT |
| OpenCV (`opencv-python-headless`) | 5.0.0.93 (vLLM) | Apache-2.0 |
| uv | 0.13.0 (environment setup) | MIT or Apache-2.0 |
| pytest | 8.4.2 (tests only) | MIT |

## 4. System components

| Component | Use | Licence |
|---|---|---|
| NVIDIA CUDA runtime libraries (`nvidia-*` wheels) | GPU computation; installed with PyTorch and vLLM | NVIDIA proprietary licence (redistributable runtime components) |
| Docker image `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` | Base system of the GPU training and inference environment (Ubuntu 24.04, Python 3.12) | Image of the provider; contains PyTorch (BSD-3-Clause) and the CUDA runtime (NVIDIA licence) |
| Kaggle GPU image | Base system of the Kaggle T4 runs (Python 3.13, torch 2.11.0+cu128) | Provided by Kaggle |
