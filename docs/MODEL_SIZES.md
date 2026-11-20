# Model sizes

Reference document. To reproduce the submissions, start with [VERIFICATION.md](VERIFICATION.md).

*Version 0.6. Covers every model needed at **inference** to produce the two final submissions of the primary line
(primary submission A: five M3 draws, the fifth with the large photo on the rows with small script, plus the extra
second-stage columns; primary submission B: four M3 draws, no extra columns). Both submissions use exactly the same
models. Sizes of the base models come from the public file lists of the pinned Hugging Face revisions (retrieved
2026-11-14); sizes of our own weights from the stored files. GB means 10^9 bytes, GiB 2^30 bytes. File list with
checksums: [weights_manifest.json](weights_manifest.json).*

## 1. The limits

The host's statements in the competition forum:
- "Any model of the same size or smaller than Nemotron 550B." (rule on model size)
- "We would consider any model up-to 512 GB weights in 8-bit or lower quants irrespective of VRAM required for
  inference" (earlier statement on the total size)
- "Combined model together need to be ~512GB" and, on ensembles, "overall total models used weights should remain around
  Nemotron 550B only to be fair to all participants" (2026-11-10)

We therefore check three numbers: the parameters of each model against 550 B, the total parameters against 550 B, and
the total size of the weights against 512 GB, in two readings: the files as deployed, and the size in 8-bit.

## 2. Models used at inference

| Model | Revision or file | Format of the files | Format at run time | Parameters | Bytes on disk | Used for |
|---|---|---|---|---|---:|---|
| `Qwen/Qwen3.5-397B-A17B-FP8` | `ea5b4f81096f3901c91dea97f81324302495781d` (107 files) | FP8 E4M3 block-scaled, some tensors BF16 | as stored (vLLM, tensor parallel 4) | 397 B nominal (17 B active per token); 403,397,928,944 stored values (400.72 B FP8, 2.68 B BF16, 8,640 F32) | 406,198,638,888 | 397B protocol answers on the test rows |
| `Qwen/Qwen3.8-27B` | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` (32 files) | BF16 | BF16; for one channel quantised to 4-bit NF4 **at load time** (bitsandbytes), from the same files | 27,781,427,952 | 55,586,114,863 | zero-shot scorer, reasoning texts and their scoring, protocol answers, base of M3 |
| `Qwen/Qwen3.5-9B` | `c202236235762e1c871ad0ccb60c8ee5ba337b9a` (16 files) | BF16 | FP16 (converted at load time on Kaggle T4) | 9,653,104,368 | 19,329,393,661 | base of the fine-tuned 9B scorer |
| M3 adapter (LoRA r 32 on layers 40–63 of the 27B) | `adapter_model.safetensors` + `adapter_config.json` | F32 | merged into the 27B at run time, BF16 | 75,104,256 | 300,458,355 | the five M3 draws (draw h reads the photo at 2,048 px with the same adapter); M3 behind the protocol notes and behind the 397B notes |
| 9B adapter `v1_full` (LoRA r 32 on layers 20–31 of the 9B) | `adapter_model.safetensors` + `adapter_config.json` | F32 | FP16 with the base | 27,721,728 | 110,908,078 | the 9B scorer on the test rows (plain and rotation-corrected reading) |
| Orientation network (two small CNNs) | `orient_cnn.pt` | F32 | CPU | 853,080 (2 × 426,540) | 3,436,615 | rotation list of the test photos |

Our adapters and the orientation network are delivered in the private input package (`weights/`, download link in our
email to the host); the base models are public under Apache 2.0 and are referenced by revision and file checksum.

**Not models:** the blend weights and the second-stage files (JSON, under 0.2 MB in total), the rule that marks blank photos
(a brightness threshold), the row list "character height ≤ 20 px" (a threshold on a stored image statistic), the extra
second-stage columns (computed on the CPU from frozen protocol answers, the taxonomy and the training labels), and the
mechanical comparison of a final answer with the answer key.

**Counted once although used several times.** The 27B base model is one set of files. It runs in BF16 for the reasoning,
protocol and M3 steps, in 4-bit for the zero-shot scorer (quantised in memory when loaded) and as the base of the M3
adapter, which the runner merges at run time into a temporary copy (about 55.6 GB on the disk of the GPU machine; not a
delivered file). The M3 adapter is one file used by seven jobs (five draws, two rankings behind notes). The larger photo
of draw h changes the input, not the model.

## 3. Totals

**Reading (i): the files as deployed**

| | Bytes | GB | GiB |
|---|---:|---:|---:|
| Qwen3.5-397B-A17B-FP8 | 406,198,638,888 | 406.20 | 378.30 |
| Qwen3.8-27B | 55,586,114,863 | 55.59 | 51.77 |
| Qwen3.5-9B | 19,329,393,661 | 19.33 | 18.00 |
| M3 adapter, 9B adapter, orientation network | 414,803,048 | 0.41 | 0.39 |
| **Total** | **481,528,950,460** | **481.53** | **448.46** |

Against 512 GB the margin is **30.5 GB (6.0 %)**; if 512 GB is read as 512 GiB (549.8 GB), it is 68.2 GB. The total
includes all non-weight files of the three revisions (tokenizers, configurations, licence files; 93 MB together). The
vision encoders are part of the weight shards and therefore included.

**Reading (ii): 8-bit or lower**

Each stored value counted with at most one byte (the BF16 and F32 values of the three base models and the F32 adapters
counted as 8-bit):

| | Values (= bytes at 8 bit) | GB |
|---|---:|---:|
| Qwen3.5-397B-A17B-FP8 | 403,397,928,944 | 403.40 |
| Qwen3.8-27B | 27,781,427,952 | 27.78 |
| Qwen3.5-9B | 9,653,104,368 | 9.65 |
| Adapters and orientation network | 103,679,064 | 0.10 |
| **Total** | **440,936,140,328** | **440.94** |

Margin against 512 GB: **71.1 GB (13.9 %)**. Counting the FP8 release with its actual file size instead (BF16 parts at two
bytes) gives 443.7 GB (margin 68.3 GB).

**Parameters**

| Check | Value | Limit | Margin |
|---|---:|---:|---:|
| Largest single model | 397 B nominal; 403.4 B stored values | 550 B | ≥ 146.6 B |
| All models together | 434.4 B with the nominal 397 B; 440.9 B with stored values | 550 B | ≥ 109.1 B |

The stored count of the FP8 release is the number of values in all tensors of the checkpoint as reported by the Hub; it
is larger than the nominal size in the model name. We have not broken the difference down; either count is below the
limit.

## 4. Models used only in development

| Model | Use | Size |
|---|---|---|
| Qwen3.5-397B-A17B, served by a hosted sampling API | protocol answers on the 1,500 training rows (training side of the 397B channel and of the extra columns) | same model as the FP8 release; the precision used by the provider is not documented by us. In BF16 the weights would take about 807 GB (estimate: two bytes per stored value of the FP8 release). Not delivered and not needed at inference; in the 8-bit reading it equals the FP8 release |
| Closed model through a commercial API (Claude Haiku 5.5; Sonnet 5.5 for comparison measurements) | pseudo-labels for the unlabelled pool, frozen | size not published by the provider; not a model of the solution |
| Qwen3.8-27B and Inkling-Small through the hosted sampling API | measurements on training rows only | not used by the submissions |
| M3 fold adapters (five, LoRA, 300.46 MB each) and 9B fold adapters (five, 110.91 MB each) | out-of-fold predictions of the training rows, on which the blend and second stage are fitted (the fold adapters also produced the out-of-fold draw h at 2,048 px) | 2.06 GB together; needed to retrain or re-verify the training side, not to produce the test predictions. One 9B fold adapter is also loaded by a short pre-check job of the 9B test configurations |

## 5. What could put the margin at risk

| Risk | Effect on reading (i) | Effect on reading (ii) |
|---|---|---|
| Counting M3 as a second 27B model (merged weights) | 537.1 GB, **above 512 GB** | 468.7 GB, below |
| Counting the 4-bit load of the 27B as another model | about +13.9 GB | about +13.9 GB at 4 bit; below 512 GB |
| Mixing GiB and GB (log files print GiB as "GB": 378.3 for the FP8 release) | understated by 28 GB if 378 is added as GB | — |
| Counting the provider-hosted BF16 copy of the 397B used in development | about +807 GB | none (same model as the FP8 release) |
| Counting non-weight files (tokenizers, vision encoders) | already included | already included |
| Counting the fold adapters or the temporary merged copies | +2.06 GB (fold adapters); merged copies are run-time files | +0.5 GB (fold adapters) |

**How we count.** Each set of base weights once, in the release we use; adapters as adapters; run-time copies and the
development-only models not at all. Under this count reading (i) is 481.5 GB and reading (ii) 440.9 GB.

We keep the margin without changing any file: this document states the count, M3 is delivered strictly as base model
plus adapter (as it is stored), and both readings are quoted. The host's original wording refers to "512 GB weights in
8-bit or lower quants", under which the total is about 441 GB.
