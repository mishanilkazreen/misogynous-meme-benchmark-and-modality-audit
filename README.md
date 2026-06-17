# Multimodal Misogyny Detection with Vision-Language Models

Benchmarking vision-language models (CLIP, LLaVA, LLaVA-Next, Qwen2-VL,
Gemini, GPT-4o-mini) on **binary misogyny classification** using the
MAMI 2022 dataset. Each model answers a single yes/no question — *is
this meme misogynistic?* — and is scored against the dataset's
`misogynous` label.

> **Preprocessing note:** MAMI memes contain no hidden visual content
> (unlike the earlier HatefulIllusion data), so the preprocessing
> filters do not help here — empirically they match or hurt the
> unfiltered (`none`) baseline for every model. The benchmark therefore
> runs `--filters none` by default. Do not run the full filter ablation
> on MAMI; pass `--filters` explicitly only if you have a specific
> reason.

The full task plan is in
[`.kiro/specs/vlm-content-moderation/tasks.md`](.kiro/specs/vlm-content-moderation/tasks.md).
Reference papers are catalogued in [`papers/README.md`](papers/README.md).

## Prerequisites

- Python 3.10 (pinned by `.python-version`)
- [`uv`](https://docs.astral.sh/uv/getting-started/installation/) for
  environments and packages
- [`gh`](https://cli.github.com/) for branch/PR management
- VS Code with extensions from `.vscode/extensions.json`
  (ruff, mypy, markdownlint, TOML)

## Setup

```bash
git clone https://github.com/mishanilkazreen/content-moderation.git
cd content-moderation
uv sync --dev
uv run pre-commit install
```

Copy `.env.example` to `.env` and fill in your credentials (see below).

## Dataset

[MAMI 2022](https://www.kaggle.com/datasets/chukwuebukaanulunko/multimodal-misogyny-detection-mami-2022)
(Multimodal Misogynous Memes, SemEval-2022 Task 5) — 11,000 memes split
into `train` (9,000), `validation` (1,000), and `test` (1,000). Each meme
carries a binary `misogynous` label, four sub-task labels (`shaming`,
`stereotype`, `objectification`, `violence`), and a text transcription.

The benchmark predicts the binary `misogynous` label only.

### Kaggle credentials

The dataset is downloaded via `kagglehub`, which authenticates with
Kaggle credentials. Add them to `.env` in the project root (gitignored).
Use **either** the username + key pair **or** a single API token:

```
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_KEY=your_kaggle_key
KAGGLE_API_TOKEN=your_kaggle_api_token
```

**Download the dataset** (run once before benchmarking):

```bash
uv run python scripts/download_dataset.py
```

**Verify the download:**

```bash
uv run python scripts/verify_dataset.py                      # full check (metadata + images)
uv run python scripts/verify_dataset.py --fast               # metadata only
uv run python scripts/verify_dataset.py --splits validation  # one split
```

Use `validation` (the smallest split) with `--limit` during development.

## VLM Benchmark

### Challenges

Two benchmark challenges are supported via `--task`:

| Flag | Challenge | Labels | Output file |
|---|---|---|---|
| `--task singleclass` (default) | **Challenge 1** — binary misogyny | yes / no | `{model}_{split}.json` |
| `--task multiclass` | **Challenge 2** — Sub-task B, multi-label sub-types | shaming, stereotype, objectification, violence | `{model}_{split}_multiclass.json` |

**Challenge 1** example (binary misogyny, CLIP, CPU):

```bash
uv run python scripts/benchmark_vlm_classification.py \
    --task singleclass --model clip --split train,validation
# Output: results/clip_train,validation.json  (~10,000 images: 9,000 train + 1,000 validation)
# --filters defaults to "none" (no preprocessing) for MAMI.
```

**Challenge 2** example (multi-label sub-types, CLIP, CPU):

```bash
uv run python scripts/benchmark_vlm_classification.py \
    --task multiclass --model clip --split train,validation
# Output: results/clip_train,validation_multiclass.json  (~10,000 images)
```

For Challenge 1, generative models answer *"Is this meme misogynistic? … yes or no."*
For Challenge 2, they receive a single multi-output prompt asking which of the four
sub-type categories apply, replying with a comma-separated list or `none`.
CLIP predicts each Challenge 2 category independently via separate binary comparisons.

The binary `misogynous` label is predicted only in Challenge 1; sub-task labels
are predicted only in Challenge 2.

### Hardware and quantization

Local generative models (LLaVA, LLaVA-Next, Qwen2-VL) run with **4-bit
NF4 quantization** by default (via `bitsandbytes`), reducing VRAM from
~14 GB to ~5 GB so they fit on 12 GB consumer GPUs. Pass
`--quantize none` for full fp16 (requires 24+ GB VRAM).

> **CUDA note:** torch is pinned to the **CUDA 12.8** build
> (`pytorch-cu128` index in `pyproject.toml`) because `bitsandbytes`
> ships native binaries up to CUDA 13.0 only. Do not bump torch to a
> CUDA 13.2 build — `bitsandbytes` 4-bit kernels will crash at
> runtime against a 13.2 runtime. CUDA 12.8 works on all RTX 30/40
> series and data-centre Ampere/Hopper GPUs.

### Compute Environment

Experiments were run across two compute environments:

| Environment | Used For | Hardware |
|---|---|---|
| **SCIAMA HPC Cluster** (University of Portsmouth) | All GPU-intensive training (VLM QLoRA fine-tuning, CLIP head training, XGBoost) and local-model evaluation (CLIP, VisualBERT, LLaVA, Qwen2-VL) | NVIDIA L40 GPUs (48 GB VRAM each), via SLURM `gpu.q` partition on `gpu10`/`gpu11`/`gpu12` nodes |
| **Local machine** (internet-enabled) | Gemini API zero-shot evaluation (requires internet for Google API calls) | CPU-only (API-based inference, no local GPU required) |

**SCIAMA Supercomputer Specifications:**
The [Dennis Sciama High Performance Compute Cluster](https://sciama.icg.port.ac.uk/sciama-wp/)
(SCIAMA) is operated by the Institute of Cosmology and Gravitation (ICG) at the
University of Portsmouth, UK. It was built in 2011 and is currently in its fourth
iteration. The cluster is named after Dennis Sciama, a leading figure in the
development of astrophysics and cosmology, and is also an acronym for *SEPnet
Computing Infrastructure for Astrophysical Modelling and Analysis*. Key specs:

- **3,648 CPU cores** across 63 compute nodes
- **14× NVIDIA A100 GPUs** (40 GB VRAM each) on `gpu01`/`gpu02` nodes (128 cores per node)
- **6× NVIDIA L40 GPUs** (48 GB VRAM each) on `gpu10`/`gpu11`/`gpu12` nodes
- **1.8 PB Lustre** high-performance parallel filesystem
- **QDR InfiniBand** networking with 100 Gb/s throughput (4× EDR)
- **4 login nodes**, 1 JupyterHub application node
- **1 DELL ML3 Tape Library** (80 tapes, 1 PB archival capacity)
- **Job scheduler:** SLURM

> **Why two environments?** The SCIAMA compute nodes run offline (no internet
> access), so cloud API evaluations (Gemini, GPT-4o-mini) must be run on an
> internet-enabled machine. All other experiments — including the full training
> pipeline and local-model benchmarking — were executed on SCIAMA's L40 GPU nodes
> via SLURM batch jobs (see `scripts/submit_experiments.slurm`).

### Model overview

| Model | Size | Download | VRAM (4-bit) | VRAM (fp16) |
|---|---|---|---|---|
| CLIP (ViT-B/32) | ~600 MB | auto on first run | n/a | ~1 GB |
| VisualBERT (vqa-coco-pre) |  ~138M (112M + 25.6M ResNet-50) | auto on first run (~550 MB) | n/a | n/a |
| LLaVA 1.5 | 7B | auto on first run (~14 GB) | ~5 GB | ~14 GB |
| LLaVA-Next | 7B | auto on first run (~14 GB) | ~5 GB | ~14 GB |
| Qwen2-VL | 7B | auto on first run (~15 GB) | ~5 GB | ~14 GB |
| Gemini / GPT-4o-mini | cloud | API (key required) | n/a | n/a |

### Orchestrator (all models)

`benchmark_vlm_classification.py` runs one or more models on the
unfiltered images (`--filters none`, the default for MAMI) and writes a
per-model result file.

```bash
# Quick smoke test (16 images from train+validation, CPU, no API key)
uv run python scripts/benchmark_vlm_classification.py --model clip --split train,validation --limit 16

# Every model, full labelled set (train + validation, ~10,000 images)
uv run python scripts/benchmark_vlm_classification.py --model all --split train,validation
```

Output: `results/{model}_{split}.json` (e.g. `clip_train,validation.json`).

### Per-model scripts

Each model also has a standalone script. Output is
`results/{model}_{split}.json`.

```bash
# CLIP (CPU or GPU, no API key)
uv run python scripts/benchmark_clip.py --split train,validation
uv run python scripts/benchmark_clip.py --task multiclass --split train,validation

# VisualBERT (CPU or GPU, no API key)
uv run python scripts/benchmark_visualbert.py --split train,validation
uv run python scripts/benchmark_visualbert.py --task multiclass --split train,validation

# LLaVA / LLaVA-Next / Qwen2-VL (GPU)
uv run python scripts/benchmark_llava.py     --split train,validation --device cuda --limit 16
uv run python scripts/benchmark_llavanext.py --split train,validation --device cuda
uv run python scripts/benchmark_qwen2vl.py   --split train,validation --device cuda

# Gemini (cloud, GEMINI_API_KEY required)
uv run python scripts/benchmark_gemini.py --split train,validation --limit 5
uv run python scripts/benchmark_gemini.py --split train,validation --workers 40

# GPT-4o-mini (cloud, OPENAI_API_KEY required)
uv run python scripts/benchmark_gpt4omini.py --split train,validation --limit 5
```

### Options common to per-model scripts

| Flag | Default | Description |
|---|---|---|
| `--split` | `validation` | `train`, `validation`, `test`, or comma-separated e.g. `train,validation` |
| `--filters` | `none` | Comma-separated preprocessing filters. Default `none` for MAMI (no hidden visual content); pass extras only for a deliberate ablation |
| `--limit` | none | Cap number of images (useful for quick checks) |
| `--device` | varies | `cuda` or `cpu` |
| `--batch-size` | `4` | Images per forward pass (LLaVA/LLaVA-Next/Qwen2-VL only) |
| `--quantize` | `4bit` | `4bit`, `8bit`, or `none` (local generative models only) |
| `--workers` | `20` | Parallel API threads (Gemini/GPT-4o-mini only) |
| `--model-id` | per model | Override the HF/model checkpoint |

### Cloud API keys

Gemini and GPT-4o-mini read their keys from `.env` (gitignored):

```
GEMINI_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here
```

## OCR-Augmented Baselines & Fine-Tuning

For pipelines that do not rely purely on zero-shot inference, we support extracting text transcripts via
PaddleOCR, generating CLIP embeddings, training XGBoost classifiers, and fine-tuning both CLIP and VLMs.

See [RUN_EXPERIMENTS.md](RUN_EXPERIMENTS.md) for detailed execution instructions, expected VRAM usage, and completion
times.

### 1. Extract OCR Transcripts & Embeddings

```bash
# Extract ViT-L-14 / ViT-B-32 embeddings + PaddleOCR transcripts
uv run python scripts/extract_embeddings.py --split train,validation,test --model ViT-L-14-quickgelu --use-ocr --ocr-engine paddleocr
```

### 2. Train XGBoost Fusion Classifier

```bash
uv run python scripts/train_classifier.py --model ViT-L-14-quickgelu --task singleclass --classifier xgboost --use-ocr --ocr-engine paddleocr
```

### 3. Fine-Tune CLIP Classification Head

```bash
uv run python scripts/train_clip.py --model ViT-L-14-quickgelu --epochs 5 --task singleclass --device cuda --use-ocr --ocr-engine paddleocr
```

### 4. Fine-Tune VLMs via QLoRA

```bash
uv run python scripts/train_vlm.py --model-id Qwen/Qwen2-VL-2B-Instruct --epochs 3 --quantize 4bit --device cuda --task singleclass --use-ocr --ocr-engine paddleocr
```

### 5. Running the Pipeline

You can trigger the entire pipeline sequentially (setup, extraction, training, and benchmarking) using the
automation script:

```powershell
.\scripts\run_all_experiments.ps1
```

Results and checkpoints are written to the `results/` folder (gitignored).

## Code quality

```bash
uv run ruff check --fix .
uv run ruff format .
uv run mypy models/ utils/ scripts/
uv run pytest
uv run python scripts/lint_markdown.py
uv run pre-commit run --all-files
```

## Benchmark Results & SOTA Comparison

The performance of our zero-shot and fine-tuned models on MAMI 2022 is tracked
and compared against SOTA literature benchmarks in the following resources:

- **GitHub Gist (Results & SOTA Comparison):** [Gist Link]([redacted-gist-link])
- **Detailed Literature Review:** [literature_review.md](file://[redacted-path]/papers/literature_review.md)

## Project layout

```text
content-moderation/
├── models/
│   ├── yolo/           # Ultralytics YOLO wrappers (legacy, HatefulIllusion)
│   └── vlm/            # CLIP, VisualBERT, LLaVA, Qwen2-VL classifiers
├── utils/              # dataset (MAMI), preprocessing, augmentation, OCR
├── scripts/
│   ├── download_dataset.py             # fetch MAMI 2022 via kagglehub
│   ├── verify_dataset.py               # validate the local dataset
│   ├── benchmark_visualbert.py          # one-shot untrained baseline (CPU)
│   ├── benchmark_llava.py
│   ├── benchmark_llavanext.py
│   ├── benchmark_qwen2vl.py
│   ├── benchmark_gpt4omini.py          # cloud
│   ├── benchmark_gemini.py             # cloud
│   └── benchmark_vlm_classification.py # orchestrator (all models)
├── results/            # JSON outputs (gitignored)
└── tests/
    ├── unit/
    └── tasks/          # acceptance-gate tests (one per task)
```
