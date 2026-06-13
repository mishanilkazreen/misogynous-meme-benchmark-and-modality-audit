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

### Model overview

| Model | Size | Download | VRAM (4-bit) | VRAM (fp16) |
|---|---|---|---|---|
| CLIP (ViT-B/32) | ~600 MB | auto on first run | n/a | ~1 GB |
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
uv run python scripts/benchmark_clip.py --task multiclass --split validation

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

```bash
uv run python scripts/benchmark_yolo.py --mode pretrained --model yolov8n.pt --subset digits
uv run python models/yolo/trainer.py --model yolov8n.pt --subset digits
```

Results are written to `results/yolo_benchmark.json` (gitignored).

## Code quality

```bash
uv run ruff check --fix .
uv run ruff format .
uv run mypy models/ utils/ scripts/
uv run pytest
uv run python scripts/lint_markdown.py
uv run pre-commit run --all-files
```

## Project layout

```text
content-moderation/
├── models/
│   ├── yolo/           # Ultralytics YOLO wrappers (legacy, HatefulIllusion)
│   └── vlm/            # CLIP, LLaVA, Qwen2-VL classifiers
├── utils/              # dataset (MAMI), preprocessing, augmentation, OCR
├── scripts/
│   ├── download_dataset.py             # fetch MAMI 2022 via kagglehub
│   ├── verify_dataset.py               # validate the local dataset
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
