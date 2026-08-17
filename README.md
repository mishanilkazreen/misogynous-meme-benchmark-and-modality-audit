# Multimodal Misogyny Detection with Vision-Language Models

Benchmarking vision-language models (CLIP, LLaVA, Qwen2-VL, Gemini) and
traditional ML baselines (XGBoost, LightGBM) on misogyny classification
using the MAMI 2022 dataset.

Two tasks:

- **Task A** (binary): Is this meme misogynistic? (yes/no)
- **Task B** (multilabel): Which sub-types? (shaming, stereotype, objectification, violence)

## Setup

```bash
git clone https://github.com/mishanilkazreen/misogynous-meme-benchmark-and-modality-audit.git
cd content-moderation
uv venv --python 3.10
uv sync --dev
uv run pre-commit install
```

Copy `.env.example` to `.env` and add your Kaggle credentials, then:

```bash
uv run python scripts/download_dataset.py
```

## Dataset

[MAMI 2022](https://www.kaggle.com/datasets/chukwuebukaanulunko/multimodal-misogyny-detection-mami-2022)
(SemEval-2022 Task 5): 11,000 memes split into train (9,000),
validation (1,000), and test (1,000).

## Running Experiments

See [`RUN_EXPERIMENTS.md`](RUN_EXPERIMENTS.md) for the full execution
guide with timing estimates.

Quick examples:

```bash
# CLIP zero-shot (CPU)
uv run python scripts/benchmark_clip.py --split validation --task singleclass

# Qwen2-VL fine-tuned (GPU)
uv run python scripts/benchmark_qwen2vl.py --split validation --task multiclass --device cuda

# XGBoost Fusion (CPU, uses pre-extracted embeddings)
uv run python scripts/train_classifier.py --model ViT-L-14-quickgelu --task singleclass --classifier xgboost --use-ocr --ocr-engine paddleocr
```

## Compute Environment

| Environment | Used For |
|---|---|
| **SCIAMA HPC** (University of Portsmouth) | All GPU training and local-model evaluation. NVIDIA L40 GPUs (48 GB VRAM). |
| **Local machine** | Gemini/GPT-4o-mini API evaluation (requires internet). |

> torch is pinned to CUDA 12.8 (`cu128`) because `bitsandbytes` requires it.

## Code Quality

```bash
uv run ruff check --fix .
uv run ruff format .
uv run mypy models/ utils/ scripts/
uv run pytest
uv run pre-commit run --all-files
```

## Project Layout

```text
content-moderation/
├── models/vlm/          # CLIP, VisualBERT, LLaVA, Qwen2-VL classifiers
├── utils/               # dataset, preprocessing, augmentation, OCR
├── scripts/             # benchmarks, training, evaluation, SLURM jobs
├── auto_benchmark/      # Tabular ML classifiers (XGBoost, LightGBM, etc.)
├── results/             # JSON outputs, figures (mostly gitignored)
└── tests/               # Unit and property tests
```
