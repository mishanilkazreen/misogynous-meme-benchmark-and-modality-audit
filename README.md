# Multimodal Misogyny Detection with Vision-Language Models

Benchmarking vision-language models (CLIP, LLaVA, Qwen2-VL, Gemini) and
traditional ML baselines (XGBoost, LightGBM) on misogyny classification
using the MAMI 2022 dataset.

Two tasks:

- **Task A** (binary): Is this meme misogynistic? (yes/no)
- **Task B** (multilabel): Which sub-types? (shaming, stereotype, objectification, violence)

## Setup

```bash
git clone https://github.com/mishanilkazreen/content-moderation.git
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

## Results

- **Gist (consolidated table):** [Results & SOTA Comparison]([redacted-gist-link])
- **Literature review:** [`papers/literature_review.md`](papers/literature_review.md)

## Google NotebookLM Integration

- **Notebook URL:** [MAMI 2022 Misogyny detection](https://notebooklm.google.com/notebook/[redacted-notebook-id])
- **Notebook ID:** `[redacted-notebook-id]`
- **MCP Connection & Troubleshooting:**
  - MCP executable: `[redacted-path]` (`uv tool install notebooklm-mcp-server`).
  - Auth setup: Save cookies from personal account tab into `~/cookies.txt` and run:
    `[redacted-auth-command]`.
  - Config requirement: `mcp_config.json` requires `FASTMCP_SHOW_SERVER_BANNER: "false"` & `FASTMCP_LOG_LEVEL: "WARNING"`.
  - Disconnect Pulse Secure VPN (`[redacted-ip]`) before calling Google RPC endpoints.

## Project Task Tracker

All tasks from the
[codebase review]([redacted-gist-link]).

| ID | Task | Issue | Assignee | Status |
| :---: | :--- | :---: | :--- | :--- |
| 1-3 | Fix trad ML pipeline (ViT-L-14 + PaddleOCR), add multiclass, rerun | [#84](https://github.com/mishanilkazreen/content-moderation/issues/84) | Mani | Submitted to SCIAMA |
| 4 | Qwen2-VL-2B zero-shot Task B | [#93](https://github.com/mishanilkazreen/content-moderation/issues/93) | Mani | Submitted to SCIAMA |
| 5 | CLIP ViT-B-32 fine-tuned Task B | [#94](https://github.com/mishanilkazreen/content-moderation/issues/94) | Mani | Submitted to SCIAMA |
| 6-8 | Advanced metrics + consolidated table | [#65](https://github.com/mishanilkazreen/content-moderation/issues/65) | Mani | Blocked on 1-5 |
| 9 | CLIP ViT-B-32 zero-shot baselines | [#95](https://github.com/mishanilkazreen/content-moderation/issues/95) | Mani | Submitted to SCIAMA |
| 10 | SOTA literature table with timings | [#88](https://github.com/mishanilkazreen/content-moderation/issues/88) + [#68](https://github.com/mishanilkazreen/content-moderation/issues/68) | Mishanil + Louis | In progress |
| 11 | SOTA explainability survey | [#88](https://github.com/mishanilkazreen/content-moderation/issues/88) | Mishanil + Louis | Done |
| 12 | Modality-Level SHAP Attribution | [#89](https://github.com/mishanilkazreen/content-moderation/issues/89) | Mani | Blocked on #84 |
| 13 | CLIP Concept Activation Vectors | [#89](https://github.com/mishanilkazreen/content-moderation/issues/89) | Mani | Blocked on #84 |

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
├── papers/              # Literature review, references, XAI review
├── results/             # JSON outputs, figures (mostly gitignored)
└── tests/               # Unit and property tests
```
