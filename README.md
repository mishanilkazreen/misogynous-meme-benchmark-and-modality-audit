# Vision-Language Models and Classical Classifiers: A Benchmark and Modality Audit in Misogynous Meme Detection

Multimodal content moderation on social media platforms presents a technical
challenge, especially in the context of internet memes where visual and textual
modalities interact implicitly. In this paper, we conduct a systematic benchmark
of over 30 classification architectures on the SemEval-2022 Task 5 (MAMI)
dataset, including closed-source frontier foundation models (Gemini 1.5 Pro), finetuned open-weights Vision-Language Models (Qwen2-VL, LLaVA-1.5), fine-tuned
contrastive encoders (CLIP), and 23 classical machine learning models trained on
pre-extracted CLIP embeddings. Our findings expose key insights: (1) Gemini
1.5 Pro in a zero-shot setting establishes top performance on binary misogyny
detection (Task A) with a Macro-F1 of 0.8829, outperforming all custom-trained
literature pipelines; (2) for multi-label sub-type classification (Task B), a simple
tuned Support Vector Machine (SVM-RBF) on frozen CLIP features achieves a
MAMI Score B of 0.7321, matching or exceeding the SemEval winner (0.7310); and
(3) we audit the validation-test discrepancy, proving via SHAP feature importance
that fine-tuned classifiers overfit by memorising visual templates (72.42% image
importance vs. 27.58% text importance).

---

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

## Citation

*This paper is currently under review. Citation details will be updated upon publication.*

If you use this work, please contact the corresponding author at `mani.ghahremani@port.ac.uk` for citation details
until the paper is published.