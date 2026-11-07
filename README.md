<!-- cspell:ignore harmeme leaderboard -->
# Vision-Language Models and Classical Classifiers: A Benchmark and Modality Audit in Misogynous Meme Detection

Code, results and logs for the paper above (under review).

We benchmark 46 model configurations on the SemEval-2022 Task 5 (MAMI) misogynous meme dataset. The models are a
closed model (Gemini 3.1 Flash-Lite), fine-tuned open Vision-Language Models (Qwen2-VL, LLaVA-1.5), CLIP encoders, and
23 classical classifiers trained on frozen CLIP embeddings. We also add a seven-model InternVL scaling series, a SigLIP
backbone, and a smaller test on the HarMeme (Harm-C) dataset.

Main results:

- **MAMI Task A:** zero-shot Gemini has the highest score (Score A 0.8818).
- **MAMI Task B:** an SVM-RBF on frozen CLIP features reaches Score B 0.7321, within sampling variation of the 2022
  leaderboard (0.7310).
- **Harm-C:** the same kind of classifier reaches 83.26% Macro-F1 when trained on Harm-C labels. Zero-shot Gemini reaches
  46.42%, or 58.31% with a fuller definition of harm. Gemini's lead on MAMI does not carry over.
- **Image versus text:** SHAP gives up to 72.42% of the weight to the image features (XGBoost). Masking the text inside
  the images shows that part of this weight comes from text read by the image encoder.
- **Templates:** 1.8% to 13.1% of test memes share a template with a training meme. Retraining on a template-disjoint
  split does not support template memorisation as the main cause of the drop from validation to test.

---

## Setup

```bash
git clone https://github.com/mishanilkazreen/misogynous-meme-benchmark-and-modality-audit.git
cd misogynous-meme-benchmark-and-modality-audit
uv venv --python 3.10
uv sync --dev
uv run pre-commit install
```

Copy `.env.example` to `.env` and add your Kaggle credentials (and `GEMINI_API_KEY` for the Gemini runs), then:

```bash
uv run python scripts/download_dataset.py
```

All paths in the code are relative to the repository root, so the repository can be cloned to any folder.

## Windows, macOS and Linux

- Every `uv run python ...` command in this repository is the same on all three systems.
- The `.sh` scripts need bash (Linux, macOS, or Git Bash/WSL on Windows). The scripts in `sciama/` run on the Linux
  cluster. `sciama/sync_from_sciama.sh`, `scripts/predownload_models.sh` and `scripts/run_all_experiments.sh` have
  PowerShell versions (`.ps1`) next to them.
- Cluster scripts read machine-specific folders from environment variables, not from the code:
  `PROJECT_ROOT` (the repository folder on the cluster) and `SCIAMA_REMOTE_ROOT` (the same folder, seen from your
  machine). Set `HARMEME_DIR` only if you want the Harm-C data outside `data/harmeme/`.
- `auto_benchmark/` expects to be run from its own folder, so its relative paths resolve there.

## Dataset

The SemEval-2022 Task 5 (MAMI) dataset consists of 11,000 memes split into train (9,000),
validation (1,000), and test (1,000):

- **Official SemEval-2022 Task 5 Repository:** [MIND-Lab/SemEval2022-Task-5-Multimedia-Automatic-Misogyny-Identification-MAMI-](https://github.com/MIND-Lab/SemEval2022-Task-5-Multimedia-Automatic-Misogyny-Identification-MAMI-)
- **Kaggle Mirror (used by `scripts/download_dataset.py` for automated setup):** [MAMI 2022 on Kaggle](https://www.kaggle.com/datasets/chukwuebukaanulunko/multimodal-misogyny-detection-mami-2022)

Two tasks:

- **Task A** (binary): Is this meme misogynistic? (yes/no)
- **Task B** (multilabel): Which sub-types? (shaming, stereotype, objectification, violence)

The second dataset, HarMeme (Harm-C), has 3,544 COVID-19 memes (3,013 train, 177 validation, 354 test) labelled harmful
or harmless. It is downloaded by `uv run python -m scripts.revision.harmeme_eval download` from the
[MOMENTA repository](https://github.com/LCS2-IIITD/MOMENTA). Details are in [`data/harmeme/README.md`](data/harmeme/README.md).
The datasets are not redistributed here.

## Running Experiments

See [`RUN_EXPERIMENTS.md`](RUN_EXPERIMENTS.md) for the full execution
guide with timing estimates. Section 11 covers the Harm-C runs.

Quick examples:

```bash
# CLIP zero-shot (CPU)
uv run python scripts/benchmark_clip.py --split validation --task singleclass

# Qwen2-VL fine-tuned (GPU)
uv run python scripts/benchmark_qwen2vl.py --split validation --task multiclass --device cuda

# XGBoost Fusion (CPU, uses pre-extracted embeddings)
uv run python scripts/train_classifier.py --model ViT-L-14-quickgelu --task singleclass --classifier xgboost --text-source ocr --ocr-engine paddleocr

# Harm-C: CLIP features and the two classifiers (no API key needed)
uv run python -m scripts.revision.harmeme_eval download
uv run python -m scripts.revision.harmeme_eval extract
uv run python -m scripts.revision.harmeme_eval classify
```

## Compute Environment

| Environment | Used For |
|---|---|
| **SCIAMA HPC** (University of Portsmouth) | All GPU training and local-model evaluation. NVIDIA L40 GPUs (48 GB VRAM). |
| **Local machine** | Gemini/GPT-4o-mini API evaluation (requires internet). |

> torch is pinned to CUDA 12.8 (`cu128`) because `bitsandbytes` requires it.

GPU jobs are documented in [`sciama/README.md`](sciama/README.md) (cluster guide) and
[`sciama/JOBS.md`](sciama/JOBS.md) (job definitions).

## Results

- **Results overview:** [`results/README.md`](results/README.md)
- **Prompt templates:** [`docs/PROMPTS.md`](docs/PROMPTS.md) (every prompt, including the Harm-C prompts)
- **Harm-C results and statistics:** [`results/revision/dis01_harmeme/`](results/revision/dis01_harmeme/)

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
misogynous-meme-benchmark-and-modality-audit/
├── models/vlm/          # CLIP, VisualBERT, LLaVA, Qwen2-VL classifiers
├── utils/               # dataset, preprocessing, augmentation, OCR
├── scripts/             # benchmarks, training, evaluation, SLURM jobs
├── scripts/revision/    # revision experiments, including the Harm-C evaluation
├── auto_benchmark/      # Tabular ML classifiers (XGBoost, LightGBM, etc.)
├── results/             # JSON outputs, figures, logs (tracked in git)
└── tests/               # Unit and property tests
```

## Citation

*This paper is currently under review. Citation details will be updated upon publication.*

If you use this work, please contact the corresponding author at `mani.ghahremani@port.ac.uk` for citation details
until the paper is published.
