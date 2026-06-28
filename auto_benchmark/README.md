# Autobenchmark

`autobenchmark` is an automated machine learning benchmarking and explainability framework for tabular, text,
image, object detection, and Vision-Language tasks. It allows you to profile datasets, train and compare a wide
range of ML models, estimate prediction uncertainty, and run local/global explainers, all driven by modular
configuration files.

---

## Key Features

- **Unified CLI (`autobench`)**: Standardized commands for data profiling, model training, explainability,
  object detection, image classification, and VLM zero-shot benchmarking.
- **Multimodal Benchmarking**:
  - **Tabular & Text**: 20+ ML models (Scikit-Learn, LightGBM, XGBoost) with TF-IDF and SBERT embedding extraction.
  - **Object Detection**: Benchmark YOLO models (v5, v8, v11, v12, v26) on custom datasets.
  - **Image Classification**: Fine-tune PyTorch networks (ResNet, MobileNet, EfficientNet) or perform
    feature extraction with traditional ML classifiers.
  - **Vision-Language Models (VLM)**: Benchmark local VLMs (Florence-2, Qwen2-VL) on zero-shot image classification tasks.
- **Performance & Speed Optimizations**:
  - **GPU SBERT Embeddings**: Automatically utilizes PyTorch CUDA for up to 50x speedups in text embeddings.
  - **Embedding Caching**: Hashes and caches SBERT embeddings to disk to reuse them instantly on subsequent runs.
  - **Parallel Benchmark cap**: Parallelizes the outer model training loop while capping nested
    GridSearchCV/Cross-Val jobs to `n_jobs=1` to prevent CPU oversubscription and thrashing.
  - **XGBoost GPU Support**: Auto-enables CUDA device acceleration and hist-tree algorithms on NVIDIA GPUs.
  - **Optimized LIME Samples**: Caps LIME text and tabular samples to 500 for lightning-fast explainability reports.
  - **Florence-2 Compatibility**: Integrates dynamic configuration monkey-patches to run Microsoft
    Florence-2 models seamlessly on newer `transformers` (v5+) releases.

---

## Installation & Setup

1. Make sure you have [uv](https://github.com/astral-sh/uv) installed.
2. Initialize and sync dependencies:

   ```bash
   uv sync
   ```

3. Run the automated test suite to verify the framework:

   ```bash
   uv run python -m unittest discover -s tests -p "test_*.py"
   ```

---

## How to Run: Unified CLI

Instead of running separate script files, use the unified `autobench` command from the root folder.

### 1. Data Profiling & Analysis

Analyze tabular/text columns, class distributions, and correlation maps:

```bash
uv run autobench profile --data config/data/data1.yaml
```

Outputs are saved to `results/data_analysis/<data_config_name>/`.

### 2. Tabular/Text Training & Benchmarking

Train 20+ classifiers, run hyperparameter searches, estimate prediction uncertainty, and serialize models:

```bash
uv run autobench train --model config/model/model1.yaml
```

Outputs (fitted models, `predictions.csv`, `evaluation.csv`, and uncertainty reports) are saved to `results/model_results/<model_config_name>/`.

### 3. Explainability Pipelines

Generate SHAP beeswarm/bar plots, tabular & textual LIME highlights, and global surrogate rules:

```bash
uv run autobench explain --explainer config/explainer/explainer1.yaml
```

Outputs are saved to `results/explanation_results/<explainer_config_name>/`.

### 4. YOLO Object Detection Benchmarking (`detect`)

Train and benchmark multiple YOLO variants (v5/v8/v11/v12/v26) on object detection datasets:

```bash
uv run autobench detect --model config/model/yolo_model1.yaml
```

Supports auto-resolving CUDA devices. Outputs are saved in `results/model_results/<yolo_model_config_name>/`.

### 5. Image Classification Benchmarking (`classify`)

Fine-tune deep networks or extract ResNet features to train tabular classifiers on image datasets:

```bash
uv run autobench classify --model config/model/image_class_model1.yaml
```

Outputs are saved in `results/model_results/<image_classification_model_config_name>/`.

### 6. Vision-Language Model Benchmarking (`vlmbench`)

Evaluate local Vision-Language Models (e.g. Florence-2, Qwen2-VL) on zero-shot image classification tasks:

```bash
uv run autobench vlmbench --model config/model/vlm_model1.yaml
```

Performs prompt-based prediction, uses custom keyword regex boundary parsing, measures latency, and
produces ranked metric spreadsheets and bar plots. Outputs are saved to
`results/model_results/<vlm_model_config_name>/`.

---

## Directory Structure

```
c:\Github\auto_benchmark\
├── config/                      # Configuration YAML files
│   ├── init_config.yaml         # Base directories and OS options
│   ├── data/                    # Data configurations (tabular, text, image, vlm)
│   ├── model/                   # Model configurations (classifiers, yolo, vlm)
│   └── explainer/               # Explainer configurations
├── data_files/                  # Contains raw data, images, and labels CSVs
├── src/
│   └── autobenchmark/           # Source code modules
│       ├── cli.py               # Central CLI Parser & Registry
│       ├── data.py              # Data loaders, SBERT encoding & caching
│       ├── models.py            # Tabular/Text ML estimators & GridSearch
│       ├── explain.py           # SHAP, LIME, and Surrogate Tree explainers
│       ├── yolo_models.py       # YOLO benchmarking (v5/v8/v11/v12/v26)
│       ├── image_classification.py # PyTorch Image Classification
│       ├── vlm_bench.py         # VLM Zero-Shot Benchmarking
│       └── uncertainty.py       # Shannon Entropy calculations
├── results/                     # All generated outputs and charts
└── tests/                       # Unit tests (fully mocked, offline-compatible)
```
