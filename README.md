# VLM Content Moderation System

A PyTorch-based content moderation system for detecting embedded hateful content in images
using YOLO and Vision Language Models (VLM).

## Overview

This system detects hateful messages (textual slurs, derogatory terms, visual hate symbols)
that are subtly embedded within seemingly harmless images. It supports two detection approaches:

1. **YOLO-based detection**: Fast, computationally efficient object detection with transfer learning
2. **VLM dual-pathway**: Advanced detection using Vision Transformers with preprocessing

## Quick Start

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended)
- uv package manager

### Installation

1. Clone the repository:

```bash
git clone https://github.com/mishanilkazreen/content-moderation.git
cd content-moderation
```

2. Install uv (if not already installed):

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

3. Create and activate virtual environment:

```bash
# Create venv with uv
uv venv .venv --python 3.10

# Activate (Windows)
.venv\Scripts\activate

# Activate (Linux/macOS)
source .venv/bin/activate
```

4. Install dependencies:

```bash
# Install project with dev dependencies
uv sync --dev

# Add PyTorch with CPU support (optional)
uv add torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

5. Set up pre-commit hooks:

```bash
uv run pre-commit install
```

### Dataset Setup (HatefulIllusion)

We use the [HatefulIllusion dataset][hf-dataset] from Hugging Face for training and evaluation.

[hf-dataset]: https://huggingface.co/datasets/yiting/HatefulIllusion_Dataset

```python
from datasets import load_dataset

# Load different subsets
ds_digits = load_dataset("yiting/HatefulIllusion_Dataset", "digits")      # 300 samples
ds_slangs = load_dataset("yiting/HatefulIllusion_Dataset", "hate_slangs") # 690 samples
ds_symbols = load_dataset("yiting/HatefulIllusion_Dataset", "hate_symbols") # 1170 samples
```

**Dataset Subsets:**

| Subset | Samples | Description |
|--------|---------|-------------|
| digits | 300 | Hidden digits (0-9) embedded in images |
| hate_slangs | 690 | Hidden hateful text/slurs |
| hate_symbols | 1170 | Hidden hate symbols |

Each image has:

- `message`: The hidden content embedded in the image
- `visibility`: How visible the hidden content is (0-2 = low, 3-5 = high)
- `condition_image`: Source image showing the hidden content location
- `prompt`: Description of the surface scene

### Training the Model

```bash
# Train detection model with transfer learning
python scripts/train_yolo_detection.py --epochs 25 --batch-size 8

# Model saved to checkpoints/best_detector.pt
```

### Running Tests

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=models --cov=utils

# Run specific test types
uv run pytest tests/unit/
uv run pytest tests/property/
```

### Code Quality

```bash
# Run linting and formatting
uv run ruff check .
uv run ruff format .

# Run type checking
uv run mypy models/ utils/

# Run all pre-commit hooks
uv run pre-commit run --all-files
```

## Model Architecture

### Transfer Learning with ResNet18

The detection model uses a pretrained ResNet18 backbone for feature extraction, which provides:

- **Pretrained weights**: Learned from ImageNet (1M+ images), providing robust low-level features
- **Reduced overfitting**: Pretrained features generalize better than random initialization
- **Faster convergence**: Model starts with useful representations

### Staged Training

Training proceeds in two stages:

1. **Stage 1 (Epochs 0-4)**: Backbone frozen, only train classification and bbox heads
   - Preserves pretrained features
   - Allows heads to adapt to the task
   - Uses higher learning rate (10x)

2. **Stage 2 (Epochs 5+)**: Unfreeze backbone, fine-tune entire model
   - Adapts backbone features to specific domain
   - Uses lower learning rate to avoid catastrophic forgetting

### Regularization Techniques

To prevent overfitting on the small dataset:

- **Dropout (0.5)**: Randomly drops 50% of neurons during training
- **Weight decay (0.01)**: L2 regularization on model weights
- **Early stopping**: Stops training when validation accuracy stops improving

### Data Augmentation

Training images are augmented to increase effective dataset size:

- **Horizontal flip**: 50% chance, with bbox coordinate adjustment
- **Brightness/contrast**: Random factor 0.8-1.2
- **Gaussian noise**: Adds robustness to image variations

### Bounding Box Extraction

Bounding boxes are automatically extracted from condition images using OpenCV:

1. Convert to grayscale
2. Threshold to find dark pixels (the hidden content)
3. Find contours and compute bounding rectangle
4. Scale coordinates from 512x512 to 1024x1024 (main image size)

## Current Results

| Metric | Value |
|--------|-------|
| Classification Accuracy | 51.67% |
| Bounding Box IoU | 90.33% |
| Training Samples | 240 (digits only) |

**IoU (Intersection over Union)**: Measures bounding box overlap accuracy. 1.0 = perfect match.

## Project Structure

```text
content-moderation/
├── models/
│   ├── yolo/           # YOLO detection models
│   ├── vlm/            # VLM dual-pathway models
│   └── explainability/ # Heatmap and visualization
├── scripts/
│   ├── train_yolo_detection.py  # Detection training with transfer learning
│   ├── visualize_yolo_detection.py  # Visualization
│   └── visualize_transformations.py # Data augmentation visualization
├── utils/
│   ├── dataset.py      # DatasetManager, HatefulIllusionDataset
│   ├── preprocessing.py # Image preprocessing pipeline
│   ├── augmentation.py  # Data augmentation
│   └── ocr.py          # OCR utilities
├── checkpoints/        # Saved model weights
├── tests/
│   ├── unit/           # Unit tests
│   ├── property/       # Property-based tests (Hypothesis)
│   └── integration/    # Integration tests
├── pyproject.toml      # Project configuration and dependencies
└── uv.lock            # Locked dependency versions
```

## Development

### Python Version Requirements

This project requires Python 3.10 or 3.11 for ML library compatibility.

```bash
# Check available Python versions
uv python list

# Create environment with specific Python version
uv venv --python 3.10

# Activate (Windows)
.venv\Scripts\activate

# Activate (Linux/macOS)
source .venv/bin/activate

# Install dependencies
uv sync --dev
```

### CI/CD Pipeline

- **GitHub Actions**: Automated testing and quality checks
- **Pre-commit hooks**: Code formatting, linting, and type checking
- **Coverage reporting**: Test coverage tracking

### Local Development

```bash
# Run all quality checks
uv run ruff check .           # Linting
uv run ruff format .          # Formatting
uv run mypy models/ utils/    # Type checking
uv run pytest tests/          # Tests

# Run pre-commit on all files
uv run pre-commit run --all-files
```

### Troubleshooting

**Python version issues:**

```bash
# Force Python 3.10
rm -rf venv .venv
uv venv --python 3.10 venv
```

**PyTorch not working:**

```bash
# Reinstall with CUDA support
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

See `.kiro/specs/vlm-content-moderation/` for detailed requirements and design documentation.

## Future Improvements

1. **Use all dataset subsets**: Expand from 300 to 2160 samples
2. **Multi-task learning**: Train on digits, slangs, and symbols together
3. **Ensemble methods**: Combine multiple model predictions
4. **Test-time augmentation**: Average predictions over augmented inputs
5. **Self-supervised pretraining**: Learn representations from unlabeled data

## Third-Party Licenses

This project uses the following open source libraries:

### Core Framework (Permissive Licenses)

| Package | License | Purpose |
|---------|---------|---------|
| PyTorch | BSD-3-Clause | Neural network framework |
| torchvision | BSD-3-Clause | Image transforms and pretrained models |
| transformers | Apache-2.0 | CLIP and VLM model support |
| numpy | BSD-3-Clause | Numerical computing |
| Pillow | HPND | Image loading and saving |
| opencv-python | Apache-2.0 | Image processing (blur, equalization) |

### OCR (Permissive Licenses)

| Package | License | Purpose |
|---------|---------|---------|
| EasyOCR | Apache-2.0 | Text extraction from images |
| pytesseract | Apache-2.0 | Alternative OCR wrapper |

### Dataset Management (Permissive Licenses)

| Package | License | Purpose |
|---------|---------|---------|
| datasets | Apache-2.0 | HuggingFace dataset loading |
| huggingface_hub | Apache-2.0 | Model and dataset downloads |

### Testing (Permissive/Weak Copyleft)

| Package | License | Purpose |
|---------|---------|---------|
| pytest | MIT | Test framework |
| pytest-cov | MIT | Coverage reporting |
| hypothesis | MPL-2.0 | Property-based testing |

### Development Dependencies (Permissive Licenses)

| Package | License | Purpose |
|---------|---------|---------|
| ruff | MIT | Fast Python linter and formatter |
| pre-commit | MIT | Git hooks framework |
| mypy | MIT | Static type checking |

### License Summary

- All core dependencies use permissive licenses (MIT, BSD, Apache-2.0)
- No GPL or AGPL dependencies that would require source disclosure
- Hypothesis uses MPL-2.0 (weak copyleft) - only affects modifications to Hypothesis itself

For detailed license information, see `.kiro/steering/dependency-licenses.md`.
