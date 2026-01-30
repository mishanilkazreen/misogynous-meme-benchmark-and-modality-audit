# VLM Content Moderation System

A content moderation system for detecting hateful content deliberately embedded within seemingly harmless images using
YOLO detection and Vision Language Models.

See [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md) for research context and planned dual-pathway VLM architecture.

## Prerequisites

- **Python**: 3.10 or 3.11 (required for PyTorch compatibility)
- **uv**: Package manager ([installation guide](https://docs.astral.sh/uv/getting-started/installation/))
- **GPU**: CUDA-capable GPU recommended (CPU fallback available)

## Quick Start

```bash
# Clone repository
git clone https://github.com/mishanilkazreen/content-moderation.git
cd content-moderation

# Install uv (if needed)
curl -LsSf https://astral.sh/uv/install.sh | sh  # macOS/Linux
# or: powershell -c "irm https://astral.sh/uv/install.ps1 | iex"  # Windows

# Create environment and install dependencies
uv venv --python 3.10
source .venv/bin/activate  # Linux/macOS
# or: .venv\Scripts\activate  # Windows
uv sync --dev

# Install PyTorch (CPU version shown, see below for GPU)
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Set up pre-commit hooks
uv run pre-commit install
```

**GPU Installation**: Replace CPU index URL with:

- CUDA 11.8: `https://download.pytorch.org/whl/cu118`
- CUDA 12.1: `https://download.pytorch.org/whl/cu121`

## Dataset

Uses [HatefulIllusion dataset](https://huggingface.co/datasets/yiting/HatefulIllusion_Dataset) (2,160 images across 3 subsets):

| Subset       | Samples | Content                |
|--------------|---------|------------------------|
| digits       | 300     | Hidden digits (0-9)    |
| hate_slangs  | 690     | Hidden hateful text    |
| hate_symbols | 1,170   | Hidden hate symbols    |

Dataset loads automatically during training. Each image includes visibility level (0-5) and condition image showing
hidden content location.

## Training and Evaluation

```bash
# Train YOLO detector (ResNet18 backbone with transfer learning)
python scripts/train_yolo_detection.py --epochs 25 --batch-size 8
# Saves model to: checkpoints/best_detector.pt

# Visualize detection results (requires trained model)
python scripts/visualize_yolo_detection.py --samples 10
# Output: yolo_test_results.png

# Visualize data augmentation pipeline (no training needed)
python scripts/visualize_transformations.py
# Output: transformation_visualization.png
```

**Current Results** (digits subset only, 240 training samples):

- Classification Accuracy: 51.67%
- Bounding Box IoU: 90.33%

**Target**: 93.8% accuracy (Qu et al. 2025) using full dataset with CLIP fine-tuning.

## Testing

```bash
# Run all tests
uv run pytest

# Run with coverage report
uv run pytest --cov=models --cov=utils --cov-report=html

# Run specific test suites
uv run pytest tests/unit/          # Unit tests
uv run pytest tests/property/      # Property-based tests (Hypothesis)
uv run pytest tests/integration/   # Integration tests
```

Coverage reports: `htmlcov/index.html`

## Code Quality

```bash
# Linting and formatting
uv run ruff check .                # Check for issues
uv run ruff check --fix .          # Auto-fix issues
uv run ruff format .               # Format code

# Type checking
uv run mypy models/ utils/

# Run all pre-commit hooks
uv run pre-commit run --all-files
```

**Standards**: See [.kiro/steering/linting-standards.md](.kiro/steering/linting-standards.md) for detailed requirements.

## CI/CD

GitHub Actions runs on all PRs and pushes to `main`:

**CI Checks** (`.github/workflows/ci.yml`):

- Ruff linting and formatting
- Mypy type checking
- Pytest with coverage (Python 3.10 and 3.11)
- Pre-commit hooks validation
- Coverage upload to Codecov

**CD Pipeline** (`.github/workflows/cd.yml`):

- Package building and validation
- Distribution artifact upload

## Project Structure

```text
content-moderation/
├── models/
│   ├── yolo/              # YOLO detection (detector.py, trainer.py, evaluator.py)
│   ├── vlm/               # VLM models (planned)
│   └── explainability/    # Visualization tools (planned)
├── utils/
│   ├── dataset.py         # HatefulIllusionDataset, DatasetManager
│   ├── preprocessing.py   # Blur, histogram equalization
│   ├── augmentation.py    # Training augmentations
│   └── ocr.py            # OCR utilities
├── scripts/
│   ├── train_yolo_detection.py        # Training script
│   ├── visualize_yolo_detection.py    # Results visualization
│   └── visualize_transformations.py   # Augmentation preview
├── tests/                 # Unit, property, and integration tests
├── .github/workflows/     # CI/CD pipelines
└── pyproject.toml        # Dependencies and configuration
```

## Architecture Details

**Current Implementation**: YOLO detector with ResNet18 backbone

- **Transfer Learning**: ImageNet pretrained weights
- **Staged Training**: Frozen backbone (epochs 0-4), then fine-tuning (5+)
- **Regularization**: Dropout (0.5), weight decay (0.01), early stopping
- **Augmentation**: Horizontal flip, brightness/contrast, Gaussian noise
- **Bbox Extraction**: Automatic from condition images via OpenCV contours

See [.kiro/steering/model-training-guidelines.md](.kiro/steering/model-training-guidelines.md) for training details.

**Planned**: Dual-pathway VLM ensemble (see [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md))

## Development

### Adding Dependencies

```bash
uv add package_name           # Production dependency
uv add --dev package_name     # Development dependency
```

Update [.kiro/steering/dependency-licenses.md](.kiro/steering/dependency-licenses.md) when adding dependencies.

### Troubleshooting

**Python version issues:**

```bash
uv python list                # Check available versions
uv venv --python 3.10         # Recreate with specific version
```

**PyTorch installation:**

```bash
# Verify installation
python -c "import torch; print(torch.__version__)"

# Reinstall if needed
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## License and Dependencies

All dependencies use permissive licenses (MIT, BSD-3-Clause, Apache-2.0) or weak copyleft (MPL-2.0 for Hypothesis).

**Key Dependencies:**

- PyTorch, torchvision (BSD-3-Clause)
- transformers (Apache-2.0)
- opencv-python (Apache-2.0)
- pytest, ruff, mypy (MIT)

See [.kiro/steering/dependency-licenses.md](.kiro/steering/dependency-licenses.md) for complete license information.

## References

Qu et al. (2025). "HatefulIllusion: Evaluating and Mitigating Hateful Illusions in Vision Language Models"

- Dataset: <https://huggingface.co/datasets/yiting/HatefulIllusion_Dataset>
- Paper: <https://arxiv.org/pdf/2507.22617>
