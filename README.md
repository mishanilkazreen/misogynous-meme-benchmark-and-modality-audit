# VLM Content Moderation System

A PyTorch-based content moderation system for detecting embedded hateful content in images using YOLO and Vision Language Models (VLM).

## Overview

This system detects hateful messages (textual slurs, derogatory terms, visual hate symbols) that are subtly embedded within seemingly harmless images. It supports two detection approaches:

1. **YOLO-based detection**: Fast, computationally efficient object detection
2. **VLM dual-pathway**: Advanced detection using Vision Transformers with preprocessing

## Quick Start

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended)
- Kaggle account (for dataset download)

### Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd content-moderation
```

2. Install kagglehub globally (required for dataset download):

```bash
pip install kagglehub
```

3. Create and activate virtual environment:

```bash
# Create venv
python -m venv venv

# Activate (Windows)
.\venv\Scripts\activate

# Activate (Linux/macOS)
source venv/bin/activate
```

4. Install project dependencies:

```bash
pip install -r requirements.txt
```

### Dataset Setup (MMHS150K)

We use the [MMHS150K dataset](https://www.kaggle.com/datasets/victorcallejasf/multimodal-hate-speech) for training and evaluation.

**Option 1: Download via Python**

```python
from utils import download_mmhs150k_dataset

# Downloads to ~/.cache/kagglehub/datasets/
path = download_mmhs150k_dataset()
print(f"Dataset at: {path}")
```

**Option 2: Download via CLI**

```bash
python -c "from utils import download_mmhs150k_dataset; print(download_mmhs150k_dataset())"
```

**Note:** First-time download requires Kaggle authentication. You'll be prompted to log in or provide your Kaggle API credentials.

**Dataset Structure:**

```text
mmhs150k/
├── img_resized/          # Images (shortest side = 500px)
├── img_txt/              # Pre-extracted OCR text per image
├── splits/
│   ├── train_ids.txt     # ~112K images
│   ├── val_ids.txt       # ~19K images
│   └── test_ids.txt      # ~19K images
├── MMHS150K_GT.json      # Ground truth annotations
└── hatespeech_keywords.txt
```

**Annotation Format (MMHS150K_GT.json):**

Each image has 3 annotator labels (0-5):
- 0: NotHate
- 1: Racist
- 2: Sexist
- 3: Homophobe
- 4: Religion
- 5: OtherHate

### Usage Example

```python
from utils import DatasetManager

# Point to downloaded dataset
manager = DatasetManager("/path/to/mmhs150k")

# Load training data
train_dataset = manager.load_dataset(split="train")
print(f"Training samples: {len(train_dataset)}")

# Get dataset statistics
stats = manager.get_dataset_stats(split="train")
print(f"Hate images: {stats['hate_images']}")
print(f"Fleiss Kappa: {stats['fleiss_kappa']:.3f}")

# Validate annotation quality (should be >= 0.783)
kappa = manager.validate_annotations(split="train")
print(f"Annotation agreement: {kappa:.3f}")

# Check dataset meets minimum size (5000+ images)
print(f"Meets size requirement: {manager.supports_minimum_size(5000)}")
```

### Running Tests

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/unit/test_dataset.py -v
```

## Project Structure

```text
content-moderation/
├── models/
│   ├── yolo/           # YOLO detection models
│   ├── vlm/            # VLM dual-pathway models
│   └── explainability/ # Heatmap and visualization
├── utils/
│   ├── dataset.py      # DatasetManager, MMHS150KDataset
│   └── ...
├── tests/
│   ├── unit/           # Unit tests
│   ├── property/       # Property-based tests (Hypothesis)
│   └── integration/    # Integration tests
├── requirements.txt
└── pytest.ini
```

## Development

See `.kiro/specs/vlm-content-moderation/` for:
- `requirements.md` - Detailed requirements
- `design.md` - Technical design and architecture
- `tasks.md` - Implementation tasks and progress
