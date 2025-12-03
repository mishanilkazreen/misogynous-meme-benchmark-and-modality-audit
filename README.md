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

```python
from utils import download_mmhs150k_dataset

# Downloads to ~/.cache/kagglehub/datasets/
path = download_mmhs150k_dataset()
print(f"Dataset at: {path}")
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

### Running Tests

```bash
pytest
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
