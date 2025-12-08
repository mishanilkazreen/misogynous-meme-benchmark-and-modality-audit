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

### Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd content-moderation
```

2. Create and activate virtual environment:

```bash
# Create venv
python -m venv venv

# Activate (Windows)
.\venv\Scripts\activate

# Activate (Linux/macOS)
source venv/bin/activate
```

3. Install project dependencies:

```bash
pip install -r requirements.txt
```

### Dataset Setup (HatefulIllusion)

We use the [HatefulIllusion dataset](https://huggingface.co/datasets/yiting/HatefulIllusion_Dataset) from Hugging Face for training and evaluation.

```python
from utils import download_hateful_illusion_dataset

# Downloads to ~/.cache/huggingface/datasets/
path = download_hateful_illusion_dataset()
```

Alternative download methods:

```python
# Using datasets library directly
from datasets import load_dataset
ds = load_dataset("yiting/HatefulIllusion_Dataset", "digits")

# Using pandas
import pandas as pd
df = pd.read_json("hf://datasets/yiting/HatefulIllusion_Dataset/digits/metadata.jsonl", lines=True)
```

**Dataset Structure:**

- `image`: Path to the image with embedded content
- `message`: The embedded message (digit 0-9)
- `condition_image`: Path to the message image
- `prompt`: Description of the surface scene
- `visibility`: Visibility level (1-5, higher = more visible)

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
│   ├── dataset.py      # DatasetManager, HatefulIllusionDataset
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
