# VLM Content Moderation System

A PyTorch-based content moderation system for detecting embedded hateful content in images using YOLO and Vision Language Models (VLM).

## Overview

This system detects hateful messages (textual slurs, derogatory terms, visual hate symbols) that are subtly embedded within seemingly harmless images. It supports two detection approaches:

1. **YOLO-based detection**: Fast, computationally efficient object detection
2. **VLM dual-pathway**: Advanced detection using Vision Transformers with preprocessing

## Quick Start

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- Tesseract OCR (for text extraction)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd content-moderation
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install Tesseract OCR:
- **macOS**: `brew install tesseract`
- **Ubuntu/Debian**: `sudo apt-get install tesseract-ocr`
- **Windows**: Download from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)

### Dataset Setup (MMHS150K)

We use the [MMHS150K dataset](https://www.kaggle.com/datasets/victorcallejasf/multimodal-hate-speech) for training and evaluation.

**Option 1: Using kagglehub (recommended)**

```bash
# Install kagglehub globally (not in venv)
pip install kagglehub

# Download via Python
python -c "from utils.dataset import download_mmhs150k_dataset; print(download_mmhs150k_dataset())"
```

This downloads to `~/.cache/kagglehub/datasets/victorcallejasf/multimodal-hate-speech/`.

**Option 2: Manual download**

1. Download from [Kaggle](https://www.kaggle.com/datasets/victorcallejasf/multimodal-hate-speech)
2. Extract to a directory of your choice

**Dataset Structure:**
```
mmhs150k/
├── img_resized/          # Images (shortest side = 500px)
├── img_txt/              # Pre-extracted OCR text per image
├── splits/
│   ├── train_ids.txt
│   ├── val_ids.txt
│   └── test_ids.txt
├── MMHS150K_GT.json      # Ground truth annotations
└── hatespeech_keywords.txt
```

**Using the dataset:**
```python
from utils.dataset import DatasetManager

# Point to your dataset location
manager = DatasetManager("/path/to/mmhs150k")
train_dataset = manager.load_dataset(split="train")

# Get dataset statistics
stats = manager.get_dataset_stats()
print(f"Total images: {stats['total_images']}")
print(f"Fleiss Kappa: {stats['fleiss_kappa']:.3f}")
```

### Running Tests

```bash
pytest
```

## Development

See `.kiro/specs/vlm-content-moderation/` for requirements, design, and tasks.