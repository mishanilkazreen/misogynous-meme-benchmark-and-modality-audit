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

### Dataset Setup

Download your hateful content dataset and organize it like this:

```
data/
├── train/
│   ├── image001.jpg
│   ├── image002.jpg
│   └── ...
├── val/
│   └── ...
├── test/
│   └── ...
└── annotations/
    ├── train/
    │   ├── image001.txt  # YOLO format: <class> <x_center> <y_center> <width> <height>
    │   └── ...
    └── labels.json       # VLM format: {"image_id": "...", "message_type": "textual", ...}
```

**Example annotation (YOLO format)** - `data/annotations/train/image001.txt`:
```
0 0.5 0.3 0.2 0.15
```
Classes: 0=textual, 1=symbolic

**Example annotation (VLM format)** - `data/annotations/labels.json`:
```json
{
  "image001.jpg": {
    "has_hateful_content": true,
    "message_type": "textual",
    "visibility_level": "low"
  }
}
```

### Running Tests

```bash
pytest
```

## Development

See `.kiro/specs/vlm-content-moderation/` for requirements, design, and tasks.