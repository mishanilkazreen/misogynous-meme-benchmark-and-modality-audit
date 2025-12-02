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

### Running Tests

```bash
# Run all tests
pytest

# Run only unit tests
pytest -m unit

# Run only property-based tests
pytest -m property

# Run with coverage
pytest --cov=models --cov=utils
```

## Project Structure

See [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) for detailed information about the codebase organization.

## Development

This project follows a spec-driven development approach. See `.kiro/specs/vlm-content-moderation/` for:
- `requirements.md`: System requirements and acceptance criteria
- `design.md`: Technical design and architecture
- `tasks.md`: Implementation task list

## License

[Add license information]