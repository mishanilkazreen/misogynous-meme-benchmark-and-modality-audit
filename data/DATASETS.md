# Dataset Directory

This directory contains datasets for training and evaluating the content moderation models.

## Directory Structure

```
data/
├── raw/                    # Original, unprocessed images
│   ├── train/             # Training images
│   ├── val/               # Validation images
│   └── test/              # Test images
│
├── processed/             # Preprocessed images
│   ├── blur_equalized/   # Images with blur + histogram equalization
│   └── augmented/        # Augmented training images
│
└── annotations/           # Annotation files
    ├── yolo/             # YOLO format (bounding boxes)
    └── vlm/              # VLM format (image-level labels)
```

## Dataset Requirements

According to Requirements 1.1-1.3:

- **Minimum size**: 5,000 annotated images
- **Content types**: 
  - High visibility + textual
  - High visibility + symbolic
  - Low visibility + textual
  - Low visibility + symbolic
- **Annotation quality**: Fleiss Kappa ≥ 0.783

## Annotation Formats

### YOLO Format (Bounding Boxes)

```
<class_id> <x_center> <y_center> <width> <height>
```

Example: `annotations/yolo/image001.txt`
```
0 0.5 0.3 0.2 0.15
1 0.7 0.6 0.1 0.1
```

Classes:
- 0: Textual hate content
- 1: Symbolic hate content

### VLM Format (Image-Level Labels)

JSON format with metadata:

```json
{
  "image_id": "image001.jpg",
  "has_hateful_content": true,
  "message_type": "textual",
  "visibility_level": "low",
  "extracted_text": "sample text"
}
```

## Adding Datasets

1. Place raw images in `data/raw/train/`, `data/raw/val/`, or `data/raw/test/`
2. Add corresponding annotations in `data/annotations/`
3. Run preprocessing pipeline to generate processed versions
4. Validate annotation quality using `DatasetManager.validate_annotations()`

## Notes

- Raw data files are excluded from version control (see `.gitignore`)
- Ensure proper licensing and ethical considerations for hate content datasets
- Follow data privacy and security guidelines when handling sensitive content
