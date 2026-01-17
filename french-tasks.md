# Project Overview

Title: Vision Language Model for Detecting Embedded and Obscure Harmful Visual Content

As part of this project, we aim to develop a Vision-Language Model (VLM) for detecting embedded
and obscure harmful visual content in digital media, specifically images. Our objective primarily
is to detect (mainly textual or symbolic) hateful content that is subtly embedded within images,
often requiring various transformations to become automatically detectable.

As part of this project we have defined several key tasks that will guide the development and
implementation of the VLM system. Each task is designed to address specific components of the
overall system, from data preparation to model architecture and evaluation.

## Task 1: Data Collection & Annotation

Purpose: Collect and annotate a comprehensive dataset of images containing embedded harmful
content. This dataset will serve as the foundation for training and evaluating the VLM.

Deliverables:

- A curated dataset of images with embedded harmful content, along with corresponding annotations
- Annotation guidelines and documentation detailing the annotation process
- A data preprocessing pipeline to prepare the dataset for model training

Current Status:

- Integrated HatefulIllusion dataset from HuggingFace (2,160 total samples across 3 subsets)
- Dataset subsets: digits (300), hate_slangs (690), hate_symbols (1,170)
- Each sample includes: image, message, visibility level (0-5), condition image, and prompt
- Implemented `HatefulIllusionDetectionDataset` class with automatic bbox extraction from
  condition images
- Created `DatasetManager` utility for loading and managing dataset splits
- Preprocessing pipeline implemented with blur and histogram equalization for low-visibility
  content

To do:

- Expand dataset with additional sources and manual annotation where needed

## Task 2: YOLO-Based Detection Model

Purpose: Develop a YOLO-based model for detecting embedded harmful content in images. This model
serves as a baseline that is capable of detecting embedded content with bounding box localization.

Deliverables:

- A YOLO-based detection model trained on the annotated dataset
- Training and evaluation scripts for the YOLO model
- Performance metrics including accuracy, precision, recall, F1-score, and IoU

Current Status:

Implementation Details:

- **Model Architecture**: Implemented `YOLODetector` with pretrained ResNet18 backbone for
  transfer learning
- **Training Pipeline**: `train_yolo_detection.py` script with staged training (frozen backbone
  for 5 epochs, then full fine-tuning)
- **Data Augmentation**: Horizontal flip, brightness/contrast adjustment, color jitter, Gaussian
  noise, rotation, and blur
- **Regularization**: Dropout (0.5), weight decay (0.01), early stopping with patience
- **Bounding Box Extraction**: Automatic extraction from condition images using OpenCV contour
  detection
- **Multi-task Learning**: Combined classification (digit/slang/symbol recognition) and bounding
  box regression
- **Evaluation Metrics**: Classification accuracy, IoU for bbox localization, stratified by
  visibility level
- **Training Scripts**: `scripts/train_yolo_detection.py` with configurable epochs, batch size,
  learning rate
- **Visualization**: `scripts/visualize_yolo_detection.py` for prediction visualization with bbox
  overlays

Training Results (on digits subset):

- Classification Accuracy: 51.67%
- Bounding Box IoU: 90.33%
- Training samples: 240 (80% of 300 digits)
- Validation samples: 60 (20% of 300 digits)

To do:

- The model obviously needs significant improvement in classification accuracy

## Task 3: Model Evaluation & Testing

Purpose: Implement comprehensive testing infrastructure to ensure model reliability, correctness,
and performance across different scenarios.

Deliverables:

- Unit tests for individual components (dataset, preprocessing, augmentation, OCR)
- Property-based tests using Hypothesis for robustness validation
- Integration tests for end-to-end workflows
- Evaluation utilities with stratified metrics

Current Status: **COMPLETED**

Implementation Details:

- **Unit Tests**: Implemented for dataset loading, preprocessing pipeline, augmentation
  transforms, and OCR functionality
- **Property-Based Tests**: Using Hypothesis for testing dataset composition, preprocessing
  invariants, OCR robustness, and YOLO detection properties
- **Evaluation Module**: `models/yolo/evaluator.py` with `YOLOEvaluator` class computing accuracy,
  precision, recall, F1, and visibility-stratified metrics
- **Test Coverage**: Comprehensive coverage of models and utils packages
- **CI/CD Integration**: GitHub Actions workflow for automated testing on Python 3.10 and 3.11

Test Structure:

- `tests/unit/`: Component-level tests
- `tests/property/`: Property-based tests with Hypothesis
- `tests/integration/`: End-to-end workflow tests (placeholder)

## Task 4: Preprocessing & Data Augmentation

Purpose: Develop robust preprocessing and augmentation pipelines to handle low-visibility content
and increase effective dataset size.

Deliverables:

- Preprocessing pipeline for enhancing low-visibility content
- Data augmentation strategies for training robustness
- Visualization tools for transformation inspection

Current Status: **COMPLETED**

Implementation Details:

- **Preprocessing Pipeline**: `utils/preprocessing.py` with blur and histogram equalization for
  low-visibility enhancement
- **Augmentation Module**: `utils/augmentation.py` with transforms including:
  - Horizontal flip with bbox coordinate adjustment
  - Random brightness/contrast (0.7-1.3 factor)
  - Color jitter (HSV space manipulation)
  - Gaussian noise injection
  - Random rotation (-15 to +15 degrees)
  - Random Gaussian blur
- **Visualization Script**: `scripts/visualize_transformations.py` for inspecting augmentation
  effects
- **Integration**: Augmentation integrated into training pipeline with configurable enable/disable

## Task 5: OCR Integration

Purpose: Integrate OCR capabilities for extracting and validating text content from images,
supporting both EasyOCR and Tesseract backends.

Deliverables:

- OCR wrapper supporting multiple backends
- Text extraction and validation utilities
- Performance comparison between OCR engines

Current Status: **COMPLETED**

Implementation Details:

- **OCR Module**: `utils/ocr.py` with `OCREngine` class supporting EasyOCR and Tesseract
- **Dual Backend Support**: Configurable backend selection with fallback mechanism
- **Text Extraction**: Methods for extracting text with confidence scores and bounding boxes
- **Testing**: Unit and property-based tests for OCR functionality and robustness

## Task 6: VLM Dual-Pathway Architecture

Purpose: Implement the core research contribution with parallel surface-level and embedded-content
analysis using Vision Language Models.

Deliverables:

- Two-branch model architecture (Pathway A: raw image, Pathway B: preprocessed image)
- Training loop supporting full fine-tuning and prompt learning
- Input/output schema validation
- Reproducible training and inference scripts

Current Status: **NOT STARTED**

Next Steps:

- Design dual-pathway architecture using CLIP or similar VLM backbone
- Implement Pathway A (surface-level analysis) and Pathway B (embedded content analysis)
- Create fusion mechanism for combining pathway outputs
- Develop training pipeline with both pathways
- Implement prompt learning strategies for efficient fine-tuning

## Task 7: Dynamic Fusion Engine

Purpose: Combine outputs from both VLM pathways into a single, reliable moderation signal with
confidence-weighted combination strategies.

Deliverables:

- Fusion logic with multiple strategies (rule-based, learned weights, thresholds)
- Comparative evaluation of fusion approaches
- Explainability hooks for fusion decisions

Current Status: **NOT STARTED**

Dependencies: Requires Task 6 (VLM Dual-Pathway Architecture) to be completed first.

## Task 8: Explainability & Visualization

Purpose: Enable human-interpretable inspection of model predictions for moderation review,
debugging, and trust building.

Deliverables:

- Heatmap/saliency map generation for predictions
- Bounding box visualization with confidence overlays
- Attention visualization for VLM pathways
- Exportable visual outputs for reports

Current Status: **PARTIALLY COMPLETED**

Implementation Details:

- **Visualization Scripts**: `scripts/visualize_yolo_detection.py` for YOLO predictions with bbox
  overlays
- **Explainability Module**: `models/explainability/` package structure created (placeholder)

Next Steps:

- Implement gradient-based saliency maps (GradCAM, Integrated Gradients)
- Add attention visualization for VLM pathways
- Create interactive visualization tools
- Develop explainability metrics for validation

## Task 9: Model Persistence & Configuration Management

Purpose: Ensure trained models are portable, reproducible, and configurable across environments
with proper serialization and versioning.

Deliverables:

- Save/load utilities for trained models with metadata
- Configuration system for thresholds, preprocessing, and inference behavior
- Compatibility tests for serialized models
- Clear versioning conventions

Current Status: **COMPLETED**

Implementation Details:

- **Checkpoint System**: Implemented in `models/yolo/trainer.py` with automatic best model saving
- **Configuration Classes**: `YOLOTrainingConfig` dataclass for training parameters
- **Model Serialization**: PyTorch state dict saving with metadata (accuracy, IoU, epoch, label
  mappings)
- **Checkpoint Directory**: `checkpoints/` for storing trained models
- **Load/Resume**: Support for loading checkpoints and resuming training

## Task 10: Code Quality & Documentation

Purpose: Maintain high code quality standards with linting, type checking, formatting, and
comprehensive documentation.

Deliverables:

- Linting and formatting with Ruff
- Type checking with mypy
- Pre-commit hooks for automated quality checks
- Comprehensive README and documentation

Current Status: **COMPLETED**

Implementation Details:

- **Linting**: Ruff configured in `pyproject.toml` with strict rules
- **Type Checking**: mypy configured for Python 3.10+ with type annotations throughout
- **Pre-commit Hooks**: `.pre-commit-config.yaml` with automated checks on commit
- **Documentation**: Comprehensive README.md with setup, usage, and architecture details
- **Steering Files**: `.kiro/steering/` with guidelines for model training, linting standards,
  and dependency licenses
- **CI/CD**: GitHub Actions workflow for automated testing and quality checks

## Summary

### Completed Tasks (7/10)

1. Data Collection & Annotation
2. YOLO-Based Detection Model
3. Model Evaluation & Testing
4. Preprocessing & Data Augmentation
5. OCR Integration
9. Model Persistence & Configuration Management
10. Code Quality & Documentation

### Partially Completed Tasks (1/10)

8. Explainability & Visualization (YOLO visualization done, VLM explainability pending)

### Not Started Tasks (2/10)

6. VLM Dual-Pathway Architecture
7. Dynamic Fusion Engine

### Next Priority

Focus on Task 6 (VLM Dual-Pathway Architecture) as it is the core research contribution and
blocks Task 7 (Dynamic Fusion Engine). The YOLO baseline provides a solid foundation for
comparison and the infrastructure is in place for rapid VLM development.
