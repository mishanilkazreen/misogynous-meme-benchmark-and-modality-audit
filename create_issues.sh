#!/bin/bash

# Script to create GitHub issues from tasks.md
# Run after: gh auth login

echo "Creating GitHub issues for VLM Content Moderation tasks..."

# Task 1
gh issue create --title "Set up project structure and dependencies" \
  --body "Create directory structure for data, models, utils, and tests
- Set up PyTorch environment with required dependencies (torch, torchvision, transformers, opencv, numpy)
- Set up Hypothesis for property-based testing

**Requirements:** 1.1, 4.3
**Task:** 1" \
  --label "setup,task-1"

# Task 2.1
gh issue create --title "Create DatasetManager class" \
  --body "Implement dataset loading and validation:
- Support minimum 5000 images
- Implement Fleiss Kappa calculation for annotation quality validation
- Support both bounding box annotations (YOLO) and image-level labels (VLM)

**Requirements:** 1.1, 1.3
**Task:** 2.1" \
  --label "data-pipeline,task-2"

# Task 2.2
gh issue create --title "Property test: Dataset size support" \
  --body "Write property-based test for dataset loading.

**Property 1:** Dataset size support
**Validates:** Requirements 1.1
**Task:** 2.2" \
  --label "testing,property-test,task-2"

# Task 2.3
gh issue create --title "Property test: Annotation quality threshold" \
  --body "Write property-based test for annotation quality.

**Property 3:** Annotation quality threshold
**Validates:** Requirements 1.3
**Task:** 2.3" \
  --label "testing,property-test,task-2"

# Task 2.4
gh issue create --title "Implement PreprocessingPipeline class" \
  --body "Implement image preprocessing:
- Gaussian blur with configurable kernel size
- Histogram equalization
- Ensure preprocessing maintains image dimensions
- Support batch preprocessing

**Requirements:** 1.5, 2.2
**Task:** 2.4" \
  --label "data-pipeline,task-2"

# Task 2.5
gh issue create --title "Property test: Pathway B preprocessing order" \
  --body "Write property-based test for preprocessing order.

**Property 11:** Pathway B preprocessing order
**Validates:** Requirements 3.2
**Task:** 2.5" \
  --label "testing,property-test,task-2"

# Task 2.6
gh issue create --title "Implement data augmentation utilities" \
  --body "Implement data augmentation:
- Support rotation, scaling, brightness adjustments
- Ensure balanced distribution across content types (high/low visibility, textual/symbolic)

**Requirements:** 1.2
**Task:** 2.6" \
  --label "data-pipeline,task-2"

# Task 2.7
gh issue create --title "Property test: Dataset composition completeness" \
  --body "Write property-based test for dataset composition.

**Property 2:** Dataset composition completeness
**Validates:** Requirements 1.2
**Task:** 2.7" \
  --label "testing,property-test,task-2"

# Task 2.8
gh issue create --title "Implement OCR pipeline" \
  --body "Implement OCR for text extraction:
- Integrate Tesseract or EasyOCR for text extraction from images
- Clean and normalize extracted text

**Requirements:** 1.6
**Task:** 2.8" \
  --label "data-pipeline,ocr,task-2"

# Task 3.1
gh issue create --title "Create YOLODetector class" \
  --body "Implement YOLO detection model:
- Implement YOLO architecture (backbone + neck + detection head)
- Support configurable input sizes up to 4K resolution
- Implement Non-Maximum Suppression (NMS)

**Requirements:** 2.1, 2.4
**Task:** 3.1" \
  --label "yolo,model,task-3"

# Task 3.2
gh issue create --title "Property test: YOLO detection output format" \
  --body "Write property-based test for YOLO output format.

**Property 6:** YOLO detection output format
**Validates:** Requirements 2.1
**Task:** 3.2" \
  --label "testing,property-test,yolo,task-3"

# Task 3.3
gh issue create --title "Implement YOLO training loop" \
  --body "Implement YOLO training:
- Support bounding box annotations
- Implement optional preprocessing (blur + equalization)
- Add early stopping and checkpointing
- Target mAP ≥ 0.70

**Requirements:** 1.4, 2.2, 2.3
**Task:** 3.3" \
  --label "yolo,training,task-3"

# Task 3.4
gh issue create --title "Property test: YOLO training data format" \
  --body "Write property-based test for training data format.

**Property 4:** YOLO training data format
**Validates:** Requirements 1.4
**Task:** 3.4" \
  --label "testing,property-test,yolo,task-3"

# Task 3.5
gh issue create --title "Property test: Preprocessing configuration support" \
  --body "Write property-based test for preprocessing configuration.

**Property 7:** Preprocessing configuration support
**Validates:** Requirements 2.2
**Task:** 3.5" \
  --label "testing,property-test,task-3"

# Task 3.6
gh issue create --title "Implement YOLO evaluation metrics" \
  --body "Implement evaluation metrics:
- Calculate mAP, precision, recall, F1
- Stratify metrics by visibility level (high/low)
- Measure inference time

**Requirements:** 2.3, 2.5
**Task:** 3.6" \
  --label "yolo,evaluation,task-3"

# Task 3.7
gh issue create --title "Property test: Visibility-stratified evaluation" \
  --body "Write property-based test for visibility-stratified evaluation.

**Property 9:** Visibility-stratified evaluation
**Validates:** Requirements 2.5
**Task:** 3.7" \
  --label "testing,property-test,task-3"

# Task 3.8
gh issue create --title "Property test: YOLO inference performance" \
  --body "Write property-based test for inference performance.

**Property 8:** YOLO inference performance
**Validates:** Requirements 2.4
**Task:** 3.8" \
  --label "testing,property-test,yolo,task-3"

# Task 4.1
gh issue create --title "Create ExplainabilityModule class" \
  --body "Implement explainability for YOLO:
- Generate heatmaps from YOLO detection confidence maps
- Implement heatmap overlay with color coding
- Ensure heatmap dimensions match input image

**Requirements:** 6.1, 6.3
**Task:** 4.1" \
  --label "explainability,yolo,task-4"

# Task 4.2
gh issue create --title "Property test: Heatmap generation" \
  --body "Write property-based test for heatmap generation.

**Property 20:** Heatmap generation
**Validates:** Requirements 6.1
**Task:** 4.2" \
  --label "testing,property-test,explainability,task-4"

# Task 4.3
gh issue create --title "Property test: Heatmap overlay visualization" \
  --body "Write property-based test for heatmap overlay.

**Property 22:** Heatmap overlay visualization
**Validates:** Requirements 6.3
**Task:** 4.3" \
  --label "testing,property-test,explainability,task-4"

# Task 4.4
gh issue create --title "Implement bounding box visualization" \
  --body "Implement bounding box visualization:
- Output bounding box coordinates for all detections
- Handle multiple detections with distinct visualizations
- Ensure coordinates are within image bounds

**Requirements:** 6.2, 6.4, 6.5
**Task:** 4.4" \
  --label "explainability,visualization,task-4"

# Task 4.5
gh issue create --title "Property test: Detection output completeness" \
  --body "Write property-based test for detection output completeness.

**Property 21:** Detection output completeness
**Validates:** Requirements 6.2, 6.5
**Task:** 4.5" \
  --label "testing,property-test,task-4"

# Task 4.6
gh issue create --title "Property test: Multiple detection visualization" \
  --body "Write property-based test for multiple detection visualization.

**Property 23:** Multiple detection visualization
**Validates:** Requirements 6.4
**Task:** 4.6" \
  --label "testing,property-test,explainability,task-4"

# Task 5.1
gh issue create --title "Create PathwayA class (Frozen CLIP Multi-Modal)" \
  --body "Implement Pathway A:
- Load pre-trained CLIP (Vision and Text encoders)
- Freeze all parameters
- Image Head: Extract 512-dim embeddings from raw images
- Text Head: Extract 512-dim embeddings from OCR text

**Requirements:** 3.1, 1.6
**Task:** 5.1" \
  --label "vlm,pathway-a,task-5"

# Task 5.2
gh issue create --title "Create PathwayB class (trainable ViT)" \
  --body "Implement Pathway B:
- Implement trainable ViT architecture
- Apply preprocessing (blur + equalization) before feature extraction
- Extract 512-dim embeddings
- Implement confidence calculation

**Requirements:** 3.1, 3.2
**Task:** 5.2" \
  --label "vlm,pathway-b,task-5"

# Task 5.3
gh issue create --title "Property test: VLM dual-pathway execution" \
  --body "Write property-based test for dual-pathway execution.

**Property 10:** VLM dual-pathway execution
**Validates:** Requirements 3.1
**Task:** 5.3" \
  --label "testing,property-test,vlm,task-5"

# Task 5.4
gh issue create --title "Property test: VLM Pathway B preprocessing" \
  --body "Write property-based test for Pathway B preprocessing.

**Property 5:** VLM Pathway B preprocessing
**Validates:** Requirements 1.5
**Task:** 5.4" \
  --label "testing,property-test,vlm,task-5"

# Task 5.5
gh issue create --title "Implement VLMDualPathway class" \
  --body "Combine pathways:
- Combine Pathway A and Pathway B
- Process images through both pathways in parallel

**Requirements:** 3.1
**Task:** 5.5" \
  --label "vlm,task-5"

# Task 5.6
gh issue create --title "Implement FPTL training for Pathway B" \
  --body "Implement FPTL training:
- Implement Full Fine-Tuning for backbone
- Implement Prompt Learning components
- Keep Pathway A frozen during training
- Target accuracy ≥ 0.938 for low visibility content

**Requirements:** 3.3, 3.4
**Task:** 5.6" \
  --label "vlm,training,task-5"

# Task 5.7
gh issue create --title "Property test: FPTL training methodology" \
  --body "Write property-based test for FPTL methodology.

**Property 12:** FPTL training methodology
**Validates:** Requirements 3.3
**Task:** 5.7" \
  --label "testing,property-test,vlm,task-5"

# Task 6.1
gh issue create --title "Create DynamicFusionEngine class" \
  --body "Implement dynamic fusion:
- Implement confidence-weighted embedding fusion (Visual A + Text A + Visual B)
- Compute weights based on Pathway B confidence
- Output risk score normalized to [0, 1]

**Requirements:** 3.1
**Task:** 6.1" \
  --label "vlm,fusion,task-6"

# Task 6.2
gh issue create --title "Implement explainability for VLM" \
  --body "Implement VLM explainability:
- Implement Layer-wise Relevance Propagation (LRP) or Class Activation Mapping (CAM)
- Generate localization coordinates

**Requirements:** 6.1
**Task:** 6.2" \
  --label "vlm,explainability,task-6"

# Task 6.3
gh issue create --title "Implement VLM evaluation metrics" \
  --body "Implement VLM evaluation:
- Calculate accuracy, precision, recall, and F1-score (Weighted)
- Benchmark against Unimodal Baselines (Text-only vs Image-only)
- Stratify by visibility level

**Requirements:** 3.5
**Task:** 6.3" \
  --label "vlm,evaluation,task-6"

# Task 6.4
gh issue create --title "Property test: VLM comparative evaluation" \
  --body "Write property-based test for VLM comparative evaluation.

**Property 13:** VLM comparative evaluation
**Validates:** Requirements 3.5
**Task:** 6.4" \
  --label "testing,property-test,vlm,task-6"

# Task 7.1
gh issue create --title "Create ModelPersistence class" \
  --body "Implement model persistence:
- Implement PyTorch model serialization with metadata
- Implement model loading with architecture validation

**Requirements:** 4.1, 4.2, 4.3, 4.4
**Task:** 7.1" \
  --label "persistence,task-7"

# Task 7.2
gh issue create --title "Property test: Model serialization round-trip" \
  --body "Write property-based test for serialization round-trip.

**Property 14:** Model serialization round-trip
**Validates:** Requirements 4.1, 4.2, 4.5
**Task:** 7.2" \
  --label "testing,property-test,persistence,task-7"

# Task 7.3
gh issue create --title "Property test: YOLO serialization format" \
  --body "Write property-based test for YOLO serialization format.

**Property 15:** YOLO serialization format
**Validates:** Requirements 4.3
**Task:** 7.3" \
  --label "testing,property-test,yolo,persistence,task-7"

# Task 7.4
gh issue create --title "Property test: Model compatibility validation" \
  --body "Write property-based test for model compatibility validation.

**Property 16:** Model compatibility validation
**Validates:** Requirements 4.4
**Task:** 7.4" \
  --label "testing,property-test,persistence,task-7"

# Task 7.5
gh issue create --title "Create ConfigurationManager class" \
  --body "Implement configuration management:
- Implement confidence threshold configuration
- Implement minimum bounding box size configuration
- Implement threshold validation
- Apply configuration updates without retraining

**Requirements:** 7.1, 7.2, 7.3, 7.4
**Task:** 7.5" \
  --label "configuration,task-7"

# Task 7.6
gh issue create --title "Property test: Threshold configuration and application" \
  --body "Write property-based test for threshold configuration.

**Property 24:** Threshold configuration and application
**Validates:** Requirements 7.1, 7.3, 7.5
**Task:** 7.6" \
  --label "testing,property-test,configuration,task-7"

# Task 7.7
gh issue create --title "Property test: Minimum bounding box filtering" \
  --body "Write property-based test for bbox filtering.

**Property 25:** Minimum bounding box filtering
**Validates:** Requirements 7.2
**Task:** 7.7" \
  --label "testing,property-test,configuration,task-7"

# Task 7.8
gh issue create --title "Property test: Threshold validation" \
  --body "Write property-based test for threshold validation.

**Property 26:** Threshold validation
**Validates:** Requirements 7.4
**Task:** 7.8" \
  --label "testing,property-test,configuration,task-7"

# Task 8.1
gh issue create --title "Create InferenceAPI class" \
  --body "Implement inference API:
- Support single image processing
- Support batch processing with parallelization
- Output ModerationResult with all required fields
- Track performance statistics

**Requirements:** 5.1, 5.5
**Task:** 8.1" \
  --label "inference,api,task-8"

# Task 8.2
gh issue create --title "Property test: Inference produces results" \
  --body "Write property-based test for inference.

**Property 17:** Inference produces results
**Validates:** Requirements 5.1
**Task:** 8.2" \
  --label "testing,property-test,inference,task-8"

# Task 8.3
gh issue create --title "Property test: Batch processing parallelization" \
  --body "Write property-based test for batch parallelization.

**Property 19:** Batch processing parallelization
**Validates:** Requirements 5.5
**Task:** 8.3" \
  --label "testing,property-test,inference,task-8"

# Task 8.4
gh issue create --title "Implement detection classification" \
  --body "Implement classification:
- Classify message type (textual/symbolic)
- Classify visibility level (high/low)
- Include classifications in output

**Requirements:** 5.2, 5.3
**Task:** 8.4" \
  --label "inference,classification,task-8"

# Task 8.5
gh issue create --title "Property test: Detection classification completeness" \
  --body "Write property-based test for classification completeness.

**Property 18:** Detection classification completeness
**Validates:** Requirements 5.2, 5.3
**Task:** 8.5" \
  --label "testing,property-test,inference,task-8"

# Task 8.6
gh issue create --title "Implement false positive rate monitoring" \
  --body "Implement FPR monitoring:
- Test on clean images (no embedded content)
- Target false positive rate < 0.15

**Requirements:** 5.4
**Task:** 8.6" \
  --label "evaluation,monitoring,task-8"

echo "Done! All issues created."
