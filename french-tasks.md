# Project Overview

Title: Vision Language Model for Detecting Embedded and Obscure Harmful Visual
Content

User-generated content platforms increasingly encounter images where harmful
material is deliberately embedded within normal-looking scenes, making detection
difficult for both humans and automated systems. Examples can be found in recent
papers:

- <https://arxiv.org/pdf/2507.22617>
- <https://arxiv.org/abs/2005.04790>

Recent advances in image generation techniques have made creating such content
increasingly feasible, raising concerns about existing moderation systems designed
primarily for explicit content. AI image generators enable bad actors to create
content that evades current detection models, as discussed in this BBC article:
<https://www.bbc.co.uk/news/articles/c2lp5pn9e1qo>

Current vision-language models are optimised for surface-level visual semantics
and under-represent secondary visual structures, limiting their ability to detect
subtle forms of harmful content as generative capabilities grow.

This project investigates whether architectural changes to vision-language models,
combined with targeted preprocessing and rigorous evaluation, can improve
robustness to this emerging class of harmful content while preserving
interpretability and reproducibility.

## Dual-Pathway Ensemble Approach

The core research contribution is an ensemble approach combining multiple detection
pathways to identify harmful content at different visual representation levels,
addressing single-pathway models' limitations.

The ensemble consists of two parallel Vision Language Model (VLM) pathways:

1. Raw Image Pathway: Uses a frozen contrastive language image Pre-training the
   Vision Transformer (CLIP-ViT) backbone to capture surface-level visual
   semantics and detect immediately visible harmful content. Integrates Optical
   Character Recognition (OCR) using EasyOCR to extract and analyse embedded
   textual content.

2. Preprocessed Image Pathway: Applies Gaussian blur and histogram equalisation,
   then uses a trainable Vision Transformer (ViT) to amplify and detect
   low-visibility harmful material that has been deliberately obscured or embedded.

Each pathway produces detection outputs with confidence scores and class
predictions. A meta-learner (fusion engine) combines both pathways into a unified
moderation decision, weighting signals based on their reliability and input
characteristics.

Developers may be exposed to sensitive material related to harmful content.
Appropriate guidance and support will be provided to ensure responsible handling.

## Current Status and Outstanding Work

The project has established a substantial technical foundation, including dataset
integration, baseline models, preprocessing pipelines, and an evaluation framework.
The remaining work focuses on core research components for the dual-pathway
approach.

### Completed Components

Uses the HatefulIllusion dataset from Hugging Face (2,160 images: 300 hidden
digits, 690 hate slangs, and 1,170 hate symbols). A YOLO-based classifier baseline
implemented with a ResNet18 backbone achieves 51.67% classification accuracy and
90.33% bounding box IoU on the digits' subset. Partial explainability is
implemented for YOLO, including bounding box predictions and confidence overlays.
Dataset: Qu et al. (2025), "HatefulIllusion: Evaluating and Mitigating Hateful
Illusions in Vision Language Models"
<https://huggingface.co/datasets/yiting/HatefulIllusion_Dataset>

### Outstanding Work

The YOLO model requires a complete dataset training to approach the 93.8% accuracy
target achieved by Qu et al. (2025) using Full Fine-Tuning and Prompt Learning
(FPTL) on CLIP. VLM explainability mechanisms remain outstanding: no support for
visualising attention, saliency, or pathway-specific contributions. These
components are essential for human inspection of model decisions and validating the
system's response to embedded visual structures.

### Planned Components

Four core components remain:

1. VLM Dual-Pathway Architecture: Design and implement a two-branch model with
   CLIP-ViT for raw images and trainable ViT for preprocessed images. Implement
   the FPTL training methodology from Qu et al. (2025) for Pathway B. Ensure
   independent pathway execution and define input/output schemas for consistency.

2. Existing Content Moderation Tools Integration: Investigate and integrate
   established models using KerasHub or similar libraries. Evaluate ShieldGemma
   (google/shieldgemma-2-4b-it) and other HuggingFace moderation tools. Compare
   performance independently and within ensemble architecture. Explore fine-tuning
   and assess incorporation as additional ensemble members.

3. Dynamic Fusion Engine: Implement a meta-learner combining VLM pathway outputs
   into a unified moderation decision. Compute normalised risk score [0, 1]
   integrating surface-level and embedded-content signals. Implement multiple
   strategies: rule-based fusion, linear methods (logistic regression, linear
   stacking), tree-based models (gradient boosted decision trees), and neural
   approaches (attention-weighted multi-layer perceptrons). Enable systematic
   comparison of strategies.

4. Explainability, Transparency, and Trust: Implement comprehensive explainability
   for interpretable, trustworthy decisions. Use gradient-based saliency methods
   (Grad-CAM, Integrated Gradients) to generate visual heatmaps showing influential
   image regions. Add attention visualisation for pathway-specific contributions.
   Generate heatmaps and bounding boxes with confidence scores and localisation
   coordinates for moderators. Implement exportable outputs for reports and
   analysis. Expose explainability hooks for inspecting fusion decisions during
   experimentation. Ensure outputs meet regulatory compliance for auditable AI
   systems.
