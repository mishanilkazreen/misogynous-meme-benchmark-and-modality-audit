# Project Overview

Title: Vision Language Model for Detecting Embedded and Obscure Harmful Visual
Content

User-generated content platforms such as Roblox are expected to increasingly
encounter images in which harmful material is deliberately embedded within
otherwise normal-looking scenes. In such content, harmful textual or symbolic
content is typically embedded in a way to make it difficult for both humans and
automated systems to detect using standard inspection methods. You can see
examples of such content in recent papers:

- <https://arxiv.org/pdf/2507.22617>
- <https://arxiv.org/abs/2005.04790>

While this type of content is not yet widespread on a large scale, recent advances
in image generation and manipulation techniques have made the creation of such
content increasingly feasible. This trend raises concerns about the robustness of
existing automated moderation systems, which are primarily designed to detect
explicit content. The prevalence of this problem is underscored by the rise of AI
image generators, which bad actors are using to create and disseminate content
that is deliberately embedded or obscured to overwhelm and evade current detection
models. What gives the project additional urgency is the imperfect state of safety
measures in place in online platforms like Roblox, as discussed in this article on
BBC: <https://www.bbc.co.uk/news/articles/c2lp5pn9e1qo>

Most current vision-language models, however, are optimised for capturing dominant,
surface-level visual semantics. As a result, these models tend to under-represent
secondary visual structures, which in turn may limit the ability of existing
moderation pipelines to detect increasingly subtle forms of harmful content as
generative capabilities continue to grow.

This project investigates whether architectural changes to vision-language models,
combined with targeted preprocessing and rigorous evaluation, can improve
robustness to this emerging class of harmful visual content while preserving
interpretability and reproducibility.

## Dual-Pathway Ensemble Approach

The core research contribution is an ensemble approach that combines multiple
detection pathways to identify harmful content at different levels of visual
representation. This methodology addresses the limitation of single-pathway models
that may miss either surface-level or embedded harmful content.

The ensemble consists of two parallel Vision Language Model (VLM) pathways
processing the same input image:

1. Raw Image Pathway: Operates on the original, unmodified image using a frozen
   Contrastive Language-Image Pre-training Vision Transformer (CLIP-ViT) backbone
   to capture dominant surface-level visual semantics and detect explicit harmful
   content that is immediately visible. This pathway also integrates Optical
   Character Recognition (OCR) using EasyOCR to extract and analyze textual content
   embedded within images.

2. Preprocessed Image Pathway: Operates on a transformed version of the image
   (applying Gaussian blur and histogram equalisation) using a trainable Vision
   Transformer (ViT) to amplify low-visibility content and detect harmful material
   that has been deliberately obscured or embedded.

Each pathway independently produces detection outputs, including confidence scores
and class predictions. A meta-learner (fusion engine) combines the outputs from
both pathways into a unified moderation decision, addressing the challenge of
integrating potentially conflicting signals. The fusion component learns to weight
and combine these signals appropriately based on their reliability and input
characteristics.

The project is structured around key tasks that guide development from data
preparation to model architecture and evaluation. As part of this work, the
developers may be exposed to sensitive material related to harmful content.
Appropriate guidance and support will be provided to ensure that this material is
handled responsibly.

## Current Status and Outstanding Work

The project has already established a substantial technical foundation, including
dataset integration, baseline models, preprocessing pipelines, and a complete
evaluation and testing framework. The remaining work focuses on the core research
components required to evaluate the proposed dual-pathway approach.

The project currently uses the HatefulIllusion dataset from Hugging Face,
containing 2,160 images across three subsets: 300 images with hidden digits, 690
with hate slangs, and 1,170 with hate symbols. The dataset was introduced in the
paper "HatefulIllusion: Evaluating and Mitigating Hateful Illusions in Vision
Language Models" by Qu et al. (2025). Dataset:
<https://huggingface.co/datasets/yiting/HatefulIllusion_Dataset>

A YOLO-based classifier baseline has been implemented using transfer learning with
a ResNet18 backbone, achieving 51.67% classification accuracy and 90.33% bounding
box IoU on the digits subset (240 training samples). The model requires training
on the complete dataset and significant improvement to approach the research target
of 93.8% accuracy achieved by Qu et al. (2025) using Full Fine-Tuning and Prompt
Learning (FPTL) methodology on CLIP.

### Partially Completed Components

Explainability and visualisation have been partially implemented for the You Only
Look Once (YOLO) baseline model, including visualisation of the bounding box
predictions and confidence overlays. However, explainability mechanisms for the
Vision Language Model (VLM) components remain outstanding. Specifically, there is
currently no support for visualising attention, saliency, or pathway-specific
contributions within the architecture. Completing this component is essential for
enabling human inspection of model decisions and for validating that the system is
responding to embedded visual structures rather than surface-level artefacts.

### Planned Components

The remaining implementation work consists of four core components:

1. VLM Dual-Pathway Architecture: Design and implement the two-branch model
   architecture with CLIP-ViT for the raw image pathway and trainable ViT for the
   preprocessed image pathway. Implement the Full Fine-Tuning and Prompt Learning
   (FPTL) training methodology from Qu et al. (2025) to train Pathway B on the
   HatefulIllusion dataset. Ensure both pathways are executable independently and
   expose representations suitable for downstream combination. Define and validate
   input and output schemas for both pathways to ensure consistency across training
   and evaluation.

2. Existing Content Moderation Tools Integration: Investigate and integrate
   established content moderation models using KerasHub or similar libraries.
   Evaluate models such as ShieldGemma (google/shieldgemma-2-4b-it) and other
   state-of-the-art moderation tools from HuggingFace. Compare their performance on
   the HatefulIllusion dataset both independently and as part of the ensemble
   architecture. Explore fine-tuning these models on the dataset and assess whether
   they can be incorporated into the dual-pathway system or used as additional
   ensemble members to improve overall detection accuracy.

3. Dynamic Fusion Engine: Implement the meta-learner component that combines
   outputs from both VLM pathways into a unified moderation decision. The fusion
   engine must compute a normalised risk score in the range [0, 1] by integrating
   surface-level and embedded-content signals. Implement multiple combination
   strategies including rule-based fusion (threshold-based logic or weighted
   averaging), linear methods (logistic regression or linear stacking), tree-based
   models (Gradient Boosted Decision Trees), and neural approaches
   (attention-weighted Multi-Layer Perceptrons). The component must be configurable
   to support systematic comparison of different combination strategies.

4. Explainability, Transparency, and Trust: Implement comprehensive explainability
   mechanisms to ensure model decisions are interpretable and trustworthy.
   Implement gradient-based saliency methods including Gradient-weighted Class
   Activation Mapping (Grad-CAM) and Integrated Gradients to generate visual
   heatmaps showing which image regions influenced the model's decision. Add
   attention visualisation for pathway-specific contributions to enable inspection
   of how each pathway contributes to the final decision. Generate visual heatmaps
   and bounding boxes that show moderators which image regions triggered detection,
   with confidence scores and localization coordinates. Implement exportable visual
   outputs suitable for reports and experimental analysis. The component should
   expose lightweight explainability hooks that allow fusion decisions to be
   inspected during experimentation, supporting review and debugging of combined
   model outputs. Ensure all explainability outputs meet regulatory compliance
   requirements for auditable AI systems.
