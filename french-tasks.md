# Project Overview

Title: Vision Language Model for Detecting Embedded and Obscure Harmful Visual
Content

User-generated content platforms are expected to increasingly encounter images in
which harmful material is deliberately embedded within otherwise normal-looking
scenes. In such content, harmful textual or symbolic content is typically embedded
in a way to make it difficult for both humans and automated systems to detect
using standard inspection methods.

While this type of content is not yet widespread on a large scale, recent advances
in image generation and manipulation techniques have made the creation of such
content increasingly feasible. This trend raises concerns about the robustness of
existing automated moderation systems, which are primarily designed to detect
explicit content. The urgency of this problem is further underscored by the rise
of AI image generators, which bad actors are using to create and disseminate
content that is deliberately embedded or obscured to overwhelm and evade current
detection models, as reported by the BBC in
[AI image generators: The new battleground for explicit and harmful content](https://www.bbc.co.uk/news/articles/c2lp5pn9e1qo).
This trend highlights the critical need for more robust, AI-powered moderation
tools.

Most current vision-language models, however, are optimised for capturing dominant,
surface-level visual semantics. As a result, these models tend to under-represent
secondary visual structures, which in turn may limit the ability of existing
moderation pipelines to detect increasingly subtle forms of harmful content as
generative capabilities continue to grow.

This project investigates whether architectural changes to vision-language models,
combined with targeted preprocessing and rigorous evaluation, can improve
robustness to this emerging class of harmful visual content while preserving
interpretability and reproducibility.

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

The project uses the HatefulIllusion dataset from Hugging Face, containing 2,160
images across three subsets: 300 images with hidden digits, 690 with hate slangs,
and 1,170 with hate symbols. The dataset was introduced in the paper "HatefulIllusion:
Evaluating and Mitigating Hateful Illusions in Vision Language Models" by Qu et al.
(2025). Dataset: <https://huggingface.co/datasets/yiting/HatefulIllusion_Dataset>

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

The core research contribution of this project is an ensemble approach that
combines multiple detection pathways to identify harmful content at different
levels of visual representation. This methodology addresses the limitation of
single-pathway models that may miss either surface-level or embedded harmful
content.

The ensemble consists of two parallel vision language model pathways processing
the same input image:

1. Raw Image Pathway: Operates on the original, unmodified image using a frozen
   CLIP-ViT backbone to capture dominant surface-level visual semantics and
   detect explicit harmful content that is immediately visible.

2. Preprocessed Image Pathway: Operates on a transformed version of the image
   (applying Gaussian blur and histogram equalisation) using a trainable ViT to
   amplify low-visibility content and detect harmful material that has been
   deliberately obscured or embedded.

Each pathway independently produces detection outputs, including confidence scores
and class predictions. The implementation requires designing and implementing the
two-branch architecture with clearly defined inputs and outputs for each pathway,
ensuring both pathways are executable independently and expose representations
suitable for downstream combination. A training pipeline must support full
fine-tuning and prompt-learning strategies (FPTL methodology from Qu et al., 2025),
along with reproducible training and inference scripts.

To combine the outputs from both pathways into a unified moderation decision, a
meta-learner (fusion engine) is required. The meta-learner addresses the challenge
of integrating potentially conflicting signals: one pathway may detect harmful
content while the other does not, or both may detect different types of threats.
The fusion component must learn to weight and combine these signals appropriately
based on their reliability and the characteristics of the input. Developers may
implement the meta-learner using various approaches, ranging from simple to
sophisticated. Rule-based fusion uses threshold-based logic or weighted averaging
of pathway confidence scores. Linear methods such as logistic regression or linear
stacking can learn optimal pathway weights. Tree-based models like Gradient Boosted
Decision Trees (XGBoost, LightGBM) can capture non-linear interactions between
pathways. Neural approaches such as attention-weighted Multi-Layer Perceptrons can
dynamically weight pathway contributions based on input characteristics. The fusion
stage will compute a normalised risk score in the range [0, 1] and must be
configurable to support systematic comparison of different combination strategies.

Explainability and visualisation capabilities are essential for enabling human
inspection of model decisions and validating that the system responds to embedded
visual structures rather than surface-level artefacts. The implementation requires
gradient-based saliency methods (Grad-CAM and Integrated Gradients) to generate
heatmap-style explanations, attention visualisation for pathway-specific
contributions, and exportable visual outputs suitable for reports and experimental
analysis. The component should expose lightweight explainability hooks that allow
fusion decisions to be inspected during experimentation, supporting review and
debugging of combined model outputs.
