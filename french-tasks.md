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
components required to evaluate the proposed dual-pathway approach (an ensemble
approach in which each pathway is capable of detecting harmful content in a
different layer of the image – more information in the Planned Components section
of this document and provided in the repository).

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

1. **Raw Image Pathway**: Operates on the original, unmodified image to capture
   dominant surface-level visual semantics and detect explicit harmful content
   that is immediately visible.

2. **Preprocessed Image Pathway**: Operates on a transformed version of the image
   (applying blur and histogram equalisation) to amplify low-visibility content
   and detect harmful material that has been deliberately obscured or embedded.

Each pathway independently produces detection outputs, including confidence scores
and class predictions. Developing this dual-pathway architecture, along with its
training and evaluation pipeline, is the first step of development in this
project.

To combine the outputs from both pathways into a unified moderation decision, a
meta-learner (fusion engine) is required. The meta-learner addresses the challenge
of integrating potentially conflicting signals: one pathway may detect harmful
content while the other does not, or both may detect different types of threats.
The fusion component must learn to weight and combine these signals appropriately
based on their reliability and the characteristics of the input.

Developers may implement the meta-learner using various approaches, ranging from
simple to sophisticated. Rule-based fusion uses threshold-based logic or weighted
averaging of pathway confidence scores. Linear methods such as logistic regression
or linear stacking can learn optimal pathway weights. Tree-based models like
Gradient Boosted Decision Trees (XGBoost, LightGBM) can capture non-linear
interactions between pathways. Neural approaches such as attention-weighted
Multi-Layer Perceptrons can dynamically weight pathway contributions based on
input characteristics.

The fusion stage will explore multiple combination strategies and will be
evaluated experimentally to understand how surface-level and embedded content
signals interact. This work is central to assessing whether the proposed ensemble
architecture offers measurable advantages over single-path baselines.

## Technical Scope

### Explainability and Visualisation

This task involves extending the existing visualisation tooling to support
explainability for the model predictions. The required work includes implementing a
gradient-based saliency method (Grad-Cam and Integrated Gradients) to generate a
heatmap-style explanation for the model's output, and adding attention
visualisation for the model pathways to enable inspection of pathway-specific
contributions.

Additional work includes producing exportable visual outputs suitable for use in
reports and experimental analysis, and developing lightweight validation measures
to assess the consistency of explanation outputs across samples and visibility
levels.

### VLM Dual-Pathway Architecture

The work required includes designing and implementing a two-branch model
architecture (raw image & preprocessed version of the image) using a CLIP-based
backbone, with clearly defined inputs and outputs for each pathway. Both pathways
should be executable independently and expose representations suitable for
downstream combination. A training pipeline must be implemented to support full
fine-tuning and prompt-learning strategies, along with reproducible training and
inference scripts.

The task also includes defining and validating input and output schemas for both
pathways, ensuring consistency across training and evaluation.

### Dynamic Fusion Engine

This task involves implementing the fusion component that combines outputs from
the two VLM pathways into a single moderation decision. The fusion engine will
compute a normalised risk score in the range [0, 1] by integrating surface-level
and embedded-content signals produced by the two pathways.

The work includes defining the inputs and outputs of the fusion stage and
implementing multiple combination strategies: rule-based approaches, learned weight
combinations, and threshold-based methods. These should operate on pathway-level
outputs and be configurable to support systematic comparison under consistent
experimental conditions.

Additionally, the component should expose lightweight explainability hooks that
allow fusion decisions to be inspected and analysed during experimentation,
supporting review and debugging of combined model outputs.

This component can be developed in parallel at the interface and evaluation level,
but implementation and validation require the models to produce stable pathway
outputs.
