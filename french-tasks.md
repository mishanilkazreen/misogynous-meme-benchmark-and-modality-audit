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
enabling human inspection of mode decisions and for validating that the system is
responding to embedded visual structures rather than surface-level artefacts.

### Planned Components

The implementation of the vision language model dual pathway represents the core
research contribution of the project. This component involves designing and
implementing a model architecture in which the same input image is processed in
parallel by two distinct pathways: one operating on the raw image to capture the
surface-level semantics, and the other operating on a transformed version of the
image to amplify low-visibility content. Developing this architecture, along with
its training and evaluation pipeline, is the first step of development in this
project.

The dynamic fusion engine (potentially developed as a sophisticated model like a
Gradient Boosted Decision Tree or Attention-Weighted Multi-Layer Perceptron or
even a simple Logistic Regression/Linear Stacking or Rule-Based and Threshold
Fusion) combines the outputs of the two vision language model pathways into a
single moderation decision. The fusion stage will explore multiple combination
strategies and will be evaluated experimentally to understand how surface-level
and embedded content signals interact. This work is central to assessing whether
the proposed architecture offers measurable advantages over single-path baselines.

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
