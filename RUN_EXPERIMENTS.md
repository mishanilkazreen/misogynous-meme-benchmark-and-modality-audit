# Guide to Running Content Moderation & OCR-augmented VLM Experiments

This guide outlines the steps required to replicate and run all the experiments on a machine with a powerful GPU (e.g. 12GB+ VRAM).

---

## 1. Setup Environment & Dependencies

First, ensure that Python and `uv` are installed. Then sync the environment including the GPU/VLM dependencies:

```bash
# Sync dependencies
uv sync --group vlm-gpu

# Setup Kaggle Credentials
# Create a `.env` file in the root directory and add your Kaggle username and key
# to allow downloading the MAMI 2022 dataset via kagglehub:
#
# KAGGLE_USERNAME=your_kaggle_username
# KAGGLE_KEY=your_kaggle_key
```

---

## 2. Download the Dataset

Download the MAMI 2022 dataset to the local cache folder managed by `kagglehub`:

```bash
uv run python scripts/download_dataset.py
```

---

## 3. Extract OCR Transcripts & Embeddings

Because the `.npz` embedding and OCR files are too large to be saved in the Git repository, you must first extract the text transcripts using the OCR engines. This script also extracts the CLIP image and text tower embeddings:

```bash
# Extract ViT-L-14 embeddings + PaddleOCR transcripts (train, validation, test)
uv run python scripts/extract_embeddings.py \
    --split train,validation,test \
    --model ViT-L-14-quickgelu \
    --use-ocr \
    --ocr-engine paddleocr

# Extract ViT-B-32 embeddings + PaddleOCR transcripts (train, validation, test)
uv run python scripts/extract_embeddings.py \
    --split train,validation,test \
    --model ViT-B-32-quickgelu \
    --use-ocr \
    --ocr-engine paddleocr
```
*Outputs will be saved in `results/embeddings/` (e.g. `train_vit_l_14_quickgelu_ocr_paddleocr.npz`).*

---

## 4. Train the XGBoost Fusion Classifiers

Once embeddings and OCR transcripts are extracted, you can train the Supervised Embedding Fusion Classifier (XGBoost head on top of CLIP + OCR representations):

```bash
# Challenge 1: Train binary misogyny detector
uv run python scripts/train_classifier.py \
    --model ViT-L-14-quickgelu \
    --task singleclass \
    --classifier xgboost \
    --use-ocr \
    --ocr-engine paddleocr

# Challenge 2: Train multiclass subtype detector
uv run python scripts/train_classifier.py \
    --model ViT-L-14-quickgelu \
    --task multiclass \
    --classifier xgboost \
    --use-ocr \
    --ocr-engine paddleocr
```
*Outputs will be saved as `.pkl` files in `results/models/`.*

## 5. Direct Fine-Tuning of the CLIP Classification Head

Fine-tune the CLIP model classification head directly using the PaddleOCR transcripts:

```bash
# Challenge 1: Fine-tune the ViT-L-14 model for Binary misogyny
uv run python scripts/train_clip.py \
    --model ViT-L-14-quickgelu \
    --epochs 5 \
    --batch-size 16 \
    --loss-mode classification \
    --task singleclass \
    --device cuda \
    --use-ocr \
    --ocr-engine paddleocr

# Challenge 2: Fine-tune the ViT-L-14 model for Multiclass subtypes
uv run python scripts/train_clip.py \
    --model ViT-L-14-quickgelu \
    --epochs 5 \
    --batch-size 16 \
    --loss-mode classification \
    --task multiclass \
    --device cuda \
    --use-ocr \
    --ocr-engine paddleocr
```
*Outputs will be saved as `.pth` files in `results/models/` (e.g. `finetuned_clip_classification_singleclass_vit_l_14_quickgelu.pth`).*

---

## 6. Local Generative VLM QLoRA Fine-Tuning

Fine-tune the local generative Vision-Language Models (VLMs) using **QLoRA** (4-bit quantization + LoRA adapters) with PaddleOCR transcripts injected dynamically into the prompt prefix:

```bash
# Challenge 1: Fine-tune Qwen2-VL-2B-Instruct for Binary misogyny
uv run python scripts/train_vlm.py \
    --model-id Qwen/Qwen2-VL-2B-Instruct \
    --epochs 3 \
    --batch-size 2 \
    --quantize 4bit \
    --device cuda \
    --task singleclass \
    --use-ocr \
    --ocr-engine paddleocr

# Challenge 2: Fine-tune Qwen2-VL-2B-Instruct for Multiclass subtypes
uv run python scripts/train_vlm.py \
    --model-id Qwen/Qwen2-VL-2B-Instruct \
    --epochs 3 \
    --batch-size 2 \
    --quantize 4bit \
    --device cuda \
    --task multiclass \
    --use-ocr \
    --ocr-engine paddleocr
```
*Outputs will be saved as LoRA adapters under `results/models/lora_qwen2_vl_2b_instruct_<task>/`.*

---

## 7. Run Benchmarks & Evaluations

Run evaluations on the validation split using the trained adapters and checkpoints:

```bash
# ===========================================================================
# EVALUATING FINE-TUNED CLIP
# ===========================================================================

# Challenge 1: Binary misogyny
uv run python scripts/benchmark_clip.py \
    --split validation \
    --device cuda \
    --model-path results/models/finetuned_clip_classification_singleclass_vit_l_14_quickgelu.pth \
    --use-ocr \
    --ocr-engine paddleocr

# Challenge 2: Multiclass subtypes
uv run python scripts/benchmark_clip.py \
    --split validation \
    --device cuda \
    --model-path results/models/finetuned_clip_classification_multiclass_vit_l_14_quickgelu.pth \
    --use-ocr \
    --ocr-engine paddleocr \
    --task multiclass


# ===========================================================================
# EVALUATING FINE-TUNED GENERATIVE VLMS (Qwen2-VL-2B)
# ===========================================================================

# Challenge 1: Binary misogyny
uv run python scripts/benchmark_qwen2vl.py \
    --model-id Qwen/Qwen2-VL-2B-Instruct \
    --split validation \
    --use-ocr \
    --ocr-engine paddleocr \
    --lora-path results/models/lora_qwen2_vl_2b_instruct_singleclass

# Challenge 2: Multiclass subtypes
uv run python scripts/benchmark_qwen2vl.py \
    --model-id Qwen/Qwen2-VL-2B-Instruct \
    --split validation \
    --use-ocr \
    --ocr-engine paddleocr \
    --task multiclass \
    --lora-path results/models/lora_qwen2_vl_2b_instruct_multiclass
```
