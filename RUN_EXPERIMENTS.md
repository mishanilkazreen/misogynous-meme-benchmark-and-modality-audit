# Guide to Running Content Moderation & OCR-augmented VLM Experiments

This guide outlines the steps required to replicate and run all experiments on a machine with a powerful GPU (e.g. 12GB+ VRAM). All experiments are ordered **fastest first** to allow quick verification of pipeline components before committing to hours-long VLM training runs.

An automation script is provided under `scripts/run_all_experiments.ps1`. You can trigger it directly in PowerShell:
```powershell
# Run the entire pipeline sequentially (estimated time: ~15 hours)
.\scripts\run_all_experiments.ps1
```

---

## Expected Execution Times per Model and Task (12GB+ VRAM GPU)

This table shows the execution time estimates for running the experiments sequentially on a single 12GB VRAM GPU.

| Phase | Model / Operation | Task / Split | Expected Time | Notes / Details |
|---|---|---|---|---|
| **Setup** | Dependency Sync (`uv sync`) | N/A | **30s – 3m** | Dependent on network speed. |
| **Download** | Dataset Download (`download_dataset.py`) | N/A | **1m – 5m** | Downloads ~1.5 GB MAMI 2022 dataset. |
| **Preprocessing** | OCR & CLIP Embedding Extraction (`extract_embeddings.py`) | 11k samples (all splits) | **~15 – 18m** | Primarily dominated by PaddleOCR inference. |
| **XGBoost**| XGBoost Classifier (`train_classifier.py`) | Binary (`singleclass`) | **~15 seconds** | Extremely lightweight CPU/GPU training. |
| | XGBoost Classifier (`train_classifier.py`) | Multiclass (`multiclass`) | **~15 seconds** | Trains 4 independent binary classifiers. |
| **CLIP Head** | CLIP classification head (`train_clip.py`, 5 epochs) | ViT-B-32-quickgelu (Binary) | **~3 – 4 minutes** | Standard PyTorch classification loop. |
| | CLIP classification head (`train_clip.py`, 5 epochs) | ViT-L-14-quickgelu (Binary) | **~10 minutes** | Larger image/text towers. |
| | CLIP classification head (`train_clip.py`, 5 epochs) | ViT-L-14-quickgelu (Multiclass) | **~10 minutes** | Larger image/text towers. |
| **VLM (2B)** | Qwen2-VL-2B-Instruct QLoRA (`train_vlm.py`, 3 epochs) | Binary (`singleclass`) | **~1h 10 minutes** | 4.5k steps/epoch, batch size 2. |
| | Qwen2-VL-2B-Instruct QLoRA (`train_vlm.py`, 3 epochs) | Multiclass (`multiclass`) | **~1h 10 minutes** | 4.5k steps/epoch, batch size 2. |
| **LLaVA (7B)** | LLaVA-1.5-7b-hf QLoRA (`train_vlm.py`, 3 epochs) | Binary (`singleclass`) | **~2h 15 minutes** | 4.5k steps/epoch, batch size 2. |
| | LLaVA-1.5-7b-hf QLoRA (`train_vlm.py`, 3 epochs) | Multiclass (`multiclass`) | **~2h 15 minutes** | 4.5k steps/epoch, batch size 2. |
| **Qwen (7B)** | Qwen2-VL-7B-Instruct QLoRA (`train_vlm.py`, 3 epochs) | Binary (`singleclass`) | **~2h 30 minutes** | 4.5k steps/epoch, batch size 2. |
| | Qwen2-VL-7B-Instruct QLoRA (`train_vlm.py`, 3 epochs) | Multiclass (`multiclass`) | **~2h 30 minutes** | 4.5k steps/epoch, batch size 2. |
| **Evaluation** | CLIP Cosine Similarity / head (`benchmark_clip.py`) | 1k Validation samples | **~10 seconds** | Linear/cosine classification. |
| | Qwen2-VL-2B-Instruct (`benchmark_qwen2vl.py`) | 1k Validation samples | **~3 minutes** | Generative inference (batch size 4). |
| | LLaVA-1.5-7b-hf (`benchmark_llava.py`) | 1k Validation samples | **~5 minutes** | Generative inference (batch size 4). |
| | Qwen2-VL-7B-Instruct (`benchmark_qwen2vl.py`) | 1k Validation samples | **~6 minutes** | Generative inference (batch size 4). |

---

## 1. Setup Environment & Dependencies (Estimated Time: 1–3 minutes)

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

## 2. Download the Dataset (Estimated Time: 1–5 minutes)

Download the MAMI 2022 dataset to the local cache folder managed by `kagglehub`:

```bash
uv run python scripts/download_dataset.py
```

---

## 3. Extract OCR Transcripts & Embeddings (Estimated Time: ~15–18 minutes)

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

## 4. Train the XGBoost Fusion Classifiers (Estimated Time: ~30 seconds)

Once embeddings and OCR transcripts are extracted, train the Supervised Embedding Fusion Classifier (XGBoost head on top of CLIP + OCR representations):

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

---

## 5. Direct Fine-Tuning of the CLIP Classification Head (Estimated Time: ~24 minutes)

Fine-tune the CLIP model classification head directly using the PaddleOCR transcripts:

```bash
# ===========================================================================
# 5.1. CLIP ViT-B-32 Training
# ===========================================================================

# Challenge 1: Fine-tune ViT-B-32 model for Binary misogyny (~3 minutes)
uv run python scripts/train_clip.py \
    --model ViT-B-32-quickgelu \
    --epochs 5 \
    --batch-size 16 \
    --loss-mode classification \
    --task singleclass \
    --device cuda \
    --use-ocr \
    --ocr-engine paddleocr

# ===========================================================================
# 5.2. CLIP ViT-L-14 Training
# ===========================================================================

# Challenge 1: Fine-tune ViT-L-14 model for Binary misogyny (~10 minutes)
uv run python scripts/train_clip.py \
    --model ViT-L-14-quickgelu \
    --epochs 5 \
    --batch-size 16 \
    --loss-mode classification \
    --task singleclass \
    --device cuda \
    --use-ocr \
    --ocr-engine paddleocr

# Challenge 2: Fine-tune ViT-L-14 model for Multiclass subtypes (~10 minutes)
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

## 6. Generative VLM QLoRA Fine-Tuning (Estimated Time: ~12 hours)

Fine-tune the local generative Vision-Language Models (VLMs) using **QLoRA** (4-bit quantization + LoRA adapters) with PaddleOCR transcripts injected dynamically into the prompt prefix:

```bash
# ===========================================================================
# 6.1. Fine-Tune Qwen2-VL-2B-Instruct (Estimated Time: ~1h 10m per task)
# ===========================================================================

# Challenge 1: Binary misogyny
uv run python scripts/train_vlm.py \
    --model-id Qwen/Qwen2-VL-2B-Instruct \
    --epochs 3 \
    --batch-size 2 \
    --quantize 4bit \
    --device cuda \
    --task singleclass \
    --use-ocr \
    --ocr-engine paddleocr

# Challenge 2: Multiclass subtypes
uv run python scripts/train_vlm.py \
    --model-id Qwen/Qwen2-VL-2B-Instruct \
    --epochs 3 \
    --batch-size 2 \
    --quantize 4bit \
    --device cuda \
    --task multiclass \
    --use-ocr \
    --ocr-engine paddleocr

# ===========================================================================
# 6.2. Fine-Tune LLaVA-1.5-7b (Estimated Time: ~2h 15m per task)
# ===========================================================================

# Challenge 1: Binary misogyny
uv run python scripts/train_vlm.py \
    --model-id llava-hf/llava-1.5-7b-hf \
    --epochs 3 \
    --batch-size 2 \
    --quantize 4bit \
    --device cuda \
    --task singleclass \
    --use-ocr \
    --ocr-engine paddleocr

# Challenge 2: Multiclass subtypes
uv run python scripts/train_vlm.py \
    --model-id llava-hf/llava-1.5-7b-hf \
    --epochs 3 \
    --batch-size 2 \
    --quantize 4bit \
    --device cuda \
    --task multiclass \
    --use-ocr \
    --ocr-engine paddleocr

# ===========================================================================
# 6.3. Fine-Tune Qwen2-VL-7B-Instruct (Estimated Time: ~2h 30m per task)
# ===========================================================================

# Challenge 1: Binary misogyny
uv run python scripts/train_vlm.py \
    --model-id Qwen/Qwen2-VL-7B-Instruct \
    --epochs 3 \
    --batch-size 2 \
    --quantize 4bit \
    --device cuda \
    --task singleclass \
    --use-ocr \
    --ocr-engine paddleocr

# Challenge 2: Multiclass subtypes
uv run python scripts/train_vlm.py \
    --model-id Qwen/Qwen2-VL-7B-Instruct \
    --epochs 3 \
    --batch-size 2 \
    --quantize 4bit \
    --device cuda \
    --task multiclass \
    --use-ocr \
    --ocr-engine paddleocr
```
*Outputs will be saved as LoRA adapters under `results/models/lora_<model_name_clean>_<task>/`.*

---

## 7. Run Benchmarks & Evaluations (Estimated Time: ~30 minutes)

Run evaluations on the 1,000-sample validation split using the trained adapters and checkpoints:

```bash
# ===========================================================================
# 7.1. EVALUATING FINE-TUNED CLIP (Estimated Time: ~10 seconds per model)
# ===========================================================================

# Challenge 1: Binary misogyny (ViT-B-32)
uv run python scripts/benchmark_clip.py \
    --split validation \
    --device cuda \
    --model-path results/models/finetuned_clip_classification_singleclass_vit_b_32_quickgelu.pth \
    --use-ocr \
    --ocr-engine paddleocr

# Challenge 1: Binary misogyny (ViT-L-14)
uv run python scripts/benchmark_clip.py \
    --split validation \
    --device cuda \
    --model-path results/models/finetuned_clip_classification_singleclass_vit_l_14_quickgelu.pth \
    --use-ocr \
    --ocr-engine paddleocr

# Challenge 2: Multiclass subtypes (ViT-L-14)
uv run python scripts/benchmark_clip.py \
    --split validation \
    --device cuda \
    --model-path results/models/finetuned_clip_classification_multiclass_vit_l_14_quickgelu.pth \
    --use-ocr \
    --ocr-engine paddleocr \
    --task multiclass


# ===========================================================================
# 7.2. EVALUATING FINE-TUNED QWEN2-VL-2B (Estimated Time: ~3 minutes per task)
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


# ===========================================================================
# 7.3. EVALUATING FINE-TUNED LLaVA-1.5-7B (Estimated Time: ~5 minutes per task)
# ===========================================================================

# Challenge 1: Binary misogyny
uv run python scripts/benchmark_llava.py \
    --model-id llava-hf/llava-1.5-7b-hf \
    --split validation \
    --use-ocr \
    --ocr-engine paddleocr \
    --lora-path results/models/lora_llava_1.5_7b_hf_singleclass

# Challenge 2: Multiclass subtypes
uv run python scripts/benchmark_llava.py \
    --model-id llava-hf/llava-1.5-7b-hf \
    --split validation \
    --use-ocr \
    --ocr-engine paddleocr \
    --task multiclass \
    --lora-path results/models/lora_llava_1.5_7b_hf_multiclass


# ===========================================================================
# 7.4. EVALUATING FINE-TUNED QWEN2-VL-7B (Estimated Time: ~6 minutes per task)
# ===========================================================================

# Challenge 1: Binary misogyny
uv run python scripts/benchmark_qwen2vl.py \
    --model-id Qwen/Qwen2-VL-7B-Instruct \
    --split validation \
    --use-ocr \
    --ocr-engine paddleocr \
    --lora-path results/models/lora_qwen2_vl_7b_instruct_singleclass

# Challenge 2: Multiclass subtypes
uv run python scripts/benchmark_qwen2vl.py \
    --model-id Qwen/Qwen2-VL-7B-Instruct \
    --split validation \
    --use-ocr \
    --ocr-engine paddleocr \
    --task multiclass \
    --lora-path results/models/lora_qwen2_vl_7b_instruct_multiclass
```
