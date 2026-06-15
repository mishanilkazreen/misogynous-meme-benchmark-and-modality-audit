# run_all_experiments.ps1
# PowerShell script to run all content moderation & OCR-augmented VLM experiments sequentially, fastest first.

$ErrorActionPreference = "Stop"

# Set offline mode environment variables to speed up HF model loading from local cache
$env:HF_HUB_OFFLINE = 1
$env:HF_OFFLINE = 1

Write-Host "=== Starting all content moderation experiments ===" -ForegroundColor Green

# 1. Setup & Download
Write-Host "[1/7] Syncing dependencies and downloading dataset..." -ForegroundColor Yellow
uv sync --group vlm-gpu
uv run python scripts/download_dataset.py

# 2. Extract OCR & Embeddings
Write-Host "[2/7] Extracting OCR transcripts and embeddings (this will take ~18 minutes)..." -ForegroundColor Yellow
uv run python scripts/extract_embeddings.py --split train,validation,test --model ViT-L-14-quickgelu --use-ocr --ocr-engine paddleocr
uv run python scripts/extract_embeddings.py --split train,validation,test --model ViT-B-32-quickgelu --use-ocr --ocr-engine paddleocr

# 3. Train & Evaluate XGBoost Classifiers (~30 seconds)
Write-Host "[3/7] Training XGBoost Fusion Classifiers..." -ForegroundColor Yellow
uv run python scripts/train_classifier.py --model ViT-L-14-quickgelu --task singleclass --classifier xgboost --use-ocr --ocr-engine paddleocr
uv run python scripts/train_classifier.py --model ViT-L-14-quickgelu --task multiclass --classifier xgboost --use-ocr --ocr-engine paddleocr

# 4. Train & Evaluate CLIP classification heads (~24 minutes)
Write-Host "[4/7] Training and evaluating CLIP classification heads..." -ForegroundColor Yellow
# Binary ViT-B-32
uv run python scripts/train_clip.py --model ViT-B-32-quickgelu --epochs 5 --batch-size 16 --loss-mode classification --task singleclass --device cuda --use-ocr --ocr-engine paddleocr
uv run python scripts/benchmark_clip.py --split validation --device cuda --model-path results/models/finetuned_clip_classification_singleclass_vit_b_32_quickgelu.pth --use-ocr --ocr-engine paddleocr
# Binary ViT-L-14
uv run python scripts/train_clip.py --model ViT-L-14-quickgelu --epochs 5 --batch-size 16 --loss-mode classification --task singleclass --device cuda --use-ocr --ocr-engine paddleocr
uv run python scripts/benchmark_clip.py --split validation --device cuda --model-path results/models/finetuned_clip_classification_singleclass_vit_l_14_quickgelu.pth --use-ocr --ocr-engine paddleocr
# Multiclass ViT-L-14
uv run python scripts/train_clip.py --model ViT-L-14-quickgelu --epochs 5 --batch-size 16 --loss-mode classification --task multiclass --device cuda --use-ocr --ocr-engine paddleocr
uv run python scripts/benchmark_clip.py --split validation --device cuda --model-path results/models/finetuned_clip_classification_multiclass_vit_l_14_quickgelu.pth --use-ocr --ocr-engine paddleocr --task multiclass

# 5. Train & Evaluate Qwen2-VL-2B (~2.5 hours)
Write-Host "[5/7] Training and evaluating Qwen2-VL-2B (QLoRA)..." -ForegroundColor Yellow
# Binary
uv run python scripts/train_vlm.py --model-id Qwen/Qwen2-VL-2B-Instruct --epochs 3 --batch-size 2 --quantize 4bit --device cuda --task singleclass --use-ocr --ocr-engine paddleocr
uv run python scripts/benchmark_qwen2vl.py --model-id Qwen/Qwen2-VL-2B-Instruct --split validation --use-ocr --ocr-engine paddleocr --lora-path results/models/lora_qwen2_vl_2b_instruct_singleclass
# Multiclass
uv run python scripts/train_vlm.py --model-id Qwen/Qwen2-VL-2B-Instruct --epochs 3 --batch-size 2 --quantize 4bit --device cuda --task multiclass --use-ocr --ocr-engine paddleocr
uv run python scripts/benchmark_qwen2vl.py --model-id Qwen/Qwen2-VL-2B-Instruct --split validation --use-ocr --ocr-engine paddleocr --task multiclass --lora-path results/models/lora_qwen2_vl_2b_instruct_multiclass

# 6. Train & Evaluate LLaVA-1.5-7B (~4.5 hours)
Write-Host "[6/7] Training and evaluating LLaVA-1.5-7B (QLoRA)..." -ForegroundColor Yellow
# Binary
uv run python scripts/train_vlm.py --model-id llava-hf/llava-1.5-7b-hf --epochs 3 --batch-size 2 --quantize 4bit --device cuda --task singleclass --use-ocr --ocr-engine paddleocr
uv run python scripts/benchmark_llava.py --model-id llava-hf/llava-1.5-7b-hf --split validation --use-ocr --ocr-engine paddleocr --lora-path results/models/lora_llava_1.5_7b_hf_singleclass
# Multiclass
uv run python scripts/train_vlm.py --model-id llava-hf/llava-1.5-7b-hf --epochs 3 --batch-size 2 --quantize 4bit --device cuda --task multiclass --use-ocr --ocr-engine paddleocr
uv run python scripts/benchmark_llava.py --model-id llava-hf/llava-1.5-7b-hf --split validation --use-ocr --ocr-engine paddleocr --task multiclass --lora-path results/models/lora_llava_1.5_7b_hf_multiclass

# 7. Train & Evaluate Qwen2-VL-7B (~5.2 hours)
Write-Host "[7/7] Training and evaluating Qwen2-VL-7B (QLoRA)..." -ForegroundColor Yellow
# Binary
uv run python scripts/train_vlm.py --model-id Qwen/Qwen2-VL-7B-Instruct --epochs 3 --batch-size 2 --quantize 4bit --device cuda --task singleclass --use-ocr --ocr-engine paddleocr
uv run python scripts/benchmark_qwen2vl.py --model-id Qwen/Qwen2-VL-7B-Instruct --split validation --use-ocr --ocr-engine paddleocr --lora-path results/models/lora_qwen2_vl_7b_instruct_singleclass
# Multiclass
uv run python scripts/train_vlm.py --model-id Qwen/Qwen2-VL-7B-Instruct --epochs 3 --batch-size 2 --quantize 4bit --device cuda --task multiclass --use-ocr --ocr-engine paddleocr
uv run python scripts/benchmark_qwen2vl.py --model-id Qwen/Qwen2-VL-7B-Instruct --split validation --use-ocr --ocr-engine paddleocr --task multiclass --lora-path results/models/lora_qwen2_vl_7b_instruct_multiclass

Write-Host "=== All experiments completed successfully! ===" -ForegroundColor Green
