# rerun_leftover_models.ps1
# Re-runs every model that did NOT finish in the original run (which was killed after
# the XGBoost stage). Qwen2-VL-2B is intentionally excluded.
#
# Already completed (NOT repeated here):
#   - dependency sync + dataset download
#   - ViT-L-14 OCR + embeddings (results/embeddings/*_ocr_paddleocr.npz)
#   - XGBoost binary + multiclass classifiers
#
# Still to run (this script):
#   - CLIP ViT-B-32 head (binary)            [fine-tuned]
#   - CLIP ViT-L-14 head (binary + multiclass) [fine-tuned]
#   - LLaVA-1.5-7B (binary + multiclass)     [QLoRA fine-tuned]
#   - Qwen2-VL-7B (binary + multiclass)      [QLoRA fine-tuned]
#
# OCR REUSE: this script does NOT run scripts/extract_embeddings.py. OCR is computed only
# by that script and saved to results/embeddings/*_ocr_paddleocr.npz. The train/benchmark
# scripts below load those existing transcripts via --use-ocr --ocr-engine paddleocr (they
# glob results/embeddings for any *_ocr_paddleocr.npz for the split), so OCR is read from
# your already-computed ViT-L-14 files and never recalculated.
#
# Prerequisite: ViT-B-32-quickgelu, LLaVA-1.5-7B and Qwen2-VL-7B must be in the local HF
# cache (they are). Run scripts/predownload_models.ps1 first only if anything is missing.

$ErrorActionPreference = "Stop"
# Abort on the first failing native command (uv/python) instead of cascading silently.
$PSNativeCommandUseErrorActionPreference = $true

# Offline mode intentionally left OFF: cached weights still load, and a still-missing model
# downloads instead of crashing. Uncomment to force strict offline behavior.
# $env:HF_HUB_OFFLINE = 1
# $env:HF_OFFLINE = 1

Write-Host "=== Re-running leftover models (CLIP B-32 + L-14, LLaVA-7B, Qwen2-VL-7B) ===" -ForegroundColor Green
Write-Host "Reusing existing paddleocr OCR transcripts from results/embeddings (no OCR recompute)." -ForegroundColor Cyan

# 1. CLIP classification heads (~24 min total)
Write-Host "[1/3] Training and evaluating CLIP classification heads..." -ForegroundColor Yellow
# Binary ViT-B-32
uv run python scripts/train_clip.py --model ViT-B-32-quickgelu --epochs 5 --batch-size 16 --loss-mode classification --task singleclass --device cuda --use-ocr --ocr-engine paddleocr
uv run python scripts/benchmark_clip.py --split validation --device cuda --model-path results/models/finetuned_clip_classification_singleclass_vit_b_32_quickgelu.pth --use-ocr --ocr-engine paddleocr
# Binary ViT-L-14 (batch 4: full fine-tune of ViT-L-14 in fp32 exceeds 12 GB at batch 16)
uv run python scripts/train_clip.py --model ViT-L-14-quickgelu --epochs 5 --batch-size 4 --loss-mode classification --task singleclass --device cuda --use-ocr --ocr-engine paddleocr
uv run python scripts/benchmark_clip.py --split validation --device cuda --model-path results/models/finetuned_clip_classification_singleclass_vit_l_14_quickgelu.pth --use-ocr --ocr-engine paddleocr
# Multiclass ViT-L-14 (batch 4 for the same VRAM reason)
uv run python scripts/train_clip.py --model ViT-L-14-quickgelu --epochs 5 --batch-size 4 --loss-mode classification --task multiclass --device cuda --use-ocr --ocr-engine paddleocr
uv run python scripts/benchmark_clip.py --split validation --device cuda --model-path results/models/finetuned_clip_classification_multiclass_vit_l_14_quickgelu.pth --use-ocr --ocr-engine paddleocr --task multiclass

# 2. LLaVA-1.5-7B (QLoRA) - binary + multiclass (~4.5 h)
Write-Host "[2/3] Training and evaluating LLaVA-1.5-7B (QLoRA)..." -ForegroundColor Yellow
# Binary
uv run python scripts/train_vlm.py --model-id llava-hf/llava-1.5-7b-hf --epochs 3 --batch-size 2 --quantize 4bit --device cuda --task singleclass --use-ocr --ocr-engine paddleocr
uv run python scripts/benchmark_llava.py --model-id llava-hf/llava-1.5-7b-hf --split validation --use-ocr --ocr-engine paddleocr --lora-path results/models/lora_llava_1.5_7b_hf_singleclass
# Multiclass
uv run python scripts/train_vlm.py --model-id llava-hf/llava-1.5-7b-hf --epochs 3 --batch-size 2 --quantize 4bit --device cuda --task multiclass --use-ocr --ocr-engine paddleocr
uv run python scripts/benchmark_llava.py --model-id llava-hf/llava-1.5-7b-hf --split validation --use-ocr --ocr-engine paddleocr --task multiclass --lora-path results/models/lora_llava_1.5_7b_hf_multiclass

# 3. Qwen2-VL-7B (QLoRA) - binary + multiclass (~5.2 h)
Write-Host "[3/3] Training and evaluating Qwen2-VL-7B (QLoRA)..." -ForegroundColor Yellow
# Binary
uv run python scripts/train_vlm.py --model-id Qwen/Qwen2-VL-7B-Instruct --epochs 3 --batch-size 2 --quantize 4bit --device cuda --task singleclass --use-ocr --ocr-engine paddleocr
uv run python scripts/benchmark_qwen2vl.py --model-id Qwen/Qwen2-VL-7B-Instruct --split validation --use-ocr --ocr-engine paddleocr --lora-path results/models/lora_qwen2_vl_7b_instruct_singleclass
# Multiclass
uv run python scripts/train_vlm.py --model-id Qwen/Qwen2-VL-7B-Instruct --epochs 3 --batch-size 2 --quantize 4bit --device cuda --task multiclass --use-ocr --ocr-engine paddleocr
uv run python scripts/benchmark_qwen2vl.py --model-id Qwen/Qwen2-VL-7B-Instruct --split validation --use-ocr --ocr-engine paddleocr --task multiclass --lora-path results/models/lora_qwen2_vl_7b_instruct_multiclass

Write-Host "=== Leftover models completed successfully! ===" -ForegroundColor Green
