#!/bin/bash
# scripts/run_test_evaluation.sh
# Sequential evaluation of zero-shot and fine-tuned models on the MAMI 2022 test split.

set -e # Abort on the first failing command

# Set offline mode environment variables to load models from local Lustre cache.
export HF_HUB_OFFLINE=1
export HF_OFFLINE=1

echo "=== Starting Test Set Evaluation ==="

# 1. Ensure embeddings for the test split are fully extracted
echo "[1/10] Verifying test embeddings..."
uv run python scripts/extract_embeddings.py --split test --model ViT-L-14-quickgelu --use-ocr --ocr-engine paddleocr
uv run python scripts/extract_embeddings.py --split test --model ViT-B-32-quickgelu --use-ocr --ocr-engine paddleocr

# 2. Train & Evaluate XGBoost Classifiers (Validation + Test splits saved to separate JSON files)
echo "[2/10] Evaluating XGBoost Fusion Classifiers..."
uv run python scripts/train_classifier.py --model ViT-L-14-quickgelu --task singleclass --classifier xgboost --use-ocr --ocr-engine paddleocr
uv run python scripts/train_classifier.py --model ViT-L-14-quickgelu --task multiclass --classifier xgboost --use-ocr --ocr-engine paddleocr

# 3. CLIP Zero-Shot Baselines
echo "[3/10] Evaluating Zero-Shot CLIP Baselines..."
uv run python scripts/benchmark_clip.py --split test --device cuda --use-ocr --ocr-engine paddleocr
uv run python scripts/benchmark_clip.py --split test --device cuda --use-ocr --ocr-engine paddleocr --task multiclass

# 4. CLIP Fine-Tuned Heads
echo "[4/10] Evaluating Fine-Tuned CLIP Heads..."
# Binary ViT-B-32
uv run python scripts/benchmark_clip.py --split test --device cuda --model-path results/models/finetuned_clip_classification_singleclass_vit_b_32_quickgelu.pth --use-ocr --ocr-engine paddleocr
# Binary ViT-L-14
uv run python scripts/benchmark_clip.py --split test --device cuda --model-path results/models/finetuned_clip_classification_singleclass_vit_l_14_quickgelu.pth --use-ocr --ocr-engine paddleocr
# Multiclass ViT-L-14
uv run python scripts/benchmark_clip.py --split test --device cuda --model-path results/models/finetuned_clip_classification_multiclass_vit_l_14_quickgelu.pth --use-ocr --ocr-engine paddleocr --task multiclass

# 5. Qwen2-VL-2B (Zero-Shot)
echo "[5/10] Evaluating Zero-Shot Qwen2-VL-2B..."
uv run python scripts/benchmark_qwen2vl.py --model-id Qwen/Qwen2-VL-2B-Instruct --split test --use-ocr --ocr-engine paddleocr

# 6. Qwen2-VL-2B QLoRA Fine-Tuned
echo "[6/10] Evaluating QLoRA Fine-Tuned Qwen2-VL-2B..."
# Binary
uv run python scripts/benchmark_qwen2vl.py --model-id Qwen/Qwen2-VL-2B-Instruct --split test --use-ocr --ocr-engine paddleocr --lora-path results/models/lora_qwen2_vl_2b_instruct_singleclass
# Multiclass
uv run python scripts/benchmark_qwen2vl.py --model-id Qwen/Qwen2-VL-2B-Instruct --split test --use-ocr --ocr-engine paddleocr --task multiclass --lora-path results/models/lora_qwen2_vl_2b_instruct_multiclass

# 7. LLaVA-1.5-7B (Zero-Shot)
echo "[7/10] Evaluating Zero-Shot LLaVA-1.5-7B..."
uv run python scripts/benchmark_llava.py --model-id llava-hf/llava-1.5-7b-hf --split test --use-ocr --ocr-engine paddleocr
uv run python scripts/benchmark_llava.py --model-id llava-hf/llava-1.5-7b-hf --split test --use-ocr --ocr-engine paddleocr --task multiclass

# 8. LLaVA-1.5-7B QLoRA Fine-Tuned
echo "[8/10] Evaluating QLoRA Fine-Tuned LLaVA-1.5-7B..."
# Binary
uv run python scripts/benchmark_llava.py --model-id llava-hf/llava-1.5-7b-hf --split test --use-ocr --ocr-engine paddleocr --lora-path results/models/lora_llava_1.5_7b_hf_singleclass
# Multiclass
uv run python scripts/benchmark_llava.py --model-id llava-hf/llava-1.5-7b-hf --split test --use-ocr --ocr-engine paddleocr --task multiclass --lora-path results/models/lora_llava_1.5_7b_hf_multiclass

# 9. Qwen2-VL-7B QLoRA Fine-Tuned
echo "[9/10] Evaluating QLoRA Fine-Tuned Qwen2-VL-7B..."
# Binary
uv run python scripts/benchmark_qwen2vl.py --model-id Qwen/Qwen2-VL-7B-Instruct --split test --use-ocr --ocr-engine paddleocr --lora-path results/models/lora_qwen2_vl_7b_instruct_singleclass
# Multiclass
uv run python scripts/benchmark_qwen2vl.py --model-id Qwen/Qwen2-VL-7B-Instruct --split test --use-ocr --ocr-engine paddleocr --task multiclass --lora-path results/models/lora_qwen2_vl_7b_instruct_multiclass

# 10. VisualBERT Zero-Shot
echo "[10/10] Evaluating Zero-Shot VisualBERT..."
uv run python scripts/benchmark_visualbert.py --split test --device cuda
uv run python scripts/benchmark_visualbert.py --split test --device cuda --task multiclass

echo "=== All test set evaluations completed successfully! ==="
