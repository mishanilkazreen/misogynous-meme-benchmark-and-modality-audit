#!/bin/bash
#SBATCH --job-name=anna-gist-tasks
#SBATCH --partition=gpu.q
#SBATCH --nodelist=gpu10,gpu11,gpu12
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=runs/slurm_logs/anna_tasks_%j.log
#SBATCH --error=runs/slurm_logs/anna_tasks_%j.err

# Ensure the logs directory exists
mkdir -p runs/slurm_logs

echo "=== Anna Gist Tasks - SLURM Job Started ==="
echo "Node: $(hostname)"
echo "Date: $(date)"
nvidia-smi
echo ""

# Environment setup
module purge
module load system/sciama
module load services/slurm
module load anaconda3
module load uv

export HF_HOME=/mnt/lustre2/mres/ghahrem/.cache/huggingface
export KAGGLEHUB_CACHE=/mnt/lustre2/mres/ghahrem/.cache/kagglehub

conda activate /mnt/lustre2/mres/ghahrem/envs/vlm-moderation
cd /mnt/lustre2/mres/ghahrem/content-moderation

echo "Environment activated. Python: $(python --version)"
echo ""

# ============================================================================
# TASK 1: Issue #84 - Fix Trad ML Pipeline & Rerun (Gist IDs 1-3)
# Runs all 22 classifiers on ViT-L-14 + PaddleOCR embeddings
# Expected time: ~5 minutes (CPU-based, embedding-only classifiers)
# ============================================================================
echo ">>> [1/8] Issue #84: Running auto_benchmark Task A (binary)..."
cd /mnt/lustre2/mres/ghahrem/content-moderation/auto_benchmark
uv run python scripts/run_benchmark.py \
    --init config/init_config.yaml \
    --model config/model/mami_tabular_model.yaml
echo "    Task A benchmark complete."
echo ""

echo ">>> [2/8] Issue #84: Running auto_benchmark Task B (multiclass)..."
uv run python scripts/run_benchmark.py \
    --init config/init_config.yaml \
    --model config/model/mami_tabular_model_multiclass.yaml
echo "    Task B benchmark complete."
echo ""

cd /mnt/lustre2/mres/ghahrem/content-moderation

# ============================================================================
# TASK 2: Issue #95 - CLIP ViT-B-32 Zero-Shot Baselines (Gist ID 9)
# Runs zero-shot CLIP ViT-B-32-quickgelu on validation+test for both tasks
# Expected time: ~20 seconds
# ============================================================================
echo ">>> [3/8] Issue #95: CLIP ViT-B-32 zero-shot Task A (validation+test)..."
uv run python scripts/benchmark_clip.py \
    --split validation,test \
    --model-name ViT-B-32-quickgelu \
    --task singleclass \
    --device cuda
echo "    CLIP ViT-B-32 Task A complete."
echo ""

echo ">>> [4/8] Issue #95: CLIP ViT-B-32 zero-shot Task B (validation+test)..."
uv run python scripts/benchmark_clip.py \
    --split validation,test \
    --model-name ViT-B-32-quickgelu \
    --task multiclass \
    --device cuda
echo "    CLIP ViT-B-32 Task B complete."
echo ""

# ============================================================================
# TASK 3: Issue #93 - Qwen2-VL-2B Zero-Shot Task B (Gist ID 4)
# Runs zero-shot Qwen2-VL-2B on validation+test for multiclass
# Two configs: image-only and OCR-augmented
# Expected time: ~6 minutes total
# ============================================================================
echo ">>> [5/8] Issue #93: Qwen2-VL-2B zero-shot multiclass (image-only, validation+test)..."
uv run python scripts/benchmark_qwen2vl.py \
    --model-id Qwen/Qwen2-VL-2B-Instruct \
    --split validation,test \
    --task multiclass \
    --quantize 4bit \
    --device cuda
echo "    Qwen2-VL-2B image-only multiclass complete."
echo ""

echo ">>> [6/8] Issue #93: Qwen2-VL-2B zero-shot multiclass (OCR-augmented, validation+test)..."
uv run python scripts/benchmark_qwen2vl.py \
    --model-id Qwen/Qwen2-VL-2B-Instruct \
    --split validation,test \
    --task multiclass \
    --quantize 4bit \
    --device cuda \
    --use-ocr \
    --ocr-engine paddleocr
echo "    Qwen2-VL-2B OCR multiclass complete."
echo ""

# ============================================================================
# TASK 4: Issue #94 - CLIP ViT-B-32 Fine-Tuned Task B (Gist ID 5)
# Train + evaluate CLIP ViT-B-32 classification head for multiclass
# Two configs: OCR-augmented and Non-OCR (ground-truth transcripts)
# Expected time: ~8 minutes total
# ============================================================================
echo ">>> [7/8] Issue #94: Train CLIP ViT-B-32 multiclass head (OCR-augmented)..."
uv run python scripts/train_clip.py \
    --model ViT-B-32-quickgelu \
    --epochs 5 \
    --batch-size 16 \
    --loss-mode classification \
    --task multiclass \
    --device cuda \
    --use-ocr \
    --ocr-engine paddleocr
echo "    Training complete. Evaluating..."

uv run python scripts/benchmark_clip.py \
    --split validation,test \
    --device cuda \
    --model-path results/models/finetuned_clip_classification_multiclass_vit_b_32_quickgelu_ocr_paddleocr.pth \
    --task multiclass \
    --use-ocr \
    --ocr-engine paddleocr
echo "    CLIP ViT-B-32 OCR multiclass eval complete."
echo ""

echo ">>> [8/8] Issue #94: Train CLIP ViT-B-32 multiclass head (Non-OCR)..."
uv run python scripts/train_clip.py \
    --model ViT-B-32-quickgelu \
    --epochs 5 \
    --batch-size 16 \
    --loss-mode classification \
    --task multiclass \
    --device cuda
echo "    Training complete. Evaluating..."

uv run python scripts/benchmark_clip.py \
    --split validation,test \
    --device cuda \
    --model-path results/models/finetuned_clip_classification_multiclass_vit_b_32_quickgelu.pth \
    --task multiclass
echo "    CLIP ViT-B-32 non-OCR multiclass eval complete."
echo ""

# ============================================================================
# SUMMARY
# ============================================================================
echo "=== All Tasks Complete ==="
echo "Date: $(date)"
echo ""
echo "Expected output files:"
echo "  Issue #84:"
echo "    auto_benchmark/results/model_results/mami_tabular_model/mami_tabular_model_evaluation.csv"
echo "    auto_benchmark/results/model_results/mami_tabular_model_multiclass/mami_tabular_model_multiclass_evaluation.csv"
echo ""
echo "  Issue #95:"
echo "    results/validation/clip_validation_vit_b_32_quickgelu.json"
echo "    results/test/clip_test_vit_b_32_quickgelu.json"
echo "    results/validation/clip_validation_multiclass_vit_b_32_quickgelu.json"
echo "    results/test/clip_test_multiclass_vit_b_32_quickgelu.json"
echo ""
echo "  Issue #93:"
echo "    results/validation/qwen2vl_validation_multiclass.json"
echo "    results/test/qwen2vl_test_multiclass.json"
echo "    results/validation/qwen2vl_validation_ocr_paddleocr_multiclass.json"
echo "    results/test/qwen2vl_test_ocr_paddleocr_multiclass.json"
echo ""
echo "  Issue #94:"
echo "    results/models/finetuned_clip_classification_multiclass_vit_b_32_quickgelu_ocr_paddleocr.pth"
echo "    results/models/finetuned_clip_classification_multiclass_vit_b_32_quickgelu.pth"
echo "    results/validation/clip_validation_finetuned_multiclass_vit_b_32_quickgelu*.json"
echo "    results/test/clip_test_finetuned_multiclass_vit_b_32_quickgelu*.json"
echo ""
echo "=== DONE ==="
