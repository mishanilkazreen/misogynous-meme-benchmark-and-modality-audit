#!/bin/bash
#SBATCH --job-name=issue93-ocr-fix
#SBATCH --partition=gpu.q
#SBATCH --nodelist=gpu10,gpu11,gpu12
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=runs/slurm_logs/issue93_ocr_fix_%j.log
#SBATCH --error=runs/slurm_logs/issue93_ocr_fix_%j.err

mkdir -p runs/slurm_logs

echo "=== Issue #93 OCR Fix - SLURM Job Started ==="
echo "Date: $(date)"
echo "Node: $(hostname)"
nvidia-smi
echo ""

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
# Issue #93 OCR-Augmented Re-run (separate splits to match NPZ files)
# The previous run used --split validation,test but OCR embeddings
# exist as separate validation and test NPZ files.
# ============================================================================

echo ">>> [1/4] Qwen2-VL-2B zero-shot multiclass OCR (validation only)..."
uv run python scripts/benchmark_qwen2vl.py \
    --model-id Qwen/Qwen2-VL-2B-Instruct \
    --split validation \
    --task multiclass \
    --quantize 4bit \
    --device cuda \
    --use-ocr \
    --ocr-engine paddleocr
echo "    Validation OCR complete."
echo ""

echo ">>> [2/4] Qwen2-VL-2B zero-shot multiclass OCR (test only)..."
uv run python scripts/benchmark_qwen2vl.py \
    --model-id Qwen/Qwen2-VL-2B-Instruct \
    --split test \
    --task multiclass \
    --quantize 4bit \
    --device cuda \
    --use-ocr \
    --ocr-engine paddleocr
echo "    Test OCR complete."
echo ""

# Also re-run the image-only with separate splits for consistency
echo ">>> [3/4] Qwen2-VL-2B zero-shot multiclass image-only (validation only)..."
uv run python scripts/benchmark_qwen2vl.py \
    --model-id Qwen/Qwen2-VL-2B-Instruct \
    --split validation \
    --task multiclass \
    --quantize 4bit \
    --device cuda
echo "    Validation image-only complete."
echo ""

echo ">>> [4/4] Qwen2-VL-2B zero-shot multiclass image-only (test only)..."
uv run python scripts/benchmark_qwen2vl.py \
    --model-id Qwen/Qwen2-VL-2B-Instruct \
    --split test \
    --task multiclass \
    --quantize 4bit \
    --device cuda
echo "    Test image-only complete."
echo ""

echo "=== Issue #93 OCR Fix Complete ==="
echo "Date: $(date)"
echo ""
echo "Expected output files:"
echo "  results/validation/qwen2vl_validation_*multiclass*.json"
echo "  results/test/qwen2vl_test_*multiclass*.json"
echo ""
echo "=== DONE ==="
