#!/bin/bash
#SBATCH --job-name=anna-tasks-v2
#SBATCH --partition=gpu.q
#SBATCH --nodelist=gpu10,gpu11,gpu12
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=runs/slurm_logs/anna_tasks_v2_%j.log
#SBATCH --error=runs/slurm_logs/anna_tasks_v2_%j.err

# Corrected re-run of Anna's gist tasks after fixing:
#  - auto_benchmark npz loader crash (target_col check)
#  - auto_benchmark hardcoded C:/Github path
#  - auto_benchmark multilabel pickling (PassThroughEncoder)
#  - train_clip.py / benchmark_qwen2vl.py OCR suffix (no overwrite)
#  - separate validation/test splits (no blended metrics)
#  - CV disabled for tabular (Anna needs val/test acc+F1, not CV scores)
#
# Order: fast GPU tasks first (#95, #93, #94), slow tabular CPU job (#84) last,
# so most results are ready early even if the tabular sweep runs long.

mkdir -p runs/slurm_logs

echo "=== Anna Tasks v2 - SLURM Job Started ==="
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
# ISSUE #95: CLIP ViT-B-32 zero-shot baselines (Gist 9) - separate splits
# ============================================================================
echo ">>> [#95 1/4] CLIP ViT-B-32 zero-shot Task A - validation..."
uv run python scripts/benchmark_clip.py --split validation \
    --model-name ViT-B-32-quickgelu --task singleclass --device cuda || echo "FAILED: #95 A val"
echo ""
echo ">>> [#95 2/4] CLIP ViT-B-32 zero-shot Task A - test..."
uv run python scripts/benchmark_clip.py --split test \
    --model-name ViT-B-32-quickgelu --task singleclass --device cuda || echo "FAILED: #95 A test"
echo ""
echo ">>> [#95 3/4] CLIP ViT-B-32 zero-shot Task B - validation..."
uv run python scripts/benchmark_clip.py --split validation \
    --model-name ViT-B-32-quickgelu --task multiclass --device cuda || echo "FAILED: #95 B val"
echo ""
echo ">>> [#95 4/4] CLIP ViT-B-32 zero-shot Task B - test..."
uv run python scripts/benchmark_clip.py --split test \
    --model-name ViT-B-32-quickgelu --task multiclass --device cuda || echo "FAILED: #95 B test"
echo ""

# ============================================================================
# ISSUE #93: Qwen2-VL-2B zero-shot Task B multiclass (Gist 4)
# Separate splits + distinct OCR/image-only filenames (fixed naming).
# ============================================================================
echo ">>> [#93 1/4] Qwen2-VL-2B multiclass image-only - validation..."
uv run python scripts/benchmark_qwen2vl.py --model-id Qwen/Qwen2-VL-2B-Instruct \
    --split validation --task multiclass --quantize 4bit --device cuda || echo "FAILED: #93 img val"
echo ""
echo ">>> [#93 2/4] Qwen2-VL-2B multiclass image-only - test..."
uv run python scripts/benchmark_qwen2vl.py --model-id Qwen/Qwen2-VL-2B-Instruct \
    --split test --task multiclass --quantize 4bit --device cuda || echo "FAILED: #93 img test"
echo ""
echo ">>> [#93 3/4] Qwen2-VL-2B multiclass OCR-augmented - validation..."
uv run python scripts/benchmark_qwen2vl.py --model-id Qwen/Qwen2-VL-2B-Instruct \
    --split validation --task multiclass --quantize 4bit --device cuda \
    --use-ocr --ocr-engine paddleocr || echo "FAILED: #93 ocr val"
echo ""
echo ">>> [#93 4/4] Qwen2-VL-2B multiclass OCR-augmented - test..."
uv run python scripts/benchmark_qwen2vl.py --model-id Qwen/Qwen2-VL-2B-Instruct \
    --split test --task multiclass --quantize 4bit --device cuda \
    --use-ocr --ocr-engine paddleocr || echo "FAILED: #93 ocr test"
echo ""

# ============================================================================
# ISSUE #94: CLIP ViT-B-32 fine-tuned Task B multiclass (Gist 5)
# Train OCR + non-OCR heads (distinct checkpoint names), eval val AND test.
# ============================================================================
echo ">>> [#94 train] CLIP ViT-B-32 multiclass head (OCR-augmented)..."
uv run python scripts/train_clip.py --model ViT-B-32-quickgelu --epochs 5 \
    --batch-size 16 --loss-mode classification --task multiclass --device cuda \
    --use-ocr --ocr-engine paddleocr || echo "FAILED: #94 train ocr"
echo ""
echo ">>> [#94 train] CLIP ViT-B-32 multiclass head (non-OCR)..."
uv run python scripts/train_clip.py --model ViT-B-32-quickgelu --epochs 5 \
    --batch-size 16 --loss-mode classification --task multiclass --device cuda || echo "FAILED: #94 train non-ocr"
echo ""
echo ">>> [#94 eval] OCR head - validation..."
uv run python scripts/benchmark_clip.py --split validation --device cuda \
    --model-path results/models/finetuned_clip_classification_multiclass_vit_b_32_quickgelu_ocr_paddleocr.pth \
    --task multiclass --use-ocr --ocr-engine paddleocr || echo "FAILED: #94 ocr eval val"
echo ""
echo ">>> [#94 eval] OCR head - test..."
uv run python scripts/benchmark_clip.py --split test --device cuda \
    --model-path results/models/finetuned_clip_classification_multiclass_vit_b_32_quickgelu_ocr_paddleocr.pth \
    --task multiclass --use-ocr --ocr-engine paddleocr || echo "FAILED: #94 ocr eval test"
echo ""
echo ">>> [#94 eval] non-OCR head - validation..."
uv run python scripts/benchmark_clip.py --split validation --device cuda \
    --model-path results/models/finetuned_clip_classification_multiclass_vit_b_32_quickgelu.pth \
    --task multiclass || echo "FAILED: #94 non-ocr eval val"
echo ""
echo ">>> [#94 eval] non-OCR head - test..."
uv run python scripts/benchmark_clip.py --split test --device cuda \
    --model-path results/models/finetuned_clip_classification_multiclass_vit_b_32_quickgelu.pth \
    --task multiclass || echo "FAILED: #94 non-ocr eval test"
echo ""

# ============================================================================
# ISSUE #84: Tabular ML baselines on ViT-L-14 + PaddleOCR embeddings (Gist 1-3)
# Runs LAST because it is the slowest (25 classifiers x {val,test} x {A,B};
# multiclass wraps each base model in MultiOutputClassifier). CV disabled.
# ============================================================================
cd /mnt/lustre2/mres/ghahrem/content-moderation/auto_benchmark

echo ">>> [#84 1/4] Tabular Task A (binary) - validation eval..."
uv run python scripts/run_benchmark.py --init config/init_config.yaml \
    --model config/model/mami_tabular_model.yaml || echo "FAILED: #84 Task A val"
echo ""
echo ">>> [#84 2/4] Tabular Task A (binary) - test eval..."
uv run python scripts/run_benchmark.py --init config/init_config.yaml \
    --model config/model/mami_tabular_model_test.yaml || echo "FAILED: #84 Task A test"
echo ""
echo ">>> [#84 3/4] Tabular Task B (multiclass) - validation eval..."
uv run python scripts/run_benchmark.py --init config/init_config.yaml \
    --model config/model/mami_tabular_model_multiclass.yaml || echo "FAILED: #84 Task B val"
echo ""
echo ">>> [#84 4/4] Tabular Task B (multiclass) - test eval..."
uv run python scripts/run_benchmark.py --init config/init_config.yaml \
    --model config/model/mami_tabular_model_multiclass_test.yaml || echo "FAILED: #84 Task B test"
echo ""

echo "=== Anna Tasks v2 Complete ==="
echo "Date: $(date)"
echo "=== DONE ==="
