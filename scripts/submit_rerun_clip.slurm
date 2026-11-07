#!/bin/bash
#SBATCH --job-name=mami-rerun-clip
#SBATCH --partition=gpu.q
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=runs/slurm_logs/rerun_clip_%j.log
#SBATCH --error=runs/slurm_logs/rerun_clip_%j.err

mkdir -p runs/slurm_logs

echo "=== MAMI Rerun CLIP - SLURM Job Started ==="
echo "Date: $(date)"
echo "Node: $(hostname)"
nvidia-smi

module purge
module load system/sciama
module load services/slurm
module load anaconda3
module load uv

export HF_HOME=${HOME}/.cache/huggingface
export KAGGLEHUB_CACHE=${HOME}/.cache/kagglehub

conda activate "${CONDA_ENV:-vlm-moderation}"
cd "${PROJECT_ROOT:?set PROJECT_ROOT to the repo checkout}"

# Phase 1.3: CLIP Zero-Shot (both models, validation and test, binary and multilabel)
echo ">>> [CLIP Zero-Shot] Running baselines..."
for MODEL in ViT-B-32-quickgelu ViT-L-14-quickgelu; do
  # Task A (binary)
  uv run python scripts/benchmark_clip.py --split validation,test --task singleclass --model-name $MODEL --text-source provided --prompt-ensemble --tta --device cuda
  # Task B (multilabel)
  uv run python scripts/benchmark_clip.py --split validation,test --task multiclass --text-source provided --prompt-ensemble --tta --device cuda
done

# Phase 1.4 & 2.2: CLIP Fine-Tuned (both models, validation and test, binary and multilabel, seeds 1, 2, 3)
echo ">>> [CLIP Fine-Tuning] Training and evaluating classification heads..."
for MODEL in ViT-B-32-quickgelu ViT-L-14-quickgelu; do
  MODEL_CLEAN=$(echo $MODEL | tr '[:upper:]' '[:lower:]' | tr '-' '_')
  for SEED in 1 2 3; do
    echo "=== Training $MODEL | Task: binary | Seed: $SEED ==="
    # uv run python scripts/train_clip.py --model $MODEL --loss-mode classification --task binary --epochs 10 --batch-size 32 --text-source provided --seed $SEED --device cuda

    echo "Evaluating $MODEL | Task: binary | Seed: $SEED..."
    uv run python scripts/benchmark_clip.py --split validation,test --device cuda --model-path results/models/finetuned_clip_classification_singleclass_${MODEL_CLEAN}_seed${SEED}.pth --task singleclass

    echo "=== Training $MODEL | Task: multilabel | Seed: $SEED ==="
    # uv run python scripts/train_clip.py --model $MODEL --loss-mode classification --task multilabel --epochs 10 --batch-size 32 --text-source provided --seed $SEED --device cuda

    echo "Evaluating $MODEL | Task: multilabel | Seed: $SEED..."
    uv run python scripts/benchmark_clip.py --split validation,test --device cuda --model-path results/models/finetuned_clip_classification_multiclass_${MODEL_CLEAN}_seed${SEED}.pth --task multiclass
  done
done

echo "=== MAMI Rerun CLIP - SLURM Job Finished ==="
