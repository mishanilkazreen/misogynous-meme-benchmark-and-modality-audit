#!/bin/bash
#SBATCH --job-name=mami-rerun-vlm-2b
#SBATCH --partition=gpu.q
#SBATCH --nodelist=gpu10,gpu11,gpu12
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=runs/slurm_logs/rerun_vlm_2b_%j.log
#SBATCH --error=runs/slurm_logs/rerun_vlm_2b_%j.err

mkdir -p runs/slurm_logs

echo "=== MAMI Rerun VLM 2B - SLURM Job Started ==="
echo "Date: $(date)"
echo "Node: $(hostname)"
nvidia-smi

module purge
module load system/sciama
module load services/slurm
module load anaconda3
module load uv

export HF_HOME=/mnt/lustre2/mres/ghahrem/.cache/huggingface
export KAGGLEHUB_CACHE=/mnt/lustre2/mres/ghahrem/.cache/kagglehub

conda activate /mnt/lustre2/mres/ghahrem/envs/vlm-moderation
cd /mnt/lustre2/mres/ghahrem/content-moderation

MODEL="Qwen/Qwen2-VL-2B-Instruct"
MODEL_CLEAN="qwen2_vl_2b_instruct"

# Phase 1.6: Qwen2-VL-2B Zero-Shot (Task A & Task B)
# echo ">>> [Phase 1.6] Qwen2-VL-2B Zero-Shot..."
# uv run python scripts/benchmark_qwen2vl.py --model-id $MODEL --split validation,test --task singleclass --text-source provided --device cuda
# uv run python scripts/benchmark_qwen2vl.py --model-id $MODEL --split validation,test --task multiclass --text-source provided --device cuda

# Phase 1.7: Qwen2-VL-2B QLoRA Task A (seeds 1, 2, 3)
# echo ">>> [Phase 1.7] Qwen2-VL-2B QLoRA Task A fine-tuning..."
# for SEED in 1 2 3; do
#   echo "=== Fine-tuning Task A | Seed: $SEED ==="
#   uv run python scripts/train_vlm.py --model-id $MODEL --task singleclass --epochs 3 --batch-size 4 --lr 2e-5 --text-source provided --seed $SEED --device cuda
#
#   echo "Evaluating Task A | Seed: $SEED..."
#   uv run python scripts/benchmark_qwen2vl.py --model-id $MODEL --split validation,test --task singleclass --text-source provided --lora-path results/models/lora_${MODEL_CLEAN}_singleclass_seed${SEED} --device cuda
# done

# Phase 2.3: Qwen2-VL-2B QLoRA Task B (seeds 1, 2, 3)
echo ">>> [Phase 2.3] Qwen2-VL-2B QLoRA Task B fine-tuning..."
for SEED in 1 2 3; do
  echo "=== Fine-tuning Task B | Seed: $SEED ==="
  uv run python scripts/train_vlm.py --model-id $MODEL --task multiclass --epochs 3 --batch-size 4 --lr 2e-5 --text-source provided --seed $SEED --device cuda

  echo "Evaluating Task B | Seed: $SEED..."
  uv run python scripts/benchmark_qwen2vl.py --model-id $MODEL --split validation,test --task multiclass --text-source provided --lora-path results/models/lora_${MODEL_CLEAN}_multiclass_seed${SEED} --device cuda
done

# Phase 2b.2: Qwen2-VL-2B QLoRA Joint Training (seed 1)
echo ">>> [Phase 2b.2] Qwen2-VL-2B Joint training..."
uv run python scripts/train_vlm.py --model-id $MODEL --task joint --epochs 3 --batch-size 4 --lr 2e-5 --text-source provided --seed 1 --device cuda

echo "Evaluating Joint model on Task A..."
uv run python scripts/benchmark_qwen2vl.py --model-id $MODEL --split validation,test --task singleclass --text-source provided --lora-path results/models/lora_${MODEL_CLEAN}_joint_seed1 --device cuda

echo "Evaluating Joint model on Task B..."
uv run python scripts/benchmark_qwen2vl.py --model-id $MODEL --split validation,test --task multiclass --text-source provided --lora-path results/models/lora_${MODEL_CLEAN}_joint_seed1 --device cuda

echo "=== MAMI Rerun VLM 2B - SLURM Job Finished ==="
