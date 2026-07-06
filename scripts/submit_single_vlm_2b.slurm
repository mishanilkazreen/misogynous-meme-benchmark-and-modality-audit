#!/bin/bash
#SBATCH --job-name=mami-vlm-2b
#SBATCH --partition=gpu.q
#SBATCH --nodelist=gpu10,gpu11,gpu12
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=runs/slurm_logs/vlm_2b_%j.log
#SBATCH --error=runs/slurm_logs/vlm_2b_%j.err

# Inputs: SEED, TASK
# Example: sbatch --export=ALL,SEED=2,TASK=multiclass scripts/submit_single_vlm_2b.slurm

mkdir -p runs/slurm_logs

echo "=== MAMI VLM 2B - SLURM Job Started ==="
echo "Task: $TASK | Seed: $SEED"
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

echo "=== Fine-tuning Qwen2-VL-2B | Task: $TASK | Seed: $SEED ==="
uv run python scripts/train_vlm.py --model-id $MODEL --task $TASK --epochs 3 --batch-size 4 --lr 2e-5 --text-source provided --seed $SEED --device cuda

echo "Evaluating Qwen2-VL-2B | Task: $TASK | Seed: $SEED..."
uv run python scripts/benchmark_qwen2vl.py --model-id $MODEL --split validation,test --task $TASK --text-source provided --lora-path results/models/lora_${MODEL_CLEAN}_${TASK}_seed${SEED} --device cuda

echo "=== MAMI VLM 2B - SLURM Job Finished ==="
