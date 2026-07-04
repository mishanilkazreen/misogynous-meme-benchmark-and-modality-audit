#!/bin/bash
#SBATCH --job-name=vlm-2b-seed1-fix
#SBATCH --partition=gpu.q
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --output=runs/slurm_logs/vlm_2b_seed1_fix_%j.log
#SBATCH --error=runs/slurm_logs/vlm_2b_seed1_fix_%j.err

mkdir -p runs/slurm_logs

echo "=== MAMI VLM 2B Seed 1 Fix - SLURM Job Started ==="
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

echo "Evaluating Task A | Seed: 1..."
uv run python scripts/benchmark_qwen2vl.py --model-id $MODEL --split validation,test --task singleclass --text-source provided --lora-path results/models/lora_${MODEL_CLEAN}_singleclass_seed1 --device cuda

echo "=== MAMI VLM 2B Seed 1 Fix - SLURM Job Finished ==="
