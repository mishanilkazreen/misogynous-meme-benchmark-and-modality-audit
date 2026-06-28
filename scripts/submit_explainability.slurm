#!/bin/bash
#SBATCH --job-name=vlm-explainability
#SBATCH --partition=gpu.q
#SBATCH --nodelist=gpu10,gpu11,gpu12
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=runs/slurm_logs/explainability_%j.log
#SBATCH --error=runs/slurm_logs/explainability_%j.err

# Create directory for logs
mkdir -p runs/slurm_logs

echo "=== VLM Explainability Job Started ==="
echo "Date: $(date)"
echo "Node: $(hostname)"
nvidia-smi
echo ""

# Load environment modules on Sciama
module purge
module load system/sciama
module load services/slurm
module load anaconda3
module load uv

# Set HuggingFace and Kaggle cache paths
export HF_HOME=/mnt/lustre2/mres/ghahrem/.cache/huggingface
export KAGGLEHUB_CACHE=/mnt/lustre2/mres/ghahrem/.cache/kagglehub

# Activate conda environment
conda activate /mnt/lustre2/mres/ghahrem/envs/vlm-moderation
cd /mnt/lustre2/mres/ghahrem/content-moderation

echo "Environment activated. Running explain_with_vlm.py..."
uv run python scripts/explain_with_vlm.py --model qwen --limit 30 --device cuda --quantize 4bit --output results/vlm_explanations.jsonl

echo "=== Job Completed ==="
