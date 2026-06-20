#!/bin/bash
#SBATCH --job-name=mami-tabular-test
#SBATCH --partition=gpu.q
#SBATCH --nodelist=gpu10,gpu11,gpu12
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=runs/slurm_logs/mami_tabular_test_%j.log
#SBATCH --error=runs/slurm_logs/mami_tabular_test_%j.err

# Ensure the logs directory exists
mkdir -p runs/slurm_logs

echo "=== SLURM Tabular Test Training Job Started ==="
echo "Node: $(hostname)"
nvidia-smi

module purge
module load system/sciama
module load services/slurm
module load uv

export HF_HOME=/mnt/lustre2/mres/ghahrem/.cache/huggingface
export KAGGLEHUB_CACHE=/mnt/lustre2/mres/ghahrem/.cache/kagglehub

cd /mnt/lustre2/mres/ghahrem/content-moderation/auto_benchmark
uv run autobench train --model config/model/mami_tabular_model.yaml

echo "=== SLURM Tabular Test Training Job Finished ==="
