#!/bin/bash
#SBATCH --job-name=vlm-moderation-benchmark
#SBATCH --partition=gpu.q
#SBATCH --nodelist=gpu10,gpu11,gpu12
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=runs/slurm_experiments_%j.out
#SBATCH --error=runs/slurm_experiments_%j.err

# Ensure the runs directory exists for logging
mkdir -p runs

echo "=== SLURM Job Started ==="
echo "Node: $(hostname)"
echo "GPU resources allocated:"
nvidia-smi

# 1. Reset modules to a clean state
module purge
module load system/sciama
module load services/slurm

# 2. Load Anaconda and uv
module load anaconda3
module load uv

# 3. Configure cache directories to use the high-performance Lustre filesystem (20GB home quota bypass)
export HF_HOME=/mnt/lustre2/mres/ghahrem/.cache/huggingface
export KAGGLEHUB_CACHE=/mnt/lustre2/mres/ghahrem/.cache/kagglehub

# 4. Activate the environment stored on high-performance Lustre storage
conda activate /mnt/lustre2/mres/ghahrem/envs/vlm-moderation

# 5. Execute the pipeline
bash scripts/run_all_experiments.sh

echo "=== SLURM Job Finished ==="
