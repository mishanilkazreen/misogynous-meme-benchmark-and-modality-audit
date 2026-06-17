#!/bin/bash
#SBATCH --job-name=vlm-test-evaluation
#SBATCH --partition=gpu.q
#SBATCH --nodelist=gpu10,gpu11,gpu12
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=runs/slurm_logs/test_eval_%j.log
#SBATCH --error=runs/slurm_logs/test_eval_%j.err

# Ensure the logs directory exists
mkdir -p runs/slurm_logs

echo "=== SLURM Test Evaluation Job Started ==="
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

# 3. Configure cache directories to use high-performance Lustre storage
export HF_HOME=/mnt/lustre2/mres/ghahrem/.cache/huggingface
export KAGGLEHUB_CACHE=/mnt/lustre2/mres/ghahrem/.cache/kagglehub

# 4. Activate the conda environment stored on Lustre
conda activate /mnt/lustre2/mres/ghahrem/envs/vlm-moderation

# 5. Execute the test evaluation script
bash scripts/run_test_evaluation.sh

echo "=== SLURM Test Evaluation Job Finished ==="
