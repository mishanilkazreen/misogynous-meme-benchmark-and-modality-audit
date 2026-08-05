#!/bin/bash
#SBATCH --job-name=vlm-all-explainability
#SBATCH --partition=gpu.q
#SBATCH --nodelist=gpu10,gpu11,gpu12
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=runs/slurm_logs/explainability_all_%j.log
#SBATCH --error=runs/slurm_logs/explainability_all_%j.err

mkdir -p runs/slurm_logs

echo "=== All VLM Explainability Job Started ==="
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

echo "Running LLaVA-1.5-7B explainability..."
uv run python scripts/explain_with_vlm.py --model llava --limit 30 --device cuda --quantize 4bit --output results/llava_explanations.jsonl

echo "Running Qwen2-VL-2B explainability..."
uv run python scripts/explain_with_vlm.py --model qwen --model-id Qwen/Qwen2-VL-2B-Instruct --limit 30 --device cuda --quantize 4bit --output results/qwen2b_explanations.jsonl

echo "=== All Jobs Completed ==="
