#!/bin/bash
#SBATCH --job-name=eval-completed
#SBATCH --partition=gpu.q
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0:30:00
#SBATCH --output=runs/slurm_logs/eval_completed_%j.log
#SBATCH --error=runs/slurm_logs/eval_completed_%j.err

mkdir -p runs/slurm_logs

echo "=== MAMI Evaluation of Completed Models ==="
module purge
module load system/sciama
module load services/slurm
module load anaconda3
module load uv

conda activate /mnt/lustre2/mres/ghahrem/envs/vlm-moderation
cd /mnt/lustre2/mres/ghahrem/content-moderation

echo "Evaluating VLM 2B Task A Seed 2..."
uv run python scripts/benchmark_qwen2vl.py --model-id Qwen/Qwen2-VL-2B-Instruct --split validation,test --task singleclass --text-source provided --lora-path results/models/lora_qwen2_vl_2b_instruct_singleclass_seed2 --device cuda

echo "Evaluating VLM 2B Task A Seed 3..."
uv run python scripts/benchmark_qwen2vl.py --model-id Qwen/Qwen2-VL-2B-Instruct --split validation,test --task singleclass --text-source provided --lora-path results/models/lora_qwen2_vl_2b_instruct_singleclass_seed3 --device cuda

echo "Evaluating VLM 7B Task A Seed 1..."
uv run python scripts/benchmark_qwen2vl.py --model-id Qwen/Qwen2-VL-7B-Instruct --split validation,test --task singleclass --text-source provided --lora-path results/models/lora_qwen2_vl_7b_instruct_singleclass_seed1 --device cuda

echo "=== Finished ==="
