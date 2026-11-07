#!/bin/bash
#SBATCH --job-name=rationale-test-split
#SBATCH --partition=gpu.q
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=runs/slurm_logs/rationale_test_%j.log
#SBATCH --error=runs/slurm_logs/rationale_test_%j.err

mkdir -p runs/slurm_logs results/rationales

echo "=== Rationale Generation on Test Split Started ==="
echo "Date: $(date)"
echo "Node: $(hostname)"
nvidia-smi
echo ""

module purge
module load system/sciama
module load services/slurm
module load anaconda3
module load uv
module load cuda/25.1

export HF_HOME=${HOME}/.cache/huggingface
export KAGGLEHUB_CACHE=${HOME}/.cache/kagglehub

conda activate "${CONDA_ENV:-vlm-moderation}"
cd "${PROJECT_ROOT:?set PROJECT_ROOT to the repo checkout}"

echo "--- 1/3: Running Qwen2-VL-7B-Instruct with fine-tuned singleclass adapter on test split ---"
uv run python scripts/explain_with_vlm.py \
    --model qwen \
    --model-id Qwen/Qwen2-VL-7B-Instruct \
    --adapter results/models/lora_qwen2_vl_7b_instruct_singleclass \
    --split test \
    --limit 30 \
    --use-ocr \
    --device cuda \
    --quantize 4bit \
    --output results/rationales/qwen7b_test.jsonl

echo "--- 2/3: Running Qwen2-VL-2B-Instruct with fine-tuned singleclass adapter on test split ---"
uv run python scripts/explain_with_vlm.py \
    --model qwen \
    --model-id Qwen/Qwen2-VL-2B-Instruct \
    --adapter results/models/lora_qwen2_vl_2b_instruct_singleclass \
    --split test \
    --limit 30 \
    --use-ocr \
    --device cuda \
    --quantize 4bit \
    --output results/rationales/qwen2b_test.jsonl

echo "--- 3/3: Running LLaVA-1.5-7B with fine-tuned singleclass adapter on test split ---"
uv run python scripts/explain_with_vlm.py \
    --model llava \
    --model-id llava-hf/llava-1.5-7b-hf \
    --adapter results/models/lora_llava_1.5_7b_hf_singleclass \
    --split test \
    --limit 30 \
    --use-ocr \
    --device cuda \
    --quantize 4bit \
    --output results/rationales/llava7b_test.jsonl

echo "=== All 3 Rationale Generation Jobs Completed ==="
echo "Date: $(date)"
