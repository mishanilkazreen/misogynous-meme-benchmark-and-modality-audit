#!/bin/bash
#SBATCH --job-name=dis01-qwen-def
#SBATCH --partition=gpu.q
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=results/sciama/logs/%x_%j.log
#SBATCH --error=results/sciama/logs/%x_%j.err
# DIS-01: prompt sensitivity check. Zero-shot Qwen2-VL-7B-Instruct (4-bit NF4) on Harm-C validation/test
# using the dataset paper's definition of harm. Writes *_definition.json, so earlier outputs are kept.
# Submit ONLY through the checked wrapper:
#   bash sciama/submit.sh S6-qwen-def
source sciama/jobs/_env.sh
uv run python sciama/sciama_checks.py prereqs --job S6-qwen-def --in-job
uv run python -m scripts.revision.harmeme_eval qwen --device cuda --batch-size 4 --prompt-variant definition --output-tag _definition
echo "=== DONE dis01 S6-qwen-def ==="
