#!/bin/bash
#SBATCH --job-name=dis01-qwen-raw
#SBATCH --partition=gpu.q
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=results/sciama/logs/%x_%j.log
#SBATCH --error=results/sciama/logs/%x_%j.err
# DIS-01: re-run zero-shot Qwen2-VL-7B-Instruct (4-bit NF4) on Harm-C validation/test, keeping each
# raw model reply. Writes *_raw_responses.json so the S6 outputs are never overwritten.
# Submit ONLY through the checked wrapper:
#   bash sciama/submit.sh S6-qwen-raw
source sciama/jobs/_env.sh
uv run python sciama/sciama_checks.py prereqs --job S6-qwen-raw --in-job
uv run python -m scripts.revision.harmeme_eval qwen --device cuda --batch-size 4 --output-tag _raw_responses
echo "=== DONE dis01 S6-qwen-raw ==="
