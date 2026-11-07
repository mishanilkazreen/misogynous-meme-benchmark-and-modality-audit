#!/bin/bash
#SBATCH --job-name=dis01-harmeme
#SBATCH --partition=gpu.q
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=results/sciama/logs/%x_%j.log
#SBATCH --error=results/sciama/logs/%x_%j.err

# DIS-01 (P1): HarMeme (Harm-C) second-dataset evaluation.
# Extracts frozen CLIP ViT-L-14-quickgelu embeddings across Harm-C train/val/test (3,013 / 177 / 354),
# fits SVM-RBF and XGBoost on train and evaluates on val/test, runs zero-shot Qwen2-VL-7B-Instruct
# (4-bit NF4) on val/test, and writes verified summary artefacts.
# Submit ONLY through the checked wrapper:
#   bash sciama/submit.sh S6

source sciama/jobs/_env.sh
uv run python sciama/sciama_checks.py prereqs --job S6 --in-job

uv run python -m scripts.revision.harmeme_eval run-sciama --device cuda --batch-size 4

echo "=== DONE dis01 S6 ==="
