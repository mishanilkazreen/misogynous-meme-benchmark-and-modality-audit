#!/bin/bash
#SBATCH --job-name=exp07-llava-zeroshot
#SBATCH --partition=gpu.q
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=results/sciama/logs/%x_%j.log
#SBATCH --error=results/sciama/logs/%x_%j.err

# EXP-01/EXP-07: re-run LLaVA-1.5-7B zero-shot on each split SEPARATELY (the committed LLaVA
# zero-shot files predate the per-split evaluation fix). Task A and Task B, provided and
# PaddleOCR text. Overwrites results/{validation,test}/llava_<split>_llava_1_5_7b_hf*.json; the old
# versions stay recoverable from git history (review the diff before committing).
# Submit ONLY through the checked wrapper: bash sciama/submit.sh S1-zeroshot

source sciama/jobs/_env.sh
uv run python sciama/sciama_checks.py prereqs --job S1-zeroshot --allow-overwrite --in-job
MODEL_ID="llava-hf/llava-1.5-7b-hf"

for TASK in singleclass multiclass; do
  for TS in provided ocr; do
    for SPLIT in validation test; do
      echo "=== ZERO-SHOT llava TASK=$TASK text=$TS split=$SPLIT ==="
      uv run python scripts/benchmark_llava.py --model-id "$MODEL_ID" --task "$TASK" --split "$SPLIT" \
        --text-source "$TS" --ocr-engine paddleocr --quantize 4bit --device cuda
    done
  done
done
echo "=== DONE exp07 llava zero-shot ==="
