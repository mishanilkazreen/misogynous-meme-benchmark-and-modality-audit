#!/bin/bash
#SBATCH --job-name=exp05-ocr-qwen
#SBATCH --partition=gpu.q
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=results/sciama/logs/%x_%j.log
#SBATCH --error=results/sciama/logs/%x_%j.err

# EXP-05: OCR ablation on the FINE-TUNED Qwen2-VL adapters. Evaluates the existing
# seed 1-3 adapters (trained on provided text) with PaddleOCR text at inference, each split separately.
# Zero-shot OCR runs already exist (scripts/submit_reeval_vlm.slurm).
# Submit ONLY through the checked wrapper: bash sciama/submit.sh S2 MODEL=2b   (then MODEL=7b)
# Outputs: results/{validation,test}/qwen2vl_<split>_<model>_ocr_paddleocr[_multiclass]_finetuned_seedN.json

source sciama/jobs/_env.sh
: "${MODEL:?set MODEL (2b|7b)}"
uv run python sciama/sciama_checks.py prereqs --job S2 --model "$MODEL" --in-job
if [ "$MODEL" = "7b" ]; then
  MODEL_ID="Qwen/Qwen2-VL-7B-Instruct"; CLEAN="qwen2_vl_7b_instruct"
else
  MODEL_ID="Qwen/Qwen2-VL-2B-Instruct"; CLEAN="qwen2_vl_2b_instruct"
fi

MISSING=0
for TASK in singleclass multiclass; do
  for SEED in 1 2 3; do
    ADAPTER="results/models/lora_${CLEAN}_${TASK}_seed${SEED}"
    if [ ! -f "$ADAPTER/adapter_config.json" ]; then
      echo "MISSING adapter: $ADAPTER" >&2; MISSING=1; continue
    fi
    for SPLIT in validation test; do
      echo "=== OCR-FINETUNED $MODEL TASK=$TASK seed=$SEED split=$SPLIT ==="
      uv run python scripts/benchmark_qwen2vl.py --model-id "$MODEL_ID" --task "$TASK" --split "$SPLIT" \
        --text-source ocr --ocr-engine paddleocr --lora-path "$ADAPTER" --seed "$SEED" \
        --quantize 4bit --device cuda
    done
  done
done
# A missing adapter makes the ablation incomplete: fail the job so it is noticed.
exit "$MISSING"
