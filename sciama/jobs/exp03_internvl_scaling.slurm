#!/bin/bash
#SBATCH --job-name=exp03-internvl
#SBATCH --partition=gpu.q
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=results/sciama/logs/%x_%j.log
#SBATCH --error=results/sciama/logs/%x_%j.err

# EXP-03 (P2): InternVL scaling study, zero-shot + QLoRA per size/version.
# BLOCKED until code change C3 (sciama/JOBS.md) adds scripts/benchmark_internvl.py and InternVL support
# in scripts/train_vlm.py. sciama_checks.py prereqs enforces that.
# Submit ONLY through the checked wrapper, one job per verified model id:
#   bash sciama/submit.sh S5 MODEL_ID=<verified-hf-id> TASK=singleclass SEED=1

source sciama/jobs/_env.sh
: "${MODEL_ID:?set MODEL_ID}" "${TASK:?set TASK}" "${SEED:?set SEED}"
uv run python sciama/sciama_checks.py prereqs --job S5 --model-id "$MODEL_ID" --task "$TASK" --seed "$SEED" --in-job
CLEAN=$(echo "${MODEL_ID##*/}" | tr '[:upper:]' '[:lower:]' | tr '-' '_')

for SPLIT in validation test; do
  uv run python scripts/benchmark_internvl.py --model-id "$MODEL_ID" --task "$TASK" --split "$SPLIT" \
    --text-source provided --quantize 4bit --device cuda
done
uv run python scripts/train_vlm.py --model-id "$MODEL_ID" --task "$TASK" --epochs 3 --batch-size 4 \
  --lr 2e-5 --text-source provided --seed "$SEED" --quantize 4bit --device cuda
ADAPTER="results/models/lora_${CLEAN}_${TASK}_seed${SEED}"
require_file "$ADAPTER/adapter_config.json"
for SPLIT in validation test; do
  uv run python scripts/benchmark_internvl.py --model-id "$MODEL_ID" --task "$TASK" --split "$SPLIT" \
    --text-source provided --lora-path "$ADAPTER" --seed "$SEED" --quantize 4bit --device cuda
done
echo "=== DONE exp03 $MODEL_ID TASK=$TASK SEED=$SEED ==="
