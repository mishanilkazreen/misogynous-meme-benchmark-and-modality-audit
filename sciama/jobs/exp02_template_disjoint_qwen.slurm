#!/bin/bash
#SBATCH --job-name=exp02-tdisjoint-qwen
#SBATCH --partition=gpu.q
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=results/sciama/logs/%x_%j.log
#SBATCH --error=results/sciama/logs/%x_%j.err

# EXP-02: re-fine-tune Qwen2-VL on the TEMPLATE-DISJOINT split and evaluate on its validation/test.
# BLOCKED until code change C2 (sciama/JOBS.md) is merged: the split must be selectable through
# MAMI_SPLIT_DIR and every output name must carry MAMI_SPLIT_TAG, otherwise this job would overwrite
# the committed random-split results and adapters. sciama_checks.py prereqs enforces that.
# Submit ONLY through the checked wrapper (after L1 and C2 are on main):
#   bash sciama/submit.sh S4 MODEL=2b TASK=singleclass SEED=1

source sciama/jobs/_env.sh
: "${MODEL:?set MODEL (2b|7b)}" "${TASK:?set TASK}" "${SEED:?set SEED}"
uv run python sciama/sciama_checks.py prereqs --job S4 --model "$MODEL" --task "$TASK" --seed "$SEED" --in-job
export MAMI_SPLIT_DIR="$REPO/data/splits/template_disjoint"
export MAMI_SPLIT_TAG="tdisjoint"

if [ "$MODEL" = "7b" ]; then
  MODEL_ID="Qwen/Qwen2-VL-7B-Instruct"; CLEAN="qwen2_vl_7b_instruct"
else
  MODEL_ID="Qwen/Qwen2-VL-2B-Instruct"; CLEAN="qwen2_vl_2b_instruct"
fi
uv run python scripts/train_vlm.py --model-id "$MODEL_ID" --task "$TASK" --epochs 3 --batch-size 4 \
  --lr 2e-5 --text-source provided --seed "$SEED" --quantize 4bit --device cuda
ADAPTER="results/models/lora_${CLEAN}_${TASK}_seed${SEED}_${MAMI_SPLIT_TAG}"
require_file "$ADAPTER/adapter_config.json"
for SPLIT in validation test; do
  uv run python scripts/benchmark_qwen2vl.py --model-id "$MODEL_ID" --task "$TASK" --split "$SPLIT" \
    --text-source provided --lora-path "$ADAPTER" --seed "$SEED" --quantize 4bit --device cuda
done
echo "=== DONE exp02 MODEL=$MODEL TASK=$TASK SEED=$SEED ==="
