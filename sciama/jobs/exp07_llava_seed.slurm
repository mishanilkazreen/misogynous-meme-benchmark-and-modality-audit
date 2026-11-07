#!/bin/bash
#SBATCH --job-name=exp07-llava-seed
#SBATCH --partition=gpu.q
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=results/sciama/logs/%x_%j.log
#SBATCH --error=results/sciama/logs/%x_%j.err

# EXP-07 (and EXP-01 LLaVA rows): train one LLaVA-1.5-7B QLoRA adapter for one seed and one task,
# then evaluate it on validation and test SEPARATELY with a seed-tagged output file.
# Submit ONLY through the checked wrapper (6 jobs, 3 seeds x 2 tasks), see sciama/JOBS.md S1:
#   bash sciama/submit.sh S1-seed TASK=singleclass SEED=1
# Outputs:
#   results/models/lora_llava_1.5_7b_hf_${TASK}_seed${SEED}/   (adapter, NOT in git; see manifest step)
#   results/{validation,test}/llava_<split>_llava_1_5_7b_hf[_multiclass]_finetuned_seed${SEED}.json

source sciama/jobs/_env.sh
: "${SEED:?set SEED}" "${TASK:?set TASK (singleclass|multiclass)}"
uv run python sciama/sciama_checks.py prereqs --job S1-seed --task "$TASK" --seed "$SEED" --in-job

MODEL_ID="llava-hf/llava-1.5-7b-hf"
ADAPTER="results/models/lora_llava_1.5_7b_hf_${TASK}_seed${SEED}"

uv run python scripts/train_vlm.py --model-id "$MODEL_ID" --task "$TASK" --epochs 3 --batch-size 4 \
  --lr 2e-5 --text-source provided --seed "$SEED" --quantize 4bit --device cuda
require_file "$ADAPTER/adapter_config.json"

for SPLIT in validation test; do
  uv run python scripts/benchmark_llava.py --model-id "$MODEL_ID" --task "$TASK" --split "$SPLIT" \
    --text-source provided --lora-path "$ADAPTER" --seed "$SEED" --quantize 4bit --device cuda
done
echo "=== DONE exp07 llava TASK=$TASK SEED=$SEED ==="
