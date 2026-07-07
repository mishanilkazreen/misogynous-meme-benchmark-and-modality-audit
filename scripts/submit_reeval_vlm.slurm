#!/bin/bash
#SBATCH --job-name=mami-reeval-vlm
#SBATCH --partition=gpu.q
#SBATCH --nodelist=gpu10,gpu11,gpu12
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=runs/slurm_logs/reeval_vlm_%j.log
#SBATCH --error=runs/slurm_logs/reeval_vlm_%j.err

# Re-evaluate EXISTING Qwen2-VL adapters (no retraining) with the audit fixes:
#   * each split evaluated SEPARATELY (no validation+test conflation)
#   * seed-tagged output filenames (no multi-seed overwrite)
#   * corrected refusal / parse-status metric
# Also re-runs zero-shot Task A + Task B for both text sources.
#
# Input: MODEL in {2b, 7b}
# Example: sbatch --export=ALL,MODEL=2b scripts/submit_reeval_vlm.slurm

mkdir -p runs/slurm_logs

echo "=== MAMI VLM RE-EVAL (MODEL=$MODEL) ==="
echo "Date: $(date)"; echo "Node: $(hostname)"; nvidia-smi

module purge
module load system/sciama
module load services/slurm
module load anaconda3
module load uv

export HF_HOME=/mnt/lustre2/mres/ghahrem/.cache/huggingface
export KAGGLEHUB_CACHE=/mnt/lustre2/mres/ghahrem/.cache/kagglehub
conda activate /mnt/lustre2/mres/ghahrem/envs/vlm-moderation
cd /mnt/lustre2/mres/ghahrem/content-moderation

if [ "$MODEL" = "7b" ]; then
  MODEL_ID="Qwen/Qwen2-VL-7B-Instruct"; CLEAN="qwen2_vl_7b_instruct"
else
  MODEL_ID="Qwen/Qwen2-VL-2B-Instruct"; CLEAN="qwen2_vl_2b_instruct"
fi

BENCH="uv run python scripts/benchmark_qwen2vl.py --model-id $MODEL_ID --quantize 4bit --device cuda"

# 1. Zero-shot: Task A + Task B, provided + OCR text, each split separately.
for TASK in singleclass multiclass; do
  for TS in provided ocr; do
    for SPLIT in validation test; do
      echo "=== ZERO-SHOT $MODEL $TASK ts=$TS split=$SPLIT ==="
      $BENCH --task $TASK --text-source $TS --ocr-engine paddleocr --split $SPLIT
    done
  done
done

# 2. Fine-tuned adapters: Task A + Task B, provided text, per seed, per split.
for TASK in singleclass multiclass; do
  for SEED in 1 2 3; do
    ADAPTER="results/models/lora_${CLEAN}_${TASK}_seed${SEED}"
    if [ -d "$ADAPTER" ]; then
      for SPLIT in validation test; do
        echo "=== FINETUNED $MODEL $TASK seed=$SEED split=$SPLIT ==="
        $BENCH --task $TASK --text-source provided --lora-path "$ADAPTER" --seed $SEED --split $SPLIT
      done
    else
      echo "SKIP missing adapter: $ADAPTER"
    fi
  done
done

echo "=== RE-EVAL DONE (MODEL=$MODEL) ==="
