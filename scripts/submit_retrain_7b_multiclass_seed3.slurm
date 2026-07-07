#!/bin/bash
#SBATCH --job-name=mami-7b-mc-s3
#SBATCH --partition=gpu.q
#SBATCH --nodelist=gpu10,gpu11,gpu12
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=runs/slurm_logs/retrain_7b_mc_s3_%j.log
#SBATCH --error=runs/slurm_logs/retrain_7b_mc_s3_%j.err

# Fill the one multi-seed gap left by the cancelled jobs: Qwen2-VL-7B Task B
# (multiclass) seed 3. Trains the adapter then evaluates it per split with the
# audit-fixed benchmark (separate splits, seed-tagged output).

mkdir -p runs/slurm_logs
echo "=== RETRAIN 7B multiclass seed3 ==="; echo "Date: $(date)"; echo "Node: $(hostname)"; nvidia-smi

module purge
module load system/sciama
module load services/slurm
module load anaconda3
module load uv
export HF_HOME=/mnt/lustre2/mres/ghahrem/.cache/huggingface
export KAGGLEHUB_CACHE=/mnt/lustre2/mres/ghahrem/.cache/kagglehub
conda activate /mnt/lustre2/mres/ghahrem/envs/vlm-moderation
cd /mnt/lustre2/mres/ghahrem/content-moderation

MODEL="Qwen/Qwen2-VL-7B-Instruct"
CLEAN="qwen2_vl_7b_instruct"

echo "=== Training ==="
uv run python scripts/train_vlm.py --model-id $MODEL --task multiclass --epochs 3 \
  --batch-size 4 --lr 2e-5 --text-source provided --seed 3 --device cuda

echo "=== Evaluating (per split) ==="
for SPLIT in validation test; do
  uv run python scripts/benchmark_qwen2vl.py --model-id $MODEL --quantize 4bit --device cuda \
    --task multiclass --text-source provided \
    --lora-path results/models/lora_${CLEAN}_multiclass_seed3 --seed 3 --split $SPLIT
done

echo "=== DONE 7B multiclass seed3 ==="
