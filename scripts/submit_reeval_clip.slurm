#!/bin/bash
#SBATCH --job-name=mami-reeval-clip
#SBATCH --partition=gpu.q
#SBATCH --nodelist=gpu10,gpu11,gpu12
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=runs/slurm_logs/reeval_clip_%j.log
#SBATCH --error=runs/slurm_logs/reeval_clip_%j.err

# Re-evaluate EXISTING CLIP checkpoints (no retraining) with each split
# evaluated SEPARATELY, so validation and test metrics are no longer conflated
# into one 2000-sample file. Seeds are already encoded in the checkpoint name.

mkdir -p runs/slurm_logs
echo "=== MAMI CLIP RE-EVAL ==="; echo "Date: $(date)"; echo "Node: $(hostname)"

module purge
module load system/sciama
module load services/slurm
module load anaconda3
module load uv
export HF_HOME=/mnt/lustre2/mres/ghahrem/.cache/huggingface
conda activate /mnt/lustre2/mres/ghahrem/envs/vlm-moderation
cd /mnt/lustre2/mres/ghahrem/content-moderation

BENCH="uv run python scripts/benchmark_clip.py --device cuda --text-source provided"

# Zero-shot baselines (both backbones, both tasks, each split).
for MODELNAME in ViT-L-14-quickgelu ViT-B-32-quickgelu; do
  for TASK in singleclass multiclass; do
    for SPLIT in validation test; do
      echo "=== CLIP ZERO-SHOT $MODELNAME $TASK $SPLIT ==="
      $BENCH --model-name $MODELNAME --task $TASK --split $SPLIT
    done
  done
done

# Fine-tuned heads (both backbones, both tasks, 3 seeds, each split).
for BACKBONE in vit_l_14_quickgelu vit_b_32_quickgelu; do
  for TASK in singleclass multiclass; do
    MODELNAME=$([ "$BACKBONE" = "vit_l_14_quickgelu" ] && echo "ViT-L-14-quickgelu" || echo "ViT-B-32-quickgelu")
    for SEED in 1 2 3; do
      CKPT="results/models/finetuned_clip_classification_${TASK}_${BACKBONE}_seed${SEED}.pth"
      if [ -f "$CKPT" ]; then
        for SPLIT in validation test; do
          echo "=== CLIP FT $BACKBONE $TASK seed=$SEED $SPLIT ==="
          $BENCH --model-name $MODELNAME --task $TASK --split $SPLIT --model-path "$CKPT"
        done
      else
        echo "SKIP missing checkpoint: $CKPT"
      fi
    done
  done
done

echo "=== CLIP RE-EVAL DONE ==="
