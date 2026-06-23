#!/bin/bash
#SBATCH --job-name=issue-89-xai
#SBATCH --partition=gpu.q
#SBATCH --nodelist=gpu10,gpu11,gpu12
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=runs/slurm_logs/issue_89_xai_%j.log
#SBATCH --error=runs/slurm_logs/issue_89_xai_%j.err

# Issue #89 (Gist tasks 12-13): XAI for the XGBoost Fusion classifier.
#   Part 1 - Modality-Level SHAP attribution (run_shap_modality.py)
#   Part 2 - CLIP Concept Activation Vectors  (run_clip_cav.py)
#
# Both are CPU-light but depend on gitignored artifacts that live only on
# SCIAMA: the trained .pkl model, the ViT-L-14 + PaddleOCR test embeddings, and
# the cached open_clip ViT-L-14-quickgelu/openai weights (offline-safe; already
# downloaded during embedding extraction). Re-running SHAP is idempotent.

mkdir -p runs/slurm_logs

echo "=== Issue #89 XAI - SLURM Job Started ==="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo ""

module purge
module load system/sciama
module load services/slurm
module load anaconda3
module load uv

export HF_HOME=/mnt/lustre2/mres/ghahrem/.cache/huggingface
export KAGGLEHUB_CACHE=/mnt/lustre2/mres/ghahrem/.cache/kagglehub
# open_clip weights are pre-cached; force offline so a missing cache fails loudly
# instead of hanging on a network call the compute nodes cannot make.
export HF_HUB_OFFLINE=1

conda activate /mnt/lustre2/mres/ghahrem/envs/vlm-moderation
cd /mnt/lustre2/mres/ghahrem/content-moderation

echo "Environment activated. Python: $(python --version)"
echo ""

echo ">>> [#89 Part 1] Modality-Level SHAP attribution..."
uv run python auto_benchmark/scripts/run_shap_modality.py || echo "FAILED: #89 SHAP"
echo ""

echo ">>> [#89 Part 2] CLIP Concept Activation Vectors..."
uv run python auto_benchmark/scripts/run_clip_cav.py || echo "FAILED: #89 CAV"
echo ""

echo "Output files:"
ls -la results/shap_modality_importance.csv results/figures/shap_modality_importance.png \
    results/cav_concept_similarity.csv results/cav_global_concept_presence.csv 2>&1

echo "=== Issue #89 XAI Complete ==="
echo "Date: $(date)"
echo "=== DONE ==="
