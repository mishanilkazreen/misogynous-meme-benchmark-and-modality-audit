#!/bin/bash
#SBATCH --job-name=issue-89-xai
#SBATCH --partition=gpu.q
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=runs/slurm_logs/issue_89_xai_%j.log
#SBATCH --error=runs/slurm_logs/issue_89_xai_%j.err

# XAI for the XGBoost Fusion classifier.
#   Part 1 - Modality-Level SHAP attribution (run_shap_modality.py)
#   Part 2 - CLIP Concept-Embedding Projection (run_clip_concept_projection.py)
#
# Both are CPU-light but depend on gitignored artifacts that live only on
# SCIAMA: the trained .pkl model, the ViT-L-14 + PaddleOCR test embeddings, and
# the cached open_clip ViT-L-14-quickgelu/openai weights (offline-safe; already
# downloaded during embedding extraction). Re-running SHAP is idempotent.

mkdir -p runs/slurm_logs

echo "=== XAI - SLURM Job Started ==="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo ""

module purge
module load system/sciama
module load services/slurm
module load anaconda3
module load uv

export HF_HOME=${HOME}/.cache/huggingface
export KAGGLEHUB_CACHE=${HOME}/.cache/kagglehub
# open_clip weights are pre-cached; force offline so a missing cache fails loudly
# instead of hanging on a network call the compute nodes cannot make.
export HF_HUB_OFFLINE=1

conda activate "${CONDA_ENV:-vlm-moderation}"
cd "${PROJECT_ROOT:?set PROJECT_ROOT to the repo checkout}"

echo "Environment activated. Python: $(python --version)"
echo ""

echo ">>> [Part 1] Modality-Level SHAP attribution..."
uv run python auto_benchmark/scripts/run_shap_modality.py || echo "FAILED: SHAP"
echo ""

echo ">>> [Part 2] CLIP Concept-Embedding Projection..."
uv run python auto_benchmark/scripts/run_clip_concept_projection.py || echo "FAILED: concept projection"
echo ""

echo "Output files:"
ls -la results/shap_modality_importance.csv results/figures/shap_modality_importance.png \
    results/concept_projection_similarity.csv \
    results/concept_projection_global_presence.csv 2>&1

echo "=== XAI Complete ==="
echo "Date: $(date)"
echo "=== DONE ==="
