#!/bin/bash
#SBATCH --job-name=mami-rerun-tabular
#SBATCH --partition=gpu.q
#SBATCH --nodelist=gpu10,gpu11,gpu12
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=runs/slurm_logs/rerun_tabular_%j.log
#SBATCH --error=runs/slurm_logs/rerun_tabular_%j.err

mkdir -p runs/slurm_logs

echo "=== MAMI Rerun Tabular - SLURM Job Started ==="
echo "Date: $(date)"
echo "Node: $(hostname)"
nvidia-smi

module purge
module load system/sciama
module load services/slurm
module load anaconda3
module load uv

export HF_HOME=/mnt/lustre2/mres/ghahrem/.cache/huggingface
export KAGGLEHUB_CACHE=/mnt/lustre2/mres/ghahrem/.cache/kagglehub

conda activate /mnt/lustre2/mres/ghahrem/envs/vlm-moderation
cd /mnt/lustre2/mres/ghahrem/content-moderation

# Phase 0: Extract Embeddings (ViT-L-14 and ViT-B-32)
echo ">>> [Phase 0] Extracting embeddings..."
# Provided text
uv run python scripts/extract_embeddings.py --split train,validation,test --model ViT-L-14-quickgelu --text-source provided
uv run python scripts/extract_embeddings.py --split train,validation,test --model ViT-B-32-quickgelu --text-source provided
# PaddleOCR text
uv run python scripts/extract_embeddings.py --split train,validation,test --model ViT-L-14-quickgelu --text-source ocr --ocr-engine paddleocr
uv run python scripts/extract_embeddings.py --split train,validation,test --model ViT-B-32-quickgelu --text-source ocr --ocr-engine paddleocr
# Combined text
uv run python scripts/extract_embeddings.py --split train,validation,test --model ViT-L-14-quickgelu --text-source combined --ocr-engine paddleocr
uv run python scripts/extract_embeddings.py --split train,validation,test --model ViT-B-32-quickgelu --text-source combined --ocr-engine paddleocr

# Phase 1.1 & 1.2 & 2.1: Tabular Sweep (Task A & Task B)
echo ">>> [Phase 1 & 2] Running Tabular Sweeps..."
cd auto_benchmark

# Provided-text binary
uv run python scripts/run_benchmark.py --init config/init_config.yaml --model config/model/mami_tabular_model_provided.yaml
uv run python scripts/run_benchmark.py --init config/init_config.yaml --model config/model/mami_tabular_model_provided_test.yaml

# Combined-text binary
uv run python scripts/run_benchmark.py --init config/init_config.yaml --model config/model/mami_tabular_model_combined.yaml
uv run python scripts/run_benchmark.py --init config/init_config.yaml --model config/model/mami_tabular_model_combined_test.yaml

# PaddleOCR-text binary (mami_pre_extracted is OCR-only)
uv run python scripts/run_benchmark.py --init config/init_config.yaml --model config/model/mami_tabular_model.yaml
uv run python scripts/run_benchmark.py --init config/init_config.yaml --model config/model/mami_tabular_model_test.yaml

# Provided-text multiclass
uv run python scripts/run_benchmark.py --init config/init_config.yaml --model config/model/mami_tabular_model_multiclass_provided.yaml
uv run python scripts/run_benchmark.py --init config/init_config.yaml --model config/model/mami_tabular_model_multiclass_provided_test.yaml

# Combined-text multiclass
uv run python scripts/run_benchmark.py --init config/init_config.yaml --model config/model/mami_tabular_model_multiclass_combined.yaml
uv run python scripts/run_benchmark.py --init config/init_config.yaml --model config/model/mami_tabular_model_multiclass_combined_test.yaml

# PaddleOCR-text multiclass (mami_pre_extracted_multiclass is OCR-only)
uv run python scripts/run_benchmark.py --init config/init_config.yaml --model config/model/mami_tabular_model_multiclass.yaml
uv run python scripts/run_benchmark.py --init config/init_config.yaml --model config/model/mami_tabular_model_multiclass_test.yaml

cd ..

# Phase 3: Train XGBoost Head + XAI (SHAP & Concept Projection)
echo ">>> [Phase 3] Training XGBoost Head and running XAI..."
# Binary Provided XGBoost
uv run python scripts/train_classifier.py --model-name ViT-L-14-quickgelu --task singleclass --classifier xgboost --text-source provided --threshold-calibrate
# Multilabel Provided XGBoost
uv run python scripts/train_classifier.py --model-name ViT-L-14-quickgelu --task multiclass --classifier xgboost --text-source provided

# Run XAI scripts
cd auto_benchmark
uv run python scripts/run_shap_modality.py --model-path ../results/models/xgboost_singleclass_concat_vit_l_14_quickgelu.pkl --embeddings ../results/embeddings/test_vit_l_14_quickgelu.npz
cd ..
uv run python auto_benchmark/scripts/run_clip_concept_projection.py --model-path results/models/xgboost_singleclass_concat_vit_l_14_quickgelu.pkl --embeddings results/embeddings/test_vit_l_14_quickgelu.npz

echo "=== MAMI Rerun Tabular - SLURM Job Finished ==="
