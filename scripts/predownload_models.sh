#!/bin/bash
# scripts/predownload_models.sh
# One-off helper to fetch all HuggingFace and CLIP models on the login node.
# This ensures offline SLURM compute jobs can run without network access.

# Force online mode for this script only.
export HF_HUB_OFFLINE=0
export HF_OFFLINE=0

# Configure cache directories to use the high-performance Lustre filesystem (50GB home quota bypass)
export HF_HOME=/mnt/lustre2/mres/ghahrem/.cache/huggingface
export KAGGLEHUB_CACHE=/mnt/lustre2/mres/ghahrem/.cache/kagglehub

set -e

echo "=== Pre-downloading CLIP Models ==="
uv run python -c "
import open_clip
print('Downloading ViT-B-32-quickgelu...')
open_clip.create_model_and_transforms('ViT-B-32-quickgelu', pretrained='openai')
print('Downloading ViT-L-14-quickgelu...')
open_clip.create_model_and_transforms('ViT-L-14-quickgelu', pretrained='openai')
"

echo "=== Pre-downloading Vision-Language Models (VLMs) ==="
uv run python -c "
from huggingface_hub import snapshot_download
models = [
    'Qwen/Qwen2-VL-2B-Instruct',
    'llava-hf/llava-1.5-7b-hf',
    'Qwen/Qwen2-VL-7B-Instruct'
]
for m in models:
    print(f'Downloading {m} snapshot...')
    snapshot_download(m)
"

echo "=== Pre-download Complete! ==="
