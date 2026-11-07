#!/bin/bash
# scripts/predownload_models.sh
# One-off helper to fetch all HuggingFace and CLIP models on the login node.
# This ensures offline SLURM compute jobs can run without network access.

# Force online mode for this script only.
export HF_HUB_OFFLINE=0
export HF_OFFLINE=0

# Configure cache directories (point these at a large scratch filesystem if the home quota is small)
export HF_HOME=${HOME}/.cache/huggingface
export KAGGLEHUB_CACHE=${HOME}/.cache/kagglehub

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
    'Qwen/Qwen2-VL-7B-Instruct',
    # InternVL (EXP-03 / SCIAMA job S5, code change C3). Native-transformers
    # '-hf' repositories only (loaded via InternVLForConditionalGeneration,
    # no trust_remote_code). Each id was confirmed to resolve on the HF Hub.
    # InternVL3 has no 4B '-hf' port and InternVL2.5 has no '-hf' ports at all,
    # so those sizes/versions are intentionally omitted.
    'OpenGVLab/InternVL3-1B-hf',
    'OpenGVLab/InternVL3-2B-hf',
    'OpenGVLab/InternVL3-8B-hf',
    'OpenGVLab/InternVL3_5-1B-hf',
    'OpenGVLab/InternVL3_5-2B-hf',
    'OpenGVLab/InternVL3_5-4B-hf',
    'OpenGVLab/InternVL3_5-8B-hf'
]
for m in models:
    print(f'Downloading {m} snapshot...')
    snapshot_download(m)
"

echo "=== Pre-download Complete! ==="
