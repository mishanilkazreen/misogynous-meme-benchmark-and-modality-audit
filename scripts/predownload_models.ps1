# predownload_models.ps1
# One-off helper to fetch HuggingFace models that are NOT yet in the local cache,
# so the offline runs (run_all_experiments.ps1 / rerun_leftover_models.ps1) succeed.
# Run this once with a working internet connection.

$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $true

# Force online mode for this script only (override any inherited offline flags).
$env:HF_HUB_OFFLINE = 0
$env:HF_OFFLINE = 0

Write-Host "=== Pre-downloading missing models ===" -ForegroundColor Green

Write-Host "[1/2] open_clip ViT-B-32-quickgelu (openai)..." -ForegroundColor Yellow
uv run python -c "import open_clip; open_clip.create_model_and_transforms('ViT-B-32-quickgelu', pretrained='openai'); print('ViT-B-32-quickgelu cached')"

Write-Host "[2/2] Qwen/Qwen2-VL-2B-Instruct..." -ForegroundColor Yellow
uv run python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2-VL-2B-Instruct'); print('Qwen2-VL-2B-Instruct cached')"

Write-Host "=== Pre-download complete. Offline runs will now find these in cache. ===" -ForegroundColor Green
