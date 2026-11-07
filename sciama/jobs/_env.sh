#!/bin/bash
# Common SCIAMA environment for every job in sciama/jobs/. Sourced, never run directly.
# See sciama/README.md for the full protocol.

# -e: stop at the first failing command, so a failed step never produces a "DONE" log line.
# (-u is not used: the module and conda shell functions reference unset variables.)
set -eo pipefail

module purge
module load system/sciama services/slurm anaconda3 uv cuda/25.1

export HF_HOME=${HOME}/.cache/huggingface
export KAGGLEHUB_CACHE=${HOME}/.cache/kagglehub
# Compute nodes have no internet: fail fast instead of hanging on a download.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

conda activate "${CONDA_ENV:-vlm-moderation}"

REPO="${PROJECT_ROOT:?set PROJECT_ROOT to the repo checkout}"
cd "$REPO"
mkdir -p results/sciama/logs results/sciama/manifests

echo "=== JOB ${SLURM_JOB_NAME:-local} (${SLURM_JOB_ID:-none}) on $(hostname) at $(date -Is) ==="
echo "git commit: $(git rev-parse HEAD 2>/dev/null || echo unknown)"
nvidia-smi || true

# Fail if the dataset cache is missing (it must be downloaded on the login node first).
if [ ! -d "$KAGGLEHUB_CACHE/datasets/chukwuebukaanulunko" ]; then
  echo "FATAL: MAMI dataset not found in $KAGGLEHUB_CACHE (run the login-node setup in sciama/README.md)" >&2
  exit 1
fi

# require_file PATH: abort the job if a required input is missing.
require_file() {
  if [ ! -e "$1" ]; then
    echo "FATAL: required input missing: $1" >&2
    exit 1
  fi
}
