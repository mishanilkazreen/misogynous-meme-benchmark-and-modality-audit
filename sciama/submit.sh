#!/bin/bash
# The ONLY way to queue an experiment job on SCIAMA. Runs every prerequisite check first and refuses to
# call sbatch if any check fails, then records the submission in results/sciama/job_ledger.tsv.
# Run on the SCIAMA login node from anywhere:
#   bash sciama/submit.sh S1-seed TASK=singleclass SEED=1
#   bash sciama/submit.sh S1-zeroshot
#   bash sciama/submit.sh S2 MODEL=2b
#   bash sciama/submit.sh S3 N=200
#   bash sciama/submit.sh S4 MODEL=2b TASK=singleclass SEED=1
#   bash sciama/submit.sh S5 MODEL_ID=<verified-hf-id> TASK=singleclass SEED=1
#   bash sciama/submit.sh S6
#   bash sciama/submit.sh S6-qwen-raw
#   bash sciama/submit.sh S6-qwen-def
set -euo pipefail
REPO="${PROJECT_ROOT:?set PROJECT_ROOT to the repo checkout}"
cd "$REPO"

JOB="${1:?usage: bash sciama/submit.sh <S1-seed|S1-zeroshot|S2|S3|S4|S5|S6|S6-qwen-raw|S6-qwen-def> [KEY=VALUE ...]}"
shift
EXPORTS=""
for kv in "$@"; do
  case "$kv" in
    TASK=*|SEED=*|MODEL=*|MODEL_ID=*|N=*) export "${kv?}"; EXPORTS="$EXPORTS,$kv" ;;
    *) echo "unknown argument: $kv" >&2; exit 2 ;;
  esac
done

CHECK=(--job "$JOB")
case "$JOB" in
  S1-seed)     SCRIPT=exp07_llava_seed.slurm;             CHECK+=(--task "${TASK:?}" --seed "${SEED:?}") ;;
  S1-zeroshot) SCRIPT=exp07_llava_zeroshot.slurm;         CHECK+=(--allow-overwrite) ;;
  S2)          SCRIPT=exp05_ocr_finetuned_qwen.slurm;     CHECK+=(--model "${MODEL:?}") ;;
  S3)          SCRIPT=exp08_rationales.slurm; N="${N:-200}"; export N; EXPORTS="$EXPORTS,N=$N"
               CHECK+=(--n "$N") ;;
  S4)          SCRIPT=exp02_template_disjoint_qwen.slurm; CHECK+=(--model "${MODEL:?}" --task "${TASK:?}" --seed "${SEED:?}") ;;
  S5)          SCRIPT=exp03_internvl_scaling.slurm;       CHECK+=(--model-id "${MODEL_ID:?}" --task "${TASK:?}" --seed "${SEED:?}") ;;
  S6)          SCRIPT=dis01_harmeme.slurm ;;
  S6-qwen-raw) SCRIPT=dis01_harmeme_qwen_raw.slurm ;;
  S6-qwen-def) SCRIPT=dis01_harmeme_qwen_def.slurm ;;
  *) echo "unknown job $JOB" >&2; exit 2 ;;
esac

module purge
module load system/sciama services/slurm anaconda3 uv git
export HF_HOME=${HOME}/.cache/huggingface
export KAGGLEHUB_CACHE=${HOME}/.cache/kagglehub
export HF_HUB_OFFLINE=1
conda activate "${CONDA_ENV:-vlm-moderation}"
mkdir -p results/sciama/logs results/sciama/manifests

echo "=== Prerequisite check for $JOB ${EXPORTS#,} ==="
if ! uv run python sciama/sciama_checks.py prereqs "${CHECK[@]}"; then
  echo "NOT SUBMITTED: fix every PREREQ FAIL above, then run this command again." >&2
  exit 1
fi

OUT=$(sbatch --export="ALL${EXPORTS}" "sciama/jobs/$SCRIPT")
echo "$OUT"
JOB_ID=$(echo "$OUT" | awk '{print $NF}')
printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$(date -Is)" "$JOB_ID" "sciama/jobs/$SCRIPT" "${EXPORTS#,}" \
  "$(git rev-parse --short HEAD)" "SUBMITTED" "$JOB" >> results/sciama/job_ledger.tsv
echo "Recorded job $JOB_ID in results/sciama/job_ledger.tsv"
