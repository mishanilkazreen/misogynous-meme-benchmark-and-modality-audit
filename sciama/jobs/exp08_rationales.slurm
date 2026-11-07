#!/bin/bash
#SBATCH --job-name=exp08-rationales
#SBATCH --partition=gpu.q
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=results/sciama/logs/%x_%j.log
#SBATCH --error=results/sciama/logs/%x_%j.err

# EXP-08: rationale generation on a LARGER test sample than the original 30-sample set.
# Uses the first N test memes in dataset order (deterministic, not hand-picked). Measured cost is
# about 3 s per sample, so N=200 is roughly 10 min per model plus load time.
# Submit ONLY through the checked wrapper: bash sciama/submit.sh S3 N=200
# Outputs (new files; the 30-sample files are kept unchanged):
#   results/rationales/{qwen7b,qwen2b,llava7b}_test_n${N}.jsonl
# The Gemini counterpart for the same N is generated locally (API), see sciama/JOBS.md S3.

source sciama/jobs/_env.sh
N="${N:-200}"  # decided sample size (sciama/JOBS.md S3)
uv run python sciama/sciama_checks.py prereqs --job S3 --n "$N" --in-job
mkdir -p results/rationales

run() {  # run <model> <model-id> <adapter> <out-name>
  require_file "$3/adapter_config.json"
  uv run python scripts/explain_with_vlm.py --model "$1" --model-id "$2" --adapter "$3" \
    --split test --limit "$N" --use-ocr --device cuda --quantize 4bit \
    --output "results/rationales/$4_test_n${N}.jsonl"
}
run qwen Qwen/Qwen2-VL-7B-Instruct results/models/lora_qwen2_vl_7b_instruct_singleclass qwen7b
run qwen Qwen/Qwen2-VL-2B-Instruct results/models/lora_qwen2_vl_2b_instruct_singleclass qwen2b
run llava llava-hf/llava-1.5-7b-hf results/models/lora_llava_1.5_7b_hf_singleclass llava7b
echo "=== DONE exp08 rationales N=$N ==="
