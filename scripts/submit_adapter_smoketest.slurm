#!/bin/bash
#SBATCH --job-name=adapter-smoketest
#SBATCH --partition=gpu.q
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=runs/slurm_logs/adapter_smoketest_%j.log
#SBATCH --error=runs/slurm_logs/adapter_smoketest_%j.err

# Confirms that the QLoRA adapter actually attaches before the full rationale
# runs are submitted. Two samples only; this is a wiring check, not an experiment.

mkdir -p runs/slurm_logs /tmp/adapter_smoketest

module purge
module load system/sciama
module load services/slurm
module load anaconda3
module load uv
module load cuda/25.1

export HF_HOME=${HOME}/.cache/huggingface
export KAGGLEHUB_CACHE=${HOME}/.cache/kagglehub

cd "${PROJECT_ROOT:?set PROJECT_ROOT to the repo checkout}"

echo "=== A. base model, no adapter (control) ==="
uv run python scripts/explain_with_vlm.py \
    --model qwen \
    --model-id Qwen/Qwen2-VL-2B-Instruct \
    --split test --limit 2 --use-ocr --device cuda --quantize 4bit \
    --output /tmp/adapter_smoketest/base.jsonl

echo ""
echo "=== B. same base WITH the singleclass adapter ==="
uv run python scripts/explain_with_vlm.py \
    --model qwen \
    --model-id Qwen/Qwen2-VL-2B-Instruct \
    --adapter results/models/lora_qwen2_vl_2b_instruct_singleclass \
    --split test --limit 2 --use-ocr --device cuda --quantize 4bit \
    --output /tmp/adapter_smoketest/adapted.jsonl

echo ""
echo "=== C. deliberate mismatch, MUST fail loudly ==="
uv run python scripts/explain_with_vlm.py \
    --model qwen \
    --model-id Qwen/Qwen2-VL-2B-Instruct \
    --adapter results/models/lora_qwen2_vl_7b_instruct_singleclass \
    --split test --limit 1 --use-ocr --device cuda --quantize 4bit \
    --output /tmp/adapter_smoketest/mismatch.jsonl \
    && echo "MISMATCH GUARD FAILED: command exited 0" \
    || echo "MISMATCH GUARD OK: command refused as expected"

echo ""
echo "=== D. comparison ==="
uv run python - <<'PY'
import json

def load(p):
    try:
        with open(p, encoding="utf-8") as fh:
            return [json.loads(line) for line in fh if line.strip()]
    except FileNotFoundError:
        return []

base = load("/tmp/adapter_smoketest/base.jsonl")
adapted = load("/tmp/adapter_smoketest/adapted.jsonl")

print("base    records:", len(base), "model:", {r.get("model") for r in base})
print("adapted records:", len(adapted), "model:", {r.get("model") for r in adapted})

if base and adapted:
    identical = all(
        b["explanation"] == a["explanation"] for b, a in zip(base, adapted)
    )
    print("explanations identical:", identical)
    print("VERDICT:", "ADAPTER NOT ACTIVE" if identical else "ADAPTER ACTIVE")
    for b, a in zip(base, adapted):
        print("-" * 60)
        print("id", b["image_id"])
        print("  base   :", str(b["explanation"])[:150])
        print("  adapted:", str(a["explanation"])[:150])
PY

echo "=== smoketest finished ==="
