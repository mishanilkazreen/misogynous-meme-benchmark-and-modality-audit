#!/bin/bash
# Check progress of the Anna Gist tasks SLURM job
# Usage: bash scripts/check_progress.sh [JOB_ID]
#
# How to use:
#   1. ssh ghahrem@sciama.icg.port.ac.uk
#   2. cd /mnt/lustre2/mres/ghahrem/content-moderation
#   3. bash scripts/check_progress.sh
#
# Or with a specific job ID:
#   bash scripts/check_progress.sh 12345678

echo "=== SCIAMA Job Progress Checker ==="
echo ""

# Show running jobs
echo "--- Your SLURM Jobs ---"
squeue -u "$USER" -o "%.10i %.20j %.8T %.10M %.6D %R"
echo ""

# If a job ID is provided, show its log tail
if [ -n "$1" ]; then
    LOGFILE="runs/slurm_logs/anna_tasks_${1}.log"
    if [ -f "$LOGFILE" ]; then
        echo "--- Last 30 lines of log (Job $1) ---"
        tail -30 "$LOGFILE"
    else
        echo "Log file not found: $LOGFILE"
        echo "Available logs:"
        ls -lt runs/slurm_logs/anna_tasks_*.log 2>/dev/null | head -5
    fi
else
    # Show most recent log
    LATEST=$(ls -t runs/slurm_logs/anna_tasks_*.log 2>/dev/null | head -1)
    if [ -n "$LATEST" ]; then
        echo "--- Latest log: $LATEST ---"
        tail -30 "$LATEST"
    else
        echo "No anna_tasks logs found yet."
    fi
fi

echo ""
echo "--- Output File Check ---"

# Check which output files exist
TASKS_DONE=0
TASKS_TOTAL=8

check_file() {
    if [ -f "$1" ]; then
        echo "  [DONE] $1"
        TASKS_DONE=$((TASKS_DONE + 1))
    else
        echo "  [    ] $1"
    fi
}

echo "Issue #84 (Trad ML):"
check_file "auto_benchmark/results/model_results/mami_tabular_model/mami_tabular_model_evaluation.csv"
check_file "auto_benchmark/results/model_results/mami_tabular_model_multiclass/mami_tabular_model_multiclass_evaluation.csv"

echo "Issue #95 (CLIP ViT-B-32 zero-shot):"
check_file "results/validation/clip_validation_vit_b_32_quickgelu.json"
check_file "results/test/clip_test_vit_b_32_quickgelu.json"

echo "Issue #93 (Qwen2-VL-2B multiclass):"
check_file "results/validation/qwen2vl_validation_multiclass.json"
check_file "results/test/qwen2vl_test_multiclass.json"

echo "Issue #94 (CLIP ViT-B-32 fine-tuned multiclass):"
check_file "results/models/finetuned_clip_classification_multiclass_vit_b_32_quickgelu_ocr_paddleocr.pth"
check_file "results/models/finetuned_clip_classification_multiclass_vit_b_32_quickgelu.pth"

echo ""
echo "Progress: $TASKS_DONE / $TASKS_TOTAL files generated"
echo ""

# Simple progress bar
BAR_WIDTH=40
FILLED=$((TASKS_DONE * BAR_WIDTH / TASKS_TOTAL))
EMPTY=$((BAR_WIDTH - FILLED))
printf "["
printf "%0.s#" $(seq 1 $FILLED 2>/dev/null)
printf "%0.s-" $(seq 1 $EMPTY 2>/dev/null)
printf "] %d%%\n" $((TASKS_DONE * 100 / TASKS_TOTAL))
echo ""
