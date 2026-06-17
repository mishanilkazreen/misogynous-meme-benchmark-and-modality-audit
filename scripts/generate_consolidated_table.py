"""
Script to parse all validation and test result JSON files under results/
and print a consolidated comparison table of accuracy and F1 scores.
"""

from __future__ import annotations

import json
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
VAL_DIR = RESULTS_DIR / "validation"
TEST_DIR = RESULTS_DIR / "test"

models_info = [
    {
        "name": "Gemini 1.5 Pro (Zero-Shot)",
        "val_a": VAL_DIR / "gemini_validation.json",
        "val_b": VAL_DIR / "gemini_validation_multiclass.json",
        "test_a": TEST_DIR / "gemini_test.json",  # placeholder if evaluated
        "test_b": TEST_DIR / "gemini_test_multiclass.json",
    },
    {
        "name": "XGBoost Fusion (ViT-L-14 + PaddleOCR)",
        "val_a": VAL_DIR / "xgboost_validation_xgboost_singleclass.json",
        "val_b": VAL_DIR / "xgboost_validation_xgboost_multiclass.json",
        "test_a": TEST_DIR / "xgboost_test_xgboost_singleclass.json",
        "test_b": TEST_DIR / "xgboost_test_xgboost_multiclass.json",
    },
    {
        "name": "CLIP ViT-B-32 (Fine-Tuned)",
        "val_a": VAL_DIR / "clip_validation_finetuned_singleclass_vit_b_32_quickgelu.json",
        "val_b": None,
        "test_a": TEST_DIR / "clip_test_finetuned_singleclass_vit_b_32_quickgelu.json",
        "test_b": None,
    },
    {
        "name": "CLIP ViT-L-14 (Fine-Tuned)",
        "val_a": VAL_DIR / "clip_validation_finetuned_singleclass_vit_l_14_quickgelu.json",
        "val_b": VAL_DIR / "clip_validation_finetuned_multiclass_vit_l_14_quickgelu.json",
        "test_a": TEST_DIR / "clip_test_finetuned_singleclass_vit_l_14_quickgelu.json",
        "test_b": TEST_DIR / "clip_test_finetuned_multiclass_vit_l_14_quickgelu.json",
    },
    {
        "name": "CLIP ViT-L-14 (Zero-Shot)",
        "val_a": VAL_DIR / "clip_validation.json",
        "val_b": VAL_DIR / "clip_validation_multiclass.json",
        "test_a": TEST_DIR / "clip_test.json",
        "test_b": TEST_DIR / "clip_test_multiclass.json",
    },
    {
        "name": "Qwen2-VL-7B (QLoRA Fine-Tuned)",
        "val_a": VAL_DIR / "qwen2vl_validation_qwen2_vl_7b_instruct_finetuned.json",
        "val_b": VAL_DIR / "qwen2vl_validation_qwen2_vl_7b_instruct_multiclass_finetuned.json",
        "test_a": TEST_DIR / "qwen2vl_test_qwen2_vl_7b_instruct_finetuned.json",
        "test_b": TEST_DIR / "qwen2vl_test_qwen2_vl_7b_instruct_multiclass_finetuned.json",
    },
    {
        "name": "Qwen2-VL-7B (Zero-Shot)",
        "val_a": VAL_DIR / "qwen2vl_validation_qwen2_vl_7b_instruct.json",
        "val_b": VAL_DIR / "qwen2vl_validation_qwen2_vl_7b_instruct_multiclass.json",
        "test_a": TEST_DIR / "qwen2vl_test_qwen2_vl_7b_instruct.json",
        "test_b": TEST_DIR / "qwen2vl_test_qwen2_vl_7b_instruct_multiclass.json",
    },
    {
        "name": "Qwen2-VL-2B (QLoRA Fine-Tuned)",
        "val_a": VAL_DIR / "qwen2vl_validation_qwen2_vl_2b_instruct_finetuned.json",
        "val_b": VAL_DIR / "qwen2vl_validation_qwen2_vl_2b_instruct_multiclass_finetuned.json",
        "test_a": TEST_DIR / "qwen2vl_test_qwen2_vl_2b_instruct_finetuned.json",
        "test_b": TEST_DIR / "qwen2vl_test_qwen2_vl_2b_instruct_multiclass_finetuned.json",
    },
    {
        "name": "Qwen2-VL-2B (Zero-Shot)",
        "val_a": VAL_DIR / "qwen2vl_validation.json",
        "val_b": VAL_DIR / "qwen2vl_validation_multiclass.json",
        "test_a": TEST_DIR / "qwen2vl_test.json",
        "test_b": TEST_DIR / "qwen2vl_test_multiclass.json",
    },
    {
        "name": "LLaVA-1.5-7B (QLoRA Fine-Tuned)",
        "val_a": VAL_DIR / "llava_validation_llava_1_5_7b_hf_finetuned.json",
        "val_b": VAL_DIR / "llava_validation_llava_1_5_7b_hf_multiclass_finetuned.json",
        "test_a": TEST_DIR / "llava_test_llava_1_5_7b_hf_finetuned.json",
        "test_b": TEST_DIR / "llava_test_llava_1_5_7b_hf_multiclass_finetuned.json",
    },
    {
        "name": "LLaVA-1.5-7B (Zero-Shot)",
        "val_a": VAL_DIR / "llava_test_validation.json",  # original combined zero-shot file
        "val_b": VAL_DIR / "llava_test_validation_multiclass.json",
        "test_a": TEST_DIR / "llava_test_llava_1_5_7b_hf.json",
        "test_b": TEST_DIR / "llava_test_llava_1_5_7b_hf_multiclass.json",
    },
    {
        "name": "VisualBERT (Zero-Shot)",
        "val_a": VAL_DIR / "visualbert_validation.json",
        "val_b": VAL_DIR / "visualbert_validation_multiclass.json",
        "test_a": TEST_DIR / "visualbert_test.json",
        "test_b": TEST_DIR / "visualbert_test_multiclass.json",
    },
]


def load_metrics(filepath: Path | None) -> tuple[float | None, float | None]:
    if not filepath:
        return None, None

    if not filepath.exists():
        return None, None
    try:
        data = json.loads(filepath.read_text(encoding="utf-8"))
        if not isinstance(data, list) or len(data) == 0:
            return None, None

        # If the file contains multiple items (e.g. preprocessing filters), use 'none' filter
        row = data[0]
        for item in data:
            if item.get("filter") == "none":
                row = item
                break

        acc = row.get("exact_match_accuracy", row.get("accuracy", None))
        f1 = row.get("f1", row.get("macro_f1", None))

        acc_val = float(acc) if acc is not None else None
        f1_val = float(f1) if f1 is not None else None
        return acc_val, f1_val
    except Exception:
        return None, None


def main() -> None:
    print("# MAMI 2022 Misogyny Classification - Consolidated Validation & Test Results\n")

    headers = [
        "Model Name",
        "Val A Acc",
        "Val A F1",
        "Test A Acc",
        "Test A F1",
        "Val B EM",
        "Val B F1",
        "Test B EM",
        "Test B F1",
    ]

    col_widths = [38, 10, 10, 10, 10, 10, 10, 10, 10]

    # Print Markdown Headers
    header_str = " | ".join(f"{h:<{w}}" for h, w in zip(headers, col_widths, strict=False))
    sep_str = " | ".join("-" * w for w in col_widths)
    print(f"| {header_str} |")
    print(f"| {sep_str} |")

    for m in models_info:
        val_a_acc, val_a_f1 = load_metrics(m["val_a"])
        test_a_acc, test_a_f1 = load_metrics(m["test_a"])
        val_b_em, val_b_f1 = load_metrics(m["val_b"])
        test_b_em, test_b_f1 = load_metrics(m["test_b"])

        # Format strings
        v_a_acc = f"{val_a_acc * 100:.1f}%" if val_a_acc is not None else "N/A"
        v_a_f1 = f"{val_a_f1:.4f}" if val_a_f1 is not None else "N/A"
        t_a_acc = f"{test_a_acc * 100:.1f}%" if test_a_acc is not None else "N/A"
        t_a_f1 = f"{test_a_f1:.4f}" if test_a_f1 is not None else "N/A"

        v_b_em = f"{val_b_em * 100:.1f}%" if val_b_em is not None else "N/A"
        v_b_f1 = f"{val_b_f1:.4f}" if val_b_f1 is not None else "N/A"
        t_b_em = f"{test_b_em * 100:.1f}%" if test_b_em is not None else "N/A"
        t_b_f1 = f"{test_b_f1:.4f}" if test_b_f1 is not None else "N/A"

        row_values = [m["name"], v_a_acc, v_a_f1, t_a_acc, t_a_f1, v_b_em, v_b_f1, t_b_em, t_b_f1]
        row_str = " | ".join(
            f"{val!s:<{w}}" for val, w in zip(row_values, col_widths, strict=False)
        )
        print(f"| {row_str} |")


if __name__ == "__main__":
    main()
