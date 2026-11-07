"""Recompute paper metrics using official MAMI scorer logic and generate verified tables.

Ensures:
1. Score A is official macro-F1 across binary classes 0 and 1.
2. Accuracy is (TP + TN) / N.
3. Macro-F1 <= Accuracy is asserted for all binary models (Jensen's inequality).
4. Score B is official MAMI support-weighted macro-F1 across sub-types.
5. Exact Match (EM) is the ratio of samples matching all 4 sub-types.
6. Decision thresholds are reported explicitly.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from sklearn.metrics import accuracy_score, f1_score

from models.vlm.metrics_multilabel import compute_multilabel_metrics

ROOT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT_DIR / "results"
VAL_DIR = RESULTS_DIR / "validation"
TEST_DIR = RESULTS_DIR / "test"
AB_DIR = ROOT_DIR / "auto_benchmark" / "results" / "model_results"

SUBTYPE_LABELS = ["shaming", "stereotype", "objectification", "violence"]


def evaluate_binary_predictions(actual: list[int], preds: list[int]) -> dict[str, float]:
    acc = accuracy_score(actual, preds)
    macro_f1 = f1_score(actual, preds, average="macro", zero_division=0.0)
    pos_f1 = f1_score(actual, preds, pos_label=1, zero_division=0.0)
    neg_f1 = f1_score(actual, preds, pos_label=0, zero_division=0.0)

    # Jensen's inequality sanity check (within floating point precision)
    if macro_f1 > acc + 1e-6:
        raise ValueError(f"Jensen's inequality violated: Macro-F1 {macro_f1:.6f} > Acc {acc:.6f}")

    return {
        "acc": acc,
        "score_a": macro_f1,
        "pos_f1": pos_f1,
        "neg_f1": neg_f1,
    }


def evaluate_json_binary(path: Path | None) -> dict[str, float] | None:
    if not path or not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, list) and len(data) > 0 and "sample_predictions" in data[0]:
        preds_list = data[0]["sample_predictions"]
        actual = [int(p["ground_truth"]) for p in preds_list]
        preds = [int(p["prediction"]) for p in preds_list]
        return evaluate_binary_predictions(actual, preds)
    return None


def evaluate_json_multiclass(path: Path | None) -> dict[str, Any] | None:
    if not path or not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, list) and len(data) > 0 and "sample_predictions" in data[0]:
        preds_list = data[0]["sample_predictions"]
        gts = [p["ground_truth"] for p in preds_list]
        preds = [p["prediction"] for p in preds_list]
        metrics = compute_multilabel_metrics(preds, gts, SUBTYPE_LABELS)
        return metrics
    return None


def parse_tabular_binary_csv(path: Path) -> tuple[list[int], dict[str, list[int]]]:
    with open(path) as f:
        reader = csv.reader(f)
        header = next(reader)
        model_names = header[1:]
        actual = []
        models: dict[str, list[int]] = {m: [] for m in model_names}
        for row in reader:
            if not row:
                continue
            actual.append(int(float(row[0])))
            for m, val in zip(model_names, row[1:], strict=False):
                models[m].append(int(float(val)))
    return actual, models


def parse_tabular_multiclass_csv(
    path: Path,
) -> tuple[list[dict[str, int]], dict[str, list[dict[str, int]]]]:
    with open(path) as f:
        reader = csv.reader(f)
        header = next(reader)
        model_names = header[1:]
        actual = []
        models: dict[str, list[dict[str, int]]] = {m: [] for m in model_names}
        for row in reader:
            if not row:
                continue
            gt_vec = [int(float(x.strip())) for x in row[0].strip().strip('"').split(",")]
            actual.append({lbl: gt_vec[i] for i, lbl in enumerate(SUBTYPE_LABELS)})
            for m, cell in zip(model_names, row[1:], strict=False):
                pr_vec = [int(float(x.strip())) for x in cell.strip().strip('"').split(",")]
                models[m].append({lbl: pr_vec[i] for i, lbl in enumerate(SUBTYPE_LABELS)})
    return actual, models


def main():
    print("=== Recomputing and Auditing All MAMI Metrics ===")

    # 1. Tabular Task A & B
    tab_val_a_csv = (
        AB_DIR / "mami_tabular_model_provided" / "mami_tabular_model_provided_predictions.csv"
    )
    tab_test_a_csv = (
        AB_DIR
        / "mami_tabular_model_provided_test"
        / "mami_tabular_model_provided_test_predictions.csv"
    )
    tab_val_b_csv = (
        AB_DIR
        / "mami_tabular_model_multiclass_provided"
        / "mami_tabular_model_multiclass_provided_predictions.csv"
    )
    tab_test_b_csv = (
        AB_DIR
        / "mami_tabular_model_multiclass_provided_test"
        / "mami_tabular_model_multiclass_provided_test_predictions.csv"
    )

    val_a_act, val_a_models = parse_tabular_binary_csv(tab_val_a_csv)
    test_a_act, test_a_models = parse_tabular_binary_csv(tab_test_a_csv)
    val_b_act, val_b_models = parse_tabular_multiclass_csv(tab_val_b_csv)
    test_b_act, test_b_models = parse_tabular_multiclass_csv(tab_test_b_csv)

    all_task_a = []
    all_task_b = []

    # Non-tabular models mapping
    vlm_models = [
        {
            "name": r"\texttt{gemini-3.1-flash-lite} (Zero-Shot)",
            "origin": "Ours (API)",
            "threshold": "N/A",
            "val_a": VAL_DIR / "gemini_validation.json",
            "test_a": TEST_DIR / "gemini_test.json",
            "val_b": VAL_DIR / "gemini_validation_multiclass.json",
            "test_b": TEST_DIR / "gemini_test_multiclass.json",
        },
        {
            "name": "Qwen2-VL-7B (Zero-Shot)",
            "origin": "Ours (VLM)",
            "threshold": "N/A",
            "val_a": VAL_DIR / "qwen2vl_validation_qwen2_vl_7b_instruct.json",
            "test_a": TEST_DIR / "qwen2vl_test_qwen2_vl_7b_instruct.json",
            "val_b": VAL_DIR / "qwen2vl_validation_qwen2_vl_7b_instruct_multiclass.json",
            "test_b": TEST_DIR / "qwen2vl_test_qwen2_vl_7b_instruct_multiclass.json",
        },
        {
            "name": "Qwen2-VL-7B QLoRA",
            "origin": "Ours (VLM)",
            "threshold": "N/A",
            "val_a": VAL_DIR / "qwen2vl_validation_qwen2_vl_7b_instruct_finetuned.json",
            "test_a": TEST_DIR / "qwen2vl_test_qwen2_vl_7b_instruct_finetuned.json",
            "val_b": VAL_DIR / "qwen2vl_validation_qwen2_vl_7b_instruct_multiclass_finetuned.json",
            "test_b": TEST_DIR / "qwen2vl_test_qwen2_vl_7b_instruct_multiclass_finetuned.json",
        },
        {
            "name": "Qwen2-VL-2B (Zero-Shot)",
            "origin": "Ours (VLM)",
            "threshold": "N/A",
            "val_a": VAL_DIR / "qwen2vl_validation.json",
            "test_a": TEST_DIR / "qwen2vl_test_qwen2_vl_2b_instruct.json",
            "val_b": VAL_DIR / "qwen2vl_validation_multiclass.json",
            "test_b": TEST_DIR / "qwen2vl_test_qwen2_vl_2b_instruct_multiclass.json",
        },
        {
            "name": "Qwen2-VL-2B QLoRA",
            "origin": "Ours (VLM)",
            "threshold": "N/A",
            "val_a": VAL_DIR / "qwen2vl_validation_qwen2_vl_2b_instruct_finetuned.json",
            "test_a": TEST_DIR / "qwen2vl_test_qwen2_vl_2b_instruct_finetuned.json",
            "val_b": VAL_DIR / "qwen2vl_validation_qwen2_vl_2b_instruct_multiclass_finetuned.json",
            "test_b": TEST_DIR / "qwen2vl_test_qwen2_vl_2b_instruct_multiclass_finetuned.json",
        },
        {
            "name": "CLIP ViT-L-14 (Fine-Tuned)",
            "origin": "Ours (CLIP)",
            "threshold": "0.50",
            "val_a": VAL_DIR / "clip_validation_finetuned_singleclass_vit_l_14_quickgelu.json",
            "test_a": TEST_DIR / "clip_test_finetuned_singleclass_vit_l_14_quickgelu.json",
            "val_b": VAL_DIR / "clip_validation_finetuned_multiclass_vit_l_14_quickgelu.json",
            "test_b": TEST_DIR / "clip_test_finetuned_multiclass_vit_l_14_quickgelu.json",
        },
        {
            "name": "CLIP ViT-B-32 (Fine-Tuned)",
            "origin": "Ours (CLIP)",
            "threshold": "0.50",
            "val_a": VAL_DIR / "clip_validation_finetuned_singleclass_vit_b_32_quickgelu.json",
            "test_a": TEST_DIR / "clip_test_finetuned_singleclass_vit_b_32_quickgelu.json",
            "val_b": VAL_DIR / "clip_validation_finetuned_multiclass_vit_b_32_quickgelu.json",
            "test_b": TEST_DIR / "clip_test_finetuned_multiclass_vit_b_32_quickgelu.json",
        },
        {
            "name": "LLaVA-1.5-7B (Zero-Shot)",
            "origin": "Ours (VLM)",
            "threshold": "N/A",
            "val_a": VAL_DIR / "llava_test_validation.json",
            "test_a": TEST_DIR / "llava_test_llava_1_5_7b_hf.json",
            "val_b": VAL_DIR / "llava_test_validation_multiclass.json",
            "test_b": TEST_DIR / "llava_test_llava_1_5_7b_hf_multiclass.json",
        },
        {
            "name": "LLaVA-1.5-7B QLoRA",
            "origin": "Ours (VLM)",
            "threshold": "N/A",
            "val_a": VAL_DIR / "llava_validation_llava_1_5_7b_hf_finetuned.json",
            "test_a": TEST_DIR / "llava_test_llava_1_5_7b_hf_finetuned.json",
            "val_b": VAL_DIR / "llava_validation_llava_1_5_7b_hf_multiclass_finetuned.json",
            "test_b": TEST_DIR / "llava_test_llava_1_5_7b_hf_multiclass_finetuned.json",
        },
        {
            "name": "CLIP ViT-L-14 (Zero-Shot)",
            "origin": "Ours (CLIP)",
            "threshold": "0.50",
            "val_a": VAL_DIR / "clip_validation.json",
            "test_a": TEST_DIR / "clip_test.json",
            "val_b": VAL_DIR / "clip_validation_multiclass.json",
            "test_b": TEST_DIR / "clip_test_multiclass.json",
        },
        {
            "name": "CLIP ViT-B-32 (Zero-Shot)",
            "origin": "Ours (CLIP)",
            "threshold": "0.50",
            "val_a": VAL_DIR / "clip_validation_vit_b_32_quickgelu.json",
            "test_a": TEST_DIR / "clip_test_vit_b_32_quickgelu.json",
            "val_b": VAL_DIR / "clip_validation_multiclass_vit_b_32_quickgelu.json",
            "test_b": TEST_DIR / "clip_test_multiclass_vit_b_32_quickgelu.json",
        },
        {
            "name": "VisualBERT (Zero-Shot)",
            "origin": "Ours (VLM)",
            "threshold": "N/A",
            "val_a": VAL_DIR / "visualbert_validation.json",
            "test_a": TEST_DIR / "visualbert_test.json",
            "val_b": VAL_DIR / "visualbert_validation_multiclass.json",
            "test_b": TEST_DIR / "visualbert_test_multiclass.json",
        },
    ]

    for vm in vlm_models:
        res_val_a = evaluate_json_binary(vm["val_a"])
        res_test_a = evaluate_json_binary(vm["test_a"])
        if res_test_a:
            all_task_a.append(
                {
                    "model": vm["name"],
                    "origin": vm["origin"],
                    "threshold": vm["threshold"],
                    "test_acc": res_test_a["acc"],
                    "test_score_a": res_test_a["score_a"],
                    "val_acc": res_val_a["acc"] if res_val_a else None,
                    "val_score_a": res_val_a["score_a"] if res_val_a else None,
                }
            )

        res_val_b = evaluate_json_multiclass(vm["val_b"])
        res_test_b = evaluate_json_multiclass(vm["test_b"])
        if res_test_b:
            all_task_b.append(
                {
                    "model": vm["name"],
                    "origin": vm["origin"],
                    "threshold": vm["threshold"],
                    "test_score_b": res_test_b["mami_score_b"],
                    "test_macro_f1": res_test_b["macro_f1"],
                    "test_em": res_test_b["exact_match_accuracy"],
                    "val_score_b": res_val_b["mami_score_b"] if res_val_b else None,
                    "val_macro_f1": res_val_b["macro_f1"] if res_val_b else None,
                    "val_em": res_val_b["exact_match_accuracy"] if res_val_b else None,
                }
            )

    # Add tabular models
    name_clean = {
        "Neural Network - MLPClassifier": "MLP",
        "Support Vector Machine - RBF SVC": "SVM-RBF",
        "Support Vector Machine - LinearSVC": "Linear SVC",
        "Hist Gradient Boosting Classifier": "Hist Gradient Boosting",
        "Gradient Boosting Classifier": "Gradient Boosting",
        "Extra Tree Classifier": "Extra Tree",
        "Extra Trees Classifier": "Extra Trees",
        "AdaBoost Classifier": "AdaBoost",
        "Bagging Classifier": "Bagging Classifier",
        "KNN": "K-Nearest Neighbours",
    }

    for m in test_a_models:
        disp_name = name_clean.get(m, m)
        val_res_a = evaluate_binary_predictions(val_a_act, val_a_models[m])
        test_res_a = evaluate_binary_predictions(test_a_act, test_a_models[m])
        all_task_a.append(
            {
                "model": disp_name,
                "origin": "Ours (ML on emb)",
                "threshold": "0.50",
                "test_acc": test_res_a["acc"],
                "test_score_a": test_res_a["score_a"],
                "val_acc": val_res_a["acc"],
                "val_score_a": val_res_a["score_a"],
            }
        )

        val_res_b = compute_multilabel_metrics(val_b_models[m], val_b_act, SUBTYPE_LABELS)
        test_res_b = compute_multilabel_metrics(test_b_models[m], test_b_act, SUBTYPE_LABELS)
        all_task_b.append(
            {
                "model": disp_name,
                "origin": "Ours (ML on emb)",
                "threshold": "0.50",
                "test_score_b": test_res_b["mami_score_b"],
                "test_macro_f1": test_res_b["macro_f1"],
                "test_em": test_res_b["exact_match_accuracy"],
                "val_score_b": val_res_b["mami_score_b"],
                "val_macro_f1": val_res_b["macro_f1"],
                "val_em": val_res_b["exact_match_accuracy"],
            }
        )

    # Sort Task A by test Score A descending
    all_task_a.sort(key=lambda x: x["test_score_a"], reverse=True)
    # Sort Task B by test Score B descending
    all_task_b.sort(key=lambda x: x["test_score_b"], reverse=True)

    print("\n=== TOP TASK A MODELS ===")
    for row in all_task_a[:10]:
        val_str = (
            f"{row['val_acc'] * 100:.2f}% / {row['val_score_a']:.4f}" if row["val_acc"] else "N/A"
        )
        print(
            f"{row['model']:<35} | Thr: {row['threshold']} | Test Acc: {row['test_acc'] * 100:.2f}% | Test F1: {row['test_score_a']:.4f} | Val: {val_str}"
        )

    print("\n=== TOP TASK B MODELS ===")
    for row in all_task_b[:10]:
        val_str = (
            f"{row['val_score_b']:.4f} / {row['val_macro_f1']:.4f} / {row['val_em'] * 100:.2f}%"
            if row["val_score_b"]
            else "N/A"
        )
        print(
            f"{row['model']:<35} | Thr: {row['threshold']} | Test Score B: {row['test_score_b']:.4f} | Macro-F1: {row['test_macro_f1']:.4f} | EM: {row['test_em'] * 100:.2f}% | Val: {val_str}"
        )

    # Write out verified LaTeX table code
    with open(RESULTS_DIR / "verified_table_a1.tex", "w") as f:
        f.write("% Auto-generated verified Table A1\n")
        for r in all_task_a:
            val_acc_s = f"{r['val_acc'] * 100:.2f}\\%" if r["val_acc"] is not None else "N/A"
            val_f1_s = f"{r['val_score_a']:.4f}" if r["val_score_a"] is not None else "N/A"
            f.write(
                f"{r['model']} & {r['origin']} & {r['threshold']} & {r['test_acc'] * 100:.2f}\\% & {r['test_score_a']:.4f} & {val_acc_s} & {val_f1_s} \\\\\n"
            )

    with open(RESULTS_DIR / "verified_table_a2.tex", "w") as f:
        f.write("% Auto-generated verified Table A2\n")
        for r in all_task_b:
            val_sc_s = f"{r['val_score_b']:.4f}" if r["val_score_b"] is not None else "N/A"
            val_f1_s = f"{r['val_macro_f1']:.4f}" if r["val_macro_f1"] is not None else "N/A"
            val_em_s = f"{r['val_em'] * 100:.2f}\\%" if r["val_em"] is not None else "N/A"
            f.write(
                f"{r['model']} & {r['origin']} & {r['threshold']} & {r['test_score_b']:.4f} & {r['test_macro_f1']:.4f} & {r['test_em'] * 100:.2f}\\% & {val_sc_s} & {val_f1_s} & {val_em_s} \\\\\n"
            )

    print(
        "\nSaved LaTeX snippets to results/verified_table_a1.tex and results/verified_table_a2.tex"
    )


if __name__ == "__main__":
    main()
