"""Generate the paper-ready consolidated comparison report (GitHub issue #65).

Covers Gist tasks 6-8:

* Task A (binary misogyny): Accuracy, ROC-AUC, Precision, Recall, Macro F1.
* Task B (multi-label sub-types): per-class Precision/Recall/F1/Support plus
  Macro/Micro F1 and Exact-Match Accuracy.
* A regenerated consolidated table plus ``results/comparison_report.md`` and
  bar-chart figures under ``results/figures/``.

AUC sourcing (per the #65 audit note):
* Tabular fusion models emit ``ROC_AUC`` natively (auto_benchmark evaluation CSV).
* CLIP-family models expose per-sample softmax ``confidence`` from which the
  positive-class probability is reconstructed (``p_pos = c if pred==yes else 1-c``).
* Hard-output generative VLMs (Gemini / Qwen / LLaVA / VisualBERT) produce only
  discrete labels, so AUC is reported as ``N/A``.

Run:
    uv run python scripts/generate_consolidated_table.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
VAL_DIR = RESULTS_DIR / "validation"
TEST_DIR = RESULTS_DIR / "test"
FIGURES_DIR = RESULTS_DIR / "figures"
REPORT_PATH = RESULTS_DIR / "comparison_report.md"

# auto_benchmark tabular evaluation CSVs (ViT-L-14 + PaddleOCR fusion sweep).
AB_RESULTS = Path(__file__).resolve().parents[1] / "auto_benchmark" / "results" / "model_results"
TABULAR_VAL_CSV = AB_RESULTS / "mami_tabular_model" / "mami_tabular_model_evaluation.csv"
TABULAR_TEST_CSV = AB_RESULTS / "mami_tabular_model_test" / "mami_tabular_model_test_evaluation.csv"

SUBTYPE_LABELS = ["shaming", "stereotype", "objectification", "violence"]

models_info: list[dict[str, Any]] = [
    {
        "name": "Gemini 1.5 Pro (Zero-Shot)",
        "val_a": VAL_DIR / "gemini_validation.json",
        "val_b": VAL_DIR / "gemini_validation_multiclass.json",
        "test_a": TEST_DIR / "gemini_test.json",
        "test_b": TEST_DIR / "gemini_test_multiclass.json",
    },
    {
        # Task A metrics for this row are sourced from the auto_benchmark tabular
        # sweep (best XGBoost model), which carries a native ROC_AUC.
        "name": "XGBoost Fusion (ViT-L-14 + PaddleOCR)",
        "tabular_a": True,
        "val_b": VAL_DIR / "xgboost_validation_xgboost_multiclass.json",
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
        "test_a": TEST_DIR / "qwen2vl_test_qwen2_vl_2b_instruct.json",
        "test_b": TEST_DIR / "qwen2vl_test_qwen2_vl_2b_instruct_multiclass.json",
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
        "val_a": VAL_DIR / "llava_test_validation.json",
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


def _select_row(data: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Pick the 'none' preprocessing filter row, else the first row."""
    if not isinstance(data, list) or len(data) == 0:
        return None
    for item in data:
        if item.get("filter") == "none":
            return item
    return data[0]


def compute_auc_from_samples(row: dict[str, Any]) -> float | None:
    """Reconstruct ROC-AUC for a binary CLIP result from per-sample confidence.

    CLIP exposes the softmax confidence of its chosen label. For a 2-class
    softmax the positive-class probability is ``c`` when the prediction is the
    positive class and ``1 - c`` otherwise. That reconstructed score is
    monotonic in P(misogynous) and yields a valid ROC-AUC. Returns ``None`` if
    no usable confidences are present (e.g. hard-label generative models).
    """
    samples = row.get("sample_predictions") or []
    y_true: list[int] = []
    y_score: list[float] = []
    for s in samples:
        gt = s.get("ground_truth")
        pred = s.get("prediction")
        conf = s.get("confidence")
        if gt is None or pred is None or conf is None:
            return None
        # ground_truth / prediction are stored as ints (1 = misogynous).
        p_pos = float(conf) if int(pred) == 1 else 1.0 - float(conf)
        y_true.append(int(gt))
        y_score.append(p_pos)
    if not y_true or len(set(y_true)) < 2:
        return None
    try:
        from sklearn.metrics import roc_auc_score

        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return None


def load_task_a(filepath: Path | None) -> dict[str, float | None]:
    """Load binary (Task A) metrics from a result JSON file."""
    empty: dict[str, float | None] = {
        "acc": None,
        "auc": None,
        "precision": None,
        "recall": None,
        "f1": None,
    }
    if not filepath or not filepath.exists():
        return empty
    try:
        row = _select_row(json.loads(filepath.read_text(encoding="utf-8")))
        if row is None:
            return empty
        return {
            "acc": _as_float(row.get("exact_match_accuracy", row.get("accuracy"))),
            "auc": compute_auc_from_samples(row),
            "precision": _as_float(row.get("precision")),
            "recall": _as_float(row.get("recall")),
            "f1": _as_float(row.get("f1", row.get("macro_f1"))),
        }
    except Exception:
        return empty


def load_task_b(filepath: Path | None) -> dict[str, Any] | None:
    """Load multi-label (Task B) metrics from a result JSON file."""
    if not filepath or not filepath.exists():
        return None
    try:
        row = _select_row(json.loads(filepath.read_text(encoding="utf-8")))
        if row is None:
            return None
        return {
            "em": _as_float(row.get("exact_match_accuracy")),
            "macro_f1": _as_float(row.get("macro_f1", row.get("f1"))),
            "micro_f1": _as_float(row.get("micro_f1")),
            "macro_precision": _as_float(row.get("precision")),
            "macro_recall": _as_float(row.get("recall")),
            "per_class": row.get("per_class"),
        }
    except Exception:
        return None


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_tabular_task_a(csv_path: Path, model: str = "XGBoost") -> dict[str, float | None]:
    """Read Task A metrics (incl. native ROC_AUC) for one model from an
    auto_benchmark evaluation CSV."""
    empty: dict[str, float | None] = {
        "acc": None,
        "auc": None,
        "precision": None,
        "recall": None,
        "f1": None,
    }
    if not csv_path.exists():
        return empty
    try:
        with csv_path.open(encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r.get("Model", "").strip() == model:
                    return {
                        "acc": _as_float(r.get("Accuracy")),
                        "auc": _as_float(r.get("ROC_AUC")),
                        "precision": _as_float(r.get("Precision")),
                        "recall": _as_float(r.get("Recall")),
                        "f1": _as_float(r.get("F1")),
                    }
    except Exception:
        return empty
    return empty


def _pct(value: float | None) -> str:
    return f"{value * 100:.1f}%" if value is not None else "N/A"


def _f4(value: float | None) -> str:
    return f"{value:.4f}" if value is not None else "N/A"


def gather_rows() -> list[dict[str, Any]]:
    """Resolve every model's Task A / Task B metrics for both splits."""
    rows: list[dict[str, Any]] = []
    for m in models_info:
        if m.get("tabular_a"):
            val_a = load_tabular_task_a(TABULAR_VAL_CSV)
            test_a = load_tabular_task_a(TABULAR_TEST_CSV)
        else:
            val_a = load_task_a(m.get("val_a"))
            test_a = load_task_a(m.get("test_a"))
        rows.append(
            {
                "name": m["name"],
                "val_a": val_a,
                "test_a": test_a,
                "val_b": load_task_b(m.get("val_b")),
                "test_b": load_task_b(m.get("test_b")),
            }
        )
    return rows


def render_task_a_table(rows: list[dict[str, Any]]) -> str:
    """Markdown table: Accuracy, AUC, Precision, Recall, Macro F1 (val + test)."""
    header = (
        "| Model | Val Acc | Val AUC | Val Prec | Val Rec | Val F1 "
        "| Test Acc | Test AUC | Test Prec | Test Rec | Test F1 |"
    )
    sep = "| " + " | ".join(["---"] * 11) + " |"
    lines = [header, sep]
    for r in rows:
        va, ta = r["val_a"], r["test_a"]
        lines.append(
            "| {name} | {vacc} | {vauc} | {vp} | {vr} | {vf1} "
            "| {tacc} | {tauc} | {tp} | {tr} | {tf1} |".format(
                name=r["name"],
                vacc=_pct(va["acc"]),
                vauc=_f4(va["auc"]),
                vp=_f4(va["precision"]),
                vr=_f4(va["recall"]),
                vf1=_f4(va["f1"]),
                tacc=_pct(ta["acc"]),
                tauc=_f4(ta["auc"]),
                tp=_f4(ta["precision"]),
                tr=_f4(ta["recall"]),
                tf1=_f4(ta["f1"]),
            )
        )
    return "\n".join(lines)


def render_task_b_aggregate_table(rows: list[dict[str, Any]]) -> str:
    """Markdown table of aggregate Task B metrics on the test split."""
    header = "| Model | Exact Match | Macro F1 | Micro F1 | Macro Prec | Macro Rec |"
    sep = "| " + " | ".join(["---"] * 6) + " |"
    lines = [header, sep]
    for r in rows:
        b = r["test_b"] or r["val_b"]
        if not b:
            continue
        lines.append(
            "| {name} | {em} | {mf1} | {micf1} | {mp} | {mr} |".format(
                name=r["name"],
                em=_pct(b["em"]),
                mf1=_f4(b["macro_f1"]),
                micf1=_f4(b["micro_f1"]),
                mp=_f4(b["macro_precision"]),
                mr=_f4(b["macro_recall"]),
            )
        )
    return "\n".join(lines)


def render_task_b_per_class_tables(rows: list[dict[str, Any]]) -> str:
    """Per-class Precision/Recall/F1/Support tables (test split) per model."""
    blocks: list[str] = []
    for r in rows:
        b = r["test_b"] or r["val_b"]
        if not b or not b.get("per_class"):
            continue
        per_class = b["per_class"]
        blocks.append(f"#### {r['name']}\n")
        blocks.append("| Sub-type | Precision | Recall | F1 | Support |")
        blocks.append("| --- | --- | --- | --- | --- |")
        for lbl in SUBTYPE_LABELS:
            sc = per_class.get(lbl, {})
            blocks.append(
                "| {lbl} | {p} | {r} | {f1} | {sup} |".format(
                    lbl=lbl,
                    p=_f4(_as_float(sc.get("precision"))),
                    r=_f4(_as_float(sc.get("recall"))),
                    f1=_f4(_as_float(sc.get("f1"))),
                    sup=int(sc.get("support", 0)),
                )
            )
        blocks.append("")
    return "\n".join(blocks)


def make_figures(rows: list[dict[str, Any]]) -> None:
    """Bar charts: Task A test F1 and Task B test Macro F1 per model."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib unavailable; skipping figures")
        return

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    a_names = [r["name"] for r in rows if r["test_a"]["f1"] is not None]
    a_vals = [r["test_a"]["f1"] for r in rows if r["test_a"]["f1"] is not None]
    if a_vals:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(a_names, a_vals, color="#4C72B0")
        ax.set_xlabel("Macro F1 (Task A, test split)")
        ax.set_title("Task A: Binary Misogyny Classification - Macro F1")
        ax.set_xlim(0, 1)
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "task_a_f1_comparison.png", dpi=150)
        plt.close(fig)
        print(f"Saved figure: {FIGURES_DIR / 'task_a_f1_comparison.png'}")

    b_rows = [(r["name"], (r["test_b"] or r["val_b"])) for r in rows]
    b_rows = [(n, b) for n, b in b_rows if b and b["macro_f1"] is not None]
    if b_rows:
        b_names = [n for n, _ in b_rows]
        b_vals = [b["macro_f1"] for _, b in b_rows]
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(b_names, b_vals, color="#DD8452")
        ax.set_xlabel("Macro F1 (Task B)")
        ax.set_title("Task B: Multi-label Sub-type Classification - Macro F1")
        ax.set_xlim(0, 1)
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "task_b_macro_f1_comparison.png", dpi=150)
        plt.close(fig)
        print(f"Saved figure: {FIGURES_DIR / 'task_b_macro_f1_comparison.png'}")


def write_report(rows: list[dict[str, Any]]) -> None:
    """Write results/comparison_report.md with all #65 sections."""
    parts: list[str] = [
        "# MAMI 2022 - Paper-Ready Comparison Report",
        "",
        "Auto-generated by `scripts/generate_consolidated_table.py`",
        "(GitHub issue #65 / Gist tasks 6-8).",
        "",
        "All numbers trace back to the JSON files in `results/validation/` and",
        "`results/test/` and the auto_benchmark tabular evaluation CSVs.",
        "",
        "## Task A - Binary Misogyny Classification",
        "",
        "Metrics: Accuracy, ROC-AUC, Precision, Recall, Macro F1.",
        "",
        "AUC is `N/A` for hard-label generative VLMs (Gemini, Qwen2-VL, LLaVA,",
        "VisualBERT) which emit discrete labels with no probability score.",
        "CLIP AUC is reconstructed from per-sample softmax confidence; the",
        "XGBoost fusion AUC is the native `ROC_AUC` from the tabular sweep.",
        "",
        render_task_a_table(rows),
        "",
        "## Task B - Multi-label Sub-type Classification (Aggregate)",
        "",
        "Metrics: Exact-Match Accuracy, Macro F1, Micro F1, Macro Precision,",
        "Macro Recall (test split where available, else validation).",
        "",
        render_task_b_aggregate_table(rows),
        "",
        "## Task B - Per-Class Metrics",
        "",
        "Per-class Precision / Recall / F1 / Support for each sub-type",
        "(shaming, stereotype, objectification, violence).",
        "",
        render_task_b_per_class_tables(rows),
        "## Figures",
        "",
        "- `results/figures/task_a_f1_comparison.png` - Task A Macro F1 per model.",
        "- `results/figures/task_b_macro_f1_comparison.png` - Task B Macro F1 per model.",
        "- `results/figures/shap_modality_importance.png` - XGBoost fusion",
        "  image-vs-text reliance (issue #89).",
        "",
    ]
    REPORT_PATH.write_text("\n".join(parts) + "\n", encoding="utf-8")
    print(f"Wrote report: {REPORT_PATH}")


def main() -> None:
    rows = gather_rows()

    # Console summary (Task A).
    print("# Task A (binary) - Accuracy / AUC / Macro F1\n")
    print(f"| {'Model':<38} | {'Test Acc':<9} | {'Test AUC':<9} | {'Test F1':<9} |")
    print(f"| {'-' * 38} | {'-' * 9} | {'-' * 9} | {'-' * 9} |")
    for r in rows:
        ta = r["test_a"]
        print(
            f"| {r['name']:<38} | {_pct(ta['acc']):<9} | {_f4(ta['auc']):<9} | {_f4(ta['f1']):<9} |"
        )

    write_report(rows)
    make_figures(rows)


if __name__ == "__main__":
    main()
