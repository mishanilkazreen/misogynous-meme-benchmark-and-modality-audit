"""Compute MAMI-consistent metrics for the auto_benchmark tabular sweep.

The auto_benchmark framework reports sklearn macro-F1 for the Task B
(multi-label) sweep, which is NOT the metric the fine-tuned CLIP/VLM systems
report (``mami_score_b``, the MAMI 2022 official Sub-task B score). That makes
the consolidated leaderboard compare unlike quantities
(docs/CODE_REVIEW_ISSUES.md §1.4, audit bug 3).

This post-processor reads the sweep's per-sample prediction CSVs and recomputes
the SAME metrics used everywhere else (``models.vlm.metrics_multilabel``), so
the tabular rows line up with CLIP/VLM. It does not retrain anything.

Task B prediction CSVs store each cell as a comma-separated 4-vector string
(e.g. ``"0,1,0,0"``) with a header ``Actual,<Model1>,<Model2>,...``. Task A
CSVs store scalar 0/1 cells.

Usage:
    uv run python scripts/tabular_mami_metrics.py \\
        --predictions auto_benchmark/results/model_results/\\
mami_tabular_model_multiclass_provided_test/\\
mami_tabular_model_multiclass_provided_test_predictions.csv \\
        --task multilabel --split test --variant provided \\
        --output results/test/tabular_provided_multiclass_mami.json
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from models.vlm.metrics_multilabel import compute_multilabel_metrics

SUBTYPE_LABELS = ["shaming", "stereotype", "objectification", "violence"]


def _parse_vector_cell(cell: str) -> list[int]:
    """Parse a ``"0,1,0,0"`` style cell into a list of ints."""
    parts = [p.strip() for p in cell.strip().strip('"').split(",") if p.strip() != ""]
    return [int(float(p)) for p in parts]


def parse_multilabel_predictions_csv(
    path: str | Path,
    labels: list[str] = SUBTYPE_LABELS,
) -> dict[str, Any]:
    """Parse a Task B predictions CSV into ground-truth + per-model prediction dicts.

    Returns a dict with ``ground_truths`` (list of label dicts) and ``models``
    (mapping model name -> list of predicted label dicts).
    """
    with Path(path).open(newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        model_names = header[1:]  # first column is "Actual"
        ground_truths: list[dict[str, int]] = []
        model_preds: dict[str, list[dict[str, int]]] = {m: [] for m in model_names}
        for row in reader:
            if not row:
                continue
            gt_vec = _parse_vector_cell(row[0])
            ground_truths.append({lbl: gt_vec[i] for i, lbl in enumerate(labels)})
            for m, cell in zip(model_names, row[1:], strict=True):
                pred_vec = _parse_vector_cell(cell)
                model_preds[m].append({lbl: pred_vec[i] for i, lbl in enumerate(labels)})
    return {"ground_truths": ground_truths, "models": model_preds}


def compute_tabular_task_b_metrics(
    path: str | Path,
    labels: list[str] = SUBTYPE_LABELS,
) -> dict[str, dict[str, Any]]:
    """Compute MAMI-consistent Task B metrics for every model in a predictions CSV."""
    parsed = parse_multilabel_predictions_csv(path, labels)
    gts = parsed["ground_truths"]
    out: dict[str, dict[str, Any]] = {}
    for model, preds in parsed["models"].items():
        out[model] = compute_multilabel_metrics(preds, gts, labels)
    return out


def _parse_scalar_predictions_csv(path: str | Path) -> dict[str, Any]:
    with Path(path).open(newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        model_names = header[1:]
        actual: list[int] = []
        model_preds: dict[str, list[int]] = {m: [] for m in model_names}
        for row in reader:
            if not row:
                continue
            actual.append(int(float(row[0])))
            for m, cell in zip(model_names, row[1:], strict=True):
                model_preds[m].append(int(float(cell)))
    return {"actual": actual, "models": model_preds}


def compute_tabular_task_a_metrics(path: str | Path) -> dict[str, dict[str, float]]:
    """Compute accuracy + macro-F1 for every model in a binary predictions CSV.

    Uses the same macro-F1 definition as the rest of the pipeline. Threshold
    calibration on probabilities is handled by ``train_classifier.py``
    (``--threshold-calibrate``); the sweep CSVs only carry hard labels.
    """
    parsed = _parse_scalar_predictions_csv(path)
    actual = parsed["actual"]
    out: dict[str, dict[str, float]] = {}
    for model, preds in parsed["models"].items():
        tp = sum(1 for p, a in zip(preds, actual, strict=True) if p == 1 and a == 1)
        fp = sum(1 for p, a in zip(preds, actual, strict=True) if p == 1 and a == 0)
        fn = sum(1 for p, a in zip(preds, actual, strict=True) if p == 0 and a == 1)
        tn = sum(1 for p, a in zip(preds, actual, strict=True) if p == 0 and a == 0)
        pos_prec = tp / (tp + fp) if (tp + fp) else 0.0
        pos_rec = tp / (tp + fn) if (tp + fn) else 0.0
        pos_f1 = 2 * pos_prec * pos_rec / (pos_prec + pos_rec) if (pos_prec + pos_rec) else 0.0
        neg_prec = tn / (tn + fn) if (tn + fn) else 0.0
        neg_rec = tn / (tn + fp) if (tn + fp) else 0.0
        neg_f1 = 2 * neg_prec * neg_rec / (neg_prec + neg_rec) if (neg_prec + neg_rec) else 0.0
        n = len(actual)
        out[model] = {
            "accuracy": round((tp + tn) / n, 6) if n else 0.0,
            "macro_f1": round((pos_f1 + neg_f1) / 2, 6),
            "precision": round(pos_prec, 6),
            "recall": round(pos_rec, 6),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, help="Path to the sweep predictions CSV")
    parser.add_argument("--task", required=True, choices=["binary", "multilabel"])
    parser.add_argument("--split", required=True, help="Split label for the output JSON")
    parser.add_argument(
        "--variant", required=True, help="Text-source variant (provided/ocr/combined)"
    )
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    if args.task == "multilabel":
        metrics = compute_tabular_task_b_metrics(args.predictions)
        ranked = sorted(metrics.items(), key=lambda kv: kv[1]["mami_score_b"], reverse=True)
    else:
        metrics = compute_tabular_task_a_metrics(args.predictions)
        ranked = sorted(metrics.items(), key=lambda kv: kv[1]["macro_f1"], reverse=True)

    payload = {
        "task": args.task,
        "split": args.split,
        "variant": args.variant,
        "source_csv": str(args.predictions),
        "best_model": ranked[0][0] if ranked else None,
        "models": dict(ranked),
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if ranked:
        key = "mami_score_b" if args.task == "multilabel" else "macro_f1"
        print(
            f"{args.variant}/{args.split}/{args.task}: best={ranked[0][0]} {key}={ranked[0][1][key]}"
        )
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
