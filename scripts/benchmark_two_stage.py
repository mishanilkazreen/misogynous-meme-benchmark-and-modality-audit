"""Two-stage MAMI benchmark: run Task A first, gate Task B on positives.

The MAMI task definition says a sub-type is positive only when the meme
is misogynous. Our single-shot Task B prompt does not tell the model
this, so on benign memes the VLM either refuses or defaults to the most
common training label (see docs/CODE_REVIEW_ISSUES.md §6.2). This
orchestrator applies the constraint explicitly:

1. Read a Task A result JSON (from ``benchmark_*.py`` or the tabular
   XGBoost path). Any classifier that outputs per-image binary
   predictions works; the JSON schema is the ``sample_predictions``
   list used across the repo.
2. Read a Task B result JSON produced by any Task B benchmark.
3. For every image, force the four sub-type predictions to zero when
   the Task A prediction is ``0`` (non-misogynous); otherwise keep the
   Task B predictions unchanged.
4. Recompute the multi-label metrics under this constraint.

The result JSON matches the shape of a regular Task B benchmark output
so downstream scripts (``generate_consolidated_table.py``, the paper
tables) can consume it without any special-casing.

Usage:
    uv run python scripts/benchmark_two_stage.py \\
        --task-a results/test/qwen2vl_test_qwen2_vl_7b_instruct_finetuned.json \\
        --task-b results/test/qwen2vl_test_qwen2_vl_7b_instruct_multiclass_finetuned.json \\
        --output results/test/qwen2vl_test_two_stage_multiclass.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from models.vlm.classifier import SUBTYPE_LABELS
from models.vlm.metrics_multilabel import compute_multilabel_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _select_row(payload: list[dict] | dict) -> dict:
    """Pick the ``filter='none'`` row from a benchmark JSON payload.

    Benchmark scripts emit either a single dict or a list of per-filter
    dicts. Pick the row with ``filter='none'`` when the file is a list,
    else return the dict as-is. Preserves the convention used by
    :func:`scripts.generate_consolidated_table.load_task_b`.
    """
    if isinstance(payload, dict):
        return payload
    if not payload:
        raise ValueError("Result JSON is an empty list.")
    for row in payload:
        if row.get("filter") == "none":
            return row
    return payload[0]


def _load_task_a_predictions(path: Path) -> dict[str, int]:
    """Load per-image Task A predictions from a benchmark JSON.

    Returns a dict mapping ``image_id`` (as ``str``) to a 0/1 int
    prediction. Uses the ``sample_predictions`` array written by
    ``benchmark_*.py``; each row has an ``image_id`` and a
    ``prediction`` field (int 0/1) - see the ``build_sample_rows``
    helper in ``benchmark_vlm_classification.py``.
    """
    row = _select_row(json.loads(path.read_text(encoding="utf-8")))
    samples = row.get("sample_predictions") or []
    result: dict[str, int] = {}
    for s in samples:
        image_id = str(s.get("image_id"))
        pred = s.get("prediction")
        if pred is None:
            # A refusal or unparseable Task A response. Treat as
            # non-misogynous so Task B is forced to zero for this image.
            result[image_id] = 0
        else:
            result[image_id] = int(pred)
    return result


def _load_task_b_rows(path: Path) -> tuple[dict[str, dict[str, int]], list[dict[str, Any]]]:
    """Return (per-image sub-type prediction dict, ground-truth rows).

    The Task B benchmark JSON's ``sample_predictions`` array has
    ``ground_truth`` and ``prediction`` fields that are already dicts
    keyed by sub-type (see ``build_multiclass_sample_rows``). This
    helper unpacks them for the merge step.
    """
    row = _select_row(json.loads(path.read_text(encoding="utf-8")))
    samples = row.get("sample_predictions") or []
    preds: dict[str, dict[str, int]] = {}
    for s in samples:
        image_id = str(s.get("image_id"))
        preds[image_id] = {lbl: int(s.get("prediction", {}).get(lbl, 0)) for lbl in SUBTYPE_LABELS}
    return preds, samples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task-a",
        required=True,
        type=Path,
        help="Path to the Task A (binary) benchmark result JSON.",
    )
    parser.add_argument(
        "--task-b",
        required=True,
        type=Path,
        help="Path to the Task B (multi-label) benchmark result JSON.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Where to write the two-stage-gated result JSON.",
    )
    args = parser.parse_args()

    logger.info("Loading Task A predictions from %s", args.task_a)
    a_preds = _load_task_a_predictions(args.task_a)
    logger.info("Loading Task B predictions from %s", args.task_b)
    b_preds, b_rows = _load_task_b_rows(args.task_b)

    common_ids = set(a_preds.keys()) & set(b_preds.keys())
    if not common_ids:
        raise SystemExit(
            "No image_id overlap between the Task A and Task B result files. "
            "Check that both benchmarks ran on the same split and used the "
            "same MAMI TSV."
        )
    logger.info(
        "Merging Task A (%d images) and Task B (%d images); %d overlap.",
        len(a_preds),
        len(b_preds),
        len(common_ids),
    )

    merged_preds: list[dict[str, int]] = []
    merged_gts: list[dict[str, int]] = []
    sample_rows: list[dict[str, Any]] = []
    a_zero = 0
    for row in b_rows:
        image_id = str(row.get("image_id"))
        if image_id not in a_preds:
            continue
        a_pred = a_preds[image_id]
        gt = {lbl: int(row.get("ground_truth", {}).get(lbl, 0)) for lbl in SUBTYPE_LABELS}
        if a_pred == 0:
            # Non-misogynous -> force sub-types to zero (MAMI constraint).
            pred = dict.fromkeys(SUBTYPE_LABELS, 0)
            a_zero += 1
        else:
            pred = b_preds[image_id]
        merged_preds.append(pred)
        merged_gts.append(gt)
        sample_rows.append(
            {
                "image_id": image_id,
                "ground_truth": gt,
                "prediction": pred,
                "correct": all(pred[lbl] == gt[lbl] for lbl in SUBTYPE_LABELS),
                "task_a_prediction": a_pred,
            }
        )

    logger.info(
        "Two-stage gating zeroed out Task B for %d of %d images (%.1f %%).",
        a_zero,
        len(sample_rows),
        100.0 * a_zero / max(1, len(sample_rows)),
    )

    metrics = compute_multilabel_metrics(merged_preds, merged_gts, SUBTYPE_LABELS)
    output_payload = [
        {
            "task": "two_stage_multiclass",
            "filter": "none",
            "source_task_a": str(args.task_a),
            "source_task_b": str(args.task_b),
            "n_images": len(sample_rows),
            "n_gated_to_zero": a_zero,
            "exact_match_accuracy": metrics["exact_match_accuracy"],
            "macro_f1": metrics["macro_f1"],
            "micro_f1": metrics["micro_f1"],
            "weighted_f1": metrics["weighted_f1"],
            "precision": metrics["macro_precision"],
            "recall": metrics["macro_recall"],
            "per_class": metrics["per_class"],
            "mami_score_b": metrics.get("mami_score_b"),
            "per_label_binary_macro_f1": metrics.get("per_label_binary_macro_f1"),
            "f1": metrics.get("mami_score_b", metrics["macro_f1"]),
            "sample_predictions": sample_rows,
        }
    ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output_payload, indent=2) + "\n", encoding="utf-8")
    logger.info(
        "Two-stage MAMI score B: %.4f (macro F1: %.4f, exact match: %.4f). "
        "Wrote %s.",
        output_payload[0].get("mami_score_b") or 0.0,
        output_payload[0]["macro_f1"],
        output_payload[0]["exact_match_accuracy"],
        args.output,
    )


if __name__ == "__main__":
    main()
