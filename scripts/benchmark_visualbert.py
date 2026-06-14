"""
Benchmark VisualBERT (uclanlp/visualbert-vqa-coco-pre) on MAMI 2022 misogyny detection.

Singleclass (--task singleclass, default): binary misogyny classification via MLM cloze.
The model fills in a [MASK] token in "this meme is [MASK] toward women ." and the
logits for "yes" vs "no" determine the prediction.  This is an untrained zero-shot
baseline expected to perform near-chance.

Multiclass (--task multiclass): multi-label sub-type classification (shaming, stereotype,
objectification, violence) via four independent yes/no MLM-cloze comparisons, one per
category.  Same mechanism as singleclass — no fine-tuning.

Usage:
    uv run python scripts/benchmark_visualbert.py --split validation --limit 10
    uv run python scripts/benchmark_visualbert.py --split validation --limit 50 --device cpu
    uv run python scripts/benchmark_visualbert.py --split validation --limit 10 --task multiclass
"""

# ruff: noqa: I001  # datasets (via utils.dataset) must precede torch to avoid OpenMP segfault
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from models.vlm.classifier import (
    MISOGYNY_LABELS,
    SUBTYPE_LABELS,
    ClassificationResult,
    yesno_to_int,
)
from models.vlm.metrics_multilabel import compute_multilabel_metrics
from scripts.benchmark_vlm_classification import build_label_prevalence
from utils.dataset import DatasetManager
from utils.preprocessing import PreprocessingPipeline

try:
    from transformers import AutoTokenizer  # type: ignore[import-untyped]  # noqa: F401

    _TRANSFORMERS_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
    _TRANSFORMERS_AVAILABLE = False

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
MODEL_NAME = "visualbert"

_model_cache: dict[str, Any] = {}


def image_to_numpy(image: Any) -> np.ndarray:
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    if isinstance(image, np.ndarray):
        if image.ndim == 3 and image.shape[0] == 3:
            image = image.transpose(1, 2, 0)
        if np.issubdtype(image.dtype, np.floating):
            image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
        elif image.dtype != np.uint8:
            image = image.astype(np.uint8)
        return image
    raise ValueError(f"Unsupported image type: {type(image)}")


def collect_samples(split: str = "validation", limit: int | None = None) -> list[dict[str, Any]]:
    """Load samples from the MAMI 2022 dataset for the given split(s).

    split may be a single name or comma-separated names, e.g. 'train,validation'.
    limit caps the total number of samples across all splits combined.
    """
    manager = DatasetManager()
    samples: list[dict[str, Any]] = []
    for split_name in [s.strip() for s in split.split(",")]:
        dataset = manager.load_dataset(split=split_name)
        for index in range(len(dataset)):
            samples.append(dataset[index])
    if limit is not None:
        samples = samples[:limit]
    return samples


def _misogynous_to_label(misogynous: int) -> str:
    return "yes" if misogynous == 1 else "no"


def _get_classifier(device: str) -> Any:
    """Return a cached :class:`VisualBERTClassifier` for *device*."""
    if device not in _model_cache:
        from models.vlm.visualbert_classifier import VisualBERTClassifier

        _model_cache[device] = VisualBERTClassifier(device=device)
    return _model_cache[device]


def _aggregate(
    results: list[ClassificationResult],
    ground_truths: list[str],
    split: str,
    preprocess: str | None,
    labels: list[str],
    sample_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    n_total = len(results)
    refusal_rate = sum(1 for r in results if r.refusal) / n_total if n_total else 0.0
    avg_latency = sum(r.latency_s for r in results) / n_total if n_total else 0.0

    predictions: list[str | None] = [r.prediction for r in results]
    correct_all = sum(1 for p, gt in zip(predictions, ground_truths, strict=True) if p == gt)
    accuracy = correct_all / n_total if n_total else 0.0

    per_class_prec: list[float] = []
    per_class_rec: list[float] = []
    for label in labels:
        tp = sum(
            1
            for p, gt in zip(predictions, ground_truths, strict=True)
            if p == label and gt == label
        )
        fp = sum(
            1
            for p, gt in zip(predictions, ground_truths, strict=True)
            if p == label and gt != label
        )
        fn = sum(
            1
            for p, gt in zip(predictions, ground_truths, strict=True)
            if p != label and gt == label
        )
        per_class_prec.append(tp / (tp + fp) if (tp + fp) > 0 else 0.0)
        per_class_rec.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
    macro_prec = sum(per_class_prec) / len(per_class_prec) if per_class_prec else 0.0
    macro_rec = sum(per_class_rec) / len(per_class_rec) if per_class_rec else 0.0
    macro_f1 = (
        2 * macro_prec * macro_rec / (macro_prec + macro_rec)
        if (macro_prec + macro_rec) > 0
        else 0.0
    )

    return {
        "benchmark_date": datetime.now(timezone.utc).isoformat(),
        "model": MODEL_NAME,
        "filter": preprocess or "none",
        "split": split,
        "task": "singleclass",
        "exact_match_accuracy": accuracy,
        "precision": macro_prec,
        "recall": macro_rec,
        "f1": macro_f1,
        "avg_latency_s": avg_latency,
        "refusal_rate": refusal_rate,
        "sample_predictions": sample_rows,
    }


def _run_benchmark_multiclass(
    classifier: Any,
    samples: list[dict[str, Any]],
    split: str,
    preprocess: str | None,
) -> dict[str, Any]:
    """Run VisualBERT multiclass: 4-category sub-type classification via MLM cloze."""
    pipeline = PreprocessingPipeline() if preprocess else None

    pred_dicts: list[dict[str, int]] = []
    gt_dicts: list[dict[str, int]] = []
    latencies: list[float] = []
    sample_rows: list[dict[str, Any]] = []

    pbar = tqdm(
        total=len(samples),
        desc=f"visualbert-multiclass/{preprocess or 'none'}",
        unit="img",
    )
    for s in samples:
        arr = image_to_numpy(s["image"])
        if pipeline is not None and preprocess is not None:
            arr = pipeline.apply_transformation(arr, preprocess)

        t0 = time.perf_counter()
        subtype_pred = classifier.classify_subtypes(arr, text=s.get("text"))
        latencies.append(time.perf_counter() - t0)

        gt_dict: dict[str, int] = {
            "shaming": s.get("shaming", 0),
            "stereotype": s.get("stereotype", 0),
            "objectification": s.get("objectification", 0),
            "violence": s.get("violence", 0),
        }
        pred_dicts.append(subtype_pred)
        gt_dicts.append(gt_dict)
        exact = all(subtype_pred.get(lbl, 0) == gt_dict[lbl] for lbl in SUBTYPE_LABELS)
        sample_rows.append(
            {
                "image_id": s["image_id"],
                "ground_truth": gt_dict,
                "prediction": subtype_pred,
                "correct": exact,
                "misogynous": s.get("misogynous", 0),
            }
        )
        pbar.update(1)
    pbar.close()

    n_total = len(samples)
    avg_latency = sum(latencies) / n_total if n_total else 0.0
    ml_metrics = compute_multilabel_metrics(pred_dicts, gt_dicts, SUBTYPE_LABELS)
    label_prev = build_label_prevalence(samples, task="multiclass")

    return {
        "benchmark_date": datetime.now(timezone.utc).isoformat(),
        "model": MODEL_NAME,
        "filter": preprocess or "none",
        "split": split,
        "task": "multiclass",
        "exact_match_accuracy": ml_metrics["exact_match_accuracy"],
        "f1": ml_metrics["macro_f1"],
        "precision": ml_metrics["macro_precision"],
        "recall": ml_metrics["macro_recall"],
        "macro_f1": ml_metrics["macro_f1"],
        "micro_f1": ml_metrics["micro_f1"],
        "weighted_f1": ml_metrics["weighted_f1"],
        "per_class": ml_metrics["per_class"],
        "avg_latency_s": avg_latency,
        "refusal_rate": 0.0,
        "label_prevalence": label_prev,
        "sample_predictions": sample_rows,
    }


def run_benchmark(
    split: str = "validation",
    device: str = "cpu",
    preprocess: str | None = None,
    limit: int | None = None,
    samples: list[dict[str, Any]] | None = None,
    task: str = "singleclass",
) -> dict[str, Any]:
    """Run VisualBERT on the MAMI 2022 misogyny classification task.

    Args:
        split: Dataset split ('train', 'validation', 'test').
        device: Target device ('cpu' default; 'cuda' also supported).
        preprocess: Preprocessing filter name, or None for no filter.
        limit: Cap on number of samples.
        samples: Pre-loaded samples (skips dataset loading if provided).
        task: ``'singleclass'`` (default) or ``'multiclass'`` (4-category MLM cloze).
    """
    if not _TRANSFORMERS_AVAILABLE:
        raise RuntimeError(
            "transformers not available. Install optional group: uv sync --group vlm-gpu"
        )

    classifier = _get_classifier(device)

    if task == "multiclass":
        if samples is None:
            samples = collect_samples(split, limit=limit)
        if not samples:
            raise ValueError(f"No samples for split '{split}'")
        return _run_benchmark_multiclass(classifier, samples, split, preprocess)

    if samples is None:
        samples = collect_samples(split, limit=limit)
    if not samples:
        raise ValueError(f"No samples for split '{split}'")

    labels = MISOGYNY_LABELS  # ["yes", "no"]
    pipeline = PreprocessingPipeline() if preprocess else None
    results: list[ClassificationResult] = []
    ground_truths: list[str] = []
    sample_rows: list[dict[str, Any]] = []

    pbar = tqdm(total=len(samples), desc=f"visualbert/{preprocess or 'none'}", unit="img")
    for s in samples:
        arr = image_to_numpy(s["image"])
        if pipeline is not None and preprocess is not None:
            arr = pipeline.apply_transformation(arr, preprocess)

        result = classifier.classify(arr, labels, text=s.get("text"))
        results.append(result)

        gt = _misogynous_to_label(s["misogynous"])
        ground_truths.append(gt)
        sample_rows.append(
            {
                "image_id": s["image_id"],
                "ground_truth": yesno_to_int(gt),
                "prediction": yesno_to_int(result.prediction),
                "correct": result.prediction == gt,
                "refusal": result.refusal,
            }
        )
        pbar.update(1)
    pbar.close()

    return _aggregate(results, ground_truths, split, preprocess, labels, sample_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split",
        default="validation",
        help="Dataset split(s) to evaluate. Comma-separated for multiple: 'train,validation'"
        " (default: validation)",
    )
    parser.add_argument("--device", default="cpu", help="Torch device (default: cpu)")
    parser.add_argument(
        "--filters",
        default=None,
        help="Comma-separated filters (default: none). E.g. 'none,blur'",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--task",
        default="singleclass",
        choices=["singleclass", "multiclass"],
        help="'singleclass' = binary misogyny (default); 'multiclass' = 4-category MLM cloze",
    )
    args = parser.parse_args()

    # MAMI has no hidden visual content, so preprocessing filters do not help.
    # Default to "none"; pass --filters explicitly only for a deliberate ablation.
    filters_to_run = [f.strip() for f in args.filters.split(",")] if args.filters else ["none"]

    print(f"Split: {args.split} | Filters: {filters_to_run} | Limit: {args.limit}")
    print(f"Device: {args.device} | Task: {args.task}")

    # Load model once before collecting samples
    print("Pre-loading VisualBERT model …")
    _get_classifier(args.device)

    samples = collect_samples(args.split, limit=args.limit)
    print(f"Loaded {len(samples)} samples")

    all_results: list[dict[str, Any]] = []
    for flt in filters_to_run:
        print(f"\n--- Filter: {flt} ---")
        result = run_benchmark(
            split=args.split,
            device=args.device,
            preprocess=None if flt == "none" else flt,
            samples=samples,
            task=args.task,
        )
        all_results.append(result)
        acc = result.get("exact_match_accuracy", 0.0)
        print(f"  acc={acc:.3f}  refusals={result.get('refusal_rate', 0.0):.2%}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    split_slug = args.split.replace(",", "_")
    if args.task == "multiclass":
        out = RESULTS_DIR / f"visualbert_{split_slug}_multiclass.json"
    else:
        out = RESULTS_DIR / f"visualbert_{split_slug}.json"
    out.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"\nSaved {len(all_results)} filter rows to {out}")


if __name__ == "__main__":
    main()
