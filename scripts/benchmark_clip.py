"""
Benchmark CLIP on HatefulIllusion (closed-set zero-shot classification).

Usage:
    uv run python scripts/benchmark_clip.py --subset digits --limit 10 --device cuda
    uv run python scripts/benchmark_clip.py --subset digits --device cuda
"""

# ruff: noqa: I001  # datasets (via utils.dataset) must precede torch to avoid OpenMP segfault
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any

# utils.dataset (datasets) must be imported before models.vlm.clip_classifier (open_clip/torch)
# to avoid an OpenMP double-initialisation segfault.
from utils.dataset import DatasetManager
from models.vlm.clip_classifier import CLIPClassifier
from utils.preprocessing import PreprocessingPipeline

import numpy as np
import torch
from tqdm import tqdm

SUBSET_NAMES = ["digits", "hate_slangs", "hate_symbols"]
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


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


def collect_samples(
    subset: str, limit: int | None = None, split: str = "train"
) -> list[dict[str, Any]]:
    subsets = SUBSET_NAMES if subset == "all" else [subset]
    manager = DatasetManager()
    samples: list[dict[str, Any]] = []
    for subset_name in subsets:
        dataset = manager.load_dataset(split=split, subset=subset_name)
        count = min(len(dataset), limit) if limit is not None else len(dataset)
        for index in range(count):
            sample = dataset[index]
            sample["subset"] = subset_name
            sample["image_id"] = f"{subset_name}_{sample['image_id']}"
            samples.append(sample)
    return samples


def run_benchmark(
    subset: str,
    preprocess: str | None = None,
    limit: int | None = None,
    device: str = "cpu",
    samples: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if samples is None:
        samples = collect_samples(subset, limit=limit)
    if not samples:
        raise ValueError(f"No samples for subset '{subset}'")

    labels_by_subset_raw: dict[str, set[str]] = {}
    for s in samples:
        labels_by_subset_raw.setdefault(s["subset"], set()).add(s["message"])
    all_labels = sorted({lbl for lbls in labels_by_subset_raw.values() for lbl in lbls})

    pipeline = PreprocessingPipeline() if preprocess else None
    images = []
    for s in tqdm(samples, desc=f"clip/{preprocess or 'none'}", unit="img"):
        arr = image_to_numpy(s["image"])
        if pipeline is not None and preprocess is not None:
            arr = pipeline.apply_transformation(arr, preprocess)
        images.append(arr)

    ground_truths = [s["message"] for s in samples]
    visibility_scores = [int(s["visibility_score"]) for s in samples]

    classifier = CLIPClassifier(device=device)
    classifier.set_classes(all_labels)

    t0 = time.perf_counter()
    raw_preds = classifier.predict_batch(images)
    total_time = time.perf_counter() - t0

    predictions: list[str | None] = [p for p, _ in raw_preds]
    confidences = [c for _, c in raw_preds]
    avg_latency = total_time / len(images)
    n_total = len(predictions)

    correct_all = sum(1 for p, gt in zip(predictions, ground_truths, strict=True) if p == gt)
    accuracy = correct_all / n_total if n_total else 0.0

    per_class_prec: list[float] = []
    per_class_rec: list[float] = []
    for label in all_labels:
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

    by_visibility: dict[str, dict[str, Any]] = {}
    for v in range(1, 6):
        indices = [i for i, vs in enumerate(visibility_scores) if vs == v]
        v_preds = [predictions[i] for i in indices]
        v_gts = [ground_truths[i] for i in indices]
        v_n = len(v_preds)
        v_correct = sum(1 for p, gt in zip(v_preds, v_gts, strict=True) if p == gt)
        by_visibility[str(v)] = {
            "exact_match_accuracy": v_correct / v_n if v_n else 0.0,
            "num_images": v_n,
        }

    sample_predictions = [
        {
            "image_id": samples[i]["image_id"],
            "ground_truth": ground_truths[i],
            "prediction": predictions[i],
            "confidence": round(float(confidences[i]), 4),
            "correct": predictions[i] == ground_truths[i],
            "visibility": visibility_scores[i],
        }
        for i in range(len(samples))
    ]

    return {
        "benchmark_date": datetime.now(timezone.utc).isoformat(),
        "model": "clip",
        "filter": preprocess or "none",
        "subset": subset,
        "exact_match_accuracy": accuracy,
        "precision": macro_prec,
        "recall": macro_rec,
        "f1": macro_f1,
        "avg_latency_s": avg_latency,
        "refusal_rate": 0.0,
        "by_visibility": by_visibility,
        "sample_predictions": sample_predictions,
    }


def print_samples(result: dict[str, Any], n: int = 10) -> None:
    rows = result.get("sample_predictions", [])[:n]
    flt = result["filter"]
    print(f"\n  filter={flt}  acc={result['exact_match_accuracy']:.3f}  f1={result['f1']:.3f}")
    print(
        f"  {'image_id':<30} {'ground_truth':<15} {'prediction':<15} {'conf':<7} {'ok':<5} {'vis'}"
    )
    print("  " + "-" * 76)
    for r in rows:
        ok = "Y" if r["correct"] else "N"
        pred = str(r["prediction"]) if r["prediction"] is not None else "(none)"
        print(
            f"  {r['image_id']:<30} {r['ground_truth']:<15} {pred:<15} {r['confidence']:<7} {ok:<5} {r['visibility']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--subset",
        default="digits",
        choices=["digits", "hate_slangs", "hate_symbols", "all"],
    )
    parser.add_argument(
        "--filters",
        default=None,
        help="Comma-separated filters to run (default: all). E.g. 'none,blur,grayscale'",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    filters_to_run = (
        [f.strip() for f in args.filters.split(",")]
        if args.filters
        else ["none", *PreprocessingPipeline.TRANSFORMATIONS]
    )

    print(f"Subset: {args.subset} | Filters: {filters_to_run} | Limit: {args.limit}")
    samples = collect_samples(args.subset, limit=args.limit)
    print(f"Loaded {len(samples)} samples")

    all_results: list[dict[str, Any]] = []
    for flt in filters_to_run:
        print(f"\n--- Filter: {flt} ---")
        result = run_benchmark(
            subset=args.subset,
            preprocess=None if flt == "none" else flt,
            device=args.device,
            samples=samples,
        )
        all_results.append(result)
        print_samples(result)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"clip_{args.subset}.json"
    out.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"\nSaved {len(all_results)} filter rows to {out}")


if __name__ == "__main__":
    main()
