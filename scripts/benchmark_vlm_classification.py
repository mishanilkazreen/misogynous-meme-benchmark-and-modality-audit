"""
Orchestrator: benchmark all VLMs with preprocessing ablation on HatefulIllusion.

Outer loop: preprocessing filters from PreprocessingPipeline.TRANSFORMATIONS + "none".
Inner loop: model x subset x sample.

Writes results/vlm_classification.json with the full FR-3 schema from requirements.md.
Prints a summary table (model x filter x subset) to stdout.

Usage:
    uv run python scripts/benchmark_vlm_classification.py --model clip --subset digits --limit 10
    uv run python scripts/benchmark_vlm_classification.py --model clip,llava --subset all
    uv run python scripts/benchmark_vlm_classification.py --model all --subset digits --limit 20
"""

# ruff: noqa: I001  # datasets (via utils.dataset) must precede torch to avoid OpenMP segfault
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

from utils.dataset import DatasetManager
from models.vlm.clip_classifier import CLIPClassifier
from utils.preprocessing import PreprocessingPipeline

import numpy as np
import torch

SUBSET_NAMES = ["digits", "hate_slangs", "hate_symbols"]
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

ALL_MODELS = [
    "clip",
    "qwen2vl",
    "llava",
    "gpt4omini",
    "gemini",
]
GENERATIVE_MODELS = {"qwen2vl", "llava", "gpt4omini", "gemini"}

ALL_FILTERS = ["none", *PreprocessingPipeline.TRANSFORMATIONS]


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


_PIPELINE = PreprocessingPipeline()


def apply_filter(image: np.ndarray, filter_name: str) -> np.ndarray:
    if filter_name == "none":
        return image
    return _PIPELINE.apply_transformation(image, filter_name)


def compute_classification_metrics(
    predictions: list[str | None],
    ground_truths: list[str],
    labels: list[str],
) -> dict[str, float]:
    n = len(predictions)
    if n == 0:
        return {"exact_match_accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    correct = sum(1 for p, gt in zip(predictions, ground_truths, strict=True) if p == gt)
    accuracy = correct / n

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
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        per_class_prec.append(prec)
        per_class_rec.append(rec)

    macro_prec = sum(per_class_prec) / len(per_class_prec) if per_class_prec else 0.0
    macro_rec = sum(per_class_rec) / len(per_class_rec) if per_class_rec else 0.0
    macro_f1 = (
        2 * macro_prec * macro_rec / (macro_prec + macro_rec)
        if (macro_prec + macro_rec) > 0
        else 0.0
    )

    return {
        "exact_match_accuracy": accuracy,
        "precision": macro_prec,
        "recall": macro_rec,
        "f1": macro_f1,
    }


def build_visibility_block(
    predictions: list[str | None],
    ground_truths: list[str],
    visibility_scores: list[int],
    labels: list[str],
) -> dict[str, dict[str, Any]]:
    block: dict[str, dict[str, Any]] = {}
    for v in range(1, 6):
        indices = [i for i, vs in enumerate(visibility_scores) if vs == v]
        if not indices:
            block[str(v)] = {
                "exact_match_accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "num_images": 0,
            }
            continue
        v_preds = [predictions[i] for i in indices]
        v_gts = [ground_truths[i] for i in indices]
        metrics = compute_classification_metrics(v_preds, v_gts, labels)
        metrics["num_images"] = len(indices)  # type: ignore[assignment]
        block[str(v)] = metrics
    return block


def build_sample_rows(
    samples: list[dict[str, Any]],
    predictions: list[str | None],
    ground_truths: list[str],
    visibility_scores: list[int],
    confidences: list[float] | None = None,
) -> list[dict[str, Any]]:
    rows = []
    for i in range(len(samples)):
        row: dict[str, Any] = {
            "image_id": samples[i]["image_id"],
            "ground_truth": ground_truths[i],
            "prediction": predictions[i],
            "correct": predictions[i] == ground_truths[i],
            "visibility": visibility_scores[i],
        }
        if confidences is not None:
            row["confidence"] = round(float(confidences[i]), 4)
        rows.append(row)
    return rows


def run_clip(
    samples: list[dict[str, Any]],
    filter_name: str,
    device: str = "cpu",
) -> dict[str, Any]:
    labels = sorted({s["message"] for s in samples})
    images = [apply_filter(image_to_numpy(s["image"]), filter_name) for s in samples]
    ground_truths = [s["message"] for s in samples]
    visibility_scores = [int(s["visibility_score"]) for s in samples]
    subset = samples[0]["subset"] if len({s["subset"] for s in samples}) == 1 else "all"

    classifier = CLIPClassifier(device=device)
    classifier.set_classes(labels)

    t0 = time.perf_counter()
    raw_preds = classifier.predict_batch(images)
    total_time = time.perf_counter() - t0

    predictions: list[str | None] = [p for p, _ in raw_preds]
    confidences = [c for _, c in raw_preds]
    avg_latency = total_time / len(images)
    metrics = compute_classification_metrics(predictions, ground_truths, labels)
    by_visibility = build_visibility_block(predictions, ground_truths, visibility_scores, labels)

    return {
        "model": "clip",
        "filter": filter_name,
        "subset": subset,
        "exact_match_accuracy": metrics["exact_match_accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "avg_latency_s": avg_latency,
        "refusal_rate": 0.0,
        "by_visibility": by_visibility,
        "sample_predictions": build_sample_rows(
            samples, predictions, ground_truths, visibility_scores, confidences
        ),
    }


def run_generative_model(
    model_name: str, samples: list[dict[str, Any]], filter_name: str, device: str = "cpu"
) -> dict[str, Any] | None:
    """Dispatch to the appropriate generative model script."""
    subset = samples[0]["subset"] if len({s["subset"] for s in samples}) == 1 else "all"
    preprocess_arg = None if filter_name == "none" else filter_name

    if model_name == "qwen2vl":
        try:
            from scripts.benchmark_qwen2vl import run_benchmark  # type: ignore[import,assignment]

            return run_benchmark(  # type: ignore[call-arg]
                subset=subset, preprocess=preprocess_arg, samples=samples, device=device
            )
        except Exception as exc:
            print(f"  Skipping qwen2vl: {exc}")
            return None

    if model_name == "llava":
        try:
            from scripts.benchmark_llava import run_benchmark  # type: ignore[import,assignment]

            return run_benchmark(  # type: ignore[call-arg]
                subset=subset, preprocess=preprocess_arg, samples=samples, device=device
            )
        except Exception as exc:
            print(f"  Skipping llava: {exc}")
            return None

    if model_name == "gemini":
        try:
            from scripts.benchmark_gemini import run_benchmark  # type: ignore[import,assignment]

            return run_benchmark(subset=subset, preprocess=preprocess_arg, samples=samples)
        except Exception as exc:
            print(f"  Skipping gemini: {exc}")
            return None

    if model_name == "gpt4omini":
        try:
            from scripts.benchmark_gpt4omini import run_benchmark  # type: ignore[import,assignment]

            return run_benchmark(subset=subset, preprocess=preprocess_arg, samples=samples)
        except Exception as exc:
            print(f"  Skipping gpt4omini: {exc}")
            return None

    return None


def print_sample_predictions(result: dict[str, Any], n: int = 10) -> None:
    rows = result.get("sample_predictions", [])[:n]
    if not rows:
        return
    model = result["model"]
    flt = result["filter"]
    print(f"\n  Sample predictions ({model} | filter={flt}):")
    has_conf = "confidence" in rows[0]
    if has_conf:
        print(
            f"  {'image_id':<30} {'ground_truth':<15} {'prediction':<15} {'conf':<7} {'ok':<5} {'vis'}"
        )
        print("  " + "-" * 80)
        for r in rows:
            ok = "Y" if r["correct"] else "N"
            pred = str(r["prediction"]) if r["prediction"] is not None else "(none)"
            print(
                f"  {r['image_id']:<30} {r['ground_truth']:<15} {pred:<15}"
                f" {r['confidence']:<7} {ok:<5} {r['visibility']}"
            )
    else:
        print(f"  {'image_id':<30} {'ground_truth':<15} {'prediction':<15} {'ok':<5} {'vis'}")
        print("  " + "-" * 72)
        for r in rows:
            ok = "Y" if r["correct"] else "N"
            pred = str(r["prediction"]) if r["prediction"] is not None else "(none)"
            print(
                f"  {r['image_id']:<30} {r['ground_truth']:<15} {pred:<15} {ok:<5} {r['visibility']}"
            )


def print_summary(all_results: list[dict[str, Any]]) -> None:
    print("\n" + "=" * 80)
    print(f"{'Model':<20} {'Filter':<20} {'Subset':<15} {'Acc':<8} {'F1':<8}")
    print("=" * 80)
    for r in all_results:
        acc = f"{r['exact_match_accuracy']:.3f}" if "exact_match_accuracy" in r else "n/a"
        f1_val = r.get("f1")
        f1 = f"{f1_val:.3f}" if isinstance(f1_val, float) else "n/a"
        print(f"{r['model']:<20} {r['filter']:<20} {r['subset']:<15} {acc:<8} {f1:<8}")
    print("=" * 80)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="clip",
        help="Comma-separated model names or 'all'. Choices: " + ", ".join(ALL_MODELS),
    )
    parser.add_argument(
        "--subset",
        default="digits",
        choices=["digits", "hate_slangs", "hate_symbols", "all"],
    )
    parser.add_argument("--limit", type=int, default=None, help="Cap images per subset")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--filters",
        default=None,
        help="Comma-separated filter names to run (default: all filters)",
    )
    args = parser.parse_args()

    models_requested = (
        ALL_MODELS if args.model == "all" else [m.strip() for m in args.model.split(",")]
    )
    filters_to_run = [f.strip() for f in args.filters.split(",")] if args.filters else ALL_FILTERS

    print(f"Models: {models_requested}")
    print(f"Filters: {filters_to_run}")
    print(f"Subset: {args.subset}, Limit: {args.limit}")

    samples = collect_samples(args.subset, limit=args.limit)
    if not samples:
        raise SystemExit(f"No samples found for subset '{args.subset}'")
    print(f"Loaded {len(samples)} samples")

    all_results: list[dict[str, Any]] = []

    for filter_name in filters_to_run:
        print(f"\n--- Filter: {filter_name} ---")
        for model_name in models_requested:
            print(f"  Running {model_name} …")
            try:
                if model_name == "clip":
                    result: dict[str, Any] | None = run_clip(
                        samples, filter_name, device=args.device
                    )
                else:
                    result = run_generative_model(
                        model_name, samples, filter_name, device=args.device
                    )
                if result is not None:
                    all_results.append(result)
                    print_sample_predictions(result)
            except Exception as exc:
                print(f"  ERROR running {model_name} with filter={filter_name}: {exc}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "vlm_classification.json"
    out_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"\nWrote {len(all_results)} result rows to {out_path}")

    print_summary(all_results)


if __name__ == "__main__":
    main()
