"""
Benchmark CLIP zero-shot classifier on HatefulIllusion.

Labels are derived from the dataset's own `message` values — no hardcoding.
Results mirror the yolo_benchmark.json schema so both files can be joined in
task 7's comparison table.

Usage:
    uv run python scripts/benchmark_vlm.py --model clip --subset digits
    uv run python scripts/benchmark_vlm.py --model clip --subset all
    uv run python scripts/benchmark_vlm.py --model clip --subset hate_symbols --preprocess blur
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from models.vlm.clip_classifier import CLIPClassifier
from utils.dataset import DatasetManager
from utils.preprocessing import PreprocessingPipeline

SUBSET_NAMES = ["digits", "hate_slangs", "hate_symbols"]
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
CONFIDENCE_FLOOR = 1e-6


def results_path(subset: str, preprocess: str | None) -> Path:
    suffix = f"_{preprocess}" if preprocess else ""
    return RESULTS_DIR / f"clip_benchmark_{subset}{suffix}.json"


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


def collect_samples(subset: str, split: str = "train") -> list[dict[str, Any]]:
    subsets = SUBSET_NAMES if subset == "all" else [subset]
    manager = DatasetManager()
    samples: list[dict[str, Any]] = []
    for subset_name in subsets:
        dataset = manager.load_dataset(split=split, subset=subset_name)
        for index in range(len(dataset)):
            sample = dataset[index]
            sample["subset"] = subset_name
            sample["image_id"] = f"{subset_name}_{sample['image_id']}"
            samples.append(sample)
    return samples


def compute_vlm_metrics(
    predictions: list[tuple[str, float]],
    ground_truths: list[str],
    n_classes: int,
) -> dict[str, float]:
    """Compute exact_match_accuracy and above_chance_rate.

    above_chance_rate: fraction of images where CLIP confidence exceeded random chance
    (1/n_classes). Measures model confidence, not correctness — use exact_match_accuracy
    for accuracy comparisons. Comparable to Rekognition's any_detection_recall.
    """
    assert len(predictions) == len(ground_truths)
    if not predictions:
        return {"exact_match_accuracy": 0.0, "above_chance_rate": 0.0}

    chance_threshold = 1.0 / max(n_classes, 1)
    exact_hits = sum(
        1
        for (pred_label, _), gt in zip(predictions, ground_truths, strict=True)
        if pred_label == gt
    )
    above_chance = sum(1 for _, conf in predictions if conf > chance_threshold)

    return {
        "exact_match_accuracy": exact_hits / len(predictions),
        "above_chance_rate": above_chance / len(predictions),
    }


def build_visibility_metrics(
    predictions: list[tuple[str, float]],
    ground_truths: list[str],
    visibility_scores: list[int],
    n_classes: int,
) -> dict[str, dict[str, float]]:
    scores_seen: dict[int, list[int]] = {}
    for i, v in enumerate(visibility_scores):
        scores_seen.setdefault(v, []).append(i)

    metrics_by_visibility: dict[str, dict[str, float]] = {}
    for v, indices in scores_seen.items():
        v_preds = [predictions[i] for i in indices]
        v_gts = [ground_truths[i] for i in indices]
        metrics_by_visibility[str(v)] = compute_vlm_metrics(v_preds, v_gts, n_classes)
    return metrics_by_visibility


def run_benchmark(
    model_name: str,
    subset: str,
    clip_model: str = "ViT-L-14",
    pretrained: str = "openai",
    device: str = "cpu",
    preprocess: str | None = None,
) -> dict[str, Any]:
    samples = collect_samples(subset)
    if not samples:
        raise ValueError(f"No samples found for subset '{subset}'")

    labels = sorted({s["message"] for s in samples})
    n_classes = len(labels)
    print(f"Label set ({n_classes} classes): {labels[:10]}{'...' if n_classes > 10 else ''}")

    pipeline = PreprocessingPipeline() if preprocess else None
    images: list[np.ndarray] = []
    for s in samples:
        arr = image_to_numpy(s["image"])
        if pipeline is not None and preprocess is not None:
            arr = pipeline.apply_transformation(arr, preprocess)
        images.append(arr)

    ground_truths = [s["message"] for s in samples]
    visibility_scores = [int(s["visibility_score"]) for s in samples]

    print(f"Loading CLIP {clip_model} ({pretrained}) on {device} …")
    classifier = CLIPClassifier(model_name=clip_model, pretrained=pretrained, device=device)
    classifier.set_classes(labels)

    print(f"Running inference on {len(images)} images …")
    predictions, total_time = classifier.timed_predict_batch(images)
    average_time = total_time / len(images)

    computed_metrics = compute_vlm_metrics(predictions, ground_truths, n_classes)
    visibility_metrics = build_visibility_metrics(
        predictions, ground_truths, visibility_scores, n_classes
    )

    sample_records = [
        {
            "image_id": samples[i]["image_id"],
            "subset": samples[i]["subset"],
            "ground_truth": ground_truths[i],
            "predicted": predictions[i][0],
            "confidence": round(predictions[i][1], 4),
            "correct": predictions[i][0] == ground_truths[i],
            "visibility_score": visibility_scores[i],
            "prompt": samples[i].get("prompt", ""),
        }
        for i in range(
            len(samples)
        )  # all samples; downstream scripts must not assume a fixed count
    ]

    return {
        "benchmark_date": datetime.now(timezone.utc).isoformat(),
        "subset": subset,
        "preprocess": preprocess,
        "models": {
            model_name: {
                "clip_model": clip_model,
                "pretrained": pretrained,
                "num_images": len(samples),
                "num_classes": n_classes,
                "average_inference_time_s": average_time,
                "total_inference_time_s": total_time,
                "computed_metrics": computed_metrics,
                "visibility_metrics": visibility_metrics,
                "sample_predictions": sample_records,
            }
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="clip",
        choices=["clip"],
        help="VLM model to benchmark",
    )
    parser.add_argument(
        "--subset",
        default="digits",
        choices=["digits", "hate_slangs", "hate_symbols", "all"],
        help="HatefulIllusion subset to evaluate on",
    )
    parser.add_argument(
        "--clip-model",
        default="ViT-L-14",
        help="open_clip model architecture name",
    )
    parser.add_argument(
        "--pretrained",
        default="openai",
        help="open_clip pretrained weights tag",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="torch device string, e.g. cpu or cuda",
    )
    parser.add_argument(
        "--preprocess",
        default=None,
        choices=PreprocessingPipeline.TRANSFORMATIONS,
        help="Preprocessing filter to apply before inference",
    )
    args = parser.parse_args()

    results = run_benchmark(
        model_name=args.model,
        subset=args.subset,
        clip_model=args.clip_model,
        pretrained=args.pretrained,
        device=args.device,
        preprocess=args.preprocess,
    )

    out_path = results_path(args.subset, args.preprocess)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved benchmark results to {out_path}")


if __name__ == "__main__":
    main()
