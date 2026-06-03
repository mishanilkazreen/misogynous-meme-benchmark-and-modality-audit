"""
Benchmark YOLO-World text-prompted detector on HatefulIllusion.

YOLO-World encodes free-text class prompts with a CLIP text encoder and
localises matching regions at inference time — no retraining needed.
The metric is any_detection_recall (did any box fire?) stratified by
visibility_score, matching the task-3 YOLO benchmark schema.

Reference: Cheng et al. (2024) arXiv:2401.17270

Usage:
    uv run python scripts/benchmark_yolo_world.py --subset digits
    uv run python scripts/benchmark_yolo_world.py --subset all --device cuda
    uv run python scripts/benchmark_yolo_world.py --subset digits --conf 0.1
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

from models.vlm.yolo_world import YOLOWorldWrapper
from utils.dataset import DatasetManager
from utils.preprocessing import PreprocessingPipeline

SUBSET_NAMES = ["digits", "hate_slangs", "hate_symbols"]
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

YOLO_WORLD_CLASSES = [
    "hidden digit",
    "hidden number",
    "hate symbol",
    "hateful text",
    "extremist symbol",
    "hidden slogan",
]

DEFAULT_CHECKPOINT = "yolov8s-worldv2.pt"


def results_path(subset: str, preprocess: Optional[str]) -> Path:
    suffix = f"_{preprocess}" if preprocess else ""
    return RESULTS_DIR / f"yolo_world_benchmark_{subset}{suffix}.json"


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


def compute_any_detection_recall(fired_flags: list[bool]) -> float:
    if not fired_flags:
        return 0.0
    return sum(fired_flags) / len(fired_flags)


def build_visibility_metrics(
    fired_flags: list[bool],
    visibility_scores: list[int],
) -> dict[str, dict[str, float]]:
    indices_by_visibility: dict[int, list[int]] = {}
    for i, v in enumerate(visibility_scores):
        indices_by_visibility.setdefault(v, []).append(i)

    metrics_by_visibility: dict[str, dict[str, float]] = {}
    for v, indices in indices_by_visibility.items():
        v_flags = [fired_flags[i] for i in indices]
        metrics_by_visibility[str(v)] = {
            "any_detection_recall": compute_any_detection_recall(v_flags),
            "num_images": len(v_flags),
        }
    return metrics_by_visibility


def run_benchmark(
    subset: str,
    checkpoint: str = DEFAULT_CHECKPOINT,
    device: str = "cpu",
    conf: float = 0.25,
    preprocess: Optional[str] = None,
) -> dict[str, Any]:
    samples = collect_samples(subset)
    if not samples:
        raise ValueError(f"No samples found for subset '{subset}'")

    pipeline = PreprocessingPipeline() if preprocess else None
    images: list[np.ndarray] = []
    for s in samples:
        arr = image_to_numpy(s["image"])
        if pipeline is not None and preprocess is not None:
            arr = pipeline.apply_transformation(arr, preprocess)
        images.append(arr)

    visibility_scores = [int(s["visibility_score"]) for s in samples]

    print(f"Loading YOLO-World ({checkpoint}) on {device} …")
    detector = YOLOWorldWrapper(checkpoint=checkpoint, device=device)
    # set_classes called once — text encoding happens here, not per-image
    detector.set_classes(YOLO_WORLD_CLASSES)
    print(f"Prompt classes: {YOLO_WORLD_CLASSES}")

    print(f"Running inference on {len(images)} images (conf={conf}) …")
    fired_flags, total_time = detector.predict_batch(images, conf=conf)
    average_time = total_time / len(images)

    overall_recall = compute_any_detection_recall(fired_flags)
    visibility_metrics = build_visibility_metrics(fired_flags, visibility_scores)

    sample_records = [
        {
            "image_id": samples[i]["image_id"],
            "subset": samples[i]["subset"],
            "ground_truth": samples[i]["message"],
            "fired": fired_flags[i],
            "visibility_score": visibility_scores[i],
        }
        for i in range(len(samples))
    ]

    return {
        "benchmark_date": datetime.now(timezone.utc).isoformat(),
        "subset": subset,
        "preprocess": preprocess,
        "models": {
            "yolo_world": {
                "checkpoint": checkpoint,
                "conf_threshold": conf,
                "prompt_classes": YOLO_WORLD_CLASSES,
                "num_images": len(samples),
                "average_inference_time_s": average_time,
                "total_inference_time_s": total_time,
                "any_detection_recall": overall_recall,
                "visibility_metrics": visibility_metrics,
                "sample_predictions": sample_records,
            }
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--subset",
        default="digits",
        choices=["digits", "hate_slangs", "hate_symbols", "all"],
        help="HatefulIllusion subset to evaluate on",
    )
    parser.add_argument(
        "--checkpoint",
        default=DEFAULT_CHECKPOINT,
        help="YOLO-World checkpoint file (downloaded automatically if absent)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="torch device string, e.g. cpu or cuda",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold for detections",
    )
    parser.add_argument(
        "--preprocess",
        default=None,
        choices=PreprocessingPipeline.TRANSFORMATIONS,
        help="Preprocessing filter to apply before inference",
    )
    args = parser.parse_args()

    results = run_benchmark(
        subset=args.subset,
        checkpoint=args.checkpoint,
        device=args.device,
        conf=args.conf,
        preprocess=args.preprocess,
    )

    out_path = results_path(args.subset, args.preprocess)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved benchmark results to {out_path}")


if __name__ == "__main__":
    main()
