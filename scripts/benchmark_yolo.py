"""
Benchmark standard Ultralytics YOLO detectors on HatefulIllusion.

"Standard" here means the YOLO is trained on a fixed class list
(e.g. COCO) and fine-tuned on the target dataset, in contrast with
the text-prompted VLM detectors in `benchmark_vlm.py`.

Target models (per paper plan, task 3):
    yolov8n.pt, yolov10n.pt, yolo11n.pt, yolo12n.pt, yolo26n.pt

Metrics reported: mAP50, mAP50-95, precision, recall, F1, inference time,
stratified by visibility level (1-5) and subset (digits / hate_slangs /
hate_symbols).

Usage:
    uv run python scripts/benchmark_yolo.py --model yolov8n.pt --subset digits
    uv run python scripts/benchmark_yolo.py --all

Docs: https://docs.ultralytics.com/modes/val/
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
import torch
import yaml

from models.yolo.metrics import DetectionPrediction, GroundTruthBox, compute_detection_metrics
from models.yolo.wrapper import UltralyticsYOLO
from utils.dataset import DatasetManager

MODEL_CHECKPOINTS = [
    "yolov8n.pt",
    "yolov10n.pt",
    "yolo11n.pt",
    "yolo12n.pt",
    "yolo26n.pt",
]

SUBSET_NAMES = ["digits", "hate_slangs", "hate_symbols"]
RESULTS_PATH = Path(__file__).resolve().parents[1] / "results" / "yolo_benchmark.json"


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


def collect_samples(subset: str) -> list[dict[str, Any]]:
    subsets = SUBSET_NAMES if subset == "all" else [subset]
    manager = DatasetManager()
    samples: list[dict[str, Any]] = []
    for subset_name in subsets:
        dataset = manager.load_dataset(split="train", subset=subset_name)
        for index in range(len(dataset)):
            sample = dataset[index]
            sample["subset"] = subset_name
            sample["image_id"] = f"{subset_name}_{sample['image_id']}"
            samples.append(sample)
    return samples


def prepare_yolo_validation_dataset(samples: list[dict[str, Any]], output_dir: Path) -> Path:
    images_dir = output_dir / "images"
    labels_dir = output_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    for sample in samples:
        image_id = sample["image_id"]
        image = image_to_numpy(sample["image"])
        target_image = images_dir / f"{image_id}.png"
        if not target_image.exists():
            from PIL import Image

            Image.fromarray(image).save(target_image)

        label_path = labels_dir / f"{image_id}.txt"
        label_path.write_text("0 0.5 0.5 1.0 1.0\n", encoding="utf-8")

    data_yaml = output_dir / "data.yaml"
    with data_yaml.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            {
                "names": ["embedded_hateful_content"],
                "nc": 1,
                "val": str(images_dir),
            },
            handle,
            sort_keys=False,
        )
    return data_yaml


def build_ground_truths(samples: list[dict[str, Any]]) -> list[GroundTruthBox]:
    gts: list[GroundTruthBox] = []
    for sample in samples:
        image = sample["image"]
        if isinstance(image, torch.Tensor):
            height, width = int(image.shape[1]), int(image.shape[2])
        elif isinstance(image, np.ndarray):
            height, width = int(image.shape[0]), int(image.shape[1])
        else:
            raise ValueError("Unsupported image type for ground truth build")
        gts.append(
            GroundTruthBox(
                image_id=sample["image_id"],
                bbox=(0.0, 0.0, float(width), float(height)),
            )
        )
    return gts


def collect_predictions(
    model: UltralyticsYOLO,
    samples: list[dict[str, Any]],
) -> tuple[list[DetectionPrediction], float]:
    images = [image_to_numpy(sample["image"]) for sample in samples]
    start = datetime.now()
    results = model.predict(source=images, imgsz=640, save=False, verbose=False)
    elapsed = (datetime.now() - start).total_seconds()

    predictions: list[DetectionPrediction] = []
    for sample, result in zip(samples, results, strict=False):
        if result.boxes is None or len(result.boxes) == 0:
            continue
        xyxy = result.boxes.xyxy
        xyxy = xyxy.cpu().numpy() if hasattr(xyxy, "cpu") else np.asarray(xyxy)
        conf = result.boxes.conf
        conf = conf.cpu().numpy() if hasattr(conf, "cpu") else np.asarray(conf)
        for bbox, confidence in zip(xyxy, conf, strict=False):
            predictions.append(
                DetectionPrediction(
                    image_id=sample["image_id"],
                    bbox=(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])),
                    confidence=float(confidence),
                )
            )
    return predictions, elapsed


def build_visibility_metrics(
    predictions: list[DetectionPrediction],
    samples: list[dict[str, Any]],
    ground_truths: list[GroundTruthBox],
) -> dict[str, dict[str, float]]:
    metrics_by_visibility: dict[str, dict[str, float]] = {}
    sample_ids_by_visibility: dict[int, list[str]] = {}
    for sample in samples:
        visibility = int(sample["visibility_score"])
        sample_ids_by_visibility.setdefault(visibility, []).append(sample["image_id"])

    for visibility, image_ids in sample_ids_by_visibility.items():
        visibility_gts = [gt for gt in ground_truths if gt.image_id in image_ids]
        visibility_preds = [pred for pred in predictions if pred.image_id in image_ids]
        metrics_by_visibility[str(visibility)] = compute_detection_metrics(
            visibility_preds, visibility_gts
        )
    return metrics_by_visibility


def run_benchmark(models: list[str], subset: str) -> dict[str, Any]:
    samples = collect_samples(subset)
    if not samples:
        raise ValueError(f"No samples found for subset '{subset}'")

    results: dict[str, Any] = {
        "benchmark_date": datetime.utcnow().isoformat() + "Z",
        "subset": subset,
        "models": {},
    }

    with tempfile.TemporaryDirectory(prefix="yolo_benchmark_") as temp_dir:
        temp_path = Path(temp_dir)
        validation_data = prepare_yolo_validation_dataset(samples, temp_path)
        ground_truths = build_ground_truths(samples)

        for checkpoint in models:
            print(f"Evaluating {checkpoint} on subset {subset} ({len(samples)} images)")
            model = UltralyticsYOLO(checkpoint=checkpoint, device="cpu", verbose=False)
            val_metrics: dict[str, float | str] = {}
            try:
                val_out = model.val(data=validation_data, imgsz=640, batch=16)
                val_metrics = {
                    "mAP50": float(getattr(val_out.box, "map", 0.0)),
                    "mAP50-95": float(getattr(val_out.box, "map50_95", 0.0)),
                    "precision": float(getattr(val_out.box, "precision", 0.0)),
                    "recall": float(getattr(val_out.box, "recall", 0.0)),
                    "f1": float(getattr(val_out.box, "f1", 0.0)),
                }
            except Exception as exc:  # pragma: no cover
                val_metrics = {"error": str(exc)}

            predictions, inference_time = collect_predictions(model, samples)
            average_time = inference_time / len(samples)
            computed_metrics = compute_detection_metrics(predictions, ground_truths)
            visibility_metrics = build_visibility_metrics(predictions, samples, ground_truths)

            results["models"][checkpoint] = {
                "num_images": len(samples),
                "average_inference_time_s": average_time,
                "total_inference_time_s": inference_time,
                "val_metrics": val_metrics,
                "computed_metrics": computed_metrics,
                "visibility_metrics": visibility_metrics,
            }

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        choices=MODEL_CHECKPOINTS,
        help="Ultralytics checkpoint name, e.g. yolov8n.pt, yolo26n.pt",
    )
    parser.add_argument(
        "--subset",
        default="digits",
        choices=["digits", "hate_slangs", "hate_symbols", "all"],
        help="HatefulIllusion subset to evaluate on",
    )
    parser.add_argument("--all", action="store_true", help="Benchmark all five models")
    args = parser.parse_args()

    models = MODEL_CHECKPOINTS if args.all else [args.model]
    benchmark_results = run_benchmark(models=models, subset=args.subset)

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(benchmark_results, indent=2), encoding="utf-8")
    print(f"Saved benchmark results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
