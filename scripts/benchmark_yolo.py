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

Usage examples:
    uv run python scripts/benchmark_yolo.py --mode pretrained --model yolov8n.pt --subset digits
    uv run python scripts/benchmark_yolo.py --mode pretrained --all
    uv run python scripts/benchmark_yolo.py --mode trained --model yolov8n.pt --weights-type best --subset digits
    uv run python scripts/benchmark_yolo.py --mode trained --all --weights-type last
    uv run python scripts/benchmark_yolo.py --weights runs/detect/results/trained_models/train_yolov8n/weights/best.pt --subset digits

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
TRAINED_WEIGHTS_TYPES = ["best", "last"]
DEFAULT_TRAINED_RESULTS_DIR = (
    Path(__file__).resolve().parents[1] / "runs" / "detect" / "results" / "trained_models"
)

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


def get_trained_checkpoint_path(model_name: str, weights_type: str, trained_dir: Path) -> Path:
    stem = Path(model_name).stem
    if stem.startswith("train_"):
        stem = stem.replace("train_", "")
    trained_path = trained_dir / f"train_{stem}" / "weights" / f"{weights_type}.pt"
    if not trained_path.exists():
        raise FileNotFoundError(f"Trained weights not found for {model_name}: {trained_path}")
    return trained_path


def resolve_model_checkpoints(
    model: str,
    all_models: bool,
    mode: str,
    weights_type: str,
    trained_dir: Path,
    explicit_weights: str | None = None,
) -> list[str]:
    if explicit_weights is not None:
        if all_models:
            raise ValueError("--weights cannot be combined with --all")
        return [str(Path(explicit_weights).resolve())]

    if mode == "pretrained":
        return MODEL_CHECKPOINTS if all_models else [model]

    if mode == "trained":
        if all_models:
            return [
                str(get_trained_checkpoint_path(checkpoint, weights_type, trained_dir))
                for checkpoint in MODEL_CHECKPOINTS
            ]
        return [str(get_trained_checkpoint_path(model, weights_type, trained_dir))]

    raise ValueError(f"Unsupported mode: {mode}")


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
        "--mode",
        default="pretrained",
        choices=["pretrained", "trained"],
        help="Benchmark mode: pretrained hub checkpoint or trained local weights",
    )
    parser.add_argument(
        "--weights-type",
        default="best",
        choices=TRAINED_WEIGHTS_TYPES,
        help="When --mode trained, use either best or last trained weights",
    )
    parser.add_argument(
        "--weights",
        help="Explicit path to a single weights file (best.pt/last.pt or any .pt file)",
    )
    parser.add_argument(
        "--trained-dir",
        type=Path,
        default=DEFAULT_TRAINED_RESULTS_DIR,
        help="Base folder containing train_<model>/weights/<best|last>.pt",
    )
    parser.add_argument(
        "--subset",
        default="digits",
        choices=["digits", "hate_slangs", "hate_symbols", "all"],
        help="HatefulIllusion subset to evaluate on",
    )
    parser.add_argument("--all", action="store_true", help="Benchmark all five models")
    args = parser.parse_args()

    models = resolve_model_checkpoints(
        model=args.model,
        all_models=args.all,
        mode=args.mode,
        weights_type=args.weights_type,
        trained_dir=args.trained_dir,
        explicit_weights=args.weights,
    )
    benchmark_results = run_benchmark(models=models, subset=args.subset)

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(benchmark_results, indent=2), encoding="utf-8")
    print(f"Saved benchmark results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
