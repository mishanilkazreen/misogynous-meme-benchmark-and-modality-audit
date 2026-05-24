"""
Benchmark standard Ultralytics YOLO detectors on HatefulIllusion.

"Standard" here means the YOLO is trained on a fixed class list
(e.g. COCO) and fine-tuned on the target dataset, in contrast with
the text-prompted VLM detectors in `benchmark_vlm.py`.

Target models (per paper plan, task 3):
    yolov8n.pt, yolov10n.pt, yolo11n.pt, yolo12n.pt, yolo26n.pt

Metrics reported: mAP50, mAP50-95, precision, recall, F1, inference time,
stratified by visibility level and subset (digits / hate_slangs / hate_symbols).
HatefulIllusion has no bounding-box annotations, so a full-image proxy box
covering the entire image is used as the ground-truth for every sample.

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
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from models.yolo.metrics import DetectionPrediction, GroundTruthBox, compute_detection_metrics
from models.yolo.wrapper import UltralyticsYOLO
from utils.dataset import DatasetManager
from utils.preprocessing import PreprocessingPipeline

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


def build_ground_truths(samples: list[dict[str, Any]]) -> list[GroundTruthBox]:
    gts: list[GroundTruthBox] = []
    for sample in samples:
        image = sample["image"]
        if isinstance(image, torch.Tensor):
            height, width = int(image.shape[1]), int(image.shape[2])
        elif isinstance(image, np.ndarray):
            height, width = int(image.shape[0]), int(image.shape[1])
        else:
            arr = image_to_numpy(image)
            height, width = int(arr.shape[0]), int(arr.shape[1])
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
    preprocess: str | None = None,
) -> tuple[list[DetectionPrediction], float]:
    pipeline = PreprocessingPipeline() if preprocess else None
    images: list[np.ndarray] = []
    for s in samples:
        arr = image_to_numpy(s["image"])
        if pipeline is not None and preprocess is not None:
            arr = pipeline.apply_transformation(arr, preprocess)
        images.append(arr)
    results, elapsed = model.timed_predict(source=images, imgsz=640, save=False, verbose=False)

    predictions: list[DetectionPrediction] = []
    for sample, result in zip(samples, results, strict=True):
        if result.boxes is None or len(result.boxes) == 0:
            continue
        xyxy = result.boxes.xyxy
        xyxy = xyxy.cpu().numpy() if hasattr(xyxy, "cpu") else np.asarray(xyxy)
        conf = result.boxes.conf
        conf = conf.cpu().numpy() if hasattr(conf, "cpu") else np.asarray(conf)
        for bbox, confidence in zip(xyxy, conf, strict=True):
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
    ground_truths: list[GroundTruthBox],
    samples: list[dict[str, Any]],
) -> dict[str, dict[str, float]]:
    metrics_by_visibility: dict[str, dict[str, float]] = {}
    sample_ids_by_visibility: dict[int, list[str]] = {}
    for sample in samples:
        visibility = int(sample["visibility_score"])
        sample_ids_by_visibility.setdefault(visibility, []).append(sample["image_id"])

    gt_by_image = {gt.image_id: gt for gt in ground_truths}
    for visibility, image_ids in sample_ids_by_visibility.items():
        id_set = set(image_ids)
        visibility_preds = [pred for pred in predictions if pred.image_id in id_set]
        visibility_gts = [gt_by_image[iid] for iid in image_ids if iid in gt_by_image]
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


def run_benchmark(models: list[str], subset: str, preprocess: str | None = None) -> dict[str, Any]:
    # HatefulIllusion has no bounding-box annotations; a full-image proxy box is used as GT.
    samples = collect_samples(subset)
    if not samples:
        raise ValueError(f"No samples found for subset '{subset}'")

    ground_truths = build_ground_truths(samples)
    results: dict[str, Any] = {
        "benchmark_date": datetime.now(timezone.utc).isoformat(),
        "subset": subset,
        "preprocess": preprocess,
        "models": {},
    }

    for checkpoint in models:
        print(f"Evaluating {checkpoint} on subset {subset} ({len(samples)} images)")
        model = UltralyticsYOLO(checkpoint=checkpoint, device="cpu", verbose=False)
        predictions, inference_time = collect_predictions(model, samples, preprocess=preprocess)
        average_time = inference_time / len(samples)
        computed_metrics = compute_detection_metrics(predictions, ground_truths)
        visibility_metrics = build_visibility_metrics(predictions, ground_truths, samples)

        results["models"][checkpoint] = {
            "num_images": len(samples),
            "average_inference_time_s": average_time,
            "total_inference_time_s": inference_time,
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
    parser.add_argument(
        "--preprocess",
        default=None,
        choices=PreprocessingPipeline.TRANSFORMATIONS,
        help="Preprocessing filter to apply before inference (from PreprocessingPipeline)",
    )
    args = parser.parse_args()

    models = resolve_model_checkpoints(
        model=args.model,
        all_models=args.all,
        mode=args.mode,
        weights_type=args.weights_type,
        trained_dir=args.trained_dir,
        explicit_weights=args.weights,
    )
    benchmark_results = run_benchmark(models=models, subset=args.subset, preprocess=args.preprocess)

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(benchmark_results, indent=2), encoding="utf-8")
    print(f"Saved benchmark results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
