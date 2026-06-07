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
import yaml

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
DEFAULT_TRAINED_RESULTS_DIR = Path(__file__).resolve().parents[1] / "results" / "trained_models"

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


SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tiff", ".tif"}


def collect_samples_from_folder(
    folder: Path,
    subset_name: str,
    visibility_score: int = 3,
) -> list[dict[str, Any]]:
    """Load images from a local folder as benchmark samples.

    Produces the same dict format as collect_samples() so the full pipeline
    (collect_predictions, build_ground_truths, build_visibility_metrics) works
    without modification.

    Args:
        folder: Directory containing image files.
        subset_name: Name used as subset label and image_id prefix.
        visibility_score: Default visibility level 1-5 assigned to all images.
    """
    from PIL import Image as PILImage

    image_paths = sorted(
        p for p in folder.iterdir() if p.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    )
    if not image_paths:
        raise ValueError(f"No supported images found in {folder}")

    samples: list[dict[str, Any]] = []
    for i, img_path in enumerate(image_paths):
        pil = PILImage.open(img_path).convert("RGB")
        image = torch.from_numpy(np.array(pil).transpose(2, 0, 1)).float() / 255.0
        samples.append(
            {
                "image": image,
                "image_id": f"{subset_name}_{i}",
                "visibility_score": visibility_score,
                "subset": subset_name,
                "source_path": str(img_path),
            }
        )
    return samples


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
        # HatefulIllusion has no per-object bounding box annotations.
        # Full-image proxy GT (cx=0.5, cy=0.5, w=1.0, h=1.0) is used as documented
        # in the paper; mAP reflects detection presence, not localisation accuracy.
        label_path.write_text("0 0.5 0.5 1.0 1.0\n", encoding="utf-8")

    data_yaml = output_dir / "data.yaml"
    with data_yaml.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            {
                "names": ["embedded_hateful_content"],
                "nc": 1,
                "train": str(images_dir),
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
            arr = image_to_numpy(image)
            height, width = int(arr.shape[0]), int(arr.shape[1])
        gts.append(
            GroundTruthBox(
                image_id=sample["image_id"],
                bbox=(0.0, 0.0, float(width), float(height)),
            )
        )
    return gts


BATCH_SIZE = 64


def collect_predictions(
    model: UltralyticsYOLO,
    samples: list[dict[str, Any]],
    preprocess: str | None = None,
) -> tuple[list[DetectionPrediction], float]:
    pipeline = PreprocessingPipeline() if preprocess else None
    predictions: list[DetectionPrediction] = []
    total_elapsed = 0.0

    for batch_start in range(0, len(samples), BATCH_SIZE):
        batch = samples[batch_start : batch_start + BATCH_SIZE]
        images: list[np.ndarray] = []
        for s in batch:
            arr = image_to_numpy(s["image"])
            if pipeline is not None and preprocess is not None:
                arr = pipeline.apply_transformation(arr, preprocess)
            images.append(arr)

        results, elapsed = model.timed_predict(
            source=images, stream=True, imgsz=640, save=False, verbose=False
        )
        total_elapsed += elapsed

        for sample, result in zip(batch, results, strict=True):
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

    return predictions, total_elapsed


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


def infer_model_key_from_weights_path(
    weights_path: Path, fallback_model: str | None = None, weights_type: str | None = None
) -> str:
    if weights_type and fallback_model is not None:
        return f"{Path(fallback_model).stem}_{weights_type}"
    if weights_path.name in {"best.pt", "last.pt"} and weights_path.parent.name == "weights":
        train_folder = weights_path.parent.parent.name
        if train_folder.startswith("train_"):
            return f"{train_folder.replace('train_', '')}_{weights_path.stem}"
    return weights_path.stem


def resolve_model_checkpoints(
    model: str,
    all_models: bool,
    mode: str,
    weights_type: str,
    trained_dir: Path,
    explicit_weights: str | None = None,
    subset: str = "digits",
) -> list[tuple[str, str]]:
    if explicit_weights is not None:
        if all_models:
            raise ValueError("--weights cannot be combined with --all")
        weights_path = Path(explicit_weights).resolve()
        model_key = infer_model_key_from_weights_path(weights_path)
        return [(model_key, str(weights_path))]

    if mode == "pretrained":
        # Always use the five local yolovXn.pt files for pretrained
        return (
            [(Path(checkpoint).stem, checkpoint) for checkpoint in MODEL_CHECKPOINTS]
            if all_models
            else [(Path(model).stem, model)]
        )

    if mode == "trained":
        checkpoints = MODEL_CHECKPOINTS if all_models else [model]
        result = []
        for checkpoint in checkpoints:
            model_stem = Path(checkpoint).stem
            weights_path = (
                trained_dir / f"train_{model_stem}_{subset}" / "weights" / f"{weights_type}.pt"
            )
            if not weights_path.exists():
                raise FileNotFoundError(
                    f"Trained weights not found: {weights_path}\n"
                    f"  → Train the model first:  uv run python scripts/train_yolo.py --model {checkpoint} --subset {subset}\n"
                    f"  → Or benchmark pretrained: uv run python scripts/benchmark_yolo.py --mode pretrained --model {checkpoint} --subset {subset}"
                )
            result.append((model_stem, str(weights_path)))
        return result

    raise ValueError(f"Unsupported mode: {mode}")


def run_benchmark(
    models: list[tuple[str, str]],
    subset: str,
    preprocess: str | None = None,
    samples: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if samples is None:
        samples = collect_samples(subset)
    if not samples:
        raise ValueError(f"No samples found for subset '{subset}'")

    ground_truths = build_ground_truths(samples)
    results: dict[str, Any] = {
        "benchmark_date": datetime.now(timezone.utc).isoformat(),
        "subset": subset,
        "models": {},
    }

    for model_key, checkpoint in models:
        print(f"Evaluating {model_key} on subset {subset} ({len(samples)} images)")
        model = UltralyticsYOLO(checkpoint=checkpoint, device="cpu", verbose=False)
        predictions, inference_time = collect_predictions(model, samples, preprocess=preprocess)
        average_time = inference_time / len(samples)
        computed_metrics = compute_detection_metrics(predictions, ground_truths)
        visibility_metrics = build_visibility_metrics(predictions, ground_truths, samples)

        results["models"][model_key] = {
            "model_key": model_key,
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
        choices=[*MODEL_CHECKPOINTS, "all"],
        help="Ultralytics checkpoint name, e.g. yolov8n.pt, yolo26n.pt, or 'all' for all five models",
    )
    parser.add_argument(
        "--mode",
        default="all",
        choices=["pretrained", "trained", "all"],
        help="Benchmark mode: pretrained hub checkpoint, trained local weights, or 'all' for both (default: all)",
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
        default=None,
        help=(
            "HatefulIllusion subset (digits, hate_slangs, hate_symbols, all) "
            "or a path to a local image folder for custom evaluation."
        ),
    )
    parser.add_argument("--all", action="store_true", help="Benchmark all five models")
    parser.add_argument(
        "--preprocessing", action="store_true", help="Enable preprocessing ablation (stub)"
    )
    args = parser.parse_args()

    import gc

    preprocess = "blur_histogram" if args.preprocessing else None
    output_path = (
        RESULTS_PATH.with_name("yolo_benchmark_preprocessed.json")
        if args.preprocessing
        else RESULTS_PATH
    )

    # Custom dataset: --subset is a path to a local image folder
    _subset_as_path = Path(args.subset) if args.subset else None
    if _subset_as_path is not None and _subset_as_path.is_dir():
        folder = _subset_as_path.resolve()
        subset_name = folder.name
        custom_samples = collect_samples_from_folder(folder=folder, subset_name=subset_name)
        print(f"Loaded {len(custom_samples)} images from {folder} as subset '{subset_name}'")

        modes_to_run: list[tuple[str, str | None]] = []
        if args.mode in ("pretrained", "all"):
            modes_to_run.append(("pretrained", None))
        if args.mode in ("trained", "all"):
            for ts in SUBSET_NAMES:
                modes_to_run.append(("trained", ts))

        all_results: list[dict[str, Any]] = []
        custom_output = RESULTS_PATH.with_name(f"yolo_benchmark_{subset_name}.json")
        custom_output.parent.mkdir(parents=True, exist_ok=True)

        for mode, training_subset in modes_to_run:
            try:
                models_list = resolve_model_checkpoints(
                    model=args.model,
                    all_models=args.all or args.model == "all",
                    mode=mode,
                    weights_type=args.weights_type,
                    trained_dir=args.trained_dir,
                    explicit_weights=args.weights,
                    subset=training_subset or SUBSET_NAMES[0],
                )
                benchmark_results = run_benchmark(
                    models=models_list,
                    subset=subset_name,
                    preprocess=preprocess,
                    samples=custom_samples,
                )
                for model_key, model_result in benchmark_results["models"].items():
                    entry: dict[str, Any] = {
                        "model": model_key,
                        "subset": subset_name,
                        "mode": mode,
                        "weights_type": "pretrained" if mode == "pretrained" else args.weights_type,
                        **{k: v for k, v in model_result.items() if k != "model_key"},
                    }
                    if training_subset is not None:
                        entry["trained_on"] = training_subset
                    all_results.append(entry)
                custom_output.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
            except FileNotFoundError as exc:
                print(f"[SKIP] trained/{training_subset}: {exc}")

        print(f"Saved custom dataset benchmark results to {custom_output}")
        return

    all_models = args.all or args.model == "all" or args.preprocessing
    all_modes = args.mode == "all"
    all_subsets = args.subset is None or args.subset == "all" or args.preprocessing

    if all_models and all_subsets and all_modes:
        # Full matrix: all models x all subsets x all modes
        output_path.parent.mkdir(parents=True, exist_ok=True)
        all_results = []
        for subset in SUBSET_NAMES:
            for mode, weights_type in [("pretrained", None), ("trained", "best")]:
                try:
                    models = resolve_model_checkpoints(
                        model=args.model,
                        all_models=True,
                        mode=mode,
                        weights_type=weights_type if weights_type else "best",
                        trained_dir=args.trained_dir,
                        explicit_weights=args.weights,
                        subset=subset,
                    )
                    benchmark_results = run_benchmark(
                        models=models, subset=subset, preprocess=preprocess
                    )
                    for model_key, model_result in benchmark_results["models"].items():
                        entry = {
                            "model": model_key,
                            "subset": subset,
                            "mode": mode,
                            "weights_type": weights_type if weights_type else "pretrained",
                            **{k: v for k, v in model_result.items() if k != "model_key"},
                        }
                        all_results.append(entry)
                    # Write partial results after each mode/subset to avoid memory issues
                    output_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
                    gc.collect()
                except FileNotFoundError as exc:
                    if mode == "trained":
                        print(f"[SKIP] trained/{subset}: {exc}")
                        continue
                    raise
        print(f"Saved full-matrix benchmark results to {RESULTS_PATH}")
        return
    elif all_models and all_subsets:
        # All models, all subsets, single mode
        output_path.parent.mkdir(parents=True, exist_ok=True)
        all_results = []
        for subset in SUBSET_NAMES:
            try:
                models = resolve_model_checkpoints(
                    model=args.model,
                    all_models=True,
                    mode=args.mode,
                    weights_type=args.weights_type,
                    trained_dir=args.trained_dir,
                    explicit_weights=args.weights,
                    subset=subset,
                )
                benchmark_results = run_benchmark(
                    models=models, subset=subset, preprocess=preprocess
                )
                for model_key, model_result in benchmark_results["models"].items():
                    entry = {
                        "model": model_key,
                        "subset": subset,
                        "mode": args.mode,
                        "weights_type": "pretrained"
                        if args.mode == "pretrained"
                        else args.weights_type,
                        **{k: v for k, v in model_result.items() if k != "model_key"},
                    }
                    all_results.append(entry)
                output_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
                gc.collect()
            except FileNotFoundError as exc:
                if args.mode == "trained":
                    print(f"[SKIP] trained/{subset}: {exc}")
                    continue
                raise
        print(f"Saved all-models/all-subsets benchmark results to {RESULTS_PATH}")
        return

    elif all_models and args.subset and all_modes:
        # All models, single subset, all modes
        output_path.parent.mkdir(parents=True, exist_ok=True)
        all_results = []
        for mode, weights_type in [("pretrained", None), ("trained", "best")]:
            try:
                models = resolve_model_checkpoints(
                    model=args.model,
                    all_models=True,
                    mode=mode,
                    weights_type=weights_type if weights_type else "best",
                    trained_dir=args.trained_dir,
                    explicit_weights=args.weights,
                    subset=args.subset,
                )
                benchmark_results = run_benchmark(
                    models=models, subset=args.subset, preprocess=preprocess
                )
                for model_key, model_result in benchmark_results["models"].items():
                    entry = {
                        "model": model_key,
                        "subset": args.subset,
                        "mode": mode,
                        "weights_type": weights_type if weights_type else "pretrained",
                        **{k: v for k, v in model_result.items() if k != "model_key"},
                    }
                    all_results.append(entry)
                # Write partial results after each mode to avoid memory issues
                output_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
                gc.collect()
            except FileNotFoundError as exc:
                if mode == "trained":
                    print(f"[SKIP] trained/{args.subset}: {exc}")
                    continue
                raise
        print(f"Saved all-modes benchmark results to {RESULTS_PATH}")
        return
    elif all_models and args.subset:
        # All models, single subset, single mode
        models = resolve_model_checkpoints(
            model=args.model,
            all_models=True,
            mode=args.mode,
            weights_type=args.weights_type,
            trained_dir=args.trained_dir,
            explicit_weights=args.weights,
            subset=args.subset,
        )
        benchmark_results = run_benchmark(models=models, subset=args.subset, preprocess=preprocess)
        all_results = []
        for model_key, model_result in benchmark_results["models"].items():
            entry = {
                "model": model_key,
                "subset": args.subset,
                "mode": args.mode,
                "weights_type": "pretrained" if args.mode == "pretrained" else args.weights_type,
                **{k: v for k, v in model_result.items() if k != "model_key"},
            }
            all_results.append(entry)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
        print(f"Saved subset benchmark results to {RESULTS_PATH}")
        return
    else:
        # Single model cases
        subsets_to_run = SUBSET_NAMES if all_subsets else [args.subset]

        if all_modes and all_subsets:
            # Single model, all subsets, all modes
            output_path.parent.mkdir(parents=True, exist_ok=True)
            all_results = []
            for subset in subsets_to_run:
                for mode, weights_type in [("pretrained", None), ("trained", "best")]:
                    try:
                        models = resolve_model_checkpoints(
                            model=args.model,
                            all_models=False,
                            mode=mode,
                            weights_type=weights_type if weights_type else "best",
                            trained_dir=args.trained_dir,
                            explicit_weights=args.weights,
                            subset=subset,
                        )
                        benchmark_results = run_benchmark(
                            models=models, subset=subset, preprocess=preprocess
                        )
                        for model_key, model_result in benchmark_results["models"].items():
                            entry = {
                                "model": model_key,
                                "subset": subset,
                                "mode": mode,
                                "weights_type": weights_type if weights_type else "pretrained",
                                **{k: v for k, v in model_result.items() if k != "model_key"},
                            }
                            all_results.append(entry)
                        output_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
                        gc.collect()
                    except FileNotFoundError as exc:
                        if mode == "trained":
                            print(f"[SKIP] trained/{subset}: {exc}")
                            continue
                        raise
            print(f"Saved all-subsets/all-modes benchmark results to {RESULTS_PATH}")
        elif all_modes:
            # Single model, single subset, all modes
            subset = args.subset
            output_path.parent.mkdir(parents=True, exist_ok=True)
            all_results = []
            for mode, weights_type in [("pretrained", None), ("trained", "best")]:
                try:
                    models = resolve_model_checkpoints(
                        model=args.model,
                        all_models=False,
                        mode=mode,
                        weights_type=weights_type if weights_type else "best",
                        trained_dir=args.trained_dir,
                        explicit_weights=args.weights,
                        subset=subset,
                    )
                    benchmark_results = run_benchmark(
                        models=models, subset=subset, preprocess=preprocess
                    )
                    for model_key, model_result in benchmark_results["models"].items():
                        entry = {
                            "model": model_key,
                            "subset": subset,
                            "mode": mode,
                            "weights_type": weights_type if weights_type else "pretrained",
                            **{k: v for k, v in model_result.items() if k != "model_key"},
                        }
                        all_results.append(entry)
                    output_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
                    gc.collect()
                except FileNotFoundError as exc:
                    if mode == "trained":
                        print(f"[SKIP] trained/{subset}: {exc}")
                        continue
                    raise
            print(f"Saved all-modes benchmark results to {RESULTS_PATH}")
        elif all_subsets:
            # Single model, all subsets, single mode
            output_path.parent.mkdir(parents=True, exist_ok=True)
            all_results = []
            for subset in subsets_to_run:
                try:
                    models = resolve_model_checkpoints(
                        model=args.model,
                        all_models=False,
                        mode=args.mode,
                        weights_type=args.weights_type,
                        trained_dir=args.trained_dir,
                        explicit_weights=args.weights,
                        subset=subset,
                    )
                    benchmark_results = run_benchmark(
                        models=models, subset=subset, preprocess=preprocess
                    )
                    for model_key, model_result in benchmark_results["models"].items():
                        entry = {
                            "model": model_key,
                            "subset": subset,
                            "mode": args.mode,
                            "weights_type": "pretrained"
                            if args.mode == "pretrained"
                            else args.weights_type,
                            **{k: v for k, v in model_result.items() if k != "model_key"},
                        }
                        all_results.append(entry)
                    output_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
                    gc.collect()
                except FileNotFoundError as exc:
                    if args.mode == "trained":
                        print(f"[SKIP] trained/{subset}: {exc}")
                        continue
                    raise
            print(f"Saved all-subsets benchmark results to {RESULTS_PATH}")
        else:
            # Single model, single subset, single mode (default)
            subset = args.subset
            models = resolve_model_checkpoints(
                model=args.model,
                all_models=False,
                mode=args.mode,
                weights_type=args.weights_type,
                trained_dir=args.trained_dir,
                explicit_weights=args.weights,
                subset=subset,
            )
            benchmark_results = run_benchmark(models=models, subset=subset, preprocess=preprocess)
            all_results = []
            for model_key, model_result in benchmark_results["models"].items():
                entry = {
                    "model": model_key,
                    "subset": subset,
                    "mode": args.mode,
                    "weights_type": "pretrained"
                    if args.mode == "pretrained"
                    else args.weights_type,
                    **{k: v for k, v in model_result.items() if k != "model_key"},
                }
                all_results.append(entry)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
            print(f"Saved benchmark results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
