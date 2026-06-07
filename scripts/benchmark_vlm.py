"""
Benchmark VLM classifiers on HatefulIllusion.

Thin CLI wrapper: runs a single model with no preprocessing filter and writes
results/vlm_benchmark.json. For the full preprocessing ablation across all models
use benchmark_vlm_classification.py.

Usage:
    uv run python scripts/benchmark_vlm.py --model clip --subset digits
    uv run python scripts/benchmark_vlm.py --model llava --subset digits --device cuda
"""

# ruff: noqa: I001  # datasets (via utils.dataset) must precede torch to avoid OpenMP segfault
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

from utils.dataset import DatasetManager
from models.vlm.clip_classifier import CLIPClassifier

import numpy as np
import torch

SUBSET_NAMES = ["digits", "hate_slangs", "hate_symbols"]
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
SUPPORTED_MODELS = ["clip", "llava", "qwen2vl"]


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


def run_clip(samples: list[dict[str, Any]], device: str = "cpu") -> dict[str, Any]:
    labels = sorted({s["message"] for s in samples})
    images = [image_to_numpy(s["image"]) for s in samples]
    ground_truths = [s["message"] for s in samples]
    subset = samples[0]["subset"] if len({s["subset"] for s in samples}) == 1 else "all"

    classifier = CLIPClassifier(device=device)
    classifier.set_classes(labels)

    t0 = time.perf_counter()
    raw_preds = classifier.predict_batch(images)
    elapsed = time.perf_counter() - t0

    predictions = [p for p, _ in raw_preds]
    correct = sum(1 for p, gt in zip(predictions, ground_truths, strict=True) if p == gt)
    return {
        "model": "clip",
        "filter": "none",
        "subset": subset,
        "exact_match_accuracy": correct / len(predictions) if predictions else 0.0,
        "avg_latency_s": elapsed / len(images),
        "f1": 0.0,
        "refusal_rate": 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="clip", choices=SUPPORTED_MODELS)
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
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    samples = collect_samples(args.subset, limit=args.limit)
    if not samples:
        sys.exit(f"No samples found for subset '{args.subset}'")

    if args.model == "clip":
        result = run_clip(samples, device=args.device)
    elif args.model in ("llava", "qwen2vl"):
        sys.exit(f"Use scripts/benchmark_{args.model}.py for '{args.model}'.")
    else:
        sys.exit(f"Unknown model '{args.model}'.")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "vlm_benchmark.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
