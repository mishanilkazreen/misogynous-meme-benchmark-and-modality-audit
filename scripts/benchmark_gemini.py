"""
Benchmark Gemini 2.0 Flash on HatefulIllusion (closed-set classification).

Requires GEMINI_API_KEY environment variable. Retries with exponential backoff
on quota errors. PIL images passed directly to the Gemini SDK.

Usage:
    uv run python scripts/benchmark_gemini.py --subset digits --limit 5
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import time
from typing import Any

import numpy as np
from PIL import Image
import torch

from models.vlm.classifier import ClassificationResult, build_prompt, extract_label
from utils.dataset import DatasetManager
from utils.preprocessing import PreprocessingPipeline

try:
    import google.generativeai as genai  # type: ignore[import-untyped,import-not-found]

    _GENAI_AVAILABLE = True
except ModuleNotFoundError:
    _GENAI_AVAILABLE = False

SUBSET_NAMES = ["digits", "hate_slangs", "hate_symbols"]
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
MODEL_ID = "gemini-2.0-flash"
_MAX_RETRIES = 5
_BACKOFF_BASE_S = 2.0


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


def numpy_to_pil(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(arr)


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


def classify_with_gemini(
    model: Any, pil_image: Image.Image, prompt: str, labels: list[str]
) -> ClassificationResult:
    """Call Gemini with retry on quota errors. Maps response to closest label."""
    for attempt in range(_MAX_RETRIES):
        try:
            start = time.perf_counter()
            response = model.generate_content([prompt, pil_image])
            elapsed = time.perf_counter() - start
            text = (response.text or "").strip()
            if not text:
                return ClassificationResult(
                    prediction=None, confidence=0.0, latency_s=elapsed, refusal=True
                )
            matched = extract_label(text, labels)
            return ClassificationResult(
                prediction=matched,
                confidence=1.0 if matched else 0.0,
                latency_s=elapsed,
                refusal=False,
            )
        except Exception as exc:
            if "quota" in str(exc).lower() or "rate" in str(exc).lower():
                sleep_s = _BACKOFF_BASE_S * (2**attempt)
                print(f"  Quota error (attempt {attempt + 1}): sleeping {sleep_s:.1f}s")
                time.sleep(sleep_s)
            else:
                return ClassificationResult(
                    prediction=None, confidence=0.0, latency_s=0.0, refusal=True
                )
    return ClassificationResult(prediction=None, confidence=0.0, latency_s=0.0, refusal=True)


def run_benchmark(
    subset: str,
    preprocess: str | None = None,
    limit: int | None = None,
    samples: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if not _GENAI_AVAILABLE:
        raise RuntimeError("google-generativeai not installed. Run: uv add google-generativeai")

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable not set")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(MODEL_ID)

    if samples is None:
        samples = collect_samples(subset, limit=limit)
    if not samples:
        raise ValueError(f"No samples for subset '{subset}'")

    labels_by_subset_raw: dict[str, set[str]] = {}
    for s in samples:
        labels_by_subset_raw.setdefault(s["subset"], set()).add(s["message"])
    labels_by_subset: dict[str, list[str]] = {k: sorted(v) for k, v in labels_by_subset_raw.items()}
    all_labels = sorted({lbl for lbls in labels_by_subset.values() for lbl in lbls})

    pipeline = PreprocessingPipeline() if preprocess else None
    results: list[ClassificationResult] = []
    ground_truths: list[str] = []
    visibility_scores: list[int] = []

    for s in samples:
        arr = image_to_numpy(s["image"])
        if pipeline is not None and preprocess is not None:
            arr = pipeline.apply_transformation(arr, preprocess)
        pil = numpy_to_pil(arr)
        labels = labels_by_subset.get(s["subset"], [])
        prompt = build_prompt(s["subset"], labels)

        # Gemini needs PNG bytes for reliable colour handling
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        buf.seek(0)
        pil_for_api = Image.open(buf)

        result = classify_with_gemini(model, pil_for_api, prompt, labels)
        results.append(result)
        ground_truths.append(s["message"])
        visibility_scores.append(int(s["visibility_score"]))

    return _aggregate(results, ground_truths, visibility_scores, subset, preprocess, all_labels)


def _aggregate(
    results: list[ClassificationResult],
    ground_truths: list[str],
    visibility_scores: list[int],
    subset: str,
    preprocess: str | None,
    labels: list[str],
) -> dict[str, Any]:
    n_total = len(results)
    refusal_rate = sum(1 for r in results if r.refusal) / n_total if n_total else 0.0
    avg_latency = sum(r.latency_s for r in results) / n_total if n_total else 0.0

    # Accuracy over ALL images: refusals (prediction=None) count as wrong
    predictions: list[str | None] = [r.prediction for r in results]
    correct_all = sum(1 for p, gt in zip(predictions, ground_truths, strict=True) if p == gt)
    accuracy = correct_all / n_total if n_total else 0.0

    # Macro P/R/F1 over all images (refusals = None prediction → miss for every class)
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

    return {
        "benchmark_date": datetime.now(timezone.utc).isoformat(),
        "model": "gemini",
        "filter": preprocess or "none",
        "subset": subset,
        "exact_match_accuracy": accuracy,
        "precision": macro_prec,
        "recall": macro_rec,
        "f1": macro_f1,
        "avg_latency_s": avg_latency,
        "refusal_rate": refusal_rate,
        "by_visibility": by_visibility,
    }


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
        help="Comma-separated filters (default: all). E.g. 'none,blur,grayscale'",
    )
    parser.add_argument("--limit", type=int, default=None)
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
            samples=samples,
        )
        all_results.append(result)
        acc = result.get("exact_match_accuracy", 0.0)
        print(f"  acc={acc:.3f}  refusals={result.get('refusal_rate', 0.0):.2%}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"gemini_{args.subset}.json"
    out.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"\nSaved {len(all_results)} filter rows to {out}")


if __name__ == "__main__":
    main()
