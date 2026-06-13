"""
Benchmark Gemini 2.0 Flash on MAMI 2022 misogyny detection.

Challenge 1 (--task singleclass, default): binary misogyny classification (yes/no prompt).
Challenge 2 (--task multiclass): multi-label sub-type classification (shaming, stereotype,
objectification, violence) using a single multi-output prompt.

Requires GEMINI_API_KEY environment variable. Retries with exponential backoff
on quota errors. PIL images passed directly to the Gemini SDK.

Usage:
    uv run python scripts/benchmark_gemini.py --split validation --limit 5
    uv run python scripts/benchmark_gemini.py --split validation --limit 5 --task multiclass
"""

# ruff: noqa: I001  # datasets (via utils.dataset) must precede torch to avoid OpenMP segfault
from __future__ import annotations

import argparse
import io
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from dotenv import load_dotenv
from PIL import Image
from tqdm import tqdm

# Backward-compat aliases still available as BINARY_GROUND_TRUTH / BINARY_LABELS
from models.vlm.classifier import BINARY_GROUND_TRUTH  # noqa: F401
from models.vlm.classifier import (
    MISOGYNY_LABELS,
    SUBTYPE_LABELS,
    ClassificationResult,
    build_misogyny_prompt,
    build_subtype_prompt,
    extract_label,
    extract_subtypes,
    yesno_to_int,
)
from models.vlm.metrics_multilabel import compute_multilabel_metrics
from utils.dataset import DatasetManager
from utils.preprocessing import PreprocessingPipeline

load_dotenv()

try:
    from google import genai  # type: ignore[import-untyped,import-not-found]
    from google.genai import types as genai_types  # type: ignore[import-untyped,import-not-found]

    _GENAI_AVAILABLE = True
except ModuleNotFoundError:
    _GENAI_AVAILABLE = False

_SAFETY_SETTINGS = (
    [
        genai_types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"),
        genai_types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"),
        genai_types.SafetySetting(
            category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"
        ),
        genai_types.SafetySetting(
            category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"
        ),
    ]
    if _GENAI_AVAILABLE
    else []
)

_SYSTEM_INSTRUCTION = (
    "You are a content moderation research assistant analysing internet memes for an academic "
    "misogyny detection study. Your role is purely classificatory: answer with exactly one word "
    "— 'yes' or 'no'. Do not add explanations, warnings, or refusals."
)

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
MODEL_ID = "gemini-3.1-flash-lite"
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


def collect_samples(split: str = "validation", limit: int | None = None) -> list[dict[str, Any]]:
    """Load samples from the MAMI 2022 dataset for the given split(s).

    split may be a single name or comma-separated names, e.g. 'train,validation'.
    limit caps the total number of samples across all splits combined.
    """
    manager = DatasetManager()
    samples: list[dict[str, Any]] = []
    for split_name in [s.strip() for s in split.split(",")]:
        dataset = manager.load_dataset(split=split_name)
        for index in range(len(dataset)):
            samples.append(dataset[index])
    if limit is not None:
        samples = samples[:limit]
    return samples


def _misogynous_to_label(misogynous: int) -> str:
    return "yes" if misogynous == 1 else "no"


def classify_with_gemini(
    client: Any, image_bytes: bytes, prompt: str, labels: list[str]
) -> ClassificationResult:
    """Call Gemini with retry on quota errors. Maps response to closest label."""
    image_part = genai_types.Part.from_bytes(data=image_bytes, mime_type="image/png")
    config = genai_types.GenerateContentConfig(
        system_instruction=_SYSTEM_INSTRUCTION,
        safety_settings=_SAFETY_SETTINGS,
    )
    for attempt in range(_MAX_RETRIES):
        try:
            start = time.perf_counter()
            response = client.models.generate_content(
                model=MODEL_ID, contents=[prompt, image_part], config=config
            )
            elapsed = time.perf_counter() - start
            text = (response.text or "").strip()
            if not text:
                return ClassificationResult(
                    prediction=None, confidence=0.0, latency_s=elapsed, refusal=True
                )
            matched = extract_label(text, labels)
            if matched is None:
                print(f"  [no match] raw response: {text!r}")
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
    split: str = "validation",
    preprocess: str | None = None,
    limit: int | None = None,
    samples: list[dict[str, Any]] | None = None,
    workers: int = 20,
    task: str = "singleclass",
) -> dict[str, Any]:
    if not _GENAI_AVAILABLE:
        raise RuntimeError("google-generativeai not installed. Run: uv add google-generativeai")

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable not set")

    client = genai.Client(api_key=api_key)

    if samples is None:
        samples = collect_samples(split, limit=limit)
    if not samples:
        raise ValueError(f"No samples for split '{split}'")

    if task == "multiclass":
        return _run_benchmark_multiclass(client, samples, split, preprocess, workers)

    labels = MISOGYNY_LABELS  # ["yes", "no"]
    prompt = build_misogyny_prompt()

    pipeline = PreprocessingPipeline() if preprocess else None

    def _prepare_and_classify(idx: int, s: dict[str, Any]) -> tuple[int, ClassificationResult]:
        arr = image_to_numpy(s["image"])
        if pipeline is not None and preprocess is not None:
            arr = pipeline.apply_transformation(arr, preprocess)
        pil = numpy_to_pil(arr)
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        return idx, classify_with_gemini(client, buf.getvalue(), prompt, labels)

    ordered: list[ClassificationResult | None] = [None] * len(samples)
    with (
        tqdm(total=len(samples), desc=f"gemini/{preprocess or 'none'}", unit="img") as pbar,
        ThreadPoolExecutor(max_workers=workers) as pool,
    ):
        futures = {pool.submit(_prepare_and_classify, i, s): i for i, s in enumerate(samples)}
        for fut in as_completed(futures):
            idx, result = fut.result()
            ordered[idx] = result
            pbar.update(1)

    results: list[ClassificationResult] = [r for r in ordered if r is not None]
    ground_truths = [_misogynous_to_label(s["misogynous"]) for s in samples]

    sample_rows: list[dict[str, Any]] = [
        {
            "image_id": s["image_id"],
            "ground_truth": yesno_to_int(ground_truths[i]),
            "prediction": yesno_to_int(results[i].prediction),
            "correct": results[i].prediction == ground_truths[i],
        }
        for i, s in enumerate(samples)
    ]

    return _aggregate(results, ground_truths, split, preprocess, labels, sample_rows)


def _run_benchmark_multiclass(
    client: Any,
    samples: list[dict[str, Any]],
    split: str,
    preprocess: str | None,
    workers: int = 20,
) -> dict[str, Any]:
    """Run Gemini on multiclass: multi-label sub-type classification."""
    prompt = build_subtype_prompt()
    pipeline = PreprocessingPipeline() if preprocess else None

    def _classify_multiclass(idx: int, s: dict[str, Any]) -> tuple[int, ClassificationResult]:
        arr = image_to_numpy(s["image"])
        if pipeline is not None and preprocess is not None:
            arr = pipeline.apply_transformation(arr, preprocess)
        pil = numpy_to_pil(arr)
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        # Reuse classify_with_gemini but pass SUBTYPE_LABELS as dummy (we parse manually)
        return idx, classify_with_gemini(client, buf.getvalue(), prompt, SUBTYPE_LABELS)

    ordered: list[ClassificationResult | None] = [None] * len(samples)
    with (
        tqdm(
            total=len(samples), desc=f"gemini-multiclass/{preprocess or 'none'}", unit="img"
        ) as pbar,
        ThreadPoolExecutor(max_workers=workers) as pool,
    ):
        futures = {pool.submit(_classify_multiclass, i, s): i for i, s in enumerate(samples)}
        for fut in as_completed(futures):
            idx, result = fut.result()
            ordered[idx] = result
            pbar.update(1)

    results: list[ClassificationResult] = [r for r in ordered if r is not None]

    n_total = len(results)
    avg_latency = sum(r.latency_s for r in results) / n_total if n_total else 0.0

    pred_dicts: list[dict[str, int]] = []
    gt_dicts: list[dict[str, int]] = []
    refusals = 0
    sample_rows: list[dict[str, Any]] = []

    for i, s in enumerate(samples):
        r = results[i]
        raw_text = r.prediction or ""
        # If the result object has no prediction (refusal), treat as all zeros
        if r.refusal or not raw_text:
            subtype_pred = dict.fromkeys(SUBTYPE_LABELS, 0)
            refusals += 1
        else:
            subtype_pred = extract_subtypes(raw_text, SUBTYPE_LABELS)
            # If extract_subtypes returns all zeros for a non-empty response it may be
            # unparseable - count as refusal
            if (
                all(v == 0 for v in subtype_pred.values())
                and raw_text.strip()
                and not re.search(r"\bnone\b", raw_text.lower())
            ):
                refusals += 1

        gt_dict: dict[str, int] = {
            "shaming": s.get("shaming", 0),
            "stereotype": s.get("stereotype", 0),
            "objectification": s.get("objectification", 0),
            "violence": s.get("violence", 0),
        }
        pred_dicts.append(subtype_pred)
        gt_dicts.append(gt_dict)
        exact = all(subtype_pred.get(lbl, 0) == gt_dict[lbl] for lbl in SUBTYPE_LABELS)
        sample_rows.append(
            {
                "image_id": s["image_id"],
                "ground_truth": gt_dict,
                "prediction": subtype_pred,
                "correct": exact,
                "misogynous": s.get("misogynous", 0),
            }
        )

    ml_metrics = compute_multilabel_metrics(pred_dicts, gt_dicts, SUBTYPE_LABELS)
    refusal_rate = refusals / n_total if n_total else 0.0

    label_prev = {lbl: sum(g[lbl] for g in gt_dicts) for lbl in SUBTYPE_LABELS}

    return {
        "benchmark_date": datetime.now(timezone.utc).isoformat(),
        "model": "gemini",
        "filter": preprocess or "none",
        "split": split,
        "task": "multiclass",
        "exact_match_accuracy": ml_metrics["exact_match_accuracy"],
        "f1": ml_metrics["macro_f1"],
        "precision": ml_metrics["macro_precision"],
        "recall": ml_metrics["macro_recall"],
        "macro_f1": ml_metrics["macro_f1"],
        "micro_f1": ml_metrics["micro_f1"],
        "weighted_f1": ml_metrics["weighted_f1"],
        "per_class": ml_metrics["per_class"],
        "avg_latency_s": avg_latency,
        "refusal_rate": refusal_rate,
        "label_prevalence": label_prev,
        "sample_predictions": sample_rows,
    }


def _aggregate(
    results: list[ClassificationResult],
    ground_truths: list[str],
    split: str,
    preprocess: str | None,
    labels: list[str],
    sample_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    n_total = len(results)
    safety_block_rate = sum(1 for r in results if r.refusal) / n_total if n_total else 0.0
    refusal_rate = sum(1 for r in results if r.prediction is None) / n_total if n_total else 0.0
    avg_latency = sum(r.latency_s for r in results) / n_total if n_total else 0.0

    predictions: list[str | None] = [r.prediction for r in results]
    correct_all = sum(1 for p, gt in zip(predictions, ground_truths, strict=True) if p == gt)
    accuracy = correct_all / n_total if n_total else 0.0

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

    return {
        "benchmark_date": datetime.now(timezone.utc).isoformat(),
        "model": "gemini",
        "filter": preprocess or "none",
        "split": split,
        "task": "singleclass",
        "exact_match_accuracy": accuracy,
        "precision": macro_prec,
        "recall": macro_rec,
        "f1": macro_f1,
        "avg_latency_s": avg_latency,
        "refusal_rate": refusal_rate,
        "safety_block_rate": safety_block_rate,
        "sample_predictions": sample_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split",
        default="validation",
        help="Dataset split(s) to evaluate. Comma-separated for multiple: 'train,validation'"
        " (default: validation)",
    )
    parser.add_argument(
        "--filters",
        default=None,
        help="Comma-separated filters (default: none — MAMI has no hidden visual content). E.g. 'none,blur'",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=20, help="Parallel API request threads")
    parser.add_argument(
        "--task",
        default="singleclass",
        choices=["singleclass", "multiclass"],
        help="'singleclass' = binary misogyny (default); 'multiclass' = multi-label sub-types",
    )
    args = parser.parse_args()

    # MAMI has no hidden visual content, so preprocessing filters do not help.
    # Default to "none"; pass --filters explicitly only for a deliberate ablation.
    filters_to_run = [f.strip() for f in args.filters.split(",")] if args.filters else ["none"]

    print(
        f"Split: {args.split} | Filters: {filters_to_run} | Limit: {args.limit} | Task: {args.task}"
    )
    samples = collect_samples(args.split, limit=args.limit)
    print(f"Loaded {len(samples)} samples")

    all_results: list[dict[str, Any]] = []
    for flt in filters_to_run:
        print(f"\n--- Filter: {flt} ---")
        result = run_benchmark(
            split=args.split,
            preprocess=None if flt == "none" else flt,
            samples=samples,
            workers=args.workers,
            task=args.task,
        )
        all_results.append(result)
        acc = result.get("exact_match_accuracy", 0.0)
        print(f"  acc={acc:.3f}  refusals={result.get('refusal_rate', 0.0):.2%}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    split_slug = args.split.replace(",", "_")
    suffix = "_multiclass" if args.task == "multiclass" else ""
    out = RESULTS_DIR / f"gemini_{split_slug}{suffix}.json"
    out.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"\nSaved {len(all_results)} filter rows to {out}")


if __name__ == "__main__":
    main()
