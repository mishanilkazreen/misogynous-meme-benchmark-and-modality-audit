"""
Benchmark GPT-4o-mini on MAMI 2022 misogyny detection.

Challenge 1 (--task singleclass, default): binary misogyny classification (yes/no prompt).
Challenge 2 (--task multiclass): multi-label sub-type classification (shaming, stereotype,
objectification, violence) using a single multi-output prompt.

Images encoded as base64 data URLs. max_tokens=60 for multiclass; sleep 0.1 s between
calls. Retries with exponential backoff on RateLimitError.
Requires OPENAI_API_KEY environment variable.

Usage:
    uv run python scripts/benchmark_gpt4omini.py --split validation --limit 5
    uv run python scripts/benchmark_gpt4omini.py --split validation --limit 5 --task multiclass
"""

# ruff: noqa: I001  # datasets (via utils.dataset) must precede torch to avoid OpenMP segfault
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from dotenv import load_dotenv
from PIL import Image
from tqdm import tqdm

# Backward-compat aliases kept so old imports still resolve
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
    import openai  # type: ignore[import-untyped,import-not-found]

    _OPENAI_AVAILABLE = True
except ModuleNotFoundError:
    _OPENAI_AVAILABLE = False

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
MODEL_ID = "gpt-4o-mini"
_CALL_INTERVAL_S = 0.1
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


def numpy_to_b64(arr: np.ndarray) -> str:
    pil = Image.fromarray(arr)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


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


def classify_with_gpt(
    client: Any, b64_image: str, prompt: str, labels: list[str]
) -> ClassificationResult:
    for attempt in range(_MAX_RETRIES):
        try:
            start = time.perf_counter()
            response = client.chat.completions.create(
                model=MODEL_ID,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{b64_image}"},
                            },
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
                max_tokens=20,
            )
            elapsed = time.perf_counter() - start
            text = (response.choices[0].message.content or "").strip()
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
            if "rate" in str(exc).lower() or "429" in str(exc):
                sleep_s = _BACKOFF_BASE_S * (2**attempt)
                print(f"  Rate limit (attempt {attempt + 1}): sleeping {sleep_s:.1f}s")
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
    task: str = "singleclass",
) -> dict[str, Any]:
    if not _OPENAI_AVAILABLE:
        raise RuntimeError("openai not installed. Run: uv add openai")

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable not set")

    client = openai.OpenAI(api_key=api_key)
    if samples is None:
        samples = collect_samples(split, limit=limit)
    if not samples:
        raise ValueError(f"No samples for split '{split}'")

    if task == "multiclass":
        return _run_benchmark_multiclass(client, samples, split, preprocess)

    labels = MISOGYNY_LABELS  # ["yes", "no"]
    prompt = build_misogyny_prompt()

    pipeline = PreprocessingPipeline() if preprocess else None
    results: list[ClassificationResult] = []
    ground_truths: list[str] = []
    sample_rows: list[dict[str, Any]] = []

    pbar = tqdm(total=len(samples), desc=f"gpt4omini/{preprocess or 'none'}", unit="img")
    for s in samples:
        arr = image_to_numpy(s["image"])
        if pipeline is not None and preprocess is not None:
            arr = pipeline.apply_transformation(arr, preprocess)
        b64 = numpy_to_b64(arr)

        time.sleep(_CALL_INTERVAL_S)
        result = classify_with_gpt(client, b64, prompt, labels)
        gt = _misogynous_to_label(s["misogynous"])
        results.append(result)
        ground_truths.append(gt)
        sample_rows.append(
            {
                "image_id": s["image_id"],
                "ground_truth": yesno_to_int(gt),
                "prediction": yesno_to_int(result.prediction),
                "correct": result.prediction == gt,
            }
        )
        pbar.update(1)
    pbar.close()

    return _aggregate(results, ground_truths, split, preprocess, labels, sample_rows)


def _run_benchmark_multiclass(
    client: Any,
    samples: list[dict[str, Any]],
    split: str,
    preprocess: str | None,
) -> dict[str, Any]:
    """Run GPT-4o-mini on multiclass: multi-label sub-type classification."""
    prompt = build_subtype_prompt()
    pipeline = PreprocessingPipeline() if preprocess else None

    pred_dicts: list[dict[str, int]] = []
    gt_dicts: list[dict[str, int]] = []
    latencies: list[float] = []
    refusals = 0
    sample_rows: list[dict[str, Any]] = []

    pbar = tqdm(total=len(samples), desc=f"gpt4omini-multiclass/{preprocess or 'none'}", unit="img")
    for s in samples:
        arr = image_to_numpy(s["image"])
        if pipeline is not None and preprocess is not None:
            arr = pipeline.apply_transformation(arr, preprocess)
        b64 = numpy_to_b64(arr)

        time.sleep(_CALL_INTERVAL_S)
        # Use max_tokens=60 for the longer subtype response
        cr = _classify_gpt_multiclass(client, b64, prompt)
        latencies.append(cr.latency_s)

        raw_text = cr.prediction or ""
        if cr.refusal or not raw_text:
            subtype_pred = dict.fromkeys(SUBTYPE_LABELS, 0)
            refusals += 1
        else:
            subtype_pred = extract_subtypes(raw_text, SUBTYPE_LABELS)
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
        pbar.update(1)
    pbar.close()

    n_total = len(samples)
    avg_latency = sum(latencies) / n_total if n_total else 0.0
    ml_metrics = compute_multilabel_metrics(pred_dicts, gt_dicts, SUBTYPE_LABELS)
    label_prev = {lbl: sum(g[lbl] for g in gt_dicts) for lbl in SUBTYPE_LABELS}

    return {
        "benchmark_date": datetime.now(timezone.utc).isoformat(),
        "model": "gpt4omini",
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
        "mami_score_b": ml_metrics["mami_score_b"],
        "per_label_binary_macro_f1": ml_metrics["per_label_binary_macro_f1"],
        "avg_latency_s": avg_latency,
        "refusal_rate": refusals / n_total if n_total else 0.0,
        "label_prevalence": label_prev,
        "sample_predictions": sample_rows,
    }


def _classify_gpt_multiclass(client: Any, b64_image: str, prompt: str) -> ClassificationResult:
    """Call GPT-4o-mini for multiclass with higher max_tokens."""
    for attempt in range(_MAX_RETRIES):
        try:
            start = time.perf_counter()
            response = client.chat.completions.create(
                model=MODEL_ID,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{b64_image}"},
                            },
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
                max_tokens=60,
            )
            elapsed = time.perf_counter() - start
            text = (response.choices[0].message.content or "").strip()
            if not text:
                return ClassificationResult(
                    prediction=None, confidence=0.0, latency_s=elapsed, refusal=True
                )
            return ClassificationResult(
                prediction=text, confidence=1.0, latency_s=elapsed, refusal=False
            )
        except Exception as exc:
            if "rate" in str(exc).lower() or "429" in str(exc):
                sleep_s = _BACKOFF_BASE_S * (2**attempt)
                print(f"  Rate limit (attempt {attempt + 1}): sleeping {sleep_s:.1f}s")
                time.sleep(sleep_s)
            else:
                return ClassificationResult(
                    prediction=None, confidence=0.0, latency_s=0.0, refusal=True
                )
    return ClassificationResult(prediction=None, confidence=0.0, latency_s=0.0, refusal=True)


def _aggregate(
    results: list[ClassificationResult],
    ground_truths: list[str],
    split: str,
    preprocess: str | None,
    labels: list[str],
    sample_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    n_total = len(results)
    refusal_rate = sum(1 for r in results if r.refusal) / n_total if n_total else 0.0
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
        "model": "gpt4omini",
        "filter": preprocess or "none",
        "split": split,
        "task": "singleclass",
        "exact_match_accuracy": accuracy,
        "precision": macro_prec,
        "recall": macro_rec,
        "f1": macro_f1,
        "avg_latency_s": avg_latency,
        "refusal_rate": refusal_rate,
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
            task=args.task,
        )
        all_results.append(result)
        acc = result.get("exact_match_accuracy", 0.0)
        print(f"  acc={acc:.3f}  refusals={result.get('refusal_rate', 0.0):.2%}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    split_slug = args.split.replace(",", "_")
    suffix = "_multiclass" if args.task == "multiclass" else ""
    out = RESULTS_DIR / f"gpt4omini_{split_slug}{suffix}.json"
    out.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"\nSaved {len(all_results)} filter rows to {out}")


if __name__ == "__main__":
    main()
