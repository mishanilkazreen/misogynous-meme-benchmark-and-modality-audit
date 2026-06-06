"""
Benchmark LLaVA on HatefulIllusion (closed-set classification, GPU required).

Requires CUDA and transformers. Uses llava-hf/llava-1.5-7b-hf by default.
Refusals caught and logged as refusal_rate.

Usage:
    uv run python scripts/benchmark_llava.py --subset digits --limit 5 --device cuda
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
from PIL import Image
import torch
from tqdm import tqdm

from models.vlm.classifier import ClassificationResult, build_prompt, extract_label
from utils.dataset import DatasetManager
from utils.preprocessing import PreprocessingPipeline

try:
    from transformers import (  # type: ignore[import-untyped]
        AutoProcessor,
        LlavaForConditionalGeneration,
    )

    _TRANSFORMERS_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
    _TRANSFORMERS_AVAILABLE = False

SUBSET_NAMES = ["digits", "hate_slangs", "hate_symbols"]
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
DEFAULT_MODEL_ID = "llava-hf/llava-1.5-7b-hf"
MAX_NEW_TOKENS = 20

# Module-level cache: load once per (model_id, device), reuse across filter iterations.
_model_cache: dict[str, tuple[Any, Any]] = {}


def _load_model(model_id: str, device: str) -> tuple[Any, Any]:
    """Load processor + model with fp16 + 4-bit quant; cached after first call."""
    key = f"{model_id}:{device}"
    if key in _model_cache:
        return _model_cache[key]

    print(f"  Loading {model_id} (fp16) …")
    processor = AutoProcessor.from_pretrained(model_id, use_fast=True)

    load_kwargs: dict[str, Any] = {
        "dtype": torch.float16,
        "low_cpu_mem_usage": True,
        "device_map": device,
    }
    model = LlavaForConditionalGeneration.from_pretrained(model_id, **load_kwargs)
    model.eval()

    _model_cache[key] = (processor, model)
    return processor, model


_REFUSAL_PHRASES = [
    "i cannot",
    "i'm unable",
    "i am unable",
    "i can't",
    "as an ai",
    "inappropriate",
    "harmful",
]


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


def is_refusal(text: str) -> bool:
    lower = text.lower()
    return any(phrase in lower for phrase in _REFUSAL_PHRASES)


def run_benchmark(
    subset: str,
    model_id: str = DEFAULT_MODEL_ID,
    device: str = "cuda",
    preprocess: str | None = None,
    limit: int | None = None,
    samples: list[dict[str, Any]] | None = None,
    batch_size: int = 4,
) -> dict[str, Any]:
    if not _TRANSFORMERS_AVAILABLE:
        raise RuntimeError("transformers not available. Install: uv sync --group vlm-gpu")
    if not torch.cuda.is_available() and device.startswith("cuda"):
        raise RuntimeError("CUDA not available. Pass --device cpu for testing only.")

    processor, model = _load_model(model_id, device)

    if samples is None:
        samples = collect_samples(subset, limit=limit)
    if not samples:
        raise ValueError(f"No samples for subset '{subset}'")

    labels_by_subset_raw: dict[str, set[str]] = {}
    for s in samples:
        labels_by_subset_raw.setdefault(s["subset"], set()).add(s["message"])
    labels_sorted = {k: sorted(v) for k, v in labels_by_subset_raw.items()}
    all_labels = sorted({lbl for lbls in labels_sorted.values() for lbl in lbls})

    pipeline = PreprocessingPipeline() if preprocess else None
    results: list[ClassificationResult] = []
    ground_truths: list[str] = []
    visibility_scores: list[int] = []
    sample_rows: list[dict[str, Any]] = []

    pbar = tqdm(total=len(samples), desc=f"llava/{preprocess or 'none'}", unit="img")
    for batch_start in range(0, len(samples), batch_size):
        batch = samples[batch_start : batch_start + batch_size]

        prompts: list[str] = []
        pils: list[Image.Image] = []
        batch_labels: list[list[str]] = []
        for s in batch:
            arr = image_to_numpy(s["image"])
            if pipeline is not None and preprocess is not None:
                arr = pipeline.apply_transformation(arr, preprocess)
            pil = Image.fromarray(arr)
            labels = labels_sorted.get(s["subset"], [])
            prompt_text = build_prompt(s["subset"], labels)
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt_text},
                    ],
                }
            ]
            prompts.append(processor.apply_chat_template(conversation, add_generation_prompt=True))
            pils.append(pil)
            batch_labels.append(labels)

        inputs = processor(images=pils, text=prompts, return_tensors="pt", padding=True).to(
            model.device
        )

        t0 = time.perf_counter()
        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        elapsed = (time.perf_counter() - t0) / len(batch)

        input_len = inputs["input_ids"].shape[1]
        response_texts = processor.batch_decode(output_ids[:, input_len:], skip_special_tokens=True)

        for s, raw_text, labels in zip(batch, response_texts, batch_labels, strict=False):
            response_text = raw_text.strip()
            refusal = is_refusal(response_text)
            matched = extract_label(response_text, labels) if not refusal else None
            results.append(
                ClassificationResult(
                    prediction=matched,
                    confidence=1.0 if matched else 0.0,
                    latency_s=elapsed,
                    refusal=refusal,
                )
            )
            ground_truths.append(s["message"])
            visibility_scores.append(int(s["visibility_score"]))
            sample_rows.append(
                {
                    "image_id": s["image_id"],
                    "ground_truth": s["message"],
                    "prediction": matched,
                    "correct": matched == s["message"],
                    "refusal": refusal,
                    "visibility": int(s["visibility_score"]),
                }
            )
        pbar.update(len(batch))
    pbar.close()

    n_total = len(results)
    refusal_rate = sum(1 for r in results if r.refusal) / n_total if n_total else 0.0
    avg_latency = sum(r.latency_s for r in results) / n_total if n_total else 0.0

    predictions: list[str | None] = [r.prediction for r in results]
    correct_all = sum(1 for p, gt in zip(predictions, ground_truths, strict=True) if p == gt)
    accuracy = correct_all / n_total if n_total else 0.0

    per_class_prec: list[float] = []
    per_class_rec: list[float] = []
    for label in all_labels:
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
        "model": "llava",
        "filter": preprocess or "none",
        "subset": subset,
        "exact_match_accuracy": accuracy,
        "precision": macro_prec,
        "recall": macro_rec,
        "f1": macro_f1,
        "avg_latency_s": avg_latency,
        "refusal_rate": refusal_rate,
        "by_visibility": by_visibility,
        "sample_predictions": sample_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--subset",
        default="digits",
        choices=["digits", "hate_slangs", "hate_symbols", "all"],
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--filters",
        default=None,
        help="Comma-separated filters (default: all). E.g. 'none,blur,grayscale'",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=4, help="Images per forward pass")
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
            model_id=args.model_id,
            device=args.device,
            preprocess=None if flt == "none" else flt,
            samples=samples,
            batch_size=args.batch_size,
        )
        all_results.append(result)
        acc = result.get("exact_match_accuracy", 0.0)
        print(f"  acc={acc:.3f}  refusals={result.get('refusal_rate', 0.0):.2%}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"llava_{args.subset}.json"
    out.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"\nSaved {len(all_results)} filter rows to {out}")


if __name__ == "__main__":
    main()
