"""
Benchmark Qwen2-VL on MAMI 2022 misogyny detection (GPU required).

Challenge 1 (--task singleclass, default): binary misogyny classification (yes/no prompt).
Challenge 2 (--task multiclass): multi-label sub-type classification (shaming, stereotype,
objectification, violence) using a single multi-output prompt.

Requires CUDA and the vlm-gpu optional dependency group.
Safety refusals caught and logged as refusal_rate — not counted as incorrect.

Usage:
    uv run python scripts/benchmark_qwen2vl.py --split validation --limit 10
    uv run python scripts/benchmark_qwen2vl.py --split validation --limit 50 --quantize 4bit
    uv run python scripts/benchmark_qwen2vl.py --split validation --limit 10 --task multiclass
"""

# ruff: noqa: I001  # datasets (via utils.dataset) must precede torch to avoid OpenMP segfault
from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

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

try:
    from transformers import AutoProcessor  # type: ignore[import-untyped]
    from transformers import BitsAndBytesConfig, Qwen2VLForConditionalGeneration

    _TRANSFORMERS_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
    _TRANSFORMERS_AVAILABLE = False

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct"
DEFAULT_MAX_PIXELS = 512 * 28 * 28
MAX_NEW_TOKENS = 20

_REFUSAL_PHRASES = [
    "i cannot",
    "i'm unable",
    "i am unable",
    "i'm not able",
    "inappropriate",
    "harmful",
    "i can't",
    "as an ai",
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


def is_refusal(text: str) -> bool:
    lower = text.lower()
    return any(phrase in lower for phrase in _REFUSAL_PHRASES)


_model_cache: dict[str, tuple[Any, Any]] = {}


def _load_model(model_id: str, device: str, quantize: str = "none") -> tuple[Any, Any]:
    """Load processor + model; cached after first call.

    Args:
        model_id: HuggingFace model identifier.
        device: Target device ('cuda', 'cpu').
        quantize: Quantization mode - 'none' (fp16), '4bit', or '8bit'.
            4-bit quantization reduces VRAM from ~14 GB to ~5 GB,
            enabling inference on 12 GB consumer GPUs.
    """
    key = f"{model_id}:{device}:{quantize}"
    if key in _model_cache:
        return _model_cache[key]

    quant_label = "4-bit NF4" if quantize == "4bit" else ("8-bit" if quantize == "8bit" else "fp16")
    print(f"Loading {model_id} ({quant_label}) …")
    processor = AutoProcessor.from_pretrained(model_id, max_pixels=DEFAULT_MAX_PIXELS)

    load_kwargs: dict[str, Any] = {"device_map": device}

    if quantize == "4bit":
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    elif quantize == "8bit":
        load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    else:
        load_kwargs["dtype"] = torch.float16

    model = Qwen2VLForConditionalGeneration.from_pretrained(model_id, **load_kwargs)
    model.eval()
    _model_cache[key] = (processor, model)
    return processor, model


def run_benchmark(
    split: str = "validation",
    model_id: str = MODEL_ID,
    device: str = "cuda",
    preprocess: str | None = None,
    limit: int | None = None,
    samples: list[dict[str, Any]] | None = None,
    batch_size: int = 4,
    quantize: str = "none",
    task: str = "singleclass",
) -> dict[str, Any]:
    """Run Qwen2-VL on the MAMI 2022 misogyny classification task.

    Args:
        split: Dataset split ('train', 'validation', 'test').
        model_id: HuggingFace model identifier.
        device: Target device ('cuda' required for inference).
        preprocess: Preprocessing filter name, or None for no filter.
        limit: Cap on number of samples.
        samples: Pre-loaded samples (skips dataset loading if provided).
        batch_size: Images per forward pass.
        quantize: Quantization mode ('none', '4bit', '8bit').
        task: 'singleclass' for binary misogyny; 'multiclass' for multi-label sub-types.
    """
    if not _TRANSFORMERS_AVAILABLE:
        raise RuntimeError(
            "transformers not available. Install optional group: uv sync --group vlm-gpu"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for Qwen2-VL. Use --device cpu only for testing.")

    processor, model = _load_model(model_id, device, quantize=quantize)

    if samples is None:
        samples = collect_samples(split, limit=limit)
    if not samples:
        raise ValueError(f"No samples for split '{split}'")

    if task == "multiclass":
        return _run_benchmark_multiclass(processor, model, samples, split, preprocess, batch_size)

    labels = MISOGYNY_LABELS  # ["yes", "no"]
    prompt_text = build_misogyny_prompt()

    pipeline = PreprocessingPipeline() if preprocess else None
    results: list[ClassificationResult] = []
    ground_truths: list[str] = []
    sample_rows: list[dict[str, Any]] = []

    pbar = tqdm(total=len(samples), desc=f"qwen2vl/{preprocess or 'none'}", unit="img")
    for batch_start in range(0, len(samples), batch_size):
        batch = samples[batch_start : batch_start + batch_size]

        texts: list[str] = []
        pils: list[Image.Image] = []
        for s in batch:
            arr = image_to_numpy(s["image"])
            if pipeline is not None and preprocess is not None:
                arr = pipeline.apply_transformation(arr, preprocess)
            pil = Image.fromarray(arr)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": pil},
                        {"type": "text", "text": prompt_text},
                    ],
                }
            ]
            texts.append(
                processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            )
            pils.append(pil)

        inputs = processor(
            text=texts,
            images=pils,
            padding=True,
            return_tensors="pt",
        ).to(model.device)

        t0 = time.perf_counter()
        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
        elapsed = (time.perf_counter() - t0) / len(batch)

        input_len = inputs["input_ids"].shape[1]
        response_texts = processor.batch_decode(output_ids[:, input_len:], skip_special_tokens=True)

        for s, raw_text in zip(batch, response_texts, strict=False):
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
            gt = _misogynous_to_label(s["misogynous"])
            ground_truths.append(gt)
            sample_rows.append(
                {
                    "image_id": s["image_id"],
                    "ground_truth": yesno_to_int(gt),
                    "prediction": yesno_to_int(matched),
                    "correct": matched == gt,
                    "refusal": refusal,
                }
            )
        pbar.update(len(batch))
    pbar.close()

    return _aggregate(results, ground_truths, split, preprocess, labels, sample_rows)


def _run_benchmark_multiclass(
    processor: Any,
    model: Any,
    samples: list[dict[str, Any]],
    split: str,
    preprocess: str | None,
    batch_size: int = 4,
) -> dict[str, Any]:
    """Run Qwen2-VL on multiclass: multi-label sub-type classification."""
    prompt_text = build_subtype_prompt()
    max_new_tokens_multiclass = 60
    pipeline = PreprocessingPipeline() if preprocess else None

    pred_dicts: list[dict[str, int]] = []
    gt_dicts: list[dict[str, int]] = []
    latencies: list[float] = []
    refusals = 0
    sample_rows: list[dict[str, Any]] = []

    pbar = tqdm(total=len(samples), desc=f"qwen2vl-multiclass/{preprocess or 'none'}", unit="img")
    for batch_start in range(0, len(samples), batch_size):
        batch = samples[batch_start : batch_start + batch_size]

        texts: list[str] = []
        pils: list[Image.Image] = []
        for s in batch:
            arr = image_to_numpy(s["image"])
            if pipeline is not None and preprocess is not None:
                arr = pipeline.apply_transformation(arr, preprocess)
            pil = Image.fromarray(arr)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": pil},
                        {"type": "text", "text": prompt_text},
                    ],
                }
            ]
            texts.append(
                processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            )
            pils.append(pil)

        inputs = processor(
            text=texts,
            images=pils,
            padding=True,
            return_tensors="pt",
        ).to(model.device)

        t0 = time.perf_counter()
        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens_multiclass)
        elapsed = (time.perf_counter() - t0) / len(batch)
        latencies.extend([elapsed] * len(batch))

        input_len = inputs["input_ids"].shape[1]
        response_texts = processor.batch_decode(output_ids[:, input_len:], skip_special_tokens=True)

        for s, raw_text in zip(batch, response_texts, strict=False):
            response_text = raw_text.strip()
            refusal = is_refusal(response_text)

            if refusal or not response_text:
                subtype_pred = dict.fromkeys(SUBTYPE_LABELS, 0)
                refusals += 1
            else:
                subtype_pred = extract_subtypes(response_text, SUBTYPE_LABELS)
                if (
                    all(v == 0 for v in subtype_pred.values())
                    and response_text
                    and not re.search(r"\bnone\b", response_text.lower())
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
        pbar.update(len(batch))
    pbar.close()

    n_total = len(samples)
    avg_latency = sum(latencies) / n_total if n_total else 0.0
    ml_metrics = compute_multilabel_metrics(pred_dicts, gt_dicts, SUBTYPE_LABELS)
    label_prev = {lbl: sum(g[lbl] for g in gt_dicts) for lbl in SUBTYPE_LABELS}

    return {
        "benchmark_date": datetime.now(timezone.utc).isoformat(),
        "model": "qwen2vl",
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
        "refusal_rate": refusals / n_total if n_total else 0.0,
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
        "model": "qwen2vl",
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
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--filters",
        default=None,
        help="Comma-separated filters (default: all). E.g. 'none,blur,grayscale'",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=4, help="Images per forward pass")
    parser.add_argument(
        "--quantize",
        default="4bit",
        choices=["none", "4bit", "8bit"],
        help="Quantization mode. 4bit recommended for 12 GB GPUs (default: 4bit)",
    )
    parser.add_argument(
        "--task",
        default="singleclass",
        choices=["singleclass", "multiclass"],
        help="'singleclass' = binary misogyny (default); 'multiclass' = multi-label sub-types",
    )
    args = parser.parse_args()

    filters_to_run = (
        [f.strip() for f in args.filters.split(",")]
        if args.filters
        else ["none", *PreprocessingPipeline.TRANSFORMATIONS]
    )

    print(f"Split: {args.split} | Filters: {filters_to_run} | Limit: {args.limit}")

    # Load model FIRST (before dataset) to avoid bitsandbytes crash.
    # The 4-bit quantization CUDA kernels are sensitive to memory state;
    # pre-loading large image datasets can trigger an access violation
    # during weight conversion on Windows.
    print("Pre-loading model …")
    _load_model(args.model_id, args.device, quantize=args.quantize)

    samples = collect_samples(args.split, limit=args.limit)
    print(f"Loaded {len(samples)} samples")

    all_results: list[dict[str, Any]] = []
    for flt in filters_to_run:
        print(f"\n--- Filter: {flt} ---")
        result = run_benchmark(
            split=args.split,
            model_id=args.model_id,
            device=args.device,
            preprocess=None if flt == "none" else flt,
            samples=samples,
            batch_size=args.batch_size,
            quantize=args.quantize,
            task=args.task,
        )
        all_results.append(result)
        acc = result.get("exact_match_accuracy", 0.0)
        print(f"  acc={acc:.3f}  refusals={result.get('refusal_rate', 0.0):.2%}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    split_slug = args.split.replace(",", "_")
    suffix = "_multiclass" if args.task == "multiclass" else ""
    out = RESULTS_DIR / f"qwen2vl_{split_slug}{suffix}.json"
    out.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"\nSaved {len(all_results)} filter rows to {out}")


if __name__ == "__main__":
    main()
