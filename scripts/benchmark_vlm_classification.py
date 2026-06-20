"""
Orchestrator: benchmark all VLMs on the MAMI 2022 misogyny detection task.

Challenge 1 (--task singleclass, default): binary misogyny classification (yes/no prompt;
ground truth = misogynous field).

Challenge 2 (--task multiclass): multi-label sub-type classification over shaming,
stereotype, objectification, violence. CLIP uses independent per-category binary
comparisons; generative models use a single multi-output prompt.

Outer loop: preprocessing filters. For MAMI the default is "none" only — the memes
contain no hidden visual content, so preprocessing filters do not help (confirmed
empirically) and are off unless an explicit --filters ablation is requested.
Inner loop: model x filter x sample.

Writes results/{model}_{split}.json (singleclass) or results/{model}_{split}_multiclass.json
(multiclass) after each model completes, so a crash does not lose earlier models.

Usage:
    uv run python scripts/benchmark_vlm_classification.py --model clip --split validation --limit 16
    uv run python scripts/benchmark_vlm_classification.py --model clip --split validation \\
        --limit 16 --filters none,grayscale
    uv run python scripts/benchmark_vlm_classification.py --task multiclass --model clip \\
        --split validation --limit 16 --filters none,grayscale
    uv run python scripts/benchmark_vlm_classification.py --model all --split validation
"""

# ruff: noqa: I001  # datasets (via utils.dataset) must precede torch to avoid OpenMP segfault
from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch

from models.vlm.classifier import (
    CLIP_MISOGYNY_LABELS,
    CLIP_SUBTYPE_LABELS,
    MISOGYNY_LABELS,
    SUBTYPE_LABELS,
    build_misogyny_prompt,
    build_subtype_prompt,
    yesno_to_int,
)
from models.vlm.clip_classifier import CLIPClassifier
from models.vlm.metrics_multilabel import compute_multilabel_metrics
from utils.dataset import DatasetManager
from utils.preprocessing import PreprocessingPipeline

# --- Logging setup ---
LOG_DIR = Path(__file__).resolve().parents[1] / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "benchmark_vlm.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8"),
        logging.StreamHandler(sys.stderr),
    ],
)
# Silence noisy third-party HTTP/network debug logging.
for noisy in ("httpx", "httpcore", "urllib3", "huggingface_hub", "filelock", "datasets"):
    logging.getLogger(noisy).setLevel(logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

ALL_MODELS = [
    "clip",
    "qwen2vl",
    "llava",
    "llavanext",
    "gpt4omini",
    "gemini",
    "visualbert",
]
GENERATIVE_MODELS = {"qwen2vl", "llava", "llavanext", "gpt4omini", "gemini", "visualbert"}

ALL_FILTERS = ["none", *PreprocessingPipeline.TRANSFORMATIONS]


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

    Args:
        split: One of 'train', 'validation', 'test', or comma-separated, e.g. 'train,validation'.
        limit: Maximum number of samples to return across all splits combined (None = all).

    Returns:
        List of sample dicts with keys: image, image_id, text, misogynous,
        shaming, stereotype, objectification, violence.
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
    """Convert the binary misogynous int field to a yes/no string label."""
    return "yes" if misogynous == 1 else "no"


_PIPELINE = PreprocessingPipeline()


def apply_filter(image: np.ndarray, filter_name: str) -> np.ndarray:
    if filter_name == "none":
        return image
    return _PIPELINE.apply_transformation(image, filter_name)


def compute_classification_metrics(
    predictions: list[str | None],
    ground_truths: list[str],
    labels: list[str],
) -> dict[str, float]:
    n = len(predictions)
    if n == 0:
        return {"exact_match_accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    correct = sum(1 for p, gt in zip(predictions, ground_truths, strict=True) if p == gt)
    accuracy = correct / n

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
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        per_class_prec.append(prec)
        per_class_rec.append(rec)

    macro_prec = sum(per_class_prec) / len(per_class_prec) if per_class_prec else 0.0
    macro_rec = sum(per_class_rec) / len(per_class_rec) if per_class_rec else 0.0
    macro_f1 = (
        2 * macro_prec * macro_rec / (macro_prec + macro_rec)
        if (macro_prec + macro_rec) > 0
        else 0.0
    )

    return {
        "exact_match_accuracy": accuracy,
        "precision": macro_prec,
        "recall": macro_rec,
        "f1": macro_f1,
    }


def build_sample_rows(
    samples: list[dict[str, Any]],
    predictions: list[str | None],
    ground_truths: list[str],
    confidences: list[float] | None = None,
) -> list[dict[str, Any]]:
    """Build per-sample result rows for the singleclass (binary misogyny) task."""
    rows = []
    for i in range(len(samples)):
        row: dict[str, Any] = {
            "image_id": samples[i]["image_id"],
            "ground_truth": yesno_to_int(ground_truths[i]),
            "prediction": yesno_to_int(predictions[i]),
            "correct": predictions[i] == ground_truths[i],
        }
        if confidences is not None:
            row["confidence"] = round(float(confidences[i]), 4)
        rows.append(row)
    return rows


def build_multiclass_sample_rows(
    samples: list[dict[str, Any]],
    predictions: list[dict[str, int]],
) -> list[dict[str, Any]]:
    """Build per-sample result rows for multiclass (multi-label sub-types)."""
    rows = []
    for i, s in enumerate(samples):
        gt_dict = {
            "shaming": s.get("shaming", 0),
            "stereotype": s.get("stereotype", 0),
            "objectification": s.get("objectification", 0),
            "violence": s.get("violence", 0),
        }
        pred_dict = predictions[i]
        exact = all(pred_dict.get(lbl, 0) == gt_dict[lbl] for lbl in SUBTYPE_LABELS)
        rows.append(
            {
                "image_id": s["image_id"],
                "ground_truth": gt_dict,
                "prediction": pred_dict,
                "correct": exact,
                "misogynous": s.get("misogynous", 0),
            }
        )
    return rows


def build_label_prevalence(
    samples: list[dict[str, Any]], task: str = "singleclass"
) -> dict[str, int]:
    """Return counts of each label in the sample set.

    For the singleclass task only misogynous/non_misogynous counts are returned.
    For multiclass the four sub-type counts are returned.
    """
    if task == "multiclass":
        return {
            "shaming": sum(s.get("shaming", 0) for s in samples),
            "stereotype": sum(s.get("stereotype", 0) for s in samples),
            "objectification": sum(s.get("objectification", 0) for s in samples),
            "violence": sum(s.get("violence", 0) for s in samples),
        }
    return {
        "misogynous": sum(s.get("misogynous", 0) for s in samples),
        "non_misogynous": sum(1 - s.get("misogynous", 0) for s in samples),
    }


def load_ocr_transcripts(split: str, ocr_engine: str, embeddings_dir: Path) -> dict[str, str]:
    """Load OCR-extracted texts from any pre-existing NPZ file for the split and engine."""
    files = list(embeddings_dir.glob(f"{split}_*_{ocr_engine}.npz"))
    if not files:
        files = list(embeddings_dir.glob(f"{split}_*_ocr_{ocr_engine}.npz"))

    if not files:
        logger.warning(
            f"No pre-extracted OCR NPZ file found for split '{split}' and engine '{ocr_engine}' in {embeddings_dir}. "
            "Please run scripts/extract_embeddings.py with --use-ocr --ocr-engine {ocr_engine} first."
        )
        return {}

    file_path = files[0]
    logger.info("Loading OCR transcripts from %s...", file_path)
    data = np.load(file_path, allow_pickle=True)
    image_ids = data["image_ids"]
    raw_texts = data["raw_texts"]
    return {str(img_id): str(txt) for img_id, txt in zip(image_ids, raw_texts, strict=True)}


def run_clip(
    samples: list[dict[str, Any]],
    filter_name: str,
    split: str = "validation",
    device: str = "cpu",
    task: str = "singleclass",
    model_path: str | None = None,
    use_ocr: bool = False,
    ocr_engine: str = "easyocr",
) -> dict[str, Any]:
    """Run CLIP on the given samples with the given filter.

    singleclass: binary misogyny classification using CLIP_MISOGYNY_LABELS.
    multiclass: per-category binary classification using CLIP_SUBTYPE_LABELS.
    """
    if task == "multiclass":
        return _run_clip_multiclass(
            samples,
            filter_name,
            split=split,
            device=device,
            model_path=model_path,
            use_ocr=use_ocr,
            ocr_engine=ocr_engine,
        )

    # singleclass: binary misogyny
    clip_labels = CLIP_MISOGYNY_LABELS  # ["misogynistic meme", "not misogynistic meme"]
    ground_truths_yesno = [_misogynous_to_label(s["misogynous"]) for s in samples]
    images = [apply_filter(image_to_numpy(s["image"]), filter_name) for s in samples]

    classifier = CLIPClassifier(device=device, model_path=model_path)
    classifier.set_classes(clip_labels)

    ocr_map = None
    if use_ocr:
        ocr_map = load_ocr_transcripts(split, ocr_engine, RESULTS_DIR / "embeddings")

    t0 = time.perf_counter()
    if ocr_map:
        texts = [ocr_map.get(str(s["image_id"]), "") for s in samples]
    else:
        texts = [s.get("text", "") for s in samples]
    raw_preds = classifier.predict_batch(images, texts=texts)
    total_time = time.perf_counter() - t0

    clip_to_yesno = {
        "misogynistic meme": "yes",
        "not misogynistic meme": "no",
    }
    predictions: list[str | None] = [
        clip_to_yesno.get(p) if p is not None else None for p, _ in raw_preds
    ]
    confidences = [c for _, c in raw_preds]
    avg_latency = total_time / len(images)

    metrics = compute_classification_metrics(predictions, ground_truths_yesno, MISOGYNY_LABELS)

    return {
        "model": "clip",
        "filter": filter_name,
        "split": split,
        "task": "singleclass",
        "clip_labels": clip_labels,
        "exact_match_accuracy": metrics["exact_match_accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "avg_latency_s": avg_latency,
        "refusal_rate": 0.0,
        "label_prevalence": build_label_prevalence(samples, task="singleclass"),
        "sample_predictions": build_sample_rows(
            samples, predictions, ground_truths_yesno, confidences
        ),
    }


def _run_clip_multiclass(
    samples: list[dict[str, Any]],
    filter_name: str,
    split: str = "validation",
    device: str = "cpu",
    model_path: str | None = None,
    use_ocr: bool = False,
    ocr_engine: str = "easyocr",
) -> dict[str, Any]:
    """Run CLIP multiclass: per-category binary prediction for all four sub-types."""
    images = [apply_filter(image_to_numpy(s["image"]), filter_name) for s in samples]
    classifier = CLIPClassifier(device=device, model_path=model_path)

    ocr_map = None
    if use_ocr:
        ocr_map = load_ocr_transcripts(split, ocr_engine, RESULTS_DIR / "embeddings")

    # Per-category independent binary predictions
    category_preds: dict[str, list[int]] = {lbl: [] for lbl in SUBTYPE_LABELS}
    category_times: list[float] = []

    for category in SUBTYPE_LABELS:
        pos_phrase, neg_phrase = CLIP_SUBTYPE_LABELS[category]
        cat_labels = [pos_phrase, neg_phrase]
        classifier.set_classes(cat_labels)

        t0 = time.perf_counter()
        if ocr_map:
            texts = [ocr_map.get(str(s["image_id"]), "") for s in samples]
        else:
            texts = [s.get("text", "") for s in samples]
        raw_preds = classifier.predict_batch(images, texts=texts)
        category_times.append(time.perf_counter() - t0)

        for pred_label, _ in raw_preds:
            # Positive phrase wins → 1, negative phrase → 0
            category_preds[category].append(1 if pred_label == pos_phrase else 0)

    avg_latency = sum(category_times) / (len(images) * len(SUBTYPE_LABELS))

    # Build per-image prediction dicts
    pred_dicts: list[dict[str, int]] = [
        {lbl: category_preds[lbl][i] for lbl in SUBTYPE_LABELS} for i in range(len(samples))
    ]
    gt_dicts: list[dict[str, int]] = [
        {
            "shaming": s.get("shaming", 0),
            "stereotype": s.get("stereotype", 0),
            "objectification": s.get("objectification", 0),
            "violence": s.get("violence", 0),
        }
        for s in samples
    ]

    ml_metrics = compute_multilabel_metrics(pred_dicts, gt_dicts, SUBTYPE_LABELS)
    sample_rows = build_multiclass_sample_rows(samples, pred_dicts)

    return {
        "model": "clip",
        "filter": filter_name,
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
        "refusal_rate": 0.0,
        "label_prevalence": build_label_prevalence(samples, task="multiclass"),
        "sample_predictions": sample_rows,
    }


def free_model_vram(model_name: str) -> None:
    """Evict a generative model from its script's cache and free GPU memory.

    LLaVA and Qwen2-VL each keep a module-level ``_model_cache`` so the model
    is loaded once and reused across filters. When switching to a different
    7B model we must release the previous one or the second 4-bit load will
    exhaust VRAM and crash bitsandbytes mid-conversion (a native access
    violation that Python cannot catch).
    """
    try:
        mod: Any = None
        if model_name == "llava":
            from scripts import benchmark_llava as mod  # type: ignore[import,no-redef]
        elif model_name == "qwen2vl":
            from scripts import benchmark_qwen2vl as mod  # type: ignore[import,no-redef]
        else:
            return
        cache = getattr(mod, "_model_cache", None)
        if cache:
            cache.clear()
            logger.info("Cleared %s model cache", model_name)
    except Exception as exc:  # pragma: no cover - cleanup best-effort
        logger.warning("Could not clear %s cache: %s", model_name, exc)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    logger.info("Freed VRAM after %s", model_name)


def run_generative_model(
    model_name: str,
    samples: list[dict[str, Any]],
    filter_name: str,
    split: str = "validation",
    device: str = "cpu",
    task: str = "singleclass",
    use_ocr: bool = False,
    ocr_engine: str = "easyocr",
) -> dict[str, Any] | None:
    """Dispatch to the appropriate generative model script."""
    preprocess_arg = None if filter_name == "none" else filter_name

    if model_name == "qwen2vl":
        try:
            from scripts.benchmark_qwen2vl import run_benchmark  # type: ignore[import,assignment]

            logger.info("Starting qwen2vl (filter=%s task=%s)", filter_name, task)
            result = run_benchmark(  # type: ignore[call-arg]
                split=split,
                preprocess=preprocess_arg,
                samples=samples,
                device=device,
                quantize="4bit",
                task=task,
                use_ocr=use_ocr,
                ocr_engine=ocr_engine,
            )
            logger.info("Completed qwen2vl (filter=%s)", filter_name)
            return result
        except Exception as exc:
            logger.error(
                "qwen2vl failed (filter=%s): %s\n%s", filter_name, exc, traceback.format_exc()
            )
            print(f"  Skipping qwen2vl: {exc}")
            return None

    if model_name == "llava":
        try:
            from scripts.benchmark_llava import run_benchmark  # type: ignore[import,assignment]

            logger.info("Starting llava (filter=%s task=%s)", filter_name, task)
            result = run_benchmark(  # type: ignore[call-arg]
                split=split,
                preprocess=preprocess_arg,
                samples=samples,
                device=device,
                quantize="4bit",
                task=task,
                use_ocr=use_ocr,
                ocr_engine=ocr_engine,
            )
            logger.info("Completed llava (filter=%s)", filter_name)
            return result
        except Exception as exc:
            logger.error(
                "llava failed (filter=%s): %s\n%s", filter_name, exc, traceback.format_exc()
            )
            print(f"  Skipping llava: {exc}")
            return None

    if model_name == "llavanext":
        try:
            from scripts.benchmark_llavanext import run_benchmark  # type: ignore[import,assignment]

            return run_benchmark(  # type: ignore[call-arg]
                split=split,
                preprocess=preprocess_arg,
                samples=samples,
                device=device,
                task=task,
            )
        except Exception as exc:
            print(f"  Skipping llavanext: {exc}")
            return None

    if model_name == "gemini":
        try:
            from scripts.benchmark_gemini import run_benchmark  # type: ignore[import,assignment]

            logger.info("Starting gemini (filter=%s task=%s)", filter_name, task)
            result = run_benchmark(  # type: ignore[call-arg]
                split=split, preprocess=preprocess_arg, samples=samples, task=task
            )
            logger.info("Completed gemini (filter=%s)", filter_name)
            return result
        except Exception as exc:
            logger.error(
                "gemini failed (filter=%s): %s\n%s", filter_name, exc, traceback.format_exc()
            )
            print(f"  Skipping gemini: {exc}")
            return None

    if model_name == "gpt4omini":
        try:
            from scripts.benchmark_gpt4omini import run_benchmark  # type: ignore[import,assignment]

            logger.info("Starting gpt4omini (filter=%s task=%s)", filter_name, task)
            result = run_benchmark(  # type: ignore[call-arg]
                split=split, preprocess=preprocess_arg, samples=samples, task=task
            )
            logger.info("Completed gpt4omini (filter=%s)", filter_name)
            return result
        except Exception as exc:
            logger.error(
                "gpt4omini failed (filter=%s): %s\n%s", filter_name, exc, traceback.format_exc()
            )
            print(f"  Skipping gpt4omini: {exc}")
            return None

    if model_name == "visualbert":
        try:
            from scripts.benchmark_visualbert import run_benchmark  # type: ignore[import,assignment]

            logger.info("Starting visualbert (filter=%s task=%s)", filter_name, task)
            result = run_benchmark(  # type: ignore[call-arg]
                split=split, preprocess=preprocess_arg, samples=samples, device=device, task=task
            )
            logger.info("Completed visualbert (filter=%s)", filter_name)
            return result
        except Exception as exc:
            logger.error(
                "visualbert failed (filter=%s): %s\n%s", filter_name, exc, traceback.format_exc()
            )
            print(f"  Skipping visualbert: {exc}")
            return None

    return None


def print_sample_predictions(result: dict[str, Any], n: int = 10) -> None:
    rows = result.get("sample_predictions", [])[:n]
    if not rows:
        return
    model = result["model"]
    flt = result["filter"]
    task = result.get("task", "singleclass")
    if task == "multiclass":
        print(f"\n  Sample predictions ({model} | filter={flt} | task=multiclass):")
        print(f"  {'image_id':<30} {'GT':<30} {'PRED':<30} {'ok':<5}")
        print("  " + "-" * 95)
        for r in rows[:n]:
            ok = "Y" if r["correct"] else "N"
            gt_str = ",".join(k for k, v in r["ground_truth"].items() if v) or "none"
            pred_str = ",".join(k for k, v in r["prediction"].items() if v) or "none"
            print(f"  {r['image_id']:<30} {gt_str:<30} {pred_str:<30} {ok:<5}")
    else:
        print(f"\n  Sample predictions ({model} | filter={flt}):")
        has_conf = "confidence" in rows[0]
        if has_conf:
            print(
                f"  {'image_id':<30} {'ground_truth':<15} {'prediction':<15} {'conf':<7} {'ok':<5}"
            )
            print("  " + "-" * 72)
            for r in rows:
                ok = "Y" if r["correct"] else "N"
                pred = str(r["prediction"]) if r["prediction"] is not None else "(none)"
                print(
                    f"  {r['image_id']:<30} {r['ground_truth']:<15} {pred:<15}"
                    f" {r['confidence']:<7} {ok:<5}"
                )
        else:
            print(f"  {'image_id':<30} {'ground_truth':<15} {'prediction':<15} {'ok':<5}")
            print("  " + "-" * 65)
            for r in rows:
                ok = "Y" if r["correct"] else "N"
                pred = str(r["prediction"]) if r["prediction"] is not None else "(none)"
                print(f"  {r['image_id']:<30} {r['ground_truth']:<15} {pred:<15} {ok:<5}")


def print_summary(all_results: list[dict[str, Any]]) -> None:
    print("\n" + "=" * 90)
    print(f"{'Model':<20} {'Filter':<20} {'Split':<12} {'Task':<6} {'Acc':<8} {'F1':<8}")
    print("=" * 90)
    for r in all_results:
        acc = f"{r['exact_match_accuracy']:.3f}" if "exact_match_accuracy" in r else "n/a"
        f1_val = r.get("f1")
        f1 = f"{f1_val:.3f}" if isinstance(f1_val, float) else "n/a"
        split = r.get("split", "n/a")
        task = r.get("task", "singleclass")
        print(f"{r['model']:<20} {r['filter']:<20} {split:<12} {task:<6} {acc:<8} {f1:<8}")
    print("=" * 90)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="clip",
        help="Comma-separated model names or 'all'. Choices: " + ", ".join(ALL_MODELS),
    )
    parser.add_argument(
        "--split",
        default="validation",
        help="Dataset split(s) to evaluate. Comma-separated for multiple: 'train,validation'"
        " (default: validation)",
    )
    parser.add_argument("--limit", type=int, default=None, help="Cap number of samples")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--model-path",
        default=None,
        help="Path to fine-tuned model checkpoint file (for CLIP or other compatible models)",
    )
    parser.add_argument(
        "--filters",
        default=None,
        help=(
            "Comma-separated preprocessing filter names to run (default: 'none'). "
            "MAMI memes contain no hidden visual content, so preprocessing filters do "
            "not help and are off by default. Pass e.g. 'none,grayscale' only for an "
            "explicit ablation."
        ),
    )
    parser.add_argument(
        "--task",
        default="singleclass",
        choices=["singleclass", "multiclass"],
        help=(
            "Challenge to run: 'singleclass' = binary misogyny (default); "
            "'multiclass' = Sub-task B multi-label sub-types"
        ),
    )
    parser.add_argument(
        "--use-ocr",
        action="store_true",
        help="Use OCR-extracted text instead of dataset transcripts",
    )
    parser.add_argument(
        "--ocr-engine",
        default="easyocr",
        choices=["easyocr", "paddleocr"],
        help="OCR engine to load transcripts for",
    )
    args = parser.parse_args()

    logger.info(
        "=== Benchmark started: models=%s split=%s device=%s task=%s ===",
        args.model,
        args.split,
        args.device,
        args.task,
    )

    models_requested = (
        ALL_MODELS if args.model == "all" else [m.strip() for m in args.model.split(",")]
    )
    # MAMI has no hidden visual content, so preprocessing filters do not help and
    # are NOT run by default. Default to "none" only; pass --filters for an ablation.
    filters_to_run = [f.strip() for f in args.filters.split(",")] if args.filters else ["none"]
    invalid_filters = [f for f in filters_to_run if f not in ALL_FILTERS]
    if invalid_filters:
        raise SystemExit(f"Unknown filter(s): {invalid_filters}. Valid options: {ALL_FILTERS}")

    print(f"Models: {models_requested}")
    print(f"Filters: {filters_to_run}")
    print(f"Split: {args.split}, Limit: {args.limit}, Task: {args.task}")
    if args.task == "singleclass":
        print(f"Prompt: {build_misogyny_prompt()!r}")
    else:
        print(f"Prompt: {build_subtype_prompt()!r}")  # multiclass

    samples = collect_samples(split=args.split, limit=args.limit)
    if not samples:
        raise SystemExit(f"No samples found for split '{args.split}'")
    print(f"Loaded {len(samples)} samples")
    logger.info("Loaded %d samples for split=%s", len(samples), args.split)

    all_results: list[dict[str, Any]] = []

    # Model-outer, filter-inner: load each model once, run all filters, then
    # free its VRAM before moving to the next model.
    for model_name in models_requested:
        print(f"\n===== Model: {model_name} =====")
        logger.info("=== Model %s: running %d filters ===", model_name, len(filters_to_run))
        model_results: list[dict[str, Any]] = []

        for filter_name in filters_to_run:
            print(f"\n--- {model_name} | Filter: {filter_name} | Task: {args.task} ---")
            logger.info("Running model=%s filter=%s task=%s", model_name, filter_name, args.task)
            try:
                if model_name == "clip":
                    result: dict[str, Any] | None = run_clip(
                        samples,
                        filter_name,
                        split=args.split,
                        device=args.device,
                        task=args.task,
                        model_path=args.model_path,
                        use_ocr=args.use_ocr,
                        ocr_engine=args.ocr_engine,
                    )
                else:
                    result = run_generative_model(
                        model_name,
                        samples,
                        filter_name,
                        split=args.split,
                        device=args.device,
                        task=args.task,
                        use_ocr=args.use_ocr,
                        ocr_engine=args.ocr_engine,
                    )
                if result is not None:
                    all_results.append(result)
                    model_results.append(result)
                    print_sample_predictions(result)
                    logger.info(
                        "Result: model=%s filter=%s task=%s acc=%.4f",
                        model_name,
                        filter_name,
                        args.task,
                        result.get("exact_match_accuracy", 0.0),
                    )
            except Exception as exc:
                logger.error(
                    "FATAL error model=%s filter=%s: %s\n%s",
                    model_name,
                    filter_name,
                    exc,
                    traceback.format_exc(),
                )
                print(f"  ERROR running {model_name} with filter={filter_name}: {exc}")

        # Done with this model across all filters: release its VRAM.
        if model_name in GENERATIVE_MODELS:
            free_model_vram(model_name)

        # Persist per-model results to results/{model}_{split}[_multiclass].json
        if model_results:
            RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            split_slug = args.split.replace(",", "_")
            suffix = "_multiclass" if args.task == "multiclass" else ""
            out_path = RESULTS_DIR / f"{model_name}_{split_slug}{suffix}.json"
            out_path.write_text(json.dumps(model_results, indent=2), encoding="utf-8")
            logger.info("Wrote %d results to %s after %s", len(model_results), out_path, model_name)
            print(f"  Wrote {len(model_results)} rows to {out_path}")

    print_summary(all_results)
    logger.info("=== Benchmark completed ===")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("Interrupted by user (Ctrl+C)")
        sys.exit(1)
    except SystemExit:
        raise
    except BaseException as exc:
        # Catch everything including CUDA OOM, segfaults that Python can still catch, etc.
        logger.critical("Unhandled exception crashed the benchmark:\n%s", traceback.format_exc())
        print(f"\nFATAL: {exc}", file=sys.stderr)
        print(f"Full traceback written to: {LOG_FILE}", file=sys.stderr)
        sys.exit(1)
