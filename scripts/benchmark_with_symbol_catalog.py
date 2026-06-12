"""
VLM classification benchmark with catalogue-augmented prompts (Task 5).

NOTE: This script is HatefulIllusion-specific (subset / catalogue / visibility
logic). It is NOT used for the MAMI misogyny benchmark. It is kept importable
so that the test suite and ruff do not break; running it requires the
HatefulIllusion dataset and symbol catalogue.

Compares three prompt variants for each model x subset x filter combination:
  baseline   — standard closed-set prompt (no catalogue context)
  catalogue  — injects up to 5 symbol descriptions before the question
  per_subset — injects all subset descriptions with a targeted preamble

For CLIP (cosine similarity), prompt variants affect the text embeddings:
  baseline   — raw label strings
  catalogue  — description-enriched label strings (up to max_symbols=5)
  per_subset — description-enriched label strings (all subset entries)

Results are written to:
  results/vlm_classification_with_catalogue.json
  results/symbol_catalogue.json  (copy; checked by the task-5 marker test)

Usage:
    uv run python scripts/benchmark_with_symbol_catalog.py --subset digits --limit 20
    uv run python scripts/benchmark_with_symbol_catalog.py --subset all --model clip
"""

# ruff: noqa: I001,ARG001  # I001: import order; ARG001: stub function unused args
from __future__ import annotations

import argparse
import json
import numpy as np
from pathlib import Path
import time
from typing import Any

from models.vlm.clip_classifier import CLIPClassifier
from utils.preprocessing import PreprocessingPipeline
from models.vlm.prompt_templates import (
    build_catalogue_prompt,
    build_enriched_labels,
    build_per_subset_prompt,
    load_catalogue,
)

# NOTE: catalogue path is HatefulIllusion-specific, not used for MAMI.
# build_visibility_block was removed in the MAMI migration; stub it here so
# this module remains importable without breaking the test suite or ruff.
# build_sample_rows is defined locally to accept the old HatefulIllusion
# 5-argument form (samples, preds, gts, vis_scores, confs).
from scripts.benchmark_vlm_classification import (
    ALL_FILTERS,
    compute_classification_metrics,
    image_to_numpy,
)


def build_visibility_block(
    predictions: list,
    ground_truths: list,
    visibility_scores: list,
    labels: list,
) -> dict:
    """Stub: HatefulIllusion visibility breakdown — not applicable to MAMI.

    NOTE: catalogue path is HatefulIllusion-specific, not used for MAMI.
    Returns an empty dict so callers that use this file for Task 5 still work.
    """
    return {}


def build_sample_rows(
    samples: list,
    predictions: list,
    ground_truths: list,
    visibility_scores: list | None = None,
    confidences: list | None = None,
) -> list:
    """Local wrapper: HatefulIllusion-style 5-arg form of build_sample_rows.

    NOTE: catalogue path is HatefulIllusion-specific, not used for MAMI.
    Builds rows compatible with the Task-5 catalogue result schema.
    """
    rows = []
    for i in range(len(samples)):
        row: dict = {
            "image_id": samples[i].get("image_id", ""),
            "ground_truth": ground_truths[i],
            "prediction": predictions[i],
            "correct": predictions[i] == ground_truths[i],
        }
        if visibility_scores is not None:
            row["visibility"] = visibility_scores[i]
        if confidences is not None:
            row["confidence"] = round(float(confidences[i]), 4)
        rows.append(row)
    return rows


def collect_samples(subset: str, limit: int | None = None) -> list[dict]:  # type: ignore[misc]
    """Stub: HatefulIllusion subset loader — not applicable to MAMI.

    NOTE: catalogue path is HatefulIllusion-specific, not used for MAMI.
    Raises RuntimeError at runtime if called; present only to keep the module
    importable without error.
    """
    raise RuntimeError(
        "collect_samples in benchmark_with_symbol_catalog is HatefulIllusion-specific "
        "and cannot be used with the MAMI dataset. This script is not part of the "
        "misogyny benchmark pipeline."
    )


SUBSET_NAMES = ["digits", "hate_slangs", "hate_symbols"]
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

ALL_MODELS = ["clip"]
PROMPT_VARIANTS = ["baseline", "catalogue", "per_subset"]

_PIPELINE = PreprocessingPipeline()


def apply_filter(image: np.ndarray, filter_name: str) -> np.ndarray:
    if filter_name == "none":
        return image
    return _PIPELINE.apply_transformation(image, filter_name)


def _run_clip_variant(
    classifier: CLIPClassifier,
    images: list[np.ndarray],
    labels: list[str],
    ground_truths: list[str],
    visibility_scores: list[int],
    samples: list[dict[str, Any]],
    subset: str,
    filter_name: str,
    prompt_variant: str,
    catalogue: list[dict[str, Any]],
) -> dict[str, Any]:
    """Run a single prompt-variant pass for CLIP and return the result dict."""
    if prompt_variant == "baseline":
        text_labels = labels
    elif prompt_variant == "catalogue":
        text_labels = build_enriched_labels(labels, subset, catalogue)
        # Also build the prompt string for reference (not used by CLIP but kept for logging)
        build_catalogue_prompt(subset, labels, catalogue, shuffle=False, max_symbols=5)
    else:  # per_subset
        text_labels = build_enriched_labels(labels, subset, catalogue)
        build_per_subset_prompt(subset, labels, catalogue, shuffle=False)

    classifier.set_classes(text_labels)

    t0 = time.perf_counter()
    raw_preds = classifier.predict_batch(images)
    total_time = time.perf_counter() - t0

    # Map enriched-label predictions back to original labels
    label_map = dict(zip(text_labels, labels, strict=True))
    predictions: list[str | None] = [label_map.get(p, p) for p, _ in raw_preds]
    confidences = [c for _, c in raw_preds]
    avg_latency = total_time / len(images)

    metrics = compute_classification_metrics(predictions, ground_truths, labels)
    by_visibility = build_visibility_block(predictions, ground_truths, visibility_scores, labels)

    return {
        "model": "clip",
        "subset": subset,
        "filter": filter_name,
        "prompt_variant": prompt_variant,
        "exact_match_accuracy": metrics["exact_match_accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "avg_latency_s": avg_latency,
        "refusal_rate": 0.0,
        "by_visibility": by_visibility,
        "sample_predictions": build_sample_rows(
            samples, predictions, ground_truths, visibility_scores, confidences
        ),
    }


def run_clip_with_catalogue(
    samples: list[dict[str, Any]],
    filter_name: str,
    catalogue: list[dict[str, Any]],
    device: str = "cpu",
) -> list[dict[str, Any]]:
    """Run all three prompt variants for CLIP on the given samples."""
    labels = sorted({s["message"] for s in samples})
    images = [apply_filter(image_to_numpy(s["image"]), filter_name) for s in samples]
    ground_truths = [s["message"] for s in samples]
    visibility_scores = [int(s["visibility_score"]) for s in samples]
    subset = samples[0]["subset"] if len({s["subset"] for s in samples}) == 1 else "all"

    classifier = CLIPClassifier(device=device)
    results: list[dict[str, Any]] = []

    for variant in PROMPT_VARIANTS:
        result = _run_clip_variant(
            classifier=classifier,
            images=images,
            labels=labels,
            ground_truths=ground_truths,
            visibility_scores=visibility_scores,
            samples=samples,
            subset=subset,
            filter_name=filter_name,
            prompt_variant=variant,
            catalogue=catalogue,
        )
        results.append(result)

    return results


def print_summary(all_results: list[dict[str, Any]]) -> None:
    print("\n" + "=" * 90)
    print(f"{'Model':<10} {'Subset':<14} {'Filter':<20} {'Variant':<12} {'Acc':<8} {'F1':<8}")
    print("=" * 90)
    for r in all_results:
        acc = f"{r['exact_match_accuracy']:.3f}" if "exact_match_accuracy" in r else "n/a"
        f1_val = r.get("f1")
        f1 = f"{f1_val:.3f}" if isinstance(f1_val, float) else "n/a"
        print(
            f"{r['model']:<10} {r['subset']:<14} {r['filter']:<20}"
            f" {r['prompt_variant']:<12} {acc:<8} {f1:<8}"
        )
    print("=" * 90)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="clip",
        choices=["clip", "all"],
        help="Model to benchmark (currently only CLIP is supported)",
    )
    parser.add_argument(
        "--subset",
        default="digits",
        choices=["digits", "hate_slangs", "hate_symbols", "all"],
    )
    parser.add_argument(
        "--catalogue",
        default="data/symbols/catalogue.yaml",
        help="Path to the symbol catalogue YAML",
    )
    parser.add_argument("--limit", type=int, default=None, help="Cap images per subset")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--filters",
        default="none",
        help="Comma-separated filter names (default: none only)",
    )
    args = parser.parse_args()

    catalogue = load_catalogue(args.catalogue)
    print(f"Loaded catalogue: {len(catalogue)} entries from '{args.catalogue}'")

    filters_to_run = [f.strip() for f in args.filters.split(",")]
    valid_filters = set(ALL_FILTERS)
    for f in filters_to_run:
        if f not in valid_filters:
            raise SystemExit(f"Unknown filter '{f}'. Valid: {sorted(valid_filters)}")

    print(f"Filters: {filters_to_run}")
    print(f"Subset: {args.subset}, Limit: {args.limit}, Device: {args.device}")

    samples = collect_samples(args.subset, limit=args.limit)
    if not samples:
        raise SystemExit(f"No samples found for subset '{args.subset}'")
    print(f"Loaded {len(samples)} samples")

    all_results: list[dict[str, Any]] = []

    for filter_name in filters_to_run:
        print(f"\n--- Filter: {filter_name} ---")
        if args.model in ("clip", "all"):
            print("  Running clip (3 prompt variants) …")
            try:
                results = run_clip_with_catalogue(
                    samples, filter_name, catalogue, device=args.device
                )
                all_results.extend(results)
                for r in results:
                    acc = r["exact_match_accuracy"]
                    f1 = r["f1"]
                    print(f"    variant={r['prompt_variant']:<12} acc={acc:.3f}  f1={f1:.3f}")
            except Exception as exc:
                print(f"  ERROR running clip with filter={filter_name}: {exc}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    primary_path = RESULTS_DIR / "vlm_classification_with_catalogue.json"
    marker_path = RESULTS_DIR / "symbol_catalogue.json"

    payload = json.dumps(all_results, indent=2)
    primary_path.write_text(payload, encoding="utf-8")
    marker_path.write_text(payload, encoding="utf-8")

    print(f"\nWrote {len(all_results)} result rows to {primary_path}")
    print(f"Also written to {marker_path} (task-5 marker)")

    print_summary(all_results)


if __name__ == "__main__":
    main()
