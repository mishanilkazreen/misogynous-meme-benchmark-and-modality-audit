"""
Benchmark CLIP on MAMI 2022 (binary misogyny or multi-label sub-types).

Standalone entry point for CLIP. It reuses the orchestrator's migrated CLIP
logic (``run_clip`` in ``benchmark_vlm_classification``), so its results are
identical to running the orchestrator with ``--model clip``.

Challenge 1 (--task singleclass, default): binary misogyny via CLIP image-text
    similarity against ["misogynistic meme", "not misogynistic meme"].
Challenge 2 (--task multiclass): per-category binary prediction for the four
    sub-types (shaming, stereotype, objectification, violence).

Usage:
    uv run python scripts/benchmark_clip.py --split validation --limit 16 --filters none,grayscale
    uv run python scripts/benchmark_clip.py --task multiclass --split train,validation --device cuda
"""

# datasets (via utils.dataset) must precede torch to avoid OpenMP segfault
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.benchmark_vlm_classification import collect_samples, run_clip

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task",
        default="singleclass",
        choices=["singleclass", "multiclass"],
        help="'singleclass' = binary misogyny (default); 'multiclass' = multi-label sub-types",
    )
    parser.add_argument(
        "--split",
        default="validation",
        help="MAMI split: 'train', 'validation', 'test', or comma-separated e.g. 'train,validation'",
    )
    parser.add_argument(
        "--filters",
        default=None,
        help="Comma-separated filters to run (default: none — MAMI has no hidden visual content). E.g. 'none,blur'",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--model-path",
        default=None,
        help="Path to fine-tuned CLIP checkpoint weights file",
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

    # MAMI has no hidden visual content, so preprocessing filters do not help.
    # Default to "none"; pass --filters explicitly only for a deliberate ablation.
    filters_to_run = [f.strip() for f in args.filters.split(",")] if args.filters else ["none"]

    print(
        f"Task: {args.task} | Split: {args.split} | Filters: {filters_to_run} | "
        f"Limit: {args.limit} | Device: {args.device} | Model Path: {args.model_path}"
    )
    samples = collect_samples(args.split, limit=args.limit)
    if not samples:
        raise SystemExit(f"No samples for split '{args.split}'")
    print(f"Loaded {len(samples)} samples")

    all_results: list[dict[str, Any]] = []
    for flt in filters_to_run:
        print(f"\n--- Filter: {flt} ---")
        result = run_clip(
            samples,
            flt,
            split=args.split,
            device=args.device,
            task=args.task,
            model_path=args.model_path,
            use_ocr=args.use_ocr,
            ocr_engine=args.ocr_engine,
        )
        print(
            f"  acc={result.get('exact_match_accuracy', 0.0):.3f}  f1={result.get('f1', 0.0):.3f}"
        )
        all_results.append(result)

    split_dir = RESULTS_DIR / ("test" if "test" in args.split else "validation")
    split_dir.mkdir(parents=True, exist_ok=True)
    split_slug = args.split.replace(",", "_")
    suffix = "_multiclass" if args.task == "multiclass" else ""
    if args.model_path:
        # Include the checkpoint name so different models/tasks never overwrite each other,
        # e.g. clip_validation_finetuned_singleclass_vit_b_32_quickgelu.json
        model_slug = Path(args.model_path).stem
        prefix = "finetuned_clip_classification_"
        if model_slug.startswith(prefix):
            model_slug = model_slug[len(prefix) :]
        out = split_dir / f"clip_{split_slug}_finetuned_{model_slug}.json"
    else:
        out = split_dir / f"clip_{split_slug}{suffix}.json"
    out.write_text(json.dumps(all_results, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved {len(all_results)} filter rows to {out}")


if __name__ == "__main__":
    main()
