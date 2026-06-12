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

from scripts.benchmark_vlm_classification import ALL_FILTERS, collect_samples, run_clip

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
        help="Comma-separated filters to run (default: all). E.g. 'none,blur,grayscale'",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    filters_to_run = [f.strip() for f in args.filters.split(",")] if args.filters else ALL_FILTERS

    print(
        f"Task: {args.task} | Split: {args.split} | Filters: {filters_to_run} | "
        f"Limit: {args.limit} | Device: {args.device}"
    )
    samples = collect_samples(args.split, limit=args.limit)
    if not samples:
        raise SystemExit(f"No samples for split '{args.split}'")
    print(f"Loaded {len(samples)} samples")

    all_results: list[dict[str, Any]] = []
    for flt in filters_to_run:
        print(f"\n--- Filter: {flt} ---")
        result = run_clip(samples, flt, split=args.split, device=args.device, task=args.task)
        print(
            f"  acc={result.get('exact_match_accuracy', 0.0):.3f}  f1={result.get('f1', 0.0):.3f}"
        )
        all_results.append(result)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "_multiclass" if args.task == "multiclass" else ""
    out = RESULTS_DIR / f"clip_{args.split}{suffix}.json"
    out.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"\nSaved {len(all_results)} filter rows to {out}")


if __name__ == "__main__":
    main()
