"""
Benchmark detectors when primed with a catalogue of hateful symbols.

Pipeline (per paper plan, task 5):
    1. Load the symbol catalogue from `data/symbols/catalogue.yaml`
       (free sources first, then fallback to academically cited references).
    2. For each symbol, set it as the active class name for the detector:
         - Ultralytics YOLO: fine-tune or use classification head override
         - YOLO-World / CLIP-YOLO: call `model.set_classes([symbol_name])`
    3. Run detection on HatefulIllusion and report per-symbol recall.

Usage:
    uv run python scripts/benchmark_with_symbol_catalog.py --model yolo-world
"""

from __future__ import annotations

import argparse


def main() -> None:
    """Entry point. TODO: implement in task 5."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="yolo-world")
    parser.add_argument(
        "--catalogue", default="data/symbols/catalogue.yaml", help="Path to symbol catalogue YAML"
    )
    args = parser.parse_args()
    raise NotImplementedError(
        f"Task 5 scaffold. Implement symbol-catalogue-guided detection "
        f"using catalogue={args.catalogue} with model={args.model}."
    )


if __name__ == "__main__":
    main()
