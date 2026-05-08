"""
Benchmark standard Ultralytics YOLO detectors on HatefulIllusion.

"Standard" here means the YOLO is trained on a fixed class list
(e.g. COCO) and fine-tuned on the target dataset, in contrast with
the text-prompted VLM detectors in `benchmark_vlm.py`.

Target models (per paper plan, task 3):
    yolov8n.pt, yolov10n.pt, yolo11n.pt, yolo12n.pt, yolo26n.pt

Metrics reported: mAP50, mAP50-95, precision, recall, F1, inference time,
stratified by visibility level (1-5) and subset (digits / hate_slangs /
hate_symbols).

Usage:
    uv run python scripts/benchmark_yolo.py --model yolov8n.pt --subset digits
    uv run python scripts/benchmark_yolo.py --all

Docs: https://docs.ultralytics.com/modes/val/
"""

from __future__ import annotations

import argparse


def main() -> None:
    """Entry point. TODO: implement in task 3."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help="Ultralytics checkpoint name, e.g. yolov8n.pt, yolo26n.pt",
    )
    parser.add_argument(
        "--subset",
        default="digits",
        choices=["digits", "hate_slangs", "hate_symbols", "all"],
        help="HatefulIllusion subset to evaluate on",
    )
    parser.add_argument("--all", action="store_true", help="Benchmark all five models")
    args = parser.parse_args()
    raise NotImplementedError(
        f"Task 3 scaffold. Implement YOLO benchmarking for model={args.model} subset={args.subset}."
    )


if __name__ == "__main__":
    main()
