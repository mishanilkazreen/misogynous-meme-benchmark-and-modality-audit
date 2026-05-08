"""
Benchmark text-prompted VLM detectors on HatefulIllusion.

Target models (per paper plan, task 4):
    - YOLO-World (ultralytics.YOLOWorld, yolov8s-worldv2.pt)
      https://docs.ultralytics.com/models/yolo-world/
      Paper: https://arxiv.org/abs/2401.17270
    - CLIP + YOLO head (reproduce Qu et al. 2024 style pipeline)
      papers/qu-pv-clip-yolov8n-2024.pdf
    - YOLO-UniOW (optional, arXiv:2412.20645)
      https://github.com/THU-MIG/YOLO-UniOW

Same metrics as benchmark_yolo.py so results are directly comparable.

Usage:
    uv run python scripts/benchmark_vlm.py --model yolo-world --subset digits
"""

from __future__ import annotations

import argparse


def main() -> None:
    """Entry point. TODO: implement in task 4."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="yolo-world",
        choices=["yolo-world", "clip-yolo", "yolo-uniow"],
    )
    parser.add_argument(
        "--subset",
        default="digits",
        choices=["digits", "hate_slangs", "hate_symbols", "all"],
    )
    args = parser.parse_args()
    raise NotImplementedError(
        f"Task 4 scaffold. Implement VLM benchmarking for model={args.model} subset={args.subset}."
    )


if __name__ == "__main__":
    main()
