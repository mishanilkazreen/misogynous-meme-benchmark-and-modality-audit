"""
Generate natural-language explanations for flagged images using a VLM.

Pipeline (per paper plan, task 6):
    1. Take an image flagged by the YOLO or VLM pipeline.
    2. Prompt a VLM (e.g. LLaVA, Qwen-VL, GPT-4o) with the image + detection
       output + catalogue context, asking "Why is this image flagged?".
    3. Return a short rationale a human moderator can skim.

Outputs a JSONL file with one record per image for downstream evaluation
of explainability quality.

Usage:
    uv run python scripts/explain_with_vlm.py --model llava --input flagged.json
"""

from __future__ import annotations

import argparse


def main() -> None:
    """Entry point. TODO: implement in task 6."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="llava")
    parser.add_argument("--input", required=False, help="JSON of flagged detections")
    args = parser.parse_args()
    raise NotImplementedError(
        f"Task 6 scaffold. Implement VLM explainability for model={args.model} input={args.input}."
    )


if __name__ == "__main__":
    main()
