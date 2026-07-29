"""Generate natural-language explanations for flagged images using a VLM.

Pipeline (per paper plan, task 6):
    1. Take an image flagged by the YOLO or VLM pipeline.
    2. Prompt a VLM (e.g. LLaVA, Qwen-VL) with the image + detection
       output + catalogue context, asking "Why is this image flagged?".
    3. Return a short rationale a human moderator can skim.

Outputs a JSONL file with one record per image for downstream evaluation
of explainability quality.

Usage:
    uv run python scripts/explain_with_vlm.py --model llava --limit 30
    uv run python scripts/explain_with_vlm.py --model qwen --limit 5 --mock
"""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
from tqdm import tqdm

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.vlm.explainer import VLMExplainer
from utils.dataset import DatasetManager


def generate_mock_explanation(sample: dict) -> dict:
    """Generate a realistic mock explanation for local testing without GPU."""
    text = sample.get("text", "")
    is_misogynous = sample.get("misogynous") == 1

    if is_misogynous:
        explanation = (
            f"This meme is classified as misogynistic because the text overlay '{text}' "
            "reinforces harmful stereotypes and demeans women through derogatory context."
        )
    else:
        explanation = (
            f"This meme is not misogynistic. While it contains the text overlay '{text}', "
            "it does not demean, objectify, or express hostility toward women."
        )

    return {
        "misogynous": is_misogynous,
        "explanation": explanation,
        "raw_response": "MOCK_RESPONSE",
        "latency_s": 0.01,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", default="llava", choices=["llava", "qwen"], help="VLM type (llava or qwen)"
    )
    parser.add_argument("--model-id", default=None, help="HuggingFace model ID (optional)")
    parser.add_argument(
        "--split", default="validation", help="Dataset split to evaluate (default: validation)"
    )
    parser.add_argument(
        "--limit", type=int, default=30, help="Limit number of samples to process (default: 30)"
    )
    parser.add_argument("--device", default=None, help="Device to run on (cuda, mps, cpu)")
    parser.add_argument(
        "--quantize", default="none", choices=["none", "4bit", "8bit"], help="Quantization mode"
    )
    parser.add_argument(
        "--output", default="results/vlm_explanations.jsonl", help="Output JSONL filepath"
    )
    parser.add_argument(
        "--mock", action="store_true", help="Run in mock mode (no weights loaded, fast)"
    )
    parser.add_argument(
        "--use-ocr",
        action="store_true",
        default=True,
        help="Incorporate OCR transcripts in prompt context",
    )
    args = parser.parse_args()

    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading MAMI dataset split '{args.split}'...")
    try:
        manager = DatasetManager()
        dataset = manager.load_dataset(split=args.split)
    except Exception as e:
        print(f"Error loading dataset: {e}", file=sys.stderr)
        sys.exit(1)

    num_samples = min(args.limit, len(dataset))
    print(f"Selected {num_samples} samples for explanation generation.")

    explainer = None
    if not args.mock:
        try:
            explainer = VLMExplainer(
                model_type=args.model,
                model_id=args.model_id,
                device=args.device,
                quantize=args.quantize,
            )
        except Exception as e:
            print(f"Could not load VLM model: {e}", file=sys.stderr)
            print("Falling back to MOCK mode for execution.", file=sys.stderr)
            args.mock = True

    results = []

    # Process samples
    for i in tqdm(range(num_samples), desc="Generating Explanations"):
        sample = dataset[i]

        # Load image
        img = sample["image"]
        import torch

        if isinstance(img, torch.Tensor):
            img_np = (img.detach().cpu().numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)
            img = Image.fromarray(img_np)
        elif not isinstance(img, Image.Image):
            # If it's a filepath or array, load/convert
            if isinstance(img, (str, Path)):
                img = Image.open(img)
            elif isinstance(img, np.ndarray):
                img = Image.fromarray(img)

        ocr_text = sample.get("text", "") if args.use_ocr else None

        if args.mock:
            explanation_data = generate_mock_explanation(sample)
        else:
            assert explainer is not None
            try:
                explanation_data = explainer.explain(img, ocr_transcript=ocr_text)
            except Exception as e:
                print(f"\nError processing sample {sample['image_id']}: {e}", file=sys.stderr)
                explanation_data = {
                    "misogynous": None,
                    "explanation": f"Inference failed: {e}",
                    "raw_response": "",
                    "latency_s": 0.0,
                }

        record = {
            "image_id": str(sample["image_id"]),
            "ground_truth": int(sample["misogynous"]),
            "predicted_misogynous": explanation_data["misogynous"],
            "explanation": explanation_data["explanation"],
            "latency_s": explanation_data["latency_s"],
        }
        results.append(record)

    # Save to JSONL
    print(f"Saving results to {output_path}...")
    with output_path.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print("Explanation generation completed successfully!")


if __name__ == "__main__":
    main()
