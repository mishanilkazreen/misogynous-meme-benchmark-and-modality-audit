"""Generate natural-language explanations using Gemini 1.5 Pro zero-shot API.

Requires GEMINI_API_KEY environment variable in .env.
Outputs a JSONL file with one record per image for explanation evaluation.

Usage:
    uv run python scripts/explain_with_gemini.py --limit 30 --output results/gemini_explanations.jsonl
"""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

from dotenv import load_dotenv
import numpy as np
from PIL import Image
from tqdm import tqdm

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.vlm.prompt_templates import build_explainability_prompt
from utils.dataset import DatasetManager

try:
    from google import genai
    from google.genai import types as genai_types

    _GENAI_AVAILABLE = True
except ModuleNotFoundError:
    _GENAI_AVAILABLE = False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="validation", help="Dataset split (default: validation)")
    parser.add_argument("--limit", type=int, default=30, help="Limit samples (default: 30)")
    parser.add_argument("--model", default="gemini-1.5-pro", help="Gemini model ID")
    parser.add_argument("--output", default="results/gemini_explanations.jsonl", help="Output file")
    args = parser.parse_args()

    if not _GENAI_AVAILABLE:
        print("google-genai SDK is not installed.", file=sys.stderr)
        sys.exit(1)

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("GEMINI_API_KEY not found in environment or .env file.", file=sys.stderr)
        sys.exit(1)

    client = genai.Client(api_key=api_key)

    print(f"Loading MAMI dataset split '{args.split}'...")
    manager = DatasetManager()
    dataset = manager.load_dataset(split=args.split)
    num_samples = min(args.limit, len(dataset))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = []

    print(f"Generating Gemini explanations for {num_samples} samples...")
    for i in tqdm(range(num_samples), desc="Gemini Explanations"):
        sample = dataset[i]
        image = sample["image"]
        import torch

        if isinstance(image, torch.Tensor):
            img_np = (image.detach().cpu().numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)
            pil_img = Image.fromarray(img_np)
        elif isinstance(image, np.ndarray):
            pil_img = Image.fromarray(image).convert("RGB")
        else:
            pil_img = image.convert("RGB")

        ocr_text = sample.get("text", "")
        prompt_text = build_explainability_prompt(ocr_text)

        config = genai_types.GenerateContentConfig(
            temperature=0.0,
            safety_settings=[
                genai_types.SafetySetting(
                    category=genai_types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                    threshold=genai_types.HarmBlockThreshold.BLOCK_NONE,
                ),
                genai_types.SafetySetting(
                    category=genai_types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                    threshold=genai_types.HarmBlockThreshold.BLOCK_NONE,
                ),
                genai_types.SafetySetting(
                    category=genai_types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                    threshold=genai_types.HarmBlockThreshold.BLOCK_NONE,
                ),
                genai_types.SafetySetting(
                    category=genai_types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                    threshold=genai_types.HarmBlockThreshold.BLOCK_NONE,
                ),
            ],
        )

        t0 = time.perf_counter()
        try:
            response = client.models.generate_content(
                model=args.model,
                contents=[pil_img, prompt_text],
                config=config,
            )
            raw_text = response.text or ""
        except Exception as e:
            raw_text = f"{{\"misogynous\": false, \"explanation\": \"Error: {e}\"}}"

        latency = time.perf_counter() - t0

        # Basic parse
        is_misogynous = None
        if '"misogynous": true' in raw_text.lower():
            is_misogynous = True
        elif '"misogynous": false' in raw_text.lower():
            is_misogynous = False

        record = {
            "image_id": sample.get("file_name", str(i)),
            "ground_truth": sample.get("misogynous", 0),
            "predicted_misogynous": is_misogynous,
            "explanation": raw_text,
            "latency_s": latency,
        }
        results.append(record)

    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"Saved {len(results)} explanation records to {output_path}")


if __name__ == "__main__":
    main()
