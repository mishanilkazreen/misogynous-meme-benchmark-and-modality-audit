"""Generate natural-language explanations using the Gemini zero-shot API.

Requires GEMINI_API_KEY environment variable in .env.
Outputs a JSONL file with one record per image for explanation evaluation.

Each record carries the model identifier and dataset split it came from, so
downstream metrics can never be silently attributed to the wrong system. API
failures are recorded in a dedicated ``error`` field and never written into the
``explanation`` field, and the script exits non-zero if any request failed.

Usage:
    uv run python scripts/explain_with_gemini.py --split test --limit 30 \
        --output results/gemini_explanations.jsonl
"""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path
import re
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


def parse_response(raw_text: str) -> tuple[str, bool | None]:
    """Split a raw Gemini response into explanation prose and a binary verdict.

    The model is prompted for a JSON object. Returning the serialised envelope
    as the "explanation" would let braces and key names be scored by BLEU and
    ROUGE, so the prose is extracted here. Falls back to a regex for responses
    wrapped in markdown fences or otherwise not strictly valid JSON.
    """
    if not raw_text.strip():
        return "", None

    text = raw_text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()

    explanation = ""
    verdict: bool | None = None

    try:
        payload = json.loads(text)
        if isinstance(payload, dict):
            explanation = str(payload.get("explanation", "") or "").strip()
            raw_verdict = payload.get("misogynous")
            if isinstance(raw_verdict, bool):
                verdict = raw_verdict
    except json.JSONDecodeError:
        match = re.search(r'"explanation"\s*:\s*"(.+?)"\s*[,}]', text, re.DOTALL)
        if match:
            explanation = match.group(1).strip()

    if verdict is None:
        lowered = text.lower()
        if '"misogynous": true' in lowered or '"misogynous":true' in lowered:
            verdict = True
        elif '"misogynous": false' in lowered or '"misogynous":false' in lowered:
            verdict = False

    return explanation, verdict


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="validation", help="Dataset split (default: validation)")
    parser.add_argument("--limit", type=int, default=30, help="Limit samples (default: 30)")
    parser.add_argument("--model", default="gemini-3.1-flash-lite", help="Gemini model ID")
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

    existing: dict[str, dict] = {}
    if output_path.exists():
        with output_path.open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    rec = json.loads(line)
                    if (
                        rec.get("error") is None
                        and str(rec.get("explanation", "")).strip()
                        and rec.get("model") == args.model
                        and rec.get("split") == args.split
                    ):
                        existing[str(rec["image_id"])] = rec

    results = []
    failures: list[tuple[str, str]] = []

    print(f"Generating {args.model} explanations for {num_samples} '{args.split}' samples...")
    for i in tqdm(range(num_samples), desc="Gemini Explanations"):
        sample = dataset[i]
        sid = str(sample["image_id"])
        if sid in existing:
            results.append(existing[sid])
            continue
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

        # Encode once as PNG so the image can be passed through the SDK's typed
        # Part wrapper instead of as a loosely typed PIL object.
        buffer = io.BytesIO()
        pil_img.save(buffer, format="PNG")
        image_bytes = buffer.getvalue()

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

        contents = genai_types.Content(
            role="user",
            parts=[
                genai_types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
                genai_types.Part.from_text(text=prompt_text),
            ],
        )
        raw_text = ""
        explanation = ""
        is_misogynous: bool | None = None
        error: str | None = None
        latency = 0.0
        for attempt in range(4):
            t0 = time.perf_counter()
            error = None
            raw_text = ""
            try:
                response = client.models.generate_content(
                    model=args.model,
                    contents=contents,
                    config=config,
                )
                raw_text = response.text or ""
            except Exception as e:
                error = f"{type(e).__name__}: {e}"
            latency = time.perf_counter() - t0
            explanation, is_misogynous = parse_response(raw_text)
            if error is None and explanation:
                break
            if error is None and not explanation:
                error = "empty or unparseable response"
            if attempt < 3:
                time.sleep(2.0 * (2**attempt))

        if error is not None:
            failures.append((sid, error))

        record = {
            "image_id": sid,
            "model": args.model,
            "split": args.split,
            "ground_truth": int(sample["misogynous"]),
            "predicted_misogynous": is_misogynous,
            # Prose only. Never a serialised envelope and never an error string,
            # so lexical metrics score what the model actually argued.
            "explanation": explanation,
            "raw_response": raw_text,
            "latency_s": latency,
            "error": error,
        }
        results.append(record)

    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"Saved {len(results)} explanation records to {output_path}")
    print(f"  model: {args.model}   split: {args.split}")

    if failures:
        print(
            f"\nFAILED: {len(failures)}/{len(results)} requests did not return a usable "
            "explanation. The affected records carry an 'error' field and an empty "
            "'explanation', so they are excluded from metrics rather than scored as text.",
            file=sys.stderr,
        )
        for image_id, message in failures[:5]:
            print(f"  {image_id}: {message}", file=sys.stderr)
        if len(failures) > 5:
            print(f"  ... and {len(failures) - 5} more", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
