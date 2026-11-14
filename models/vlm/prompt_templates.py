"""Prompt-building utilities for VLM rationale generation on MAMI 2022.

Legacy symbol-catalogue helpers were retired and removed from this repository.
"""

from __future__ import annotations


def build_explainability_prompt(ocr_text: str | None = None) -> str:
    """Build a prompt requesting a joint classification and natural-language explanation.

    The VLM is asked to return a structured JSON object with:
      - 'misogynous': boolean (true/false)
      - 'explanation': string (1-2 sentence rationale referencing elements in the meme)
    """
    preamble = (
        "You are an expert AI content moderator analyzing memes for misogynistic content.\n"
        "A misogynistic meme is one that demeans, objectifies, reinforces harmful gender stereotypes, "
        "or expresses hostility or violence toward women."
    )
    if ocr_text:
        text_context = f'\nThis meme contains the text overlay: "{ocr_text.strip()}"'
    else:
        text_context = ""

    instructions = (
        "Analyze the visual elements in the image and any text present. "
        "Provide your classification and a short natural-language explanation justifying your decision.\n\n"
        "Your response MUST be a valid JSON object with the following schema:\n"
        "{\n"
        '  "misogynous": true or false,\n'
        '  "explanation": "A 1-2 sentence clear explanation referencing specific text/visual elements of the meme."\n'
        "}\n"
        "Do not include any explanation outside the JSON object. Return ONLY the JSON."
    )
    return f"{preamble}{text_context}\n\n{instructions}"
