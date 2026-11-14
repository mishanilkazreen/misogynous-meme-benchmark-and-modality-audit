"""Unit tests for MAMI 2022 rationale prompt utilities."""

from __future__ import annotations

from models.vlm.prompt_templates import build_explainability_prompt


def test_explainability_prompt_contains_json_schema() -> None:
    """Explainability prompt requires JSON fields 'misogynous' and 'explanation'."""
    prompt = build_explainability_prompt()
    assert '"misogynous"' in prompt
    assert '"explanation"' in prompt
    assert "text overlay" not in prompt


def test_explainability_prompt_includes_ocr_text() -> None:
    """Provided OCR text is embedded and stripped cleanly."""
    prompt = build_explainability_prompt("  sample meme caption  ")
    assert '"sample meme caption"' in prompt
    assert "text overlay" in prompt
