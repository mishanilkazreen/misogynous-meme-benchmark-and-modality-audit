"""Unit tests for build_prompt in models/vlm/classifier.py."""

from __future__ import annotations

from models.vlm.classifier import build_prompt


def test_build_prompt_digits_includes_labels() -> None:
    labels = ["0", "1", "2"]
    prompt = build_prompt("digits", labels)
    assert '"0"' in prompt
    assert '"1"' in prompt
    assert '"2"' in prompt


def test_build_prompt_digits_subset_instruction() -> None:
    prompt = build_prompt("digits", ["0"])
    assert "digit" in prompt.lower()


def test_build_prompt_hate_symbols_subset_instruction() -> None:
    prompt = build_prompt("hate_symbols", ["swastika"])
    assert "hate symbol" in prompt.lower() or "symbol" in prompt.lower()


def test_build_prompt_hate_slangs_subset_instruction() -> None:
    prompt = build_prompt("hate_slangs", ["slur_a"])
    assert "hidden" in prompt.lower() or "slang" in prompt.lower()


def test_build_prompt_unknown_subset_fallback() -> None:
    prompt = build_prompt("unknown_subset", ["a", "b"])
    assert '"a"' in prompt
    assert '"b"' in prompt


def test_build_prompt_options_line_format() -> None:
    labels = ["cat", "dog"]
    prompt = build_prompt("digits", labels)
    assert "Reply with exactly one of the following options:" in prompt
