"""Unit tests for the text-source helpers in ``scripts.extract_embeddings``.

Covers the ``--text-source`` refactor from docs/CODE_REVIEW_ISSUES.md §1.3
and §7.8, plus the backward-compatible ``--use-ocr`` alias.
"""

from __future__ import annotations

import pytest

from utils.text_source import (
    combine_texts,
    filename_suffix_for_source,
    resolve_text_source,
)

# ---------------------------------------------------------------------------
# resolve_text_source
# ---------------------------------------------------------------------------


def test_resolve_defaults_to_provided_when_nothing_set() -> None:
    """No ``--text-source`` and no ``--use-ocr`` -> ``provided``."""
    assert resolve_text_source(None, use_ocr_flag=False) == "provided"


def test_resolve_use_ocr_alias_maps_to_ocr() -> None:
    """Legacy ``--use-ocr`` on its own resolves to ``ocr``."""
    assert resolve_text_source(None, use_ocr_flag=True) == "ocr"


def test_resolve_explicit_provided_wins_over_use_ocr() -> None:
    """Explicit ``--text-source provided`` wins even if ``--use-ocr`` is set."""
    result = resolve_text_source("provided", use_ocr_flag=True)
    assert result == "provided"


def test_resolve_explicit_ocr_wins() -> None:
    """Explicit ``--text-source ocr`` is honoured."""
    assert resolve_text_source("ocr", use_ocr_flag=False) == "ocr"
    assert resolve_text_source("ocr", use_ocr_flag=True) == "ocr"


def test_resolve_explicit_combined_wins() -> None:
    """Explicit ``--text-source combined`` is honoured."""
    assert resolve_text_source("combined", use_ocr_flag=False) == "combined"


# ---------------------------------------------------------------------------
# combine_texts
# ---------------------------------------------------------------------------


def test_combine_texts_appends_unique_ocr_tokens() -> None:
    """Tokens in the OCR result that are absent from the provided text are appended."""
    provided = "hello world"
    ocr = "hello world watermark"
    combined = combine_texts(provided, ocr)
    assert "hello world" in combined
    assert "watermark" in combined


def test_combine_texts_dedup_is_case_insensitive() -> None:
    """A duplicate token with different casing is deduplicated."""
    provided = "Hello World"
    ocr = "HELLO world extra"
    combined = combine_texts(provided, ocr)
    # 'extra' should be appended only once
    assert combined.lower().count("hello") == 1
    assert "extra" in combined


def test_combine_texts_preserves_provided_casing() -> None:
    """The provided text's original casing is preserved in the base string."""
    provided = "Woman Driver"
    ocr = "woman driver joke"
    combined = combine_texts(provided, ocr)
    # Provided casing is unchanged; joke is appended.
    assert combined.startswith("Woman Driver")
    assert combined.endswith("joke")


def test_combine_texts_empty_provided_returns_ocr() -> None:
    """When the provided text is empty, the combined result is the OCR output."""
    provided = ""
    ocr = "some overlay text"
    combined = combine_texts(provided, ocr)
    assert combined == "some overlay text"


def test_combine_texts_empty_ocr_returns_provided() -> None:
    """When the OCR output is empty, the combined result is the provided text."""
    provided = "some overlay text"
    ocr = ""
    combined = combine_texts(provided, ocr)
    assert combined == "some overlay text"


def test_combine_texts_both_empty_returns_empty() -> None:
    """Both empty -> empty string, not whitespace."""
    assert combine_texts("", "") == ""


# ---------------------------------------------------------------------------
# filename_suffix_for_source
# ---------------------------------------------------------------------------


def test_suffix_provided_is_empty() -> None:
    """``provided`` produces no suffix, matching pre-``--use-ocr`` filenames."""
    assert filename_suffix_for_source("provided", "paddleocr") == ""
    assert filename_suffix_for_source("provided", "easyocr") == ""


def test_suffix_ocr_matches_legacy_pattern() -> None:
    """``ocr`` filenames match the historical ``_ocr_<engine>`` suffix."""
    assert filename_suffix_for_source("ocr", "paddleocr") == "_ocr_paddleocr"
    assert filename_suffix_for_source("ocr", "easyocr") == "_ocr_easyocr"


def test_suffix_combined_encodes_source_and_engine() -> None:
    """``combined`` filenames are distinct from both provided and ocr variants."""
    assert filename_suffix_for_source("combined", "paddleocr") == "_combined_paddleocr"
    assert filename_suffix_for_source("combined", "easyocr") == "_combined_easyocr"


def test_suffix_unknown_source_raises() -> None:
    """A typo in the text source triggers a ``ValueError`` rather than a silent bug."""
    with pytest.raises(ValueError):
        filename_suffix_for_source("providedd", "paddleocr")


def test_resolve_unknown_source_raises() -> None:
    """A typo in ``--text-source`` fails loud, not silent."""
    with pytest.raises(ValueError):
        resolve_text_source("providedd", use_ocr_flag=False)
