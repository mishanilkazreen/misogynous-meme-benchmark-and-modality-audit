"""Unit tests for models/vlm/prompt_templates.py."""

from __future__ import annotations

from pathlib import Path
import tempfile
from typing import Any

import pytest

from models.vlm.prompt_templates import (
    build_baseline_prompt,
    build_catalogue_prompt,
    build_enriched_labels,
    build_per_subset_prompt,
    load_catalogue,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_SAMPLE_CATALOGUE: list[dict[str, Any]] = [
    {
        "name": "swastika",
        "aliases": ["hakenkreuz", "nazi cross"],
        "subset": "hate_symbols",
        "description": "A geometric symbol used by Nazi Germany.",
        "source_url": "https://en.wikipedia.org/wiki/Swastika",
        "license": "public domain",
    },
    {
        "name": "SS bolts",
        "aliases": ["schutzstaffel runes", "lightning bolts"],
        "subset": "hate_symbols",
        "description": "Two lightning-bolt runes used by the Nazi SS.",
        "source_url": "https://en.wikipedia.org/wiki/Sig_rune",
        "license": "public domain",
    },
    {
        "name": "zero",
        "aliases": ["0", "digit 0"],
        "subset": "digits",
        "description": "The digit 0, a closed oval numeral.",
        "source_url": "https://en.wikipedia.org/wiki/0",
        "license": "public domain",
    },
    {
        "name": "one",
        "aliases": ["1", "digit 1"],
        "subset": "digits",
        "description": "The digit 1, a vertical stroke.",
        "source_url": "https://en.wikipedia.org/wiki/1",
        "license": "public domain",
    },
    {
        "name": "anti-Black racial slur",
        "aliases": ["racial slur targeting Black people"],
        "subset": "hate_slangs",
        "description": "A highly offensive racial slur targeting Black people.",
        "source_url": "https://en.wikipedia.org/wiki/Nigger",
        "license": "reference only",
    },
]

_MINIMAL_YAML = """\
symbols:
  - name: swastika
    aliases: [hakenkreuz]
    subset: hate_symbols
    description: A geometric symbol.
    source_url: https://en.wikipedia.org/wiki/Swastika
    license: public domain
"""


# ---------------------------------------------------------------------------
# load_catalogue
# ---------------------------------------------------------------------------


def test_load_catalogue_returns_list() -> None:
    with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
        f.write(_MINIMAL_YAML)
        tmp = Path(f.name)
    try:
        result = load_catalogue(tmp)
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["name"] == "swastika"
    finally:
        tmp.unlink()


def test_load_catalogue_missing_file_raises() -> None:
    with pytest.raises(FileNotFoundError, match="catalogue"):
        load_catalogue("/nonexistent/path/catalogue.yaml")


def test_load_catalogue_empty_symbols_key() -> None:
    with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
        f.write("symbols: []\n")
        tmp = Path(f.name)
    try:
        result = load_catalogue(tmp)
        assert result == []
    finally:
        tmp.unlink()


def test_load_catalogue_real_file() -> None:
    """The actual catalogue must exist and contain 10+ entries."""
    catalogue = load_catalogue("data/symbols/catalogue.yaml")
    assert len(catalogue) >= 10
    for entry in catalogue:
        for field in ("name", "aliases", "subset", "description", "source_url", "license"):
            assert field in entry, f"Missing field '{field}' in entry '{entry.get('name')}'"


# ---------------------------------------------------------------------------
# build_baseline_prompt
# ---------------------------------------------------------------------------


def test_baseline_prompt_contains_labels() -> None:
    prompt = build_baseline_prompt("digits", ["0", "1", "2"], shuffle=False)
    assert '"0"' in prompt
    assert '"1"' in prompt
    assert '"2"' in prompt


def test_baseline_prompt_ends_with_options_line() -> None:
    prompt = build_baseline_prompt("digits", ["0"], shuffle=False)
    assert "Reply with exactly one of the following options:" in prompt


def test_baseline_prompt_matches_build_prompt() -> None:
    from models.vlm.classifier import build_prompt

    labels = ["swastika", "SS bolts"]
    assert build_baseline_prompt("hate_symbols", labels, shuffle=False) == build_prompt(
        "hate_symbols", labels, shuffle=False
    )


# ---------------------------------------------------------------------------
# build_catalogue_prompt
# ---------------------------------------------------------------------------


def test_catalogue_prompt_contains_header() -> None:
    prompt = build_catalogue_prompt("hate_symbols", ["swastika"], _SAMPLE_CATALOGUE, shuffle=False)
    assert "You are analysing an image for hidden hateful content." in prompt
    assert "Known symbols in this category:" in prompt


def test_catalogue_prompt_contains_entry_descriptions() -> None:
    prompt = build_catalogue_prompt("hate_symbols", ["swastika"], _SAMPLE_CATALOGUE, shuffle=False)
    assert "- swastika:" in prompt
    assert "A geometric symbol used by Nazi Germany." in prompt


def test_catalogue_prompt_contains_labels() -> None:
    prompt = build_catalogue_prompt(
        "hate_symbols", ["swastika", "SS bolts"], _SAMPLE_CATALOGUE, shuffle=False
    )
    assert '"swastika"' in prompt or '"SS bolts"' in prompt


def test_catalogue_prompt_respects_max_symbols() -> None:
    prompt = build_catalogue_prompt(
        "hate_symbols", ["swastika"], _SAMPLE_CATALOGUE, shuffle=False, max_symbols=1
    )
    assert "- swastika:" in prompt
    assert "- SS bolts:" not in prompt


def test_catalogue_prompt_filters_by_subset() -> None:
    prompt = build_catalogue_prompt("digits", ["0", "1"], _SAMPLE_CATALOGUE, shuffle=False)
    assert "- zero:" in prompt
    assert "- swastika:" not in prompt


def test_catalogue_prompt_falls_back_to_baseline_when_no_entries() -> None:
    prompt_cat = build_catalogue_prompt("digits", ["0"], [], shuffle=False)
    prompt_base = build_baseline_prompt("digits", ["0"], shuffle=False)
    assert prompt_cat == prompt_base


def test_catalogue_prompt_ends_with_options_line() -> None:
    prompt = build_catalogue_prompt("hate_symbols", ["swastika"], _SAMPLE_CATALOGUE, shuffle=False)
    assert "Reply with exactly one of the following options:" in prompt


# ---------------------------------------------------------------------------
# build_per_subset_prompt
# ---------------------------------------------------------------------------


def test_per_subset_prompt_digits_preamble() -> None:
    prompt = build_per_subset_prompt("digits", ["0"], _SAMPLE_CATALOGUE, shuffle=False)
    assert "digit" in prompt.lower()


def test_per_subset_prompt_hate_symbols_preamble() -> None:
    prompt = build_per_subset_prompt("hate_symbols", ["swastika"], _SAMPLE_CATALOGUE, shuffle=False)
    assert "hate symbol" in prompt.lower() or "symbol" in prompt.lower()


def test_per_subset_prompt_hate_slangs_preamble() -> None:
    prompt = build_per_subset_prompt("hate_slangs", ["slur_a"], _SAMPLE_CATALOGUE, shuffle=False)
    assert "slang" in prompt.lower() or "hateful" in prompt.lower()


def test_per_subset_prompt_includes_all_subset_entries() -> None:
    prompt = build_per_subset_prompt("hate_symbols", ["swastika"], _SAMPLE_CATALOGUE, shuffle=False)
    assert "- swastika:" in prompt
    assert "- SS bolts:" in prompt


def test_per_subset_prompt_excludes_other_subsets() -> None:
    prompt = build_per_subset_prompt("digits", ["0"], _SAMPLE_CATALOGUE, shuffle=False)
    assert "- swastika:" not in prompt


def test_per_subset_prompt_unknown_subset_fallback_preamble() -> None:
    prompt = build_per_subset_prompt("unknown", ["x"], _SAMPLE_CATALOGUE, shuffle=False)
    assert "Hidden hateful content" in prompt


def test_per_subset_prompt_ends_with_options_line() -> None:
    prompt = build_per_subset_prompt("digits", ["0", "1"], _SAMPLE_CATALOGUE, shuffle=False)
    assert "Reply with exactly one of the following options:" in prompt


# ---------------------------------------------------------------------------
# build_enriched_labels
# ---------------------------------------------------------------------------


def test_enriched_labels_match_by_alias() -> None:
    enriched = build_enriched_labels(["0", "1"], "digits", _SAMPLE_CATALOGUE)
    assert len(enriched) == 2
    assert "The digit 0" in enriched[0]
    assert enriched[0].endswith("0")


def test_enriched_labels_match_by_name() -> None:
    enriched = build_enriched_labels(["swastika"], "hate_symbols", _SAMPLE_CATALOGUE)
    assert "A geometric symbol" in enriched[0]
    assert enriched[0].endswith("swastika")


def test_enriched_labels_unmatched_returned_unchanged() -> None:
    enriched = build_enriched_labels(["unknown_label"], "digits", _SAMPLE_CATALOGUE)
    assert enriched == ["unknown_label"]


def test_enriched_labels_preserves_order() -> None:
    labels = ["1", "0"]
    enriched = build_enriched_labels(labels, "digits", _SAMPLE_CATALOGUE)
    assert enriched[0].endswith("1")
    assert enriched[1].endswith("0")


def test_enriched_labels_no_cross_subset_match() -> None:
    # "swastika" is hate_symbols — should not enrich when subset=digits
    enriched = build_enriched_labels(["swastika"], "digits", _SAMPLE_CATALOGUE)
    assert enriched == ["swastika"]


def test_enriched_labels_empty_catalogue() -> None:
    labels = ["0", "1"]
    enriched = build_enriched_labels(labels, "digits", [])
    assert enriched == labels


def test_enriched_labels_case_insensitive_match() -> None:
    enriched = build_enriched_labels(["SWASTIKA"], "hate_symbols", _SAMPLE_CATALOGUE)
    assert "A geometric symbol" in enriched[0]
