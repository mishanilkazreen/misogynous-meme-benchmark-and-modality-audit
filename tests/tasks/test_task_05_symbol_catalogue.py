"""
Task 5 marker test: symbol catalogue pipeline.

Passes when `data/symbols/catalogue.yaml` exists and
`scripts/benchmark_with_symbol_catalog.py` has produced
`results/symbol_catalogue.json`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
CATALOGUE = ROOT / "data" / "symbols" / "catalogue.yaml"
RESULTS = ROOT / "results" / "symbol_catalogue.json"


@pytest.mark.xfail(reason="Task 5 not implemented yet", strict=True)
def test_symbol_catalogue_present() -> None:
    """Catalogue YAML should exist."""
    assert CATALOGUE.exists(), f"Expected {CATALOGUE}"


@pytest.mark.xfail(reason="Task 5 not implemented yet", strict=True)
def test_symbol_catalogue_results_exist() -> None:
    """Catalogue-guided results should exist."""
    assert RESULTS.exists(), f"Expected {RESULTS}"
