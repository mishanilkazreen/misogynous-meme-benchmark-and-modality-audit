"""
Task 7 marker test: final comparison and paper-ready report.

Passes when `results/comparison_report.md` exists and references both YOLO
and VLM benchmark JSONs.
"""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "results" / "comparison_report.md"


@pytest.mark.xfail(reason="Task 7 not implemented yet", strict=True)
def test_comparison_report_exists() -> None:
    """Markdown report should exist."""
    assert REPORT.exists(), f"Expected {REPORT}"


@pytest.mark.xfail(reason="Task 7 not implemented yet", strict=True)
def test_comparison_report_references_benchmarks() -> None:
    """Report should cite both benchmark JSON files."""
    text = REPORT.read_text()
    assert "yolo_benchmark.json" in text
    assert "clip_validation.json" in text
