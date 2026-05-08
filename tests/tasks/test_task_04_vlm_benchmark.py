"""
Task 4 marker test: VLM benchmark.

Passes when `scripts/benchmark_vlm.py` has produced
`results/vlm_benchmark.json` with the same metric schema as the YOLO run.
"""

from __future__ import annotations

from pathlib import Path

import pytest

RESULTS = Path(__file__).resolve().parents[2] / "results" / "vlm_benchmark.json"


@pytest.mark.xfail(reason="Task 4 not implemented yet", strict=True)
def test_vlm_benchmark_results_exist() -> None:
    """Results file should exist once task 4 is complete."""
    assert RESULTS.exists(), f"Expected {RESULTS} to exist"
