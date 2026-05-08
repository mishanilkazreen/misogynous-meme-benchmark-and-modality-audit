"""
Task 3 marker test: YOLO benchmark.

Passes when `scripts/benchmark_yolo.py --all` runs end-to-end on at least
one subset and writes metrics to `results/yolo_benchmark.json`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

RESULTS = Path(__file__).resolve().parents[2] / "results" / "yolo_benchmark.json"


@pytest.mark.xfail(reason="Task 3 not implemented yet", strict=True)
def test_yolo_benchmark_results_exist() -> None:
    """Results file should exist once task 3 is complete."""
    assert RESULTS.exists(), f"Expected {RESULTS} to exist"
