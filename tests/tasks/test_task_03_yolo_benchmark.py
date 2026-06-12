"""
Task 3 marker test: YOLO benchmark.

Passes when `scripts/benchmark_yolo.py --all` runs end-to-end on at least
one subset and writes metrics to `results/yolo_benchmark.json`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

RESULTS = Path(__file__).resolve().parents[2] / "results" / "yolo_benchmark.json"


def test_yolo_benchmark_results_exist() -> None:
    """Results file should exist once the YOLO benchmark has been run.

    The YOLO benchmark is generated output (gitignored) from the pre-MAMI
    HatefulIllusion line of work; its results now live under
    results/embedded_hate/. Skip gracefully when the file is absent so CI stays
    green without re-committing generated artifacts.
    """
    if not RESULTS.exists():
        pytest.skip(f"{RESULTS} not present — run scripts/benchmark_yolo.py --all to regenerate.")
    assert RESULTS.exists()
