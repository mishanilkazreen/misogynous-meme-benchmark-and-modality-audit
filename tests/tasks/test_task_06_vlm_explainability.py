"""
Task 6 marker test: VLM natural-language explanations.

Passes when `scripts/explain_with_vlm.py` has produced
`results/vlm_explanations.jsonl`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

RESULTS = Path(__file__).resolve().parents[2] / "results" / "vlm_explanations.jsonl"


@pytest.mark.xfail(reason="Task 6 not implemented yet", strict=True)
def test_vlm_explanations_exist() -> None:
    """Explanation JSONL should exist once task 6 is complete."""
    assert RESULTS.exists(), f"Expected {RESULTS}"
