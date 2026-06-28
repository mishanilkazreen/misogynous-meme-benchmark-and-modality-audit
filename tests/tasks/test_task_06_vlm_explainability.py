"""
Task 6 marker test: VLM natural-language explanations.

Passes when `scripts/explain_with_vlm.py` has produced
`results/vlm_explanations.jsonl`.
"""

from __future__ import annotations

from pathlib import Path

RESULTS = Path(__file__).resolve().parents[2] / "results" / "vlm_explanations.jsonl"


def test_vlm_explanations_exist() -> None:
    """Explanation JSONL should exist once task 6 is complete."""
    assert RESULTS.exists(), f"Expected {RESULTS}"
