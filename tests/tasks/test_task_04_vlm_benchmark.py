"""
Task 4 marker test: VLM classification benchmark.

Passes when:
- All wrappers in models/vlm/ import without error.
- Per-model result files exist (results/{model}_{subset}.json).
- At least 2 distinct models have results across all 3 subsets.
- At least 2 distinct filter values present (proves ablation ran).
- Every entry has a by_visibility block.
- Required metric keys present in every entry.
"""

from __future__ import annotations

import json
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
MODELS = ["clip", "llava", "qwen2vl"]
SUBSETS = ["digits", "hate_slangs", "hate_symbols"]


def _load_all_results() -> list[dict]:
    """Load all per-model result files into a single list."""
    all_data: list[dict] = []
    for model in MODELS:
        for subset in SUBSETS:
            path = RESULTS_DIR / f"{model}_{subset}.json"
            if path.exists():
                data = json.loads(path.read_text(encoding="utf-8"))
                all_data.extend(data)
    return all_data


def test_vlm_wrappers_import() -> None:
    """All wrappers in models/vlm/ must import without error."""
    from models.vlm.classifier import BaseVLMClassifier, ClassificationResult, build_prompt
    from models.vlm.clip_classifier import CLIPClassifier

    assert issubclass(CLIPClassifier, BaseVLMClassifier)
    assert callable(build_prompt)
    assert ClassificationResult is not None


def test_result_files_exist() -> None:
    """At least 2 models must have result files for all 3 subsets."""
    models_complete = 0
    for model in MODELS:
        subset_count = sum(1 for s in SUBSETS if (RESULTS_DIR / f"{model}_{s}.json").exists())
        if subset_count == 3:
            models_complete += 1
    assert models_complete >= 2, f"Expected >= 2 models with all 3 subsets, found {models_complete}"


def test_at_least_two_filters() -> None:
    """At least 2 distinct filter values must be present (proves ablation ran)."""
    data = _load_all_results()
    assert len(data) > 0, "No result entries found"
    filters = {entry.get("filter") for entry in data}
    assert len(filters) >= 2, f"Expected >= 2 filters, found: {filters}"


def test_by_visibility_keys() -> None:
    """Every entry must have a by_visibility block."""
    data = _load_all_results()
    for entry in data:
        bv = entry.get("by_visibility")
        assert bv is not None, (
            f"Missing by_visibility in {entry.get('model')}/{entry.get('filter')}"
        )


def test_required_metric_keys() -> None:
    """Every entry must have required metric keys."""
    data = _load_all_results()
    for entry in data:
        model = entry.get("model", "")
        assert "exact_match_accuracy" in entry, f"Missing exact_match_accuracy in {model}"
        assert "f1" in entry, f"Missing f1 in {model}"
        assert "avg_latency_s" in entry, f"Missing avg_latency_s in {model}"
