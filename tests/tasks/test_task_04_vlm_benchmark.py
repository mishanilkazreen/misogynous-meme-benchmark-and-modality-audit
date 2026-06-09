"""
Task 4 marker test: VLM benchmark.

Passes when:
- All wrappers in models/vlm/ import without error.
- results/vlm_classification.json exists and is valid JSON.
- At least 2 distinct model names present in results.
- At least 2 distinct filter values present in results (proves ablation ran).
- Every entry has a by_visibility block with keys "1" through "5".
- Required metric keys present in every entry.
- refusal_rate present for generative models.
"""

from __future__ import annotations

import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[2] / "results" / "vlm_classification.json"

_GENERATIVE_MODELS = {"qwen2vl", "gpt4omini", "gemini"}
_DETECTION_MODELS = {
    "yolo-world",
    "aws_rekognition",
    "google_safesearch",
    "aws-rekognition",
    "google-safesearch",
}


def test_vlm_wrappers_import() -> None:
    """All wrappers in models/vlm/ must import without error."""
    from models.vlm.classifier import BaseVLMClassifier, ClassificationResult, build_prompt
    from models.vlm.clip_classifier import CLIPClassifier
    from models.vlm.llava_classifier import LLaVAClassifier

    assert issubclass(CLIPClassifier, BaseVLMClassifier)
    assert issubclass(LLaVAClassifier, BaseVLMClassifier)
    assert callable(build_prompt)
    assert ClassificationResult is not None


def test_vlm_classification_results_exist() -> None:
    """results/vlm_classification.json must exist and be valid JSON."""
    assert RESULTS.exists(), f"Expected {RESULTS} — run benchmark_vlm_classification.py first"
    data = json.loads(RESULTS.read_text(encoding="utf-8"))
    assert isinstance(data, list), "vlm_classification.json must be a JSON array"
    assert len(data) > 0, "vlm_classification.json must not be empty"


def test_at_least_two_models() -> None:
    """At least 2 distinct model names must be present in results."""
    data = json.loads(RESULTS.read_text(encoding="utf-8"))
    models = {entry["model"] for entry in data}
    assert len(models) >= 2, f"Expected >= 2 models, found: {models}"


def test_at_least_two_filters() -> None:
    """At least 2 distinct filter values must be present (proves ablation ran)."""
    data = json.loads(RESULTS.read_text(encoding="utf-8"))
    filters = {entry["filter"] for entry in data}
    assert len(filters) >= 2, f"Expected >= 2 filters, found: {filters}"


def test_by_visibility_keys() -> None:
    """Every entry must have a by_visibility block with keys '1' through '5'."""
    data = json.loads(RESULTS.read_text(encoding="utf-8"))
    for entry in data:
        bv = entry.get("by_visibility")
        assert bv is not None, (
            f"Missing by_visibility in {entry.get('model')}/{entry.get('filter')}"
        )
        for v in ["1", "2", "3", "4", "5"]:
            assert v in bv, f"Missing visibility key '{v}' in {entry.get('model')}"


def test_required_metric_keys() -> None:
    """Every entry must have required metric keys."""
    data = json.loads(RESULTS.read_text(encoding="utf-8"))
    for entry in data:
        model = entry.get("model", "")
        is_detection = model in _DETECTION_MODELS
        if is_detection:
            assert "any_detection_recall" in entry, f"Missing any_detection_recall in {model}"
        else:
            assert "exact_match_accuracy" in entry, f"Missing exact_match_accuracy in {model}"
        assert "f1" in entry, f"Missing f1 in {model}"
        assert "avg_latency_s" in entry, f"Missing avg_latency_s in {model}"


def test_refusal_rate_for_generative_models() -> None:
    """refusal_rate key must be present for generative models."""
    data = json.loads(RESULTS.read_text(encoding="utf-8"))
    generative_entries = [e for e in data if e.get("model") in _GENERATIVE_MODELS]
    for entry in generative_entries:
        assert "refusal_rate" in entry, (
            f"Missing refusal_rate for generative model {entry['model']}"
        )
