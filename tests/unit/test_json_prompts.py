"""Unit tests for the Task B / joint JSON prompt schema.

Covers ``build_subtype_prompt``, ``build_subtype_response``,
``build_joint_prompt``, ``build_joint_response``, and the updated
``extract_subtypes`` + ``extract_joint`` parsers introduced in
docs/CODE_REVIEW_ISSUES.md §6.1 and §6.3.
"""

from __future__ import annotations

import json

import pytest

from models.vlm.classifier import (
    SUBTYPE_LABELS,
    build_joint_prompt,
    build_joint_response,
    build_subtype_prompt,
    build_subtype_response,
    extract_joint,
    extract_subtypes,
)

JOINT_KEYS = ["misogynous", *SUBTYPE_LABELS]


# ---------------------------------------------------------------------------
# build_subtype_prompt / build_subtype_response
# ---------------------------------------------------------------------------


def test_subtype_prompt_is_json_schema() -> None:
    """The prompt describes the JSON schema, not a comma-separated list."""
    prompt = build_subtype_prompt()
    for lbl in SUBTYPE_LABELS:
        assert lbl in prompt
    assert "JSON" in prompt or "json" in prompt
    assert "true" in prompt.lower()


def test_subtype_response_is_valid_json() -> None:
    """A response built from a sample row parses as JSON."""
    row = {"shaming": 1, "stereotype": 0, "objectification": 1, "violence": 0}
    resp = build_subtype_response(row)
    parsed = json.loads(resp)
    assert parsed == {
        "shaming": True,
        "stereotype": False,
        "objectification": True,
        "violence": False,
    }


# ---------------------------------------------------------------------------
# extract_subtypes: JSON path
# ---------------------------------------------------------------------------


def test_extract_subtypes_parses_canonical_json() -> None:
    """Canonical JSON output round-trips through the parser."""
    raw = '{"shaming": true, "stereotype": false, "objectification": true, "violence": false}'
    result = extract_subtypes(raw, SUBTYPE_LABELS)
    assert result == {"shaming": 1, "stereotype": 0, "objectification": 1, "violence": 0}


def test_extract_subtypes_parses_json_with_surrounding_text() -> None:
    """Chat-template artefacts before/after the JSON block do not confuse the parser."""
    raw = 'Answer: {"shaming": true, "stereotype": true, "objectification": false, "violence": false}\n'
    result = extract_subtypes(raw, SUBTYPE_LABELS)
    assert result["shaming"] == 1
    assert result["stereotype"] == 1
    assert result["objectification"] == 0
    assert result["violence"] == 0


def test_extract_subtypes_parses_json_with_numeric_values() -> None:
    """Numeric outputs from the model (0.9, 0.1) are thresholded at 0.5."""
    raw = '{"shaming": 0.9, "stereotype": 0.1, "objectification": 0.8, "violence": 0.4}'
    result = extract_subtypes(raw, SUBTYPE_LABELS)
    assert result == {"shaming": 1, "stereotype": 0, "objectification": 1, "violence": 0}


def test_extract_subtypes_missing_key_defaults_to_zero() -> None:
    """A partial JSON (some keys omitted) sets the missing keys to 0."""
    raw = '{"shaming": true, "stereotype": true}'
    result = extract_subtypes(raw, SUBTYPE_LABELS)
    assert result == {"shaming": 1, "stereotype": 1, "objectification": 0, "violence": 0}


# ---------------------------------------------------------------------------
# extract_subtypes: legacy fallback path
# ---------------------------------------------------------------------------


def test_extract_subtypes_legacy_comma_list_still_works() -> None:
    """Zero-shot models that ignore the schema and emit CSV are still parsed."""
    raw = "shaming, violence"
    result = extract_subtypes(raw, SUBTYPE_LABELS)
    assert result["shaming"] == 1
    assert result["violence"] == 1
    assert result["stereotype"] == 0
    assert result["objectification"] == 0


def test_extract_subtypes_legacy_none_still_zero() -> None:
    """Explicit 'none' still yields all-zeros."""
    result = extract_subtypes("none", SUBTYPE_LABELS)
    assert all(v == 0 for v in result.values())


def test_extract_subtypes_empty_returns_zeros() -> None:
    """Empty response yields all-zeros."""
    result = extract_subtypes("", SUBTYPE_LABELS)
    assert all(v == 0 for v in result.values())


def test_extract_subtypes_malformed_json_falls_back_to_legacy() -> None:
    """Broken JSON like ``{"shaming": ye}`` falls back to word-boundary matching."""
    raw = '{"shaming": ye, and stereotype applies}'
    result = extract_subtypes(raw, SUBTYPE_LABELS)
    # Malformed JSON -> legacy parse finds 'shaming' and 'stereotype' as words.
    assert result["shaming"] == 1
    assert result["stereotype"] == 1


# ---------------------------------------------------------------------------
# build_joint_prompt / build_joint_response / extract_joint
# ---------------------------------------------------------------------------


def test_joint_prompt_mentions_misogynous_and_all_subtypes() -> None:
    """The joint prompt describes the 5-field JSON schema."""
    prompt = build_joint_prompt()
    assert "misogynous" in prompt.lower()
    for lbl in SUBTYPE_LABELS:
        assert lbl in prompt


def test_joint_response_includes_misogynous_field() -> None:
    """The joint response is a 5-field JSON object."""
    row = {"misogynous": 1, "shaming": 0, "stereotype": 1, "objectification": 0, "violence": 0}
    resp = build_joint_response(row)
    parsed = json.loads(resp)
    assert set(parsed.keys()) == set(JOINT_KEYS)
    assert parsed["misogynous"] is True
    assert parsed["stereotype"] is True


def test_extract_joint_roundtrip() -> None:
    """A joint JSON response parses back into a dict of 0/1 ints."""
    row = {"misogynous": 1, "shaming": 1, "stereotype": 0, "objectification": 0, "violence": 1}
    resp = build_joint_response(row)
    result = extract_joint(resp, JOINT_KEYS)
    assert result == row


def test_extract_joint_malformed_returns_zeros() -> None:
    """A response without any JSON returns all-zeros rather than raising."""
    result = extract_joint("not a JSON object at all", JOINT_KEYS)
    assert result == dict.fromkeys(JOINT_KEYS, 0)


def test_extract_joint_missing_keys_default_to_zero() -> None:
    """Partial JSON sets missing keys to 0 for the joint schema too."""
    raw = '{"misogynous": true, "shaming": true}'
    result = extract_joint(raw, JOINT_KEYS)
    assert result["misogynous"] == 1
    assert result["shaming"] == 1
    assert result["stereotype"] == 0
    assert result["objectification"] == 0
    assert result["violence"] == 0


# ---------------------------------------------------------------------------
# Sample-weight helper for the balanced sampler (§6.4)
# ---------------------------------------------------------------------------


def test_compute_vlm_sample_weights_upweights_rare_positives() -> None:
    """Samples with a rare-class positive get a higher weight than base."""
    from scripts.train_vlm import compute_vlm_sample_weights

    records = [
        # Base sample, no positives -> weight 1.0
        {"shaming": 0, "stereotype": 0, "objectification": 0, "violence": 0},
        # Rare-class positive -> weight 1 + 3
        {"shaming": 1, "stereotype": 0, "objectification": 0, "violence": 0},
        # Common-class positive -> weight 1 + 1
        {"shaming": 0, "stereotype": 1, "objectification": 0, "violence": 0},
        # Both -> weight 1 + 3 + 1
        {"shaming": 1, "stereotype": 1, "objectification": 0, "violence": 0},
    ]
    weights = compute_vlm_sample_weights(records)
    assert weights[0] == pytest.approx(1.0, abs=1e-6)
    assert weights[1] == pytest.approx(4.0, abs=1e-6)
    assert weights[2] == pytest.approx(2.0, abs=1e-6)
    assert weights[3] == pytest.approx(5.0, abs=1e-6)
