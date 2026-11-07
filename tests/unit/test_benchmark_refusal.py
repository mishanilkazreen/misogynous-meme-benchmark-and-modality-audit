"""Tests for the Task B refusal / parse-status logic in benchmark_qwen2vl.

Regression coverage for the bug where a confident all-``false`` JSON object
(a valid non-misogynous prediction) was counted as a refusal, inflating the
reported ``refusal_rate`` with correct negatives.
"""

from __future__ import annotations

from scripts.benchmark_qwen2vl import subtype_parse_status

SUBTYPE_LABELS = ["shaming", "stereotype", "objectification", "violence"]


def test_all_false_json_is_valid_prediction_not_failure() -> None:
    """A confident all-false JSON object must be a valid negative, not a failure."""
    resp = '{"shaming": false, "stereotype": false, "objectification": false, "violence": false}'
    assert subtype_parse_status(resp, SUBTYPE_LABELS) == "json"


def test_positive_json_is_parsed() -> None:
    resp = '{"shaming": false, "stereotype": true, "objectification": false, "violence": false}'
    assert subtype_parse_status(resp, SUBTYPE_LABELS) == "json"


def test_json_embedded_in_text_is_parsed() -> None:
    resp = 'Answer: {"shaming": true, "stereotype": false, "objectification": false, "violence": false}'
    assert subtype_parse_status(resp, SUBTYPE_LABELS) == "json"


def test_explicit_none_is_valid_negative() -> None:
    assert subtype_parse_status("none", SUBTYPE_LABELS) == "none"


def test_free_text_label_match() -> None:
    assert subtype_parse_status("This is stereotype and shaming", SUBTYPE_LABELS) == "labels"


def test_empty_is_failure() -> None:
    assert subtype_parse_status("", SUBTYPE_LABELS) == "empty"
    assert subtype_parse_status("   ", SUBTYPE_LABELS) == "empty"


def test_unparseable_is_failure() -> None:
    assert subtype_parse_status("I think this meme is quite funny", SUBTYPE_LABELS) == "unparseable"


def test_malformed_json_falls_back_to_unparseable() -> None:
    # Truncated / malformed JSON with no recoverable label keyword.
    assert subtype_parse_status('{"sha', SUBTYPE_LABELS) == "unparseable"


def test_genuine_failure_definition_matches_status() -> None:
    """Only empty/unparseable count as parse failures; json/none/labels do not."""
    failures = {"empty", "unparseable"}
    assert subtype_parse_status("", SUBTYPE_LABELS) in failures
    assert subtype_parse_status("random prose", SUBTYPE_LABELS) in failures
    all_false = (
        '{"shaming": false, "stereotype": false, "objectification": false, "violence": false}'
    )
    assert subtype_parse_status(all_false, SUBTYPE_LABELS) not in failures
    assert subtype_parse_status("none", SUBTYPE_LABELS) not in failures
