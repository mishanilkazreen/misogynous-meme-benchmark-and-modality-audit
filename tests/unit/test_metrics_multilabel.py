"""Unit tests for the multi-label metrics in models/vlm/metrics_multilabel.py.

Covers both the general-purpose metrics (macro F1, micro F1, weighted F1,
exact-match accuracy) and the MAMI 2022 official Task B metric
(``compute_mami_score_b``) which mirrors ``compute_scoreB`` from the
shared-task evaluation script.
"""

from __future__ import annotations

import pytest

from models.vlm.classifier import SUBTYPE_LABELS
from models.vlm.metrics_multilabel import compute_mami_score_b, compute_multilabel_metrics

# ---------------------------------------------------------------------------
# Fixture: a hand-worked example whose MAMI Task B score is verifiable
# ---------------------------------------------------------------------------


@pytest.fixture
def hand_worked_case() -> tuple[list[dict[str, int]], list[dict[str, int]]]:
    """Return a 4-sample case with a known MAMI Task B score of 5/7.

    Per-sub-type breakdown (columns are shaming, stereotype, objectification,
    violence in that order):

    * Shaming     preds=[1,0,1,0] gts=[1,1,0,0] -> pos_f1=0.5 neg_f1=0.5
      binary_macro_f1=0.5 support=2
    * Stereotype  preds=[0,1,1,0] gts=[0,1,1,0] -> pos_f1=1.0 neg_f1=1.0
      binary_macro_f1=1.0 support=2
    * Object.     preds=[1,1,0,0] gts=[1,0,1,0] -> pos_f1=0.5 neg_f1=0.5
      binary_macro_f1=0.5 support=2
    * Violence    preds=[0,0,1,0] gts=[0,0,1,0] -> pos_f1=1.0 neg_f1=1.0
      binary_macro_f1=1.0 support=1

    Weighted sum: 0.5*2 + 1.0*2 + 0.5*2 + 1.0*1 = 5.0
    Total weight: 2 + 2 + 2 + 1 = 7
    MAMI Task B score: 5.0 / 7 = 0.7142857...
    """
    preds = [
        {"shaming": 1, "stereotype": 0, "objectification": 1, "violence": 0},
        {"shaming": 0, "stereotype": 1, "objectification": 1, "violence": 0},
        {"shaming": 1, "stereotype": 1, "objectification": 0, "violence": 1},
        {"shaming": 0, "stereotype": 0, "objectification": 0, "violence": 0},
    ]
    gts = [
        {"shaming": 1, "stereotype": 0, "objectification": 1, "violence": 0},
        {"shaming": 1, "stereotype": 1, "objectification": 0, "violence": 0},
        {"shaming": 0, "stereotype": 1, "objectification": 1, "violence": 1},
        {"shaming": 0, "stereotype": 0, "objectification": 0, "violence": 0},
    ]
    return preds, gts


# ---------------------------------------------------------------------------
# compute_mami_score_b
# ---------------------------------------------------------------------------


def test_mami_score_b_matches_hand_computed_value(
    hand_worked_case: tuple[list[dict[str, int]], list[dict[str, int]]],
) -> None:
    """The score for the hand-worked case must equal 5/7 within rounding."""
    preds, gts = hand_worked_case
    result = compute_mami_score_b(preds, gts, SUBTYPE_LABELS)
    assert result["mami_score_b"] == pytest.approx(5.0 / 7.0, abs=1e-6)


def test_mami_score_b_per_label_binary_macro_f1(
    hand_worked_case: tuple[list[dict[str, int]], list[dict[str, int]]],
) -> None:
    """Each per-sub-type binary-macro F1 in the hand-worked case is known."""
    preds, gts = hand_worked_case
    result = compute_mami_score_b(preds, gts, SUBTYPE_LABELS)
    per_label = result["per_label_binary_macro_f1"]
    assert per_label["shaming"] == pytest.approx(0.5, abs=1e-6)
    assert per_label["stereotype"] == pytest.approx(1.0, abs=1e-6)
    assert per_label["objectification"] == pytest.approx(0.5, abs=1e-6)
    assert per_label["violence"] == pytest.approx(1.0, abs=1e-6)


def test_mami_score_b_per_label_support(
    hand_worked_case: tuple[list[dict[str, int]], list[dict[str, int]]],
) -> None:
    """Per-sub-type support equals the number of gold positives per sub-type."""
    preds, gts = hand_worked_case
    result = compute_mami_score_b(preds, gts, SUBTYPE_LABELS)
    support = result["per_label_support"]
    assert support["shaming"] == 2
    assert support["stereotype"] == 2
    assert support["objectification"] == 2
    assert support["violence"] == 1


def test_mami_score_b_perfect_predictions_is_one() -> None:
    """A model that predicts every gold label correctly scores 1.0."""
    gts = [
        {"shaming": 1, "stereotype": 0, "objectification": 1, "violence": 0},
        {"shaming": 0, "stereotype": 1, "objectification": 0, "violence": 1},
    ]
    preds = [dict(gt) for gt in gts]
    result = compute_mami_score_b(preds, gts, SUBTYPE_LABELS)
    assert result["mami_score_b"] == pytest.approx(1.0, abs=1e-6)


def test_mami_score_b_all_negative_predictions() -> None:
    """A model that always predicts zero across sub-types with positives.

    For each sub-type with any positive gold, pos_f1=0 (no TPs) and neg_f1
    depends on TN/FN. With a mix of positives and negatives per sub-type,
    the score is well below 1.0 but above 0.
    """
    preds = [dict.fromkeys(SUBTYPE_LABELS, 0) for _ in range(4)]
    gts = [
        {"shaming": 1, "stereotype": 0, "objectification": 1, "violence": 0},
        {"shaming": 0, "stereotype": 1, "objectification": 0, "violence": 1},
        {"shaming": 1, "stereotype": 0, "objectification": 1, "violence": 0},
        {"shaming": 0, "stereotype": 1, "objectification": 0, "violence": 1},
    ]
    result = compute_mami_score_b(preds, gts, SUBTYPE_LABELS)
    # All positive-F1 = 0 (no TPs). Per-sub-type binary-macro F1 is neg_f1 / 2.
    # Each sub-type has 2 positives out of 4 samples, so neg_f1 = 2*(0.5*1)/(0.5+1) = 2/3.
    # Binary-macro F1 per sub-type = 1/3. Weighted score = 1/3.
    assert result["mami_score_b"] == pytest.approx(1.0 / 3.0, abs=1e-6)


def test_mami_score_b_empty_returns_zero() -> None:
    """Empty inputs return score 0.0."""
    result = compute_mami_score_b([], [], SUBTYPE_LABELS)
    assert result["mami_score_b"] == 0.0
    assert result["per_label_binary_macro_f1"] == dict.fromkeys(SUBTYPE_LABELS, 0.0)
    assert result["per_label_support"] == dict.fromkeys(SUBTYPE_LABELS, 0)


def test_mami_score_b_length_mismatch_raises() -> None:
    """A predictions/ground_truths length mismatch raises ``ValueError``."""
    preds = [{"shaming": 1, "stereotype": 0, "objectification": 0, "violence": 0}]
    gts: list[dict[str, int]] = []
    with pytest.raises(ValueError):
        compute_mami_score_b(preds, gts, SUBTYPE_LABELS)


def test_mami_score_b_zero_positive_support_is_zero() -> None:
    """If no sub-type has any gold positives, the score is 0.0 by convention."""
    preds = [dict.fromkeys(SUBTYPE_LABELS, 0) for _ in range(3)]
    gts = [dict.fromkeys(SUBTYPE_LABELS, 0) for _ in range(3)]
    result = compute_mami_score_b(preds, gts, SUBTYPE_LABELS)
    assert result["mami_score_b"] == 0.0


def test_mami_score_b_returns_expected_keys() -> None:
    """Result dict has the three documented keys and nothing else."""
    preds = [{"shaming": 1, "stereotype": 0, "objectification": 0, "violence": 0}]
    gts = [{"shaming": 1, "stereotype": 0, "objectification": 0, "violence": 0}]
    result = compute_mami_score_b(preds, gts, SUBTYPE_LABELS)
    assert set(result.keys()) == {
        "mami_score_b",
        "per_label_binary_macro_f1",
        "per_label_support",
    }


def test_mami_score_b_differs_from_positive_only_weighted_f1() -> None:
    """MAMI Task B and the pre-existing ``weighted_f1`` disagree in general.

    ``weighted_f1`` uses positive-class F1 per sub-type; MAMI uses the
    (positive + negative) binary-macro F1. On the all-negative-predictions
    case the two metrics diverge sharply: ``weighted_f1`` is 0.0 because
    there are no TPs, but MAMI's metric is ~1/3 because the negative-class
    F1 is non-zero. If a paper reports ``weighted_f1`` while claiming to
    match MAMI, that mismatch is visible on any real submission.
    """
    preds = [dict.fromkeys(SUBTYPE_LABELS, 0) for _ in range(4)]
    gts = [
        {"shaming": 1, "stereotype": 0, "objectification": 1, "violence": 0},
        {"shaming": 0, "stereotype": 1, "objectification": 0, "violence": 1},
        {"shaming": 1, "stereotype": 0, "objectification": 1, "violence": 0},
        {"shaming": 0, "stereotype": 1, "objectification": 0, "violence": 1},
    ]
    mami = compute_mami_score_b(preds, gts, SUBTYPE_LABELS)
    generic = compute_multilabel_metrics(preds, gts, SUBTYPE_LABELS)
    assert generic["weighted_f1"] == pytest.approx(0.0, abs=1e-6)
    assert mami["mami_score_b"] == pytest.approx(1.0 / 3.0, abs=1e-6)
    assert mami["mami_score_b"] != pytest.approx(generic["weighted_f1"], abs=1e-6)


# ---------------------------------------------------------------------------
# compute_multilabel_metrics: baseline behaviour (existing metric)
# ---------------------------------------------------------------------------


def test_multilabel_metrics_returns_expected_keys() -> None:
    """The general metric returns all documented keys including the MAMI score."""
    preds = [{"shaming": 1, "stereotype": 0, "objectification": 0, "violence": 0}]
    gts = [{"shaming": 1, "stereotype": 0, "objectification": 0, "violence": 0}]
    result = compute_multilabel_metrics(preds, gts, SUBTYPE_LABELS)
    expected_keys = {
        "per_class",
        "macro_f1",
        "micro_f1",
        "weighted_f1",
        "macro_precision",
        "macro_recall",
        "exact_match_accuracy",
        "mami_score_b",
        "per_label_binary_macro_f1",
    }
    assert set(result.keys()) == expected_keys


def test_multilabel_metrics_includes_mami_score_b(
    hand_worked_case: tuple[list[dict[str, int]], list[dict[str, int]]],
) -> None:
    """The composite metric result exposes the MAMI Task B score."""
    preds, gts = hand_worked_case
    result = compute_multilabel_metrics(preds, gts, SUBTYPE_LABELS)
    assert result["mami_score_b"] == pytest.approx(5.0 / 7.0, abs=1e-6)


def test_multilabel_metrics_perfect_predictions() -> None:
    """Perfect predictions yield all-1.0 metrics."""
    gts = [
        {"shaming": 1, "stereotype": 0, "objectification": 1, "violence": 0},
        {"shaming": 0, "stereotype": 1, "objectification": 0, "violence": 1},
    ]
    preds = [dict(gt) for gt in gts]
    result = compute_multilabel_metrics(preds, gts, SUBTYPE_LABELS)
    assert result["macro_f1"] == pytest.approx(1.0, abs=1e-6)
    assert result["micro_f1"] == pytest.approx(1.0, abs=1e-6)
    assert result["weighted_f1"] == pytest.approx(1.0, abs=1e-6)
    assert result["exact_match_accuracy"] == pytest.approx(1.0, abs=1e-6)


def test_multilabel_metrics_all_negative_predictions() -> None:
    """A model that predicts nothing gets zero positive F1 but some exact matches."""
    preds = [dict.fromkeys(SUBTYPE_LABELS, 0) for _ in range(4)]
    gts = [
        dict.fromkeys(SUBTYPE_LABELS, 0),  # exact match (both all-zero)
        {"shaming": 1, "stereotype": 0, "objectification": 0, "violence": 0},
        {"shaming": 0, "stereotype": 1, "objectification": 0, "violence": 0},
        dict.fromkeys(SUBTYPE_LABELS, 0),  # exact match
    ]
    result = compute_multilabel_metrics(preds, gts, SUBTYPE_LABELS)
    assert result["macro_f1"] == 0.0
    assert result["micro_f1"] == 0.0
    assert result["weighted_f1"] == 0.0
    # 2 of 4 samples are exact matches
    assert result["exact_match_accuracy"] == pytest.approx(0.5, abs=1e-6)


def test_multilabel_metrics_empty_input() -> None:
    """Empty inputs return the zero-metrics shape."""
    result = compute_multilabel_metrics([], [], SUBTYPE_LABELS)
    assert result["macro_f1"] == 0.0
    assert result["exact_match_accuracy"] == 0.0
    for lbl in SUBTYPE_LABELS:
        assert result["per_class"][lbl]["f1"] == 0.0
        assert result["per_class"][lbl]["support"] == 0
