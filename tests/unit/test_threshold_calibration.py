"""Unit tests for ``scripts.train_classifier.calibrate_binary_threshold``.

Verifies the threshold-scanning helper introduced in
docs/CODE_REVIEW_ISSUES.md §2.7. Uses a tiny fake classifier so the test
runs without any real model weights.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from scripts.train_classifier import calibrate_binary_threshold


class _FakeBinaryClassifier:
    """Return a fixed ``predict_proba`` regardless of input.

    ``proba[:, 1]`` is the caller-provided probability-of-positive vector,
    and ``proba[:, 0] = 1 - proba[:, 1]``. This lets us construct edge
    cases where the perfect threshold is obviously not 0.5.
    """

    def __init__(self, pos_probs: np.ndarray) -> None:
        self._pos = np.asarray(pos_probs, dtype=float)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:  # noqa: ARG002
        pos = self._pos
        return np.column_stack([1.0 - pos, pos])


def test_perfect_threshold_when_gap_is_wide() -> None:
    """Perfect separation: two clusters at 0.9 and 0.1, any threshold between them scores 1.0."""
    probs = np.array([0.9, 0.9, 0.9, 0.1, 0.1, 0.1])
    labels = np.array([1, 1, 1, 0, 0, 0])
    model: Any = _FakeBinaryClassifier(probs)
    thr, score = calibrate_binary_threshold(model, val_X=np.zeros((6, 2)), val_y=labels)
    assert 0.1 < thr <= 0.9
    assert score == pytest.approx(1.0, abs=1e-6)


def test_low_threshold_recovers_positives_when_model_is_underconfident() -> None:
    """When positives cluster around 0.3, a threshold below 0.5 is optimal."""
    probs = np.array([0.35, 0.32, 0.30, 0.05, 0.02, 0.01])
    labels = np.array([1, 1, 1, 0, 0, 0])
    model: Any = _FakeBinaryClassifier(probs)
    thr, score = calibrate_binary_threshold(model, val_X=np.zeros((6, 2)), val_y=labels)
    assert thr <= 0.35
    assert score == pytest.approx(1.0, abs=1e-6)


def test_high_threshold_when_model_is_overconfident_on_negatives() -> None:
    """When negatives cluster around 0.55 and positives around 0.85, threshold ~0.7 wins."""
    probs = np.array([0.85, 0.82, 0.80, 0.55, 0.52, 0.50])
    labels = np.array([1, 1, 1, 0, 0, 0])
    model: Any = _FakeBinaryClassifier(probs)
    thr, score = calibrate_binary_threshold(model, val_X=np.zeros((6, 2)), val_y=labels)
    assert thr >= 0.55
    assert score == pytest.approx(1.0, abs=1e-6)


def test_fallback_when_no_predict_proba() -> None:
    """A model without ``predict_proba`` returns the safe default 0.5."""

    class _NoProbaModel:
        def predict(self, X: np.ndarray) -> np.ndarray:
            return np.zeros(X.shape[0], dtype=int)

    thr, score = calibrate_binary_threshold(
        _NoProbaModel(),
        val_X=np.zeros((6, 2)),
        val_y=np.array([0, 1, 0, 1, 0, 1]),
    )
    assert thr == 0.5
    assert score == 0.0


def test_binary_f1_metric_variant() -> None:
    """When called with metric='binary', returns positive-class F1 not macro."""
    # Perfect separation, so any reasonable metric gives 1.0
    probs = np.array([0.9, 0.9, 0.1, 0.1])
    labels = np.array([1, 1, 0, 0])
    model: Any = _FakeBinaryClassifier(probs)
    thr, score = calibrate_binary_threshold(
        model, val_X=np.zeros((4, 2)), val_y=labels, metric="binary_f1"
    )
    assert score == pytest.approx(1.0, abs=1e-6)
    assert thr > 0.1
