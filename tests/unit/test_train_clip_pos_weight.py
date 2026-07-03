"""Unit tests for the Task B ``pos_weight`` computation in ``scripts.train_clip``.

The MAMI Task B sub-types are heavily imbalanced (shaming and violence
around 13 % positive each; see ``results/challenge_b_label_analysis.md``).
Plain BCE lets the model collapse to always-negative on the rare classes.
``compute_multilabel_pos_weight`` produces per-label ``N_neg / N_pos`` so
``F.binary_cross_entropy_with_logits(logits, targets, pos_weight=...)``
rebalances the loss. Fixed by docs/CODE_REVIEW_ISSUES.md §1.2.
"""

from __future__ import annotations

import pytest
import torch

from scripts.train_clip import MULTILABEL_ORDER, compute_multilabel_pos_weight


def test_pos_weight_returns_float_tensor_of_correct_shape() -> None:
    """Output is a 1-D float tensor with one entry per sub-type."""
    records = [
        {"shaming": 1, "stereotype": 0, "objectification": 0, "violence": 0},
        {"shaming": 0, "stereotype": 1, "objectification": 1, "violence": 0},
    ]
    weights = compute_multilabel_pos_weight(records)
    assert weights.dtype == torch.float32
    assert weights.shape == (len(MULTILABEL_ORDER),)


def test_pos_weight_balanced_class_is_one() -> None:
    """A perfectly balanced label (50 % positive) gets ``pos_weight = 1.0``."""
    records = [
        {"shaming": 1, "stereotype": 0, "objectification": 0, "violence": 0},
        {"shaming": 0, "stereotype": 0, "objectification": 0, "violence": 0},
    ]
    weights = compute_multilabel_pos_weight(records)
    # shaming: 1 pos, 1 neg -> weight = 1/1 = 1.0
    idx = MULTILABEL_ORDER.index("shaming")
    assert weights[idx].item() == pytest.approx(1.0, abs=1e-6)


def test_pos_weight_rare_class_gets_high_weight() -> None:
    """A label with 20 % positives gets ``pos_weight = 4.0``."""
    # 1 positive, 4 negatives -> pos_weight = 4/1 = 4.0
    records = [
        {"shaming": 1, "stereotype": 0, "objectification": 0, "violence": 0},
        {"shaming": 0, "stereotype": 0, "objectification": 0, "violence": 0},
        {"shaming": 0, "stereotype": 0, "objectification": 0, "violence": 0},
        {"shaming": 0, "stereotype": 0, "objectification": 0, "violence": 0},
        {"shaming": 0, "stereotype": 0, "objectification": 0, "violence": 0},
    ]
    weights = compute_multilabel_pos_weight(records)
    idx = MULTILABEL_ORDER.index("shaming")
    assert weights[idx].item() == pytest.approx(4.0, abs=1e-6)


def test_pos_weight_zero_positives_falls_back_to_one() -> None:
    """A label with no positives falls back to 1.0 (no rebalancing)."""
    records = [
        {"shaming": 0, "stereotype": 0, "objectification": 0, "violence": 0} for _ in range(3)
    ]
    weights = compute_multilabel_pos_weight(records)
    for lbl_idx in range(len(MULTILABEL_ORDER)):
        assert weights[lbl_idx].item() == pytest.approx(1.0, abs=1e-6)


def test_pos_weight_ordering_matches_multilabel_order() -> None:
    """Weights come back in ``MULTILABEL_ORDER`` order (matches label stacking)."""
    # Construct a case with distinct weights per label so ordering matters
    records = [
        # shaming: 1/4 pos -> weight 3
        # stereotype: 2/4 pos -> weight 1
        # objectification: 3/4 pos -> weight 1/3
        # violence: 4/4 pos -> weight 0
        {"shaming": 1, "stereotype": 1, "objectification": 1, "violence": 1},
        {"shaming": 0, "stereotype": 1, "objectification": 1, "violence": 1},
        {"shaming": 0, "stereotype": 0, "objectification": 1, "violence": 1},
        {"shaming": 0, "stereotype": 0, "objectification": 0, "violence": 1},
    ]
    weights = compute_multilabel_pos_weight(records)
    assert weights[MULTILABEL_ORDER.index("shaming")].item() == pytest.approx(3.0)
    assert weights[MULTILABEL_ORDER.index("stereotype")].item() == pytest.approx(1.0)
    assert weights[MULTILABEL_ORDER.index("objectification")].item() == pytest.approx(1 / 3)
    assert weights[MULTILABEL_ORDER.index("violence")].item() == pytest.approx(0.0)


def test_pos_weight_matches_mami_train_ballpark() -> None:
    """Sanity check against the published MAMI eval-set positive rates.

    The training-set rates are similar to the eval-set ones documented in
    ``results/challenge_b_label_analysis.md``: shaming ~13.5 %,
    stereotype ~31.3 %, objectification ~28.8 %, violence ~12.7 %.
    Reproducing them approximately here confirms the formula.
    """

    # 100 samples with the eval-set positive rates rounded to whole counts
    def _mk(pos_shaming: int, pos_ster: int, pos_obj: int, pos_viol: int) -> list[dict[str, int]]:
        out: list[dict[str, int]] = []
        for i in range(100):
            out.append(
                {
                    "shaming": 1 if i < pos_shaming else 0,
                    "stereotype": 1 if i < pos_ster else 0,
                    "objectification": 1 if i < pos_obj else 0,
                    "violence": 1 if i < pos_viol else 0,
                }
            )
        return out

    records = _mk(pos_shaming=14, pos_ster=31, pos_obj=29, pos_viol=13)
    weights = compute_multilabel_pos_weight(records)
    # Expected: shaming (86/14) ~ 6.14, stereotype (69/31) ~ 2.23,
    # objectification (71/29) ~ 2.45, violence (87/13) ~ 6.69
    assert weights[MULTILABEL_ORDER.index("shaming")].item() == pytest.approx(86 / 14, abs=1e-6)
    assert weights[MULTILABEL_ORDER.index("stereotype")].item() == pytest.approx(69 / 31, abs=1e-6)
    assert weights[MULTILABEL_ORDER.index("objectification")].item() == pytest.approx(
        71 / 29, abs=1e-6
    )
    assert weights[MULTILABEL_ORDER.index("violence")].item() == pytest.approx(87 / 13, abs=1e-6)
