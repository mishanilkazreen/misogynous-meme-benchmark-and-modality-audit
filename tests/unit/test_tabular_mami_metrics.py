"""Tests for scripts.tabular_mami_metrics.

Verifies the tabular sweep post-processor parses the comma-vector prediction
CSVs and computes MAMI-consistent metrics (docs/CODE_REVIEW_ISSUES.md §1.4).
"""

from __future__ import annotations

from pathlib import Path

from models.vlm.metrics_multilabel import compute_mami_score_b
from scripts.tabular_mami_metrics import (
    compute_tabular_task_a_metrics,
    compute_tabular_task_b_metrics,
    parse_multilabel_predictions_csv,
)

SUBTYPE_LABELS = ["shaming", "stereotype", "objectification", "violence"]


def _write(tmp_path: Path, name: str, rows: list[str]) -> Path:
    p = tmp_path / name
    p.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return p


def test_parse_multilabel_csv(tmp_path: Path) -> None:
    csv_path = _write(
        tmp_path,
        "ml.csv",
        [
            "Actual,XGBoost,LightGBM",
            '"0,1,0,0","0,1,0,0","0,0,0,0"',
            '"1,0,1,0","1,0,0,0","1,0,1,0"',
        ],
    )
    parsed = parse_multilabel_predictions_csv(csv_path)
    assert len(parsed["ground_truths"]) == 2
    assert parsed["ground_truths"][0] == {
        "shaming": 0,
        "stereotype": 1,
        "objectification": 0,
        "violence": 0,
    }
    assert parsed["models"]["XGBoost"][1] == {
        "shaming": 1,
        "stereotype": 0,
        "objectification": 0,
        "violence": 0,
    }


def test_task_b_metrics_match_canonical_metric(tmp_path: Path) -> None:
    """The post-processor's mami_score_b must equal the canonical metric fn."""
    csv_path = _write(
        tmp_path,
        "ml.csv",
        [
            "Actual,XGBoost",
            '"0,1,0,0","0,1,0,0"',
            '"1,0,1,0","1,0,1,0"',
            '"0,0,0,0","0,1,0,0"',
            '"0,0,0,1","0,0,0,0"',
        ],
    )
    metrics = compute_tabular_task_b_metrics(csv_path)
    # Recompute directly for the same predictions.
    gts = [
        {"shaming": 0, "stereotype": 1, "objectification": 0, "violence": 0},
        {"shaming": 1, "stereotype": 0, "objectification": 1, "violence": 0},
        {"shaming": 0, "stereotype": 0, "objectification": 0, "violence": 0},
        {"shaming": 0, "stereotype": 0, "objectification": 0, "violence": 1},
    ]
    preds = [
        {"shaming": 0, "stereotype": 1, "objectification": 0, "violence": 0},
        {"shaming": 1, "stereotype": 0, "objectification": 1, "violence": 0},
        {"shaming": 0, "stereotype": 1, "objectification": 0, "violence": 0},
        {"shaming": 0, "stereotype": 0, "objectification": 0, "violence": 0},
    ]
    expected = compute_mami_score_b(preds, gts, SUBTYPE_LABELS)["mami_score_b"]
    assert metrics["XGBoost"]["mami_score_b"] == expected


def test_task_a_metrics(tmp_path: Path) -> None:
    csv_path = _write(
        tmp_path,
        "bin.csv",
        [
            "Actual,XGBoost",
            "1,1",
            "1,1",
            "0,0",
            "0,1",  # one false positive
        ],
    )
    metrics = compute_tabular_task_a_metrics(csv_path)
    xgb = metrics["XGBoost"]
    assert xgb["accuracy"] == 0.75
    assert xgb["recall"] == 1.0  # both positives caught
    assert 0.0 < xgb["precision"] < 1.0  # one FP
