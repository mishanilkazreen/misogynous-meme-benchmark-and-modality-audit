"""Integration tests for ``scripts.generate_consolidated_table``.

Verifies that (a) ``load_task_b`` extracts the new MAMI metric fields from
JSON files that include them, (b) it degrades gracefully to ``None`` on
legacy JSONs that predate the metric, and (c) the rendered Task B
aggregate table has the MAMI F1 column as its headline.
"""

from __future__ import annotations

import json
from pathlib import Path

from scripts.generate_consolidated_table import (
    load_task_b,
    render_task_b_aggregate_table,
)


def _write_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_load_task_b_returns_none_for_missing_file(tmp_path: Path) -> None:
    """A missing file returns ``None`` cleanly."""
    result = load_task_b(tmp_path / "does_not_exist.json")
    assert result is None


def test_load_task_b_returns_none_for_none_path() -> None:
    """A ``None`` path returns ``None`` cleanly."""
    assert load_task_b(None) is None


def test_load_task_b_extracts_new_mami_score(tmp_path: Path) -> None:
    """When the JSON has ``mami_score_b``, ``load_task_b`` picks it up."""
    row = {
        "filter": "none",
        "task": "multiclass",
        "exact_match_accuracy": 0.4,
        "macro_f1": 0.35,
        "micro_f1": 0.45,
        "weighted_f1": 0.5,
        "precision": 0.4,
        "recall": 0.5,
        "mami_score_b": 0.62,
        "per_class": {
            "shaming": {"precision": 0.3, "recall": 0.4, "f1": 0.34, "support": 20}
        },
        "per_label_binary_macro_f1": {
            "shaming": 0.55,
            "stereotype": 0.7,
            "objectification": 0.65,
            "violence": 0.45,
        },
    }
    path = tmp_path / "model_multiclass.json"
    _write_json(path, [row])

    result = load_task_b(path)
    assert result is not None
    assert result["mami_score_b"] == 0.62
    assert result["weighted_f1"] == 0.5
    assert result["macro_f1"] == 0.35
    assert "per_label_binary_macro_f1" in result


def test_load_task_b_returns_none_mami_score_for_legacy_json(tmp_path: Path) -> None:
    """Older JSONs that predate the MAMI metric return ``None`` for it.

    The row otherwise loads fine so the legacy diagnostic columns still
    populate; the missing MAMI F1 shows as ``N/A`` in the rendered table.
    This is the intended failure mode: it flags rows that need rerun
    without breaking the report generator.
    """
    row = {
        "filter": "none",
        "task": "multiclass",
        "exact_match_accuracy": 0.4,
        "macro_f1": 0.35,
        "micro_f1": 0.45,
        "weighted_f1": 0.5,
        "precision": 0.4,
        "recall": 0.5,
        "per_class": {},
        # No mami_score_b, no per_label_binary_macro_f1
    }
    path = tmp_path / "legacy_multiclass.json"
    _write_json(path, [row])

    result = load_task_b(path)
    assert result is not None
    assert result["mami_score_b"] is None
    assert result["macro_f1"] == 0.35


def test_load_task_b_picks_filter_none_row_when_multiple(tmp_path: Path) -> None:
    """The ``filter=='none'`` row is selected in the presence of ablations."""
    payload = [
        {"filter": "blur", "task": "multiclass", "mami_score_b": 0.3},
        {"filter": "none", "task": "multiclass", "mami_score_b": 0.62},
        {"filter": "grayscale", "task": "multiclass", "mami_score_b": 0.4},
    ]
    path = tmp_path / "with_filters.json"
    _write_json(path, payload)
    result = load_task_b(path)
    assert result is not None
    assert result["mami_score_b"] == 0.62


def test_render_task_b_aggregate_table_has_mami_f1_header() -> None:
    """The rendered table's first-data column is the MAMI F1."""
    rows = [
        {
            "name": "SomeModel",
            "val_a": {"acc": None, "auc": None, "precision": None, "recall": None, "f1": None},
            "test_a": {"acc": None, "auc": None, "precision": None, "recall": None, "f1": None},
            "val_b": None,
            "test_b": {
                "em": 0.42,
                "mami_score_b": 0.68,
                "macro_f1": 0.48,
                "micro_f1": 0.52,
                "weighted_f1": 0.55,
                "macro_precision": 0.45,
                "macro_recall": 0.51,
                "per_class": None,
                "per_label_binary_macro_f1": None,
            },
        }
    ]
    table = render_task_b_aggregate_table(rows)
    lines = table.splitlines()
    assert "MAMI F1" in lines[0]
    # MAMI F1 should be the column immediately after the model name
    assert lines[0].startswith("| Model | MAMI F1 |")
    # The model row must show the MAMI F1 in the correct column
    data_line = next(line for line in lines if "SomeModel" in line)
    assert "0.6800" in data_line


def test_render_task_b_aggregate_table_shows_na_for_legacy_rows() -> None:
    """Legacy rows without a MAMI score render as N/A rather than crashing."""
    rows = [
        {
            "name": "LegacyModel",
            "val_a": {"acc": None, "auc": None, "precision": None, "recall": None, "f1": None},
            "test_a": {"acc": None, "auc": None, "precision": None, "recall": None, "f1": None},
            "val_b": None,
            "test_b": {
                "em": 0.42,
                "mami_score_b": None,  # legacy
                "macro_f1": 0.48,
                "micro_f1": 0.52,
                "weighted_f1": 0.55,
                "macro_precision": 0.45,
                "macro_recall": 0.51,
                "per_class": None,
                "per_label_binary_macro_f1": None,
            },
        }
    ]
    table = render_task_b_aggregate_table(rows)
    data_line = next(line for line in table.splitlines() if "LegacyModel" in line)
    # The MAMI F1 slot is N/A but the macro F1 diagnostic is populated.
    assert "N/A" in data_line
    assert "0.4800" in data_line
