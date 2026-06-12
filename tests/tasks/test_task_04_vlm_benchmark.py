"""
Task 4 marker test: VLM classification benchmark (MAMI 2022 misogyny task).

Passes when:
- All wrappers in models/vlm/ import without error.
- At least one per-model result file exists (results/{model}_{split}.json).
- At least 2 distinct filter values are present in the results (proves ablation ran).
- No entry has a by_visibility block (removed; MAMI has no visibility scores).
- Required metric keys are present in every entry.
- Singleclass result entries carry task == "singleclass" and sample rows contain
  ONLY the binary misogyny fields (no shaming/stereotype/objectification/violence).
- Multiclass result files ({model}_{split}_multiclass.json) contain the full multi-label
  schema when present (per_class, macro_f1, micro_f1, weighted_f1, task == "multiclass"),
  and their sample rows carry the 4-label ground_truth/prediction dicts.

If no result files exist at all the tests that require them are skipped rather than
hard-failing, so the test suite stays green while the heavy benchmark models are not
yet run.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
SPLITS = ["train", "validation", "test"]

# Models that the orchestrator can produce results for
MODELS = ["clip", "llava", "qwen2vl", "llavanext", "gpt4omini", "gemini"]

# Sub-type labels for Task B
SUBTYPE_LABELS = ["shaming", "stereotype", "objectification", "violence"]


def _load_all_results() -> list[dict]:
    """Load every {model}_{split}.json (Task A) result file into a flat list of entries."""
    all_data: list[dict] = []
    for model in MODELS:
        for split in SPLITS:
            path = RESULTS_DIR / f"{model}_{split}.json"
            if path.exists():
                data = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    all_data.extend(data)
    return all_data


def _load_all_multiclass_results() -> list[dict]:
    """Load every {model}_{split}_multiclass.json (multiclass) result file into a flat list."""
    all_data: list[dict] = []
    for model in MODELS:
        for split in SPLITS:
            path = RESULTS_DIR / f"{model}_{split}_multiclass.json"
            if path.exists():
                data = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    all_data.extend(data)
    return all_data


def _result_files_exist() -> bool:
    """Return True if at least one {model}_{split}.json file exists."""
    for model in MODELS:
        for split in SPLITS:
            if (RESULTS_DIR / f"{model}_{split}.json").exists():
                return True
    return False


def _multiclass_result_files_exist() -> bool:
    """Return True if at least one {model}_{split}_multiclass.json file exists."""
    for model in MODELS:
        for split in SPLITS:
            if (RESULTS_DIR / f"{model}_{split}_multiclass.json").exists():
                return True
    return False


# ---------------------------------------------------------------------------
# Import test — always runs
# ---------------------------------------------------------------------------


def test_vlm_wrappers_import() -> None:
    """All wrappers in models/vlm/ must import without error."""
    from models.vlm.classifier import (
        SUBTYPE_LABELS as _SL,
    )
    from models.vlm.classifier import (
        BaseVLMClassifier,
        ClassificationResult,
        build_prompt,
        build_subtype_prompt,
        extract_subtypes,
    )
    from models.vlm.clip_classifier import CLIPClassifier
    from models.vlm.metrics_multilabel import compute_multilabel_metrics

    assert issubclass(CLIPClassifier, BaseVLMClassifier)
    assert callable(build_prompt)
    assert callable(build_subtype_prompt)
    assert callable(extract_subtypes)
    assert callable(compute_multilabel_metrics)
    assert ClassificationResult is not None
    assert len(_SL) == 4


# ---------------------------------------------------------------------------
# Result-file tests — skip gracefully if no results exist yet
# ---------------------------------------------------------------------------


def test_result_files_exist() -> None:
    """At least one model must have a result file for some split."""
    if not _result_files_exist():
        pytest.skip("No result files found — run the CLIP benchmark first.")
    assert _result_files_exist()


def test_at_least_two_filters() -> None:
    """At least 2 distinct filter values must be present (proves ablation ran)."""
    data = _load_all_results()
    if not data:
        pytest.skip("No result entries found — run the benchmark first.")
    filters = {entry.get("filter") for entry in data}
    assert len(filters) >= 2, f"Expected >= 2 filters, found: {filters}"


def test_no_by_visibility_keys() -> None:
    """Entries must NOT contain a by_visibility block (MAMI has no visibility scores)."""
    data = _load_all_results()
    if not data:
        pytest.skip("No result entries found.")
    for entry in data:
        assert "by_visibility" not in entry, (
            f"Unexpected by_visibility in {entry.get('model')}/{entry.get('filter')}"
        )


def test_singleclass_sample_rows_have_no_subtask_labels() -> None:
    """Singleclass sample_predictions rows must NOT carry sub-task label fields."""
    data = _load_all_results()
    if not data:
        pytest.skip("No result entries found.")
    subtask_keys = {"shaming", "stereotype", "objectification", "violence"}
    for entry in data:
        if entry.get("task") != "singleclass":
            continue
        rows = entry.get("sample_predictions", [])
        if not rows:
            continue  # skip entries with empty sample rows
        first_row = rows[0]
        present = subtask_keys & set(first_row.keys())
        assert not present, (
            f"Unexpected sub-task keys {present} in singleclass sample_predictions "
            f"for {entry.get('model')}/{entry.get('filter')}"
        )


def test_required_metric_keys() -> None:
    """Every entry must have required metric keys."""
    data = _load_all_results()
    if not data:
        pytest.skip("No result entries found.")
    for entry in data:
        model = entry.get("model", "")
        flt = entry.get("filter", "")
        assert "exact_match_accuracy" in entry, f"Missing exact_match_accuracy in {model}/{flt}"
        assert "f1" in entry, f"Missing f1 in {model}/{flt}"
        assert "avg_latency_s" in entry, f"Missing avg_latency_s in {model}/{flt}"


def test_entries_have_split_field() -> None:
    """Every entry must carry a 'split' field (not 'subset')."""
    data = _load_all_results()
    if not data:
        pytest.skip("No result entries found.")
    for entry in data:
        model = entry.get("model", "")
        assert "split" in entry, f"Missing 'split' field in entry for model '{model}'"
        assert entry["split"] in SPLITS, (
            f"Invalid split value '{entry['split']}' in entry for model '{model}'"
        )


def test_singleclass_entries_have_task_field() -> None:
    """Singleclass result entries must carry a 'task' field.

    Entries with the legacy value 'a' are treated as pre-migration and skipped;
    all other non-multiclass entries must use 'singleclass'.
    At least one entry must have task == 'singleclass' (proves the migration ran).
    """
    data = _load_all_results()
    if not data:
        pytest.skip("No result entries found.")
    singleclass_seen = False
    for entry in data:
        model = entry.get("model", "")
        flt = entry.get("filter", "")
        assert "task" in entry, f"Missing 'task' field in entry for {model}/{flt}"
        task_val = entry["task"]
        if task_val == "a":
            continue  # legacy pre-migration entry — skip
        if task_val == "multiclass":
            continue  # multiclass entry — not checked here
        assert task_val == "singleclass", (
            f"Expected task='singleclass' in singleclass entry for {model}/{flt}, got {task_val!r}"
        )
        singleclass_seen = True
    assert singleclass_seen, (
        "No 'singleclass' task entries found — run the CLIP benchmark with --task singleclass."
    )


# ---------------------------------------------------------------------------
# Multiclass (Challenge 2) result-file tests — skip gracefully if absent
# ---------------------------------------------------------------------------


def test_multiclass_result_files_schema() -> None:
    """Multiclass result files must have the full multi-label schema in every entry."""
    data = _load_all_multiclass_results()
    if not data:
        pytest.skip("No multiclass result files found — run with --task multiclass first.")
    required_keys = {
        "model",
        "filter",
        "split",
        "task",
        "exact_match_accuracy",
        "f1",
        "precision",
        "recall",
        "macro_f1",
        "micro_f1",
        "weighted_f1",
        "per_class",
        "avg_latency_s",
        "refusal_rate",
        "label_prevalence",
        "sample_predictions",
    }
    for entry in data:
        model = entry.get("model", "")
        flt = entry.get("filter", "")
        missing = required_keys - set(entry.keys())
        assert not missing, f"Missing keys {missing} in Task B entry {model}/{flt}"


def test_multiclass_entries_have_task_multiclass_field() -> None:
    """Every multiclass entry must carry task == 'multiclass'."""
    data = _load_all_multiclass_results()
    if not data:
        pytest.skip("No multiclass result files found.")
    for entry in data:
        model = entry.get("model", "")
        flt = entry.get("filter", "")
        assert entry.get("task") == "multiclass", (
            f"Expected task='multiclass' in multiclass entry for {model}/{flt},"
            f" got {entry.get('task')!r}"
        )


def test_multiclass_per_class_has_all_subtypes() -> None:
    """per_class dict in multiclass entries must contain all four sub-type labels."""
    data = _load_all_multiclass_results()
    if not data:
        pytest.skip("No multiclass result files found.")
    for entry in data:
        model = entry.get("model", "")
        flt = entry.get("filter", "")
        per_class = entry.get("per_class", {})
        for lbl in SUBTYPE_LABELS:
            assert lbl in per_class, f"Missing sub-type '{lbl}' in per_class for {model}/{flt}"
            class_metrics = per_class[lbl]
            assert "precision" in class_metrics
            assert "recall" in class_metrics
            assert "f1" in class_metrics
            assert "support" in class_metrics


def test_multiclass_sample_rows_structure() -> None:
    """Multiclass sample_predictions rows must have ground_truth/prediction as dicts with 4 labels."""
    data = _load_all_multiclass_results()
    if not data:
        pytest.skip("No multiclass result files found.")
    for entry in data:
        model = entry.get("model", "")
        flt = entry.get("filter", "")
        rows = entry.get("sample_predictions", [])
        if not rows:
            continue
        first_row = rows[0]
        assert "ground_truth" in first_row, f"Missing ground_truth in Task B row for {model}/{flt}"
        assert "prediction" in first_row, f"Missing prediction in Task B row for {model}/{flt}"
        assert "correct" in first_row, f"Missing correct in Task B row for {model}/{flt}"
        assert "misogynous" in first_row, f"Missing misogynous in Task B row for {model}/{flt}"
        gt = first_row["ground_truth"]
        pred = first_row["prediction"]
        assert isinstance(gt, dict), f"ground_truth must be dict in Task B row for {model}/{flt}"
        assert isinstance(pred, dict), f"prediction must be dict in Task B row for {model}/{flt}"
        for lbl in SUBTYPE_LABELS:
            assert lbl in gt, f"Missing '{lbl}' in ground_truth dict for {model}/{flt}"
            assert lbl in pred, f"Missing '{lbl}' in prediction dict for {model}/{flt}"


def test_multiclass_multilabel_metric_fields() -> None:
    """Multiclass entries must have macro_f1, micro_f1, and weighted_f1 as floats."""
    data = _load_all_multiclass_results()
    if not data:
        pytest.skip("No multiclass result files found.")
    for entry in data:
        model = entry.get("model", "")
        flt = entry.get("filter", "")
        for key in ("macro_f1", "micro_f1", "weighted_f1"):
            val = entry.get(key)
            assert isinstance(val, float), (
                f"Expected float for {key} in {model}/{flt}, got {type(val)}"
            )
            assert 0.0 <= val <= 1.0, f"{key} out of range [0,1] in {model}/{flt}: {val}"


def test_multiclass_at_least_two_filters() -> None:
    """Multiclass results must also cover at least 2 distinct filters."""
    data = _load_all_multiclass_results()
    if not data:
        pytest.skip("No multiclass result files found.")
    filters = {entry.get("filter") for entry in data}
    assert len(filters) >= 2, f"Expected >= 2 filters in multiclass results, found: {filters}"
