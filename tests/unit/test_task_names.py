"""Unit tests for ``utils.task_names``.

Covers the CLI aliasing introduced in docs/CODE_REVIEW_ISSUES.md §4.1.
"""

from __future__ import annotations

import pytest

from utils.task_names import (
    CANONICAL_BINARY,
    CANONICAL_JOINT,
    CANONICAL_MULTILABEL,
    CANONICAL_PER_CATEGORY,
    TASK_CHOICES,
    canonical_task,
)


def test_binary_maps_to_canonical() -> None:
    """``--task binary`` normalises to the legacy internal name."""
    assert canonical_task("binary") == CANONICAL_BINARY


def test_multilabel_maps_to_canonical() -> None:
    """``--task multilabel`` normalises to the legacy internal name."""
    assert canonical_task("multilabel") == CANONICAL_MULTILABEL


def test_multi_label_dash_variant() -> None:
    """The hyphenated variant is accepted."""
    assert canonical_task("multi-label") == CANONICAL_MULTILABEL


def test_legacy_names_pass_through() -> None:
    """The legacy names still resolve so old SLURM scripts keep working."""
    assert canonical_task("singleclass") == CANONICAL_BINARY
    assert canonical_task("multiclass") == CANONICAL_MULTILABEL


def test_joint_is_supported() -> None:
    """``joint`` is a first-class canonical task."""
    assert canonical_task("joint") == CANONICAL_JOINT


def test_per_category_dash_and_underscore_variants() -> None:
    """Both ``per_category`` and ``per-category`` are accepted."""
    assert canonical_task("per_category") == CANONICAL_PER_CATEGORY
    assert canonical_task("per-category") == CANONICAL_PER_CATEGORY


def test_case_insensitive_input() -> None:
    """Uppercase and mixed case are normalised too."""
    assert canonical_task("BINARY") == CANONICAL_BINARY
    assert canonical_task("MultiLabel") == CANONICAL_MULTILABEL


def test_leading_trailing_whitespace_ignored() -> None:
    """Surrounding whitespace is stripped (SLURM scripts sometimes add it)."""
    assert canonical_task("  binary  ") == CANONICAL_BINARY


def test_unknown_name_raises() -> None:
    """A typo is a caller error, surfaced as ValueError."""
    with pytest.raises(ValueError):
        canonical_task("binaryy")


def test_task_choices_covers_all_canonicals() -> None:
    """``TASK_CHOICES`` includes every canonical value at least once."""
    for canonical in (CANONICAL_BINARY, CANONICAL_MULTILABEL, CANONICAL_JOINT, CANONICAL_PER_CATEGORY):
        assert canonical in TASK_CHOICES
