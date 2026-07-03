"""Canonical task-name normalisation for CLI arguments.

The pipeline historically used the terms ``singleclass`` (MAMI Task A,
binary misogyny) and ``multiclass`` (MAMI Task B, multi-label sub-types).
Both names are technically incorrect: Task A is *binary* not
"single-class", and Task B is *multi-label* not *multi-class*. This
module lets every script accept the accurate new names while keeping
the legacy names working for existing SLURM scripts and old command
lines. See docs/CODE_REVIEW_ISSUES.md §4.1.

Usage in argparse:

    from utils.task_names import canonical_task, TASK_CHOICES

    parser.add_argument(
        "--task",
        default="singleclass",
        type=canonical_task,
        choices=TASK_CHOICES,
        help=...,
    )

The ``type=canonical_task`` callable normalises the input BEFORE
argparse validates it against ``choices``, so passing either
``--task binary`` or ``--task singleclass`` lands in ``args.task ==
"singleclass"`` and every downstream branch (``if args.task ==
"singleclass": ...``) keeps working unchanged.
"""

from __future__ import annotations

# Canonical values that internal code branches on. Do NOT rename these
# without updating every ``if args.task == ...`` branch in the pipeline.
CANONICAL_BINARY = "singleclass"
CANONICAL_MULTILABEL = "multiclass"
CANONICAL_JOINT = "joint"
CANONICAL_PER_CATEGORY = "per_category"

# Every accepted CLI form. Modern paper-facing names appear on the left,
# legacy pipeline names on the right. Both are exposed so ``argparse``'s
# ``choices=`` accepts either surface.
_ALIASES: dict[str, str] = {
    # Modern -> canonical.
    "binary": CANONICAL_BINARY,
    "multilabel": CANONICAL_MULTILABEL,
    "multi-label": CANONICAL_MULTILABEL,
    # Legacy -> canonical (identity mapping keeps existing branches working).
    "singleclass": CANONICAL_BINARY,
    "multiclass": CANONICAL_MULTILABEL,
    "joint": CANONICAL_JOINT,
    "per_category": CANONICAL_PER_CATEGORY,
    "per-category": CANONICAL_PER_CATEGORY,
}

TASK_CHOICES: list[str] = sorted(set(_ALIASES.values()))


def canonical_task(name: str) -> str:
    """Return the canonical form of a task name, accepting either surface.

    Raises ``ValueError`` for unknown values so argparse surfaces the
    error to the user rather than silently defaulting.
    """
    key = name.strip().lower()
    if key not in _ALIASES:
        raise ValueError(
            f"Unknown task {name!r}. Choose one of: "
            f"{sorted(set(_ALIASES.keys()))}"
        )
    return _ALIASES[key]
