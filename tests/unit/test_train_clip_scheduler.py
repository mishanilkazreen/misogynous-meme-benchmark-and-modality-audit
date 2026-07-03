"""Unit tests for ``scripts.train_clip.build_warmup_cosine_scheduler``.

Pins the LR-schedule contract documented in docs/CODE_REVIEW_ISSUES.md
§2.3: linear ramp from 0 to the base LR over the first ``warmup_steps``
optimizer steps, then cosine decay back to 0 by the final step.
"""

from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from scripts.train_clip import build_warmup_cosine_scheduler


def _dummy_optimizer(base_lr: float = 1e-3) -> torch.optim.Optimizer:
    linear = nn.Linear(4, 2)
    return torch.optim.AdamW(linear.parameters(), lr=base_lr)


def test_scheduler_starts_at_zero() -> None:
    """LR at step 0 (before any optimizer.step()) is 0 with warmup > 0."""
    opt = _dummy_optimizer(base_lr=1e-3)
    scheduler = build_warmup_cosine_scheduler(opt, total_steps=100, warmup_steps=10)
    # Before stepping, the scheduler applies the lambda at step 0 → 0.
    assert opt.param_groups[0]["lr"] == pytest.approx(0.0, abs=1e-9)
    scheduler.step()  # advance to step 1
    assert opt.param_groups[0]["lr"] == pytest.approx(1e-4, abs=1e-9)


def test_scheduler_reaches_base_lr_at_end_of_warmup() -> None:
    """After ``warmup_steps`` optimizer.step()s, the LR equals the base LR."""
    opt = _dummy_optimizer(base_lr=1e-3)
    scheduler = build_warmup_cosine_scheduler(opt, total_steps=100, warmup_steps=10)
    for _ in range(10):
        scheduler.step()
    # LambdaLR evaluates lambda at the pre-increment step, so after 10
    # steps the scheduler has evaluated lambda at step 10 which returns 1.0
    assert opt.param_groups[0]["lr"] == pytest.approx(1e-3, abs=1e-9)


def test_scheduler_decays_to_zero_at_final_step() -> None:
    """At the final step the cosine multiplier reaches 0."""
    opt = _dummy_optimizer(base_lr=1e-3)
    scheduler = build_warmup_cosine_scheduler(opt, total_steps=20, warmup_steps=5)
    for _ in range(20):
        scheduler.step()
    assert opt.param_groups[0]["lr"] == pytest.approx(0.0, abs=1e-9)


def test_scheduler_cosine_midpoint() -> None:
    """Halfway through the decay window, LR is roughly half the base LR."""
    opt = _dummy_optimizer(base_lr=1e-3)
    scheduler = build_warmup_cosine_scheduler(opt, total_steps=200, warmup_steps=0)
    # Halfway: cos(pi/2) = 0, so 0.5 * (1 + 0) = 0.5
    for _ in range(100):
        scheduler.step()
    expected = 1e-3 * 0.5 * (1.0 + math.cos(math.pi * 100 / 200))
    assert opt.param_groups[0]["lr"] == pytest.approx(expected, abs=1e-9)


def test_scheduler_zero_warmup_still_works() -> None:
    """``warmup_steps=0`` disables warmup; LR starts at base and decays."""
    opt = _dummy_optimizer(base_lr=1e-3)
    scheduler = build_warmup_cosine_scheduler(opt, total_steps=10, warmup_steps=0)
    # At step 0, cosine(0) = 1 → LR = base_lr
    assert opt.param_groups[0]["lr"] == pytest.approx(1e-3, abs=1e-9)


def test_scheduler_negative_warmup_raises() -> None:
    """Negative warmup is a caller bug."""
    opt = _dummy_optimizer()
    with pytest.raises(ValueError):
        build_warmup_cosine_scheduler(opt, total_steps=10, warmup_steps=-1)


def test_scheduler_zero_total_steps_raises() -> None:
    """Zero total steps is a caller bug (would divide by zero later)."""
    opt = _dummy_optimizer()
    with pytest.raises(ValueError):
        build_warmup_cosine_scheduler(opt, total_steps=0, warmup_steps=5)
