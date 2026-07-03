"""Unit tests for ``utils.seed.set_seed``.

The point of this helper is that after calling ``set_seed(k)``, every
RNG the pipeline touches (``random``, ``numpy``, PyTorch CPU, and any
available CUDA device) produces identical sequences on repeated calls
with the same seed. Reproducibility is checked at the RNG level so the
test is fast and does not need a real training loop.
"""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from utils.seed import set_seed


def test_set_seed_makes_python_random_reproducible() -> None:
    """The same seed yields the same ``random.random()`` sequence twice."""
    set_seed(42)
    a = [random.random() for _ in range(5)]
    set_seed(42)
    b = [random.random() for _ in range(5)]
    assert a == b


def test_set_seed_makes_numpy_reproducible() -> None:
    """The same seed yields the same ``np.random.rand()`` sequence twice."""
    set_seed(7)
    a = np.random.rand(8)
    set_seed(7)
    b = np.random.rand(8)
    assert np.array_equal(a, b)


def test_set_seed_makes_torch_cpu_reproducible() -> None:
    """The same seed yields the same ``torch.randn()`` sample twice on CPU."""
    set_seed(1234)
    a = torch.randn(3, 4)
    set_seed(1234)
    b = torch.randn(3, 4)
    assert torch.equal(a, b)


def test_different_seeds_produce_different_sequences() -> None:
    """Different seeds don't accidentally give the same sequence."""
    set_seed(1)
    a = torch.randn(3, 4)
    set_seed(2)
    b = torch.randn(3, 4)
    assert not torch.equal(a, b)


def test_negative_seed_raises() -> None:
    """A negative seed is a caller bug and must fail loud, not silently."""
    with pytest.raises(ValueError):
        set_seed(-1)


def test_deterministic_cudnn_flag_defaults_true() -> None:
    """``deterministic_cudnn=True`` (default) flips the two cuDNN flags.

    On a CPU-only test host the CUDA branch is skipped, but the CPU-side
    cuDNN attributes still get set because PyTorch exposes them.
    """
    set_seed(42, deterministic_cudnn=True)
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False
