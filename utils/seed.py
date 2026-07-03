"""Shared seed helper for reproducible training and inference.

Every training or benchmark script that pretends to be reproducible must
seed both Python's random module and every third-party RNG that can
influence a run: numpy, PyTorch (CPU and every CUDA device), and the
cuDNN backend. The pre-fix pipeline set no seed at all in the CLIP and
VLM training loops (see docs/CODE_REVIEW_ISSUES.md §3.1), which made
single-seed reporting even noisier than it needed to be.
"""

from __future__ import annotations

import logging
import os
import random

import numpy as np

logger = logging.getLogger(__name__)


def set_seed(seed: int, *, deterministic_cudnn: bool = True) -> None:
    """Seed every RNG the training pipeline can touch.

    Args:
        seed: A non-negative integer. The same seed produces the same
            training trajectory as long as hardware, driver, PyTorch
            version, and CUDA library versions are held fixed.
        deterministic_cudnn: When True (default), set
            ``torch.backends.cudnn.deterministic = True`` and
            ``torch.backends.cudnn.benchmark = False``. Necessary for
            reproducible CUDA training at the cost of some throughput.
            Turn off if benchmarking is the primary goal.
    """
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")

    # Standard-library and numpy RNGs (drive shuffling, augmentation,
    # sklearn helpers, etc.).
    random.seed(seed)
    np.random.seed(seed)

    # Bake the seed into PYTHONHASHSEED so hash-based ordering in dicts /
    # sets (used e.g. for DataLoader worker init) is reproducible across
    # child processes spawned after this call.
    os.environ["PYTHONHASHSEED"] = str(seed)

    # PyTorch. Imported lazily so this module stays cheap to import in
    # non-training scripts (e.g. sklearn-only paths in auto_benchmark).
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic_cudnn:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        # torch not installed in this environment (e.g. a lint-only CI job);
        # skip silently, the python/numpy seeds are still set.
        pass

    logger.info("Seeded RNGs with %d (deterministic_cudnn=%s)", seed, deterministic_cudnn)
