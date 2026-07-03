"""Unit tests for ``scripts.train_clip.build_train_preprocess``.

Confirms that (a) the augmented training preprocess exists and includes
the three intended augmentations, (b) crop size and Normalize parameters
are inferred from the eval preprocess rather than hardcoded, and
(c) the resulting Compose can be applied to a PIL image without error.
"""

from __future__ import annotations

import pytest
import torch
from PIL import Image
from torchvision import transforms as T

from scripts.train_clip import build_train_preprocess


def _make_eval_preprocess(crop_size: int) -> T.Compose:
    """Reproduce open_clip's default eval preprocess (Resize -> Crop -> Norm)."""
    return T.Compose(
        [
            T.Resize(crop_size),
            T.CenterCrop(crop_size),
            T.ToTensor(),
            T.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            ),
        ]
    )


@pytest.mark.parametrize("crop_size", [224, 336])
def test_train_preprocess_uses_crop_size_from_eval(crop_size: int) -> None:
    """The RandomResizedCrop size matches the eval CenterCrop size."""
    eval_pp = _make_eval_preprocess(crop_size)
    train_pp = build_train_preprocess(eval_pp)
    crops = [t for t in train_pp.transforms if isinstance(t, T.RandomResizedCrop)]
    assert len(crops) == 1
    size = crops[0].size
    if isinstance(size, (tuple, list)):
        assert size[0] == crop_size
    else:
        assert size == crop_size


def test_train_preprocess_reuses_eval_normalize() -> None:
    """The train preprocess uses the same Normalize stats as the eval preprocess."""
    eval_pp = _make_eval_preprocess(224)
    train_pp = build_train_preprocess(eval_pp)
    eval_norm = next(t for t in eval_pp.transforms if isinstance(t, T.Normalize))
    train_norm = next(t for t in train_pp.transforms if isinstance(t, T.Normalize))
    assert train_norm.mean == eval_norm.mean
    assert train_norm.std == eval_norm.std


def test_train_preprocess_includes_horizontal_flip() -> None:
    """The train preprocess includes RandomHorizontalFlip."""
    eval_pp = _make_eval_preprocess(224)
    train_pp = build_train_preprocess(eval_pp)
    assert any(isinstance(t, T.RandomHorizontalFlip) for t in train_pp.transforms)


def test_train_preprocess_includes_color_jitter() -> None:
    """The train preprocess includes ColorJitter."""
    eval_pp = _make_eval_preprocess(224)
    train_pp = build_train_preprocess(eval_pp)
    assert any(isinstance(t, T.ColorJitter) for t in train_pp.transforms)


def test_train_preprocess_produces_correct_tensor_shape() -> None:
    """Applying the preprocess to a PIL image returns a Normalize'd tensor."""
    eval_pp = _make_eval_preprocess(224)
    train_pp = build_train_preprocess(eval_pp)
    img = Image.new("RGB", (400, 300), color=(128, 64, 200))
    tensor = train_pp(img)
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, 224, 224)


def test_train_preprocess_raises_without_normalize() -> None:
    """If the eval preprocess lacks Normalize, the builder fails loud.

    This protects against a future refactor that swaps the preprocess for
    a non-CLIP one without updating this helper.
    """
    eval_pp = T.Compose(
        [T.Resize(224), T.CenterCrop(224), T.ToTensor()]
    )  # no Normalize
    with pytest.raises(ValueError):
        build_train_preprocess(eval_pp)
