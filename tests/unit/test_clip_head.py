"""Unit tests for ``scripts.train_clip.CLIPClassifierHead``.

Covers the two head architectures introduced in
docs/CODE_REVIEW_ISSUES.md §2.1: a linear head (backwards-compatible
with pre-fix checkpoints) and an MLP head (recommended for frozen-tower
training).

The tests use a minimal fake CLIP module whose ``encode_image`` and
``encode_text`` return unit-scale tensors so the head's forward pass
can be exercised without loading real CLIP weights.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from scripts.train_clip import CLIPClassifierHead


class _FakeCLIP(nn.Module):
    """Return zero-mean random features on encode_image/encode_text."""

    def __init__(self, embed_dim: int, seed: int = 42) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self._rng = torch.Generator().manual_seed(seed)

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        return torch.randn(images.shape[0], self.embed_dim, generator=self._rng)

    def encode_text(self, tokens: torch.Tensor) -> torch.Tensor:
        return torch.randn(tokens.shape[0], self.embed_dim, generator=self._rng)


@pytest.mark.parametrize("num_classes", [2, 4])
def test_linear_head_shape(num_classes: int) -> None:
    """The linear head (hidden_dim=0) is a single Linear layer."""
    clip = _FakeCLIP(embed_dim=64)
    head = CLIPClassifierHead(clip, embed_dim=64, num_classes=num_classes, hidden_dim=0)
    # The classifier attribute is a single Linear layer.
    assert isinstance(head.classifier, nn.Linear)
    assert head.classifier.in_features == 128  # 2 * embed_dim
    assert head.classifier.out_features == num_classes


@pytest.mark.parametrize("num_classes", [2, 4])
@pytest.mark.parametrize("hidden_dim", [256, 512])
def test_mlp_head_shape(num_classes: int, hidden_dim: int) -> None:
    """The MLP head is Sequential(LayerNorm, Linear, GELU, Dropout, Linear)."""
    clip = _FakeCLIP(embed_dim=64)
    head = CLIPClassifierHead(clip, embed_dim=64, num_classes=num_classes, hidden_dim=hidden_dim)
    assert isinstance(head.classifier, nn.Sequential)
    # Five layers: LayerNorm, Linear, GELU, Dropout, Linear
    assert len(head.classifier) == 5
    assert isinstance(head.classifier[0], nn.LayerNorm)
    assert isinstance(head.classifier[1], nn.Linear)
    assert head.classifier[1].in_features == 128
    assert head.classifier[1].out_features == hidden_dim
    assert isinstance(head.classifier[2], nn.GELU)
    assert isinstance(head.classifier[3], nn.Dropout)
    assert isinstance(head.classifier[4], nn.Linear)
    assert head.classifier[4].in_features == hidden_dim
    assert head.classifier[4].out_features == num_classes


def test_forward_produces_expected_output_shape() -> None:
    """A forward pass returns ``(batch, num_classes)`` logits."""
    clip = _FakeCLIP(embed_dim=64)
    head = CLIPClassifierHead(clip, embed_dim=64, num_classes=4, hidden_dim=32)
    images = torch.zeros(3, 3, 224, 224)
    tokens = torch.zeros(3, 77, dtype=torch.long)
    logits = head(images, tokens)
    assert logits.shape == (3, 4)


def test_mlp_state_dict_keys_match_serialisation_convention() -> None:
    """The state dict uses ``classifier.<idx>.*`` keys the loader relies on.

    ``models.vlm.clip_classifier._infer_mlp_hidden_dim`` reads
    ``classifier.1.weight`` (the first Linear) to recover the hidden
    dim, and ``classifier.4.bias`` (the second Linear) to recover the
    number of classes. This test pins that contract.
    """
    clip = _FakeCLIP(embed_dim=64)
    head = CLIPClassifierHead(clip, embed_dim=64, num_classes=4, hidden_dim=128)
    keys = set(head.state_dict().keys())
    # Two Linears in the Sequential produce these key prefixes:
    assert any(k.startswith("classifier.1.") for k in keys)
    assert any(k.startswith("classifier.4.") for k in keys)


def test_linear_state_dict_keys_match_serialisation_convention() -> None:
    """The linear head's state dict uses ``classifier.weight``/``classifier.bias``.

    ``models.vlm.clip_classifier`` reads ``classifier.weight`` as the
    signal that a state dict was produced by a linear head.
    """
    clip = _FakeCLIP(embed_dim=64)
    head = CLIPClassifierHead(clip, embed_dim=64, num_classes=2, hidden_dim=0)
    keys = set(head.state_dict().keys())
    assert "classifier.weight" in keys
    assert "classifier.bias" in keys
