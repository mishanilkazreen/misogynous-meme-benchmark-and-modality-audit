"""Unit tests for ``CLIPClassifier.set_classes_ensemble`` and TTA.

Pins the contracts introduced in docs/CODE_REVIEW_ISSUES.md §7.1 and
§7.2 without requiring real CLIP weights: a fake open_clip model returns
deterministic text embeddings so we can verify the averaging logic and
the TTA branch structurally.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch


class _FakeCLIPModel:
    """Return a deterministic length-normalised vector per input token batch."""

    def __init__(self) -> None:
        self.text_projection = torch.zeros(8, 8)

    def to(self, _device):
        return self

    def eval(self):
        return self

    def encode_text(self, tokens: torch.Tensor) -> torch.Tensor:
        # Return one row per batch item; each row is (i+1)/n for i in [0..7]
        n = tokens.shape[0]
        base = torch.arange(8, dtype=torch.float32).unsqueeze(0).repeat(n, 1)
        return base + tokens[:, :1]  # add a per-batch offset


def _fake_tokenizer(labels: list[str]) -> torch.Tensor:
    """Tokenise as (n, 8) tensor of ones so shape math works."""
    return torch.ones(len(labels), 8, dtype=torch.long)


@pytest.fixture
def classifier():
    from models.vlm.clip_classifier import CLIPClassifier

    with patch("open_clip.create_model_and_transforms") as mock_create, patch(
        "open_clip.get_tokenizer"
    ) as mock_tok:
        mock_create.return_value = (_FakeCLIPModel(), None, MagicMock())
        mock_tok.return_value = _fake_tokenizer
        return CLIPClassifier(model_name="fake", device="cpu")


def test_set_classes_ensemble_stores_one_embedding_per_class(classifier) -> None:
    """Two classes -> exactly two rows in ``_text_embeddings``."""
    classifier.set_classes_ensemble(
        {
            "positive": ["a positive sample", "another positive example"],
            "negative": ["a negative sample"],
        }
    )
    assert classifier._text_embeddings is not None
    assert classifier._text_embeddings.shape[0] == 2


def test_set_classes_ensemble_preserves_label_order(classifier) -> None:
    """``_labels`` matches the insertion order of the prompt dict."""
    classifier.set_classes_ensemble(
        {"first": ["prompt one"], "second": ["prompt two"]}
    )
    assert classifier._labels == ["first", "second"]


def test_set_classes_ensemble_normalises_embeddings(classifier) -> None:
    """Each averaged class embedding is L2-normalised to unit length."""
    classifier.set_classes_ensemble(
        {"positive": ["a", "b", "c"], "negative": ["x"]}
    )
    assert classifier._text_embeddings is not None
    norms = classifier._text_embeddings.norm(dim=-1)
    for n in norms.tolist():
        assert n == pytest.approx(1.0, abs=1e-6)


def test_set_classes_ensemble_rejects_empty_prompts(classifier) -> None:
    """A class with an empty prompt list is a caller bug and fails loud."""
    with pytest.raises(ValueError):
        classifier.set_classes_ensemble({"positive": [], "negative": ["ok"]})


def test_predict_batch_accepts_tta_flag(classifier) -> None:
    """The TTA parameter must be accepted by predict_batch's signature.

    We do not exercise the full forward pass here (would need a real
    image tower), but we assert the signature so a future refactor does
    not accidentally drop the parameter.
    """
    import inspect

    sig = inspect.signature(classifier.predict_batch)
    assert "tta" in sig.parameters
    assert sig.parameters["tta"].default is False


def test_timed_predict_batch_accepts_tta_flag(classifier) -> None:
    """The convenience wrapper also exposes ``tta`` so callers keep symmetry."""
    import inspect

    sig = inspect.signature(classifier.timed_predict_batch)
    assert "tta" in sig.parameters
    assert sig.parameters["tta"].default is False
