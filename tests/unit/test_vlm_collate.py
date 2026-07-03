"""Unit tests for ``scripts.train_vlm.VLMCollate``.

Focus of these tests: verify that the collation helper forces right-padding
regardless of what the processor's tokenizer defaults to. Left-padding
combined with our ``labels[:, :prompt_len] = -100`` masking would silently
train the model to reproduce the prompt (see
``docs/CODE_REVIEW_ISSUES.md`` §1.1). Missing this check is the single
biggest reason our fine-tuned VLM Task A numbers looked broken.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from scripts.train_vlm import VLMCollate


def _make_mock_processor(padding_side: str = "left") -> MagicMock:
    """Return a MagicMock that mimics the shape of a HuggingFace processor.

    Only the attributes VLMCollate touches at construction time are set:
    the tokenizer's ``padding_side`` and ``pad_token_id``. The collate
    ``__call__`` path is NOT exercised here (it would require a real
    tokenizer stack), so this fixture is intentionally minimal.
    """
    processor = MagicMock()
    processor.tokenizer = MagicMock()
    processor.tokenizer.padding_side = padding_side
    processor.tokenizer.pad_token_id = 0
    return processor


def test_collate_forces_right_padding_when_processor_defaults_left() -> None:
    """A left-padding tokenizer (Qwen2 default) must be flipped to right."""
    processor = _make_mock_processor(padding_side="left")
    _ = VLMCollate(processor=processor, model_type="qwen2vl", task="singleclass", ocr_map=None)
    assert processor.tokenizer.padding_side == "right"


def test_collate_keeps_right_padding_when_processor_already_right() -> None:
    """A processor already configured with right-padding stays right."""
    processor = _make_mock_processor(padding_side="right")
    _ = VLMCollate(processor=processor, model_type="llava", task="multiclass", ocr_map=None)
    assert processor.tokenizer.padding_side == "right"


def test_collate_marks_first_batch_unverified_at_construction() -> None:
    """The first-batch sanity check is armed at construction time."""
    processor = _make_mock_processor(padding_side="left")
    collate = VLMCollate(
        processor=processor, model_type="qwen2vl", task="singleclass", ocr_map=None
    )
    assert collate._first_batch_verified is False


def test_collate_uses_singleclass_prompt_for_task_singleclass() -> None:
    """The singleclass task branch selects the misogyny yes/no prompt."""
    processor = _make_mock_processor(padding_side="left")
    collate = VLMCollate(
        processor=processor, model_type="qwen2vl", task="singleclass", ocr_map=None
    )
    assert "misogyn" in collate.base_prompt_text.lower()
    assert "yes" in collate.base_prompt_text.lower()


def test_collate_uses_subtype_prompt_for_task_multiclass() -> None:
    """The multiclass task branch selects the sub-type prompt."""
    processor = _make_mock_processor(padding_side="left")
    collate = VLMCollate(processor=processor, model_type="qwen2vl", task="multiclass", ocr_map=None)
    # Prompt must mention the four MAMI sub-types
    lower = collate.base_prompt_text.lower()
    for category in ("shaming", "stereotype", "objectification", "violence"):
        assert category in lower
