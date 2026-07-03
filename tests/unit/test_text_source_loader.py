"""Unit tests for ``utils.text_source.load_text_source_transcripts``.

Covers the loader that consumer scripts (training and benchmark) will use
to pick up pre-extracted transcripts from the NPZ files produced by
``scripts/extract_embeddings.py``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from utils.text_source import load_text_source_transcripts


def _write_fake_npz(path: Path, image_ids: list[str], raw_texts: list[str]) -> None:
    """Write a minimal NPZ that mimics an extract_embeddings output."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # image_embeddings / text_embeddings are unused by the loader; write
    # zero-filled tensors of a plausible shape so the file is well-formed.
    n = len(image_ids)
    np.savez_compressed(
        path,
        image_embeddings=np.zeros((n, 8), dtype=np.float32),
        text_embeddings=np.zeros((n, 8), dtype=np.float32),
        labels=np.zeros(n, dtype=np.int32),
        subtask_labels=np.zeros((n, 4), dtype=np.int32),
        image_ids=np.array(image_ids),
        raw_texts=np.array(raw_texts),
    )


def test_loader_provided_returns_empty(tmp_path: Path) -> None:
    """``provided`` needs no NPZ; the loader returns an empty dict."""
    result = load_text_source_transcripts("validation", "provided", "paddleocr", tmp_path)
    assert result == {}


def test_loader_ocr_reads_the_matching_npz(tmp_path: Path) -> None:
    """The ``ocr`` source picks up the ``_ocr_<engine>.npz`` file."""
    _write_fake_npz(
        tmp_path / "validation_vit_l_14_ocr_paddleocr.npz",
        image_ids=["1", "2"],
        raw_texts=["overlay one", "overlay two"],
    )
    result = load_text_source_transcripts("validation", "ocr", "paddleocr", tmp_path)
    assert result == {"1": "overlay one", "2": "overlay two"}


def test_loader_combined_reads_the_matching_npz(tmp_path: Path) -> None:
    """The ``combined`` source picks up the ``_combined_<engine>.npz`` file."""
    _write_fake_npz(
        tmp_path / "test_vit_b_32_combined_paddleocr.npz",
        image_ids=["10", "20"],
        raw_texts=["merged one", "merged two"],
    )
    result = load_text_source_transcripts("test", "combined", "paddleocr", tmp_path)
    assert result == {"10": "merged one", "20": "merged two"}


def test_loader_missing_file_returns_empty(tmp_path: Path) -> None:
    """When no NPZ exists for the requested source, the loader warns and returns empty.

    Callers are expected to fall back to ``sample["text"]`` in that case,
    so training does not crash on a missing OCR extract.
    """
    result = load_text_source_transcripts("validation", "combined", "paddleocr", tmp_path)
    assert result == {}


def test_loader_prefers_the_new_ocr_suffix_over_legacy(tmp_path: Path) -> None:
    """When both ``_ocr_<engine>.npz`` and the legacy ``_<engine>.npz`` exist.

    The new-style filename wins so migrations pick up the freshly-extracted
    file instead of a stale legacy one.
    """
    _write_fake_npz(
        tmp_path / "validation_vit_l_14_paddleocr.npz",  # legacy
        image_ids=["1"],
        raw_texts=["legacy"],
    )
    _write_fake_npz(
        tmp_path / "validation_vit_l_14_ocr_paddleocr.npz",  # new
        image_ids=["1"],
        raw_texts=["new"],
    )
    result = load_text_source_transcripts("validation", "ocr", "paddleocr", tmp_path)
    assert result["1"] == "new"


def test_loader_falls_back_to_legacy_when_no_new_file(tmp_path: Path) -> None:
    """Historic runs with the legacy filename are still picked up."""
    _write_fake_npz(
        tmp_path / "test_vit_l_14_paddleocr.npz",  # legacy only
        image_ids=["5"],
        raw_texts=["legacy overlay"],
    )
    result = load_text_source_transcripts("test", "ocr", "paddleocr", tmp_path)
    assert result == {"5": "legacy overlay"}
