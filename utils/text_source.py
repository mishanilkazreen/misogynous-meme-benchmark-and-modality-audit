"""Shared helpers for the ``--text-source`` refactor.

Every pipeline script that consumes text alongside images needs the same
three primitives:

* Resolve ``--text-source`` and the deprecated ``--use-ocr`` alias into one
  of ``{"provided", "ocr", "combined"}``.
* Compute the NPZ filename suffix for a given source (``""``,
  ``_ocr_<engine>``, or ``_combined_<engine>``).
* Combine MAMI's provided text with an OCR pass, keeping the provided text
  as the base and appending only tokens the OCR added.

This module centralises those primitives plus a loader
(``load_text_source_transcripts``) that resolves the right NPZ file for a
given (split, source, engine) triple. Consumer scripts should call this
loader rather than defining their own copy of ``load_ocr_transcripts``.

See docs/CODE_REVIEW_ISSUES.md §1.3 and §7.8.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

TextSource = str  # Literal["provided", "ocr", "combined"]

_VALID_SOURCES = ("provided", "ocr", "combined")


def resolve_text_source(text_source_arg: str | None, use_ocr_flag: bool) -> TextSource:
    """Resolve the effective text source, honouring the ``--use-ocr`` alias.

    Rules:

    * If ``--text-source`` is set explicitly, it wins.
    * Otherwise, if the deprecated ``--use-ocr`` flag is set, resolve to
      ``"ocr"`` for backward compatibility.
    * Otherwise, default to ``"provided"`` (MAMI leaderboard convention).

    Args:
        text_source_arg: Value of ``--text-source`` (``None`` if unset).
        use_ocr_flag: Whether the deprecated ``--use-ocr`` flag was set.

    Returns:
        One of ``{"provided", "ocr", "combined"}``.
    """
    if text_source_arg is not None:
        if text_source_arg not in _VALID_SOURCES:
            raise ValueError(
                f"Unknown text_source: {text_source_arg!r}. Choices: {list(_VALID_SOURCES)}"
            )
        if use_ocr_flag and text_source_arg == "provided":
            logger.warning("--use-ocr is ignored because --text-source=provided is set explicitly.")
        return text_source_arg
    return "ocr" if use_ocr_flag else "provided"


def combine_texts(provided: str, ocr: str) -> str:
    """Union of two texts, deduping tokens the OCR pass caught that are already present.

    The MAMI-provided transcription is manually verified and covers most
    overlays; PaddleOCR sometimes catches stylised text (watermarks, fine
    print) the annotator skipped. Keeping the provided text as the base and
    appending only new OCR tokens is a conservative union that never loses
    signal from either source.

    Case-insensitive dedup; original casing of the provided text is
    preserved. Trailing whitespace is stripped.
    """
    provided_tokens = {t.lower() for t in provided.split()}
    extra = " ".join(t for t in ocr.split() if t.lower() not in provided_tokens)
    return (provided + " " + extra).strip()


def filename_suffix_for_source(text_source: TextSource, ocr_engine: str) -> str:
    """Return the NPZ filename suffix for a given text source.

    * ``provided``: no suffix (matches the pre-``--use-ocr`` file names).
    * ``ocr``: ``_ocr_<engine>`` (matches the ``--use-ocr`` file names).
    * ``combined``: ``_combined_<engine>``.

    Raises ``ValueError`` on an unknown text source; typos in shell
    scripts fail loud instead of silently producing wrong file names.
    """
    if text_source == "provided":
        return ""
    if text_source == "ocr":
        return f"_ocr_{ocr_engine}"
    if text_source == "combined":
        return f"_combined_{ocr_engine}"
    raise ValueError(f"Unknown text_source: {text_source!r}")


def load_text_source_transcripts(
    split: str,
    text_source: TextSource,
    ocr_engine: str,
    embeddings_dir: Path,
) -> dict[str, str]:
    """Load pre-extracted ``raw_texts`` for a given (split, source, engine).

    Consumer scripts (training and benchmark) call this to obtain the
    per-image text that was written into the NPZ by
    ``scripts/extract_embeddings.py``. The returned dict maps image_id
    (as a string) to the transcribed text.

    * ``text_source == "provided"``: no NPZ lookup is needed at consumer
      time because the provided text is already available on the dataset
      sample. This function returns an empty dict; callers should fall
      back to ``sample["text"]``.
    * ``text_source == "ocr"``: search for
      ``{split}_*_ocr_{ocr_engine}.npz`` (matches the legacy pattern that
      historic scripts used).
    * ``text_source == "combined"``: search for
      ``{split}_*_combined_{ocr_engine}.npz``.

    If no matching file is found for a source that requires one, the
    function warns and returns an empty dict; callers should treat that
    as "fall back to sample['text']" so training does not crash on a
    missing OCR extract.
    """
    if text_source == "provided":
        # Nothing to load; the provided text lives on the dataset sample.
        return {}

    if text_source == "ocr":
        # Prefer the new-style ``_ocr_<engine>.npz`` files. Fall back to
        # legacy ``_<engine>.npz`` files (produced by earlier versions of
        # extract_embeddings.py) when no new-style file exists yet, so
        # historic runs keep working during the transition.
        new_files = list(embeddings_dir.glob(f"{split}_*_ocr_{ocr_engine}.npz"))
        if new_files:
            candidates = new_files
        else:
            candidates = [
                p
                for p in embeddings_dir.glob(f"{split}_*_{ocr_engine}.npz")
                if "_ocr_" not in p.name and "_combined_" not in p.name
            ]
    elif text_source == "combined":
        pattern = f"{split}_*_combined_{ocr_engine}.npz"
        candidates = list(embeddings_dir.glob(pattern))
    else:
        raise ValueError(f"Unknown text_source: {text_source!r}")

    if not candidates:
        logger.warning(
            "No pre-extracted text NPZ found for split=%s, text_source=%s, "
            "engine=%s under %s. Consumer scripts will fall back to "
            "sample['text'].",
            split,
            text_source,
            ocr_engine,
            embeddings_dir,
        )
        return {}

    file_path = candidates[0]
    logger.info("Loading text transcripts from %s...", file_path)
    data = np.load(file_path, allow_pickle=True)
    image_ids = data["image_ids"]
    raw_texts = data["raw_texts"]
    return {str(img_id): str(txt) for img_id, txt in zip(image_ids, raw_texts, strict=True)}
