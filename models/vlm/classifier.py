"""
Base classifier interface for VLM content moderation benchmarks.

Defines the shared ClassificationResult dataclass and BaseVLMClassifier ABC used
by all model wrappers (CLIP, YOLO-World, cloud APIs).

Primary task (MAMI 2022): binary misogyny classification.
  Prompt the VLM with a yes/no question about whether the meme is misogynistic;
  map yes→1 ("misogynistic"), no→0 ("not misogynistic").
  Ground truth = the ``misogynous`` field from MamiDataset.

Secondary task (MAMI 2022 Sub-task B): multi-label sub-type classification.
  Four independent binary labels: shaming, stereotype, objectification, violence.
  A single prompt asks which categories apply; response is a comma-separated list
  or "none". Each category is also predictable by CLIP via independent binary
  comparisons using CLIP_SUBTYPE_LABELS.
"""
# pylint: disable=too-few-public-methods, import-outside-toplevel

from __future__ import annotations

import abc
from dataclasses import dataclass
import re

import numpy as np

# ---------------------------------------------------------------------------
# Misogyny-classification constants (primary task — Challenge 1 / Sub-task A)
# ---------------------------------------------------------------------------

MISOGYNY_PROMPT = (
    "Is this meme misogynistic? A misogynistic meme demeans, objectifies, or expresses hostility "
    "toward women. Answer with exactly one word: yes or no."
)

# Labels used by generative models (yes/no binary)
MISOGYNY_LABELS: list[str] = ["yes", "no"]

# CLIP text labels — phrase-form works better than bare words for cosine similarity
CLIP_MISOGYNY_LABELS: list[str] = ["misogynistic meme", "not misogynistic meme"]

# Prompt ensemble for zero-shot CLIP (docs/CODE_REVIEW_ISSUES.md §7.1). Averaging
# multiple prompt embeddings per class gives a more robust text-side
# representation and typically buys 2-4 F1 points on the CLIP zero-shot rows
# without any training. Keys map to CLIP_MISOGYNY_LABELS entries; values are
# the full prompt bank averaged into a single text embedding per class.
CLIP_MISOGYNY_PROMPT_ENSEMBLE: dict[str, list[str]] = {
    "misogynistic meme": [
        "a misogynistic meme",
        "a meme that demeans women",
        "a meme expressing hostility toward women",
        "a meme objectifying women",
        "a meme stereotyping women",
        "a sexist meme",
    ],
    "not misogynistic meme": [
        "a meme that does not target women",
        "a non-misogynistic meme",
        "a neutral meme",
        "a wholesome meme",
        "a meme with no gender content",
    ],
}

# Ground-truth label when misogynous == 1
MISOGYNY_GROUND_TRUTH = "yes"

# ---------------------------------------------------------------------------
# Sub-type classification constants (Challenge 2 / Sub-task B)
# ---------------------------------------------------------------------------

# The four independent binary labels for MAMI Sub-task B
SUBTYPE_LABELS: list[str] = ["shaming", "stereotype", "objectification", "violence"]

# CLIP per-category phrase pairs: (positive_phrase, negative_phrase)
# Positive phrase wins → predict 1 for that category.
CLIP_SUBTYPE_LABELS: dict[str, tuple[str, str]] = {
    "shaming": (
        "a meme shaming or insulting a woman",
        "a meme not shaming a woman",
    ),
    "stereotype": (
        "a meme reinforcing gender stereotypes about women",
        "a meme not reinforcing gender stereotypes",
    ),
    "objectification": (
        "a meme objectifying or sexualising a woman",
        "a meme not objectifying women",
    ),
    "violence": (
        "a meme depicting or threatening violence against women",
        "a meme not depicting violence against women",
    ),
}

# Per-sub-type prompt ensemble for zero-shot CLIP. Each entry maps to
# (positive_phrase_list, negative_phrase_list); each list is averaged into
# a single text embedding before the pair is fed to
# ``CLIPClassifier.set_classes_ensemble`` (docs/CODE_REVIEW_ISSUES.md §7.1).
CLIP_SUBTYPE_PROMPT_ENSEMBLE: dict[str, tuple[list[str], list[str]]] = {
    "shaming": (
        [
            "a meme shaming or insulting a woman",
            "a meme body-shaming a woman",
            "a meme mocking a woman's appearance",
            "a meme slut-shaming a woman",
        ],
        [
            "a meme not shaming a woman",
            "a meme that does not insult a woman",
        ],
    ),
    "stereotype": (
        [
            "a meme reinforcing gender stereotypes about women",
            "a meme portraying women through traditional gender roles",
            "a meme depicting women as housewives",
            "a meme depicting women as bad drivers",
        ],
        [
            "a meme not reinforcing gender stereotypes",
            "a meme that does not stereotype women",
        ],
    ),
    "objectification": (
        [
            "a meme objectifying or sexualising a woman",
            "a meme reducing a woman to a sexual object",
            "a meme focusing on a woman's body for sexual purposes",
            "a meme sexualising a woman's appearance",
        ],
        [
            "a meme not objectifying women",
            "a meme that does not sexualise women",
        ],
    ),
    "violence": (
        [
            "a meme depicting or threatening violence against women",
            "a meme depicting physical aggression against a woman",
            "a meme depicting sexual violence against a woman",
            "a meme encouraging harm to a woman",
        ],
        [
            "a meme not depicting violence against women",
            "a meme that does not threaten women",
        ],
    ),
}


def yesno_to_int(label: str | None) -> int | None:
    """Map a yes/no label to its binary int: ``"yes"`` -> 1, ``"no"`` -> 0.

    Anything else (including ``None`` from a refusal or unmatched response)
    maps to ``None``. Used to store predictions and ground truth in result
    files as ints, consistent with the dataset's integer ``misogynous`` field
    and the sub-task labels.
    """
    if label is None:
        return None
    cleaned = label.strip().lower()
    if cleaned == "yes":
        return 1
    if cleaned == "no":
        return 0
    return None


# ---------------------------------------------------------------------------
# Backward-compat aliases (kept so existing per-model scripts compile)
# NOTE: These were previously used for HatefulIllusion closed-set tasks.
#       They are retained only to avoid import errors; new code uses the
#       MISOGYNY_* constants above.
# ---------------------------------------------------------------------------

BINARY_PROMPT = MISOGYNY_PROMPT
BINARY_LABELS: list[str] = MISOGYNY_LABELS
BINARY_GROUND_TRUTH = MISOGYNY_GROUND_TRUTH

# NOTE: _PROMPT_BASES is HatefulIllusion-specific catalogue logic.
# Kept for prompt_templates.py import compatibility; not used in MAMI benchmarks.
_PROMPT_BASES: dict[str, str] = {
    "digits": (
        "A single digit (0–9) is hidden within this image using visual steganography techniques "
        "such as colour patterns, textures, or shapes. Identify the hidden digit."
    ),
    "hate_symbols": (
        "A hate symbol is concealed within this image. It may appear as a shape, glyph, or icon "
        "embedded in the visual content. Identify the hidden symbol."
    ),
    "hate_slangs": (
        "A word or slang term is hidden within this image using visual steganography. "
        "Identify the hidden word."
    ),
}


@dataclass
class ClassificationResult:
    """Unified result returned by every VLM classifier wrapper."""

    prediction: str | None
    confidence: float
    latency_s: float
    refusal: bool


class BaseVLMClassifier(abc.ABC):
    """Abstract base class for all VLM classifiers in this benchmark."""

    @abc.abstractmethod
    def classify(
        self, image: np.ndarray, labels: list[str], text: str | None = None
    ) -> ClassificationResult:
        """Classify a single image against the given closed-set label list."""


def build_misogyny_prompt() -> str:
    """Return the binary yes/no misogyny classification prompt for MAMI 2022.

    Use this with MISOGYNY_LABELS = ["yes", "no"]; ground truth is "yes" for
    misogynous images (misogynous == 1) and "no" for non-misogynous (misogynous == 0).
    """
    return MISOGYNY_PROMPT


def build_binary_prompt() -> str:
    """Alias for build_misogyny_prompt (backward compat)."""
    return MISOGYNY_PROMPT


def build_prompt(
    subset: str | None = None, labels: list[str] | None = None, shuffle: bool = True
) -> str:
    """Build a classification prompt.

    For the MAMI misogyny task (primary use), call with no arguments or
    ``subset=None`` to obtain the standard misogyny yes/no prompt.

    The ``subset`` and ``labels`` parameters are retained for backward
    compatibility with ``prompt_templates.py`` (HatefulIllusion catalogue
    path) and generative-model scripts that still pass them.  When ``subset``
    is provided, the function falls back to the HatefulIllusion closed-set
    prompt behaviour so that ``prompt_templates.py`` continues to work without
    modification.
    """
    import random as _random

    if subset is None or subset not in _PROMPT_BASES:
        # MAMI misogyny mode — ignore labels, return the yes/no prompt
        return MISOGYNY_PROMPT

    # HatefulIllusion closed-set fallback (used by prompt_templates.py only)
    base = _PROMPT_BASES[subset]
    effective_labels = labels if labels is not None else []
    ordered = (
        _random.sample(effective_labels, len(effective_labels)) if shuffle else effective_labels
    )
    options = ", ".join(f'"{label}"' for label in ordered)
    return f"{base}\nReply with exactly one of the following options: {options}"


def extract_label(response: str, labels: list[str]) -> str | None:
    """Map a generative model's free-text response to the closest label.

    Tries exact match first (case-insensitive, strips surrounding quotes/spaces),
    then substring containment. Returns None if no label is found.
    """
    cleaned = response.strip().strip("\"'").lower()
    # Exact match
    for label in labels:
        if cleaned == label.lower():
            return label
    # Word-boundary match: prevents "0" matching inside "10", "20", etc.
    for label in labels:
        pattern = r"\b" + re.escape(label.lower()) + r"\b"
        if re.search(pattern, cleaned):
            return label
    return None


def build_subtype_prompt() -> str:
    """Return the multi-label sub-type classification prompt for MAMI Sub-task B.

    Asks the model which of the four MAMI sub-type categories apply to the meme.
    Response format: a comma-separated list of matching categories, or 'none'.
    Use extract_subtypes() to parse the response into a {label: 0/1} dict.
    """
    return (
        "This meme may target women. Which of the following categories apply to it? "
        "Categories: shaming, stereotype, objectification, violence. "
        "Reply with a comma-separated list of all categories that apply, or reply with 'none' "
        "if none of them apply. Do not add any explanation."
    )


def extract_subtypes(response: str, labels: list[str]) -> dict[str, int]:
    """Parse a generative model's free-text response into a per-label 0/1 dict.

    Uses word-boundary regex matching so that e.g. 'objectification' is not
    accidentally matched by an unrelated substring.  'none' or an empty /
    unparseable response maps all labels to 0 (callers should count this toward
    refusal_rate when the original response was empty or a refusal phrase).

    Args:
        response: Raw text returned by the model.
        labels: The candidate label strings (typically SUBTYPE_LABELS).

    Returns:
        Dict mapping each label to 1 if matched, 0 otherwise.
    """
    cleaned = response.strip().lower()

    # Empty response or explicit refusal → all zeros
    if not cleaned:
        return dict.fromkeys(labels, 0)

    # Explicit 'none' → all zeros (word-boundary so it doesn't hit 'someone' etc.)
    if re.search(r"\bnone\b", cleaned):
        return dict.fromkeys(labels, 0)

    result: dict[str, int] = {}
    any_matched = False
    for lbl in labels:
        pattern = r"\b" + re.escape(lbl.lower()) + r"\b"
        matched = bool(re.search(pattern, cleaned))
        result[lbl] = 1 if matched else 0
        if matched:
            any_matched = True

    # If nothing at all matched (e.g. refusal text without 'none'), treat as all zeros
    if not any_matched:
        return dict.fromkeys(labels, 0)

    return result
