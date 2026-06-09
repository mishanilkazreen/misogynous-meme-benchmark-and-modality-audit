"""
Prompt-building utilities for catalogue-augmented VLM classification.

Three prompt variants are provided for ablation:
  - baseline    : thin wrapper around classifier.build_prompt (no catalogue)
  - catalogue   : injects up to max_symbols descriptions before the question
  - per_subset  : injects all subset descriptions with a subset-specific preamble

All functions return a plain str ready to pass to a generative VLM.
For CLIP, callers should use build_enriched_labels() to prepend descriptions
to label strings before calling set_classes().

References:
  - Anna's suggestion: retrieval-augmented classification via symbol catalogue
  - Project task 5: hate-symbol catalogue integration
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from models.vlm.classifier import build_prompt

_SUBSET_PREAMBLES: dict[str, str] = {
    "digits": "A digit from 0 to 9 is hidden within this image using visual steganography.",
    "hate_symbols": "A hate symbol is concealed within this image.",
    "hate_slangs": "A hateful slang term is hidden within this image.",
}


def load_catalogue(path: str | Path = "data/symbols/catalogue.yaml") -> list[dict[str, Any]]:
    """Load and return the symbol catalogue as a list of entry dicts.

    Raises FileNotFoundError with a descriptive message if the file is absent.
    """
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(
            f"Symbol catalogue not found at '{resolved}'. "
            "Create data/symbols/catalogue.yaml or pass an explicit path."
        )
    with resolved.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return list(data.get("symbols", []))


def build_baseline_prompt(subset: str, labels: list[str], shuffle: bool = True) -> str:
    """Thin wrapper around build_prompt so callers can import everything from here."""
    return build_prompt(subset, labels, shuffle=shuffle)


def build_catalogue_prompt(
    subset: str,
    labels: list[str],
    catalogue: list[dict[str, Any]],
    shuffle: bool = True,
    max_symbols: int = 5,
) -> str:
    """Inject relevant symbol descriptions before the closed-set question.

    Catalogue entries are filtered by subset and truncated to max_symbols to
    keep prompts short enough for models with limited context.
    """
    relevant = [e for e in catalogue if e.get("subset") == subset][:max_symbols]

    if not relevant:
        return build_baseline_prompt(subset, labels, shuffle=shuffle)

    lines = [
        "You are analysing an image for hidden hateful content.",
        "Known symbols in this category:",
    ]
    for entry in relevant:
        name = entry.get("name", "")
        desc = entry.get("description", "").replace("\n", " ").strip()
        lines.append(f"- {name}: {desc}")

    lines.append("")
    lines.append(build_baseline_prompt(subset, labels, shuffle=shuffle))
    return "\n".join(lines)


def build_per_subset_prompt(
    subset: str,
    labels: list[str],
    catalogue: list[dict[str, Any]],
    shuffle: bool = True,
) -> str:
    """Subset-specific variant: uses all catalogue entries for the subset.

    Adds a targeted preamble that frames the detection task precisely, then
    lists every catalogue description for that subset (no max_symbols cap).
    """
    relevant = [e for e in catalogue if e.get("subset") == subset]
    preamble = _SUBSET_PREAMBLES.get(subset, "Hidden hateful content is present in this image.")

    lines = [preamble]
    if relevant:
        lines.append("Known items in this category:")
        for entry in relevant:
            name = entry.get("name", "")
            desc = entry.get("description", "").replace("\n", " ").strip()
            lines.append(f"- {name}: {desc}")
        lines.append("")

    lines.append(build_baseline_prompt(subset, labels, shuffle=shuffle))
    return "\n".join(lines)


def build_enriched_labels(
    labels: list[str],
    subset: str,
    catalogue: list[dict[str, Any]],
) -> list[str]:
    """Return label strings enriched with catalogue descriptions for CLIP.

    For each label, look for a matching catalogue entry (by name or aliases).
    If found, prepend the description so CLIP embeds richer semantics.
    If not found, return the label unchanged.
    """
    index: dict[str, dict[str, Any]] = {}
    for entry in catalogue:
        if entry.get("subset") != subset:
            continue
        name = str(entry.get("name", "")).lower()
        index[name] = entry
        for alias in entry.get("aliases", []):
            index[str(alias).lower()] = entry

    enriched: list[str] = []
    for label in labels:
        matched: dict[str, Any] | None = index.get(label.lower())
        if matched is not None:
            desc = str(matched.get("description", "")).replace("\n", " ").strip()
            enriched.append(f"{desc} {label}")
        else:
            enriched.append(label)
    return enriched
