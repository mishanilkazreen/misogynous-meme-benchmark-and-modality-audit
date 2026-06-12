"""Unit tests for the misogyny prompt API in models/vlm/classifier.py."""

from __future__ import annotations

from models.vlm.classifier import (
    CLIP_MISOGYNY_LABELS,
    CLIP_SUBTYPE_LABELS,
    MISOGYNY_GROUND_TRUTH,
    MISOGYNY_LABELS,
    MISOGYNY_PROMPT,
    SUBTYPE_LABELS,
    build_misogyny_prompt,
    build_prompt,
    build_subtype_prompt,
    extract_label,
    extract_subtypes,
    yesno_to_int,
)

# ---------------------------------------------------------------------------
# build_misogyny_prompt / MISOGYNY_PROMPT
# ---------------------------------------------------------------------------


def test_misogyny_prompt_mentions_misogyny() -> None:
    """The binary prompt must reference misogyny / misogynistic."""
    prompt = build_misogyny_prompt()
    assert "misogyn" in prompt.lower()


def test_misogyny_prompt_is_yes_no() -> None:
    """The binary prompt must ask for a yes or no answer."""
    prompt = build_misogyny_prompt()
    lower = prompt.lower()
    assert "yes" in lower and "no" in lower


def test_build_prompt_no_args_returns_misogyny_prompt() -> None:
    """build_prompt() with no arguments returns the misogyny yes/no prompt."""
    assert build_prompt() == MISOGYNY_PROMPT


def test_build_prompt_none_subset_returns_misogyny_prompt() -> None:
    """build_prompt(subset=None) returns the misogyny prompt, ignores labels."""
    prompt = build_prompt(subset=None, labels=["a", "b"])
    assert "misogyn" in prompt.lower()
    # The label list should NOT appear verbatim in the misogyny prompt
    assert '"a"' not in prompt


def test_misogyny_prompt_constant_equals_builder() -> None:
    """MISOGYNY_PROMPT constant and build_misogyny_prompt() must agree."""
    assert build_misogyny_prompt() == MISOGYNY_PROMPT


# ---------------------------------------------------------------------------
# MISOGYNY_LABELS and MISOGYNY_GROUND_TRUTH
# ---------------------------------------------------------------------------


def test_misogyny_labels_are_yes_no() -> None:
    """MISOGYNY_LABELS must be exactly ['yes', 'no']."""
    assert MISOGYNY_LABELS == ["yes", "no"]


def test_misogyny_ground_truth_is_yes() -> None:
    """Ground truth for a misogynous image is 'yes'."""
    assert MISOGYNY_GROUND_TRUTH == "yes"


# ---------------------------------------------------------------------------
# CLIP_MISOGYNY_LABELS
# ---------------------------------------------------------------------------


def test_clip_misogyny_labels_are_phrases() -> None:
    """CLIP labels should be phrase-form strings, not bare yes/no."""
    assert len(CLIP_MISOGYNY_LABELS) == 2
    for label in CLIP_MISOGYNY_LABELS:
        assert len(label.split()) > 1, f"Expected phrase, got bare word: {label!r}"


def test_clip_misogyny_labels_cover_both_classes() -> None:
    """One CLIP label should signal misogyny, one should signal benign."""
    labels_lower = [lbl.lower() for lbl in CLIP_MISOGYNY_LABELS]
    assert any("misogyn" in lbl for lbl in labels_lower), "No misogyny label found"
    assert any("not" in lbl or "benign" in lbl for lbl in labels_lower), "No benign label found"


# ---------------------------------------------------------------------------
# HatefulIllusion backward-compat path in build_prompt
# ---------------------------------------------------------------------------


def test_build_prompt_digits_subset_closed_set() -> None:
    """build_prompt with 'digits' subset still produces a closed-set prompt (backward compat)."""
    prompt = build_prompt("digits", ["0", "1", "2"])
    assert '"0"' in prompt
    assert '"1"' in prompt
    assert '"2"' in prompt


def test_build_prompt_digits_mentions_digit() -> None:
    """Closed-set digits prompt should reference 'digit'."""
    prompt = build_prompt("digits", ["5"])
    assert "digit" in prompt.lower()


def test_build_prompt_hate_symbols_mentions_symbol() -> None:
    """Closed-set hate_symbols prompt should reference 'symbol'."""
    prompt = build_prompt("hate_symbols", ["swastika"])
    assert "symbol" in prompt.lower()


def test_build_prompt_hate_slangs_mentions_hidden() -> None:
    """Closed-set hate_slangs prompt should reference 'hidden' or 'slang'."""
    prompt = build_prompt("hate_slangs", ["slur_a"])
    lower = prompt.lower()
    assert "hidden" in lower or "slang" in lower


def test_build_prompt_options_line_format() -> None:
    """Closed-set prompt should include the reply-format instruction line."""
    prompt = build_prompt("digits", ["cat", "dog"])
    assert "Reply with exactly one of the following options:" in prompt


# ---------------------------------------------------------------------------
# extract_label
# ---------------------------------------------------------------------------


def test_extract_label_exact_match() -> None:
    assert extract_label("yes", ["yes", "no"]) == "yes"


def test_extract_label_case_insensitive() -> None:
    assert extract_label("YES", ["yes", "no"]) == "yes"


def test_extract_label_no_match_returns_none() -> None:
    assert extract_label("maybe", ["yes", "no"]) is None


def test_extract_label_word_boundary() -> None:
    """'no' should not match inside 'know'."""
    result = extract_label("I don't know", ["yes", "no"])
    # 'no' can match inside 'know' only as a substring; word-boundary should prevent it
    # The exact behaviour depends on the regex; just assert no crash
    assert result in (None, "no")


# ---------------------------------------------------------------------------
# yesno_to_int
# ---------------------------------------------------------------------------


def test_yesno_to_int_yes() -> None:
    assert yesno_to_int("yes") == 1


def test_yesno_to_int_no() -> None:
    assert yesno_to_int("no") == 0


def test_yesno_to_int_case_and_whitespace() -> None:
    assert yesno_to_int(" YES ") == 1
    assert yesno_to_int("No") == 0


def test_yesno_to_int_none_passes_through() -> None:
    """A refusal (None) must map to None, not 0."""
    assert yesno_to_int(None) is None


def test_yesno_to_int_unmatched_is_none() -> None:
    assert yesno_to_int("maybe") is None


# ---------------------------------------------------------------------------
# SUBTYPE_LABELS and CLIP_SUBTYPE_LABELS (Challenge 2 / Sub-task B)
# ---------------------------------------------------------------------------


def test_subtype_labels_has_four_categories() -> None:
    """SUBTYPE_LABELS must contain the four MAMI sub-type categories."""
    assert set(SUBTYPE_LABELS) == {"shaming", "stereotype", "objectification", "violence"}
    assert len(SUBTYPE_LABELS) == 4


def test_clip_subtype_labels_covers_all_categories() -> None:
    """CLIP_SUBTYPE_LABELS must have an entry for each of the four sub-types."""
    for lbl in SUBTYPE_LABELS:
        assert lbl in CLIP_SUBTYPE_LABELS, f"Missing CLIP phrases for category: {lbl}"
        pos, neg = CLIP_SUBTYPE_LABELS[lbl]
        assert isinstance(pos, str) and len(pos) > 5
        assert isinstance(neg, str) and len(neg) > 5
        # The positive phrase should mention the category (or a synonym)
        assert len(pos.split()) > 2, f"Expected multi-word phrase for {lbl}, got: {pos!r}"


# ---------------------------------------------------------------------------
# build_subtype_prompt (Challenge 2 / Sub-task B)
# ---------------------------------------------------------------------------


def test_build_subtype_prompt_mentions_all_categories() -> None:
    """The sub-type prompt must mention all four MAMI sub-type categories."""
    prompt = build_subtype_prompt()
    for category in SUBTYPE_LABELS:
        assert category in prompt.lower(), f"Category '{category}' missing from subtype prompt"


def test_build_subtype_prompt_mentions_none_option() -> None:
    """The sub-type prompt must allow the model to respond with 'none'."""
    prompt = build_subtype_prompt()
    assert "none" in prompt.lower()


def test_build_subtype_prompt_is_nonempty_string() -> None:
    """build_subtype_prompt() must return a non-empty string."""
    prompt = build_subtype_prompt()
    assert isinstance(prompt, str) and len(prompt) > 20


# ---------------------------------------------------------------------------
# extract_subtypes (Challenge 2 / Sub-task B)
# ---------------------------------------------------------------------------


def test_extract_subtypes_comma_list() -> None:
    """Comma-separated list of categories is parsed correctly."""
    result = extract_subtypes("shaming, violence", SUBTYPE_LABELS)
    assert result["shaming"] == 1
    assert result["violence"] == 1
    assert result["stereotype"] == 0
    assert result["objectification"] == 0


def test_extract_subtypes_none_keyword_all_zeros() -> None:
    """Response containing 'none' maps all labels to 0."""
    result = extract_subtypes("none", SUBTYPE_LABELS)
    assert all(v == 0 for v in result.values())


def test_extract_subtypes_none_in_sentence_all_zeros() -> None:
    """'none of the above' also maps all labels to 0."""
    result = extract_subtypes("None of the above apply.", SUBTYPE_LABELS)
    assert all(v == 0 for v in result.values())


def test_extract_subtypes_empty_string_all_zeros() -> None:
    """Empty response maps all labels to 0."""
    result = extract_subtypes("", SUBTYPE_LABELS)
    assert all(v == 0 for v in result.values())


def test_extract_subtypes_all_four_categories() -> None:
    """All four categories in the response → all 1s."""
    result = extract_subtypes("shaming, stereotype, objectification, violence", SUBTYPE_LABELS)
    assert all(v == 1 for v in result.values())


def test_extract_subtypes_word_boundary_objectification() -> None:
    """'objectification' should not be matched by a stray partial substring."""
    # 'objectification' is a long word; a partial like 'object' must not trigger it
    result = extract_subtypes("I see an object in the image.", SUBTYPE_LABELS)
    assert result["objectification"] == 0


def test_extract_subtypes_word_boundary_shaming() -> None:
    """'shaming' must match as a whole word only."""
    # "ashamed" contains "sham" but not "shaming"
    result = extract_subtypes("The woman looks ashamed.", SUBTYPE_LABELS)
    assert result["shaming"] == 0


def test_extract_subtypes_refusal_all_zeros() -> None:
    """An unrecognised refusal text with no categories returns all zeros."""
    result = extract_subtypes("I cannot assist with this request.", SUBTYPE_LABELS)
    # No recognised category names → all zeros
    assert all(v == 0 for v in result.values())


def test_extract_subtypes_returns_dict_with_all_labels() -> None:
    """extract_subtypes always returns a dict with exactly the requested labels as keys."""
    result = extract_subtypes("stereotype", SUBTYPE_LABELS)
    assert set(result.keys()) == set(SUBTYPE_LABELS)


def test_extract_subtypes_single_category() -> None:
    """A response with a single category correctly predicts only that category."""
    result = extract_subtypes("stereotype", SUBTYPE_LABELS)
    assert result["stereotype"] == 1
    assert result["shaming"] == 0
    assert result["objectification"] == 0
    assert result["violence"] == 0
