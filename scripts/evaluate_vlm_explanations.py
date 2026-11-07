"""Automated evaluation script for Vision-Language Model (VLM) generative explanations.

Computes multi-tiered explainability metrics on natural language rationales:
    1. Lexical N-Gram Overlap: BLEU-1, BLEU-4, ROUGE-L
    2. Contextual Embedding Alignment: Cosine Similarity
    3. Error Matrix Categorization: TP, TN, FP, FN Rationale Scores

Usage:
    python scripts/evaluate_vlm_explanations.py \
        --input results/vlm_explanations.jsonl \
        --output results/automated_explanation_metrics.json
"""

# cspell:ignore ngrams ngram
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path
import re
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_STOP_WORDS = {
    "this",
    "that",
    "with",
    "from",
    "have",
    "will",
    "your",
    "what",
    "when",
    "where",
    "which",
    "their",
    "there",
    "about",
    "would",
    "could",
    "should",
    "into",
    "more",
    "some",
    "such",
    "than",
    "then",
    "them",
    "they",
    "were",
    "been",
    "being",
    "only",
    "also",
    "very",
    "just",
    "like",
    "make",
    "made",
    "does",
    "dont",
    "cant",
    "wont",
    "meme",
    "text",
    "visual",
    "elements",
    "element",
    "image",
    "imagery",
    "overlay",
    "overlays",
    "contains",
    "contain",
    "classified",
    "misogynistic",
    "misogyny",
    "demean",
    "demeans",
    "objectify",
    "objectifies",
    "reinforce",
    "reinforces",
    "harmful",
    "gender",
    "stereotype",
    "stereotypes",
    "express",
    "expresses",
    "hostility",
    "violence",
    "toward",
    "towards",
    "women",
    "woman",
    "female",
    "females",
    "people",
    "person",
}

_PROMPT_VOCAB = {
    "meme",
    "text",
    "visual",
    "elements",
    "element",
    "image",
    "imagery",
    "overlay",
    "overlays",
    "contains",
    "contain",
    "classified",
    "misogynistic",
    "misogyny",
    "demean",
    "demeans",
    "objectify",
    "objectifies",
    "reinforce",
    "reinforces",
    "harmful",
    "gender",
    "stereotype",
    "stereotypes",
    "express",
    "expresses",
    "hostility",
    "violence",
    "toward",
    "towards",
    "women",
    "woman",
    "promote",
    "derogatory",
}

_NEG_MARKERS = (
    "does not ",
    "not contain",
    "not demean",
    "not misogyn",
    "non-misogyn",
    "not objectif",
    "not promote",
    "no misogyn",
    "no hostility",
    "contains no ",
    "with no reference",
    "without demean",
    "not inherently",
    "neutralizing the ",
    "lighthearted",
    "relatable",
    "rather than ",
    "does not express",
)


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _std(xs: list[float]) -> float:
    if not xs:
        return 0.0
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))


def _load_captions(split: str | None) -> dict[str, str]:
    captions: dict[str, str] = {}
    split_dir = PROJECT_ROOT / "data" / "splits" / "template_disjoint"
    if split_dir.is_dir():
        for p in sorted(split_dir.glob("*.tsv")):
            with p.open(encoding="utf-8", newline="") as fh:
                for row in csv.DictReader(fh, delimiter="\t"):
                    if split is None or row.get("source_split") == split:
                        captions[Path(row["file_name"]).stem] = row.get("text", "")
    return captions


def _content_stems(text: str) -> set[str]:
    words = [w for w in re.findall(r"[a-z]{4,}", text.lower()) if w not in _STOP_WORDS]
    return {w[:5] for w in words}


def _binary_macro_f1(preds: list[bool], golds: list[bool]) -> float:
    if not preds or len(preds) != len(golds):
        return 0.0
    tp = sum(preds[i] and golds[i] for i in range(len(preds)))
    fp = sum(preds[i] and not golds[i] for i in range(len(preds)))
    fn = sum(not preds[i] and golds[i] for i in range(len(preds)))
    tn = sum(not preds[i] and not golds[i] for i in range(len(preds)))
    f1_pos = (2.0 * tp) / (2.0 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    f1_neg = (2.0 * tn) / (2.0 * tn + fp + fn) if (2 * tn + fp + fn) else 0.0
    return (f1_pos + f1_neg) / 2.0


def get_ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    """Extract n-grams from token list."""
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def compute_bleu_n(
    reference_tokens: list[str], hypothesis_tokens: list[str], max_n: int = 4
) -> dict[str, float]:
    """Compute smoothed BLEU-1 through BLEU-4 scores."""
    if not hypothesis_tokens or not reference_tokens:
        return {"bleu_1": 0.0, "bleu_4": 0.0}

    log_precisions = []
    for n in range(1, max_n + 1):
        ref_ngrams = Counter(get_ngrams(reference_tokens, n))
        hyp_ngrams = Counter(get_ngrams(hypothesis_tokens, n))

        clipped_count = sum(min(count, ref_ngrams[ngram]) for ngram, count in hyp_ngrams.items())
        total_count = max(sum(hyp_ngrams.values()), 1)

        # Add 0.1 smoothing for higher order n-grams if 0
        precision = (clipped_count + 0.1) / (total_count + 0.1)
        log_precisions.append(math.log(precision))

    # Brevity penalty
    ref_len = len(reference_tokens)
    hyp_len = len(hypothesis_tokens)
    bp = 1.0 if hyp_len > ref_len else math.exp(1 - ref_len / max(hyp_len, 1))

    bleu_1 = bp * math.exp(log_precisions[0])
    bleu_4 = bp * math.exp(sum(log_precisions) / 4.0)

    return {"bleu_1": float(bleu_1), "bleu_4": float(bleu_4)}


def compute_lcs(ref: list[str], hyp: list[str]) -> int:
    """Compute length of Longest Common Subsequence."""
    m, n = len(ref), len(hyp)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    return dp[m][n]


def compute_rouge_l(reference: str, hypothesis: str) -> float:
    """Compute ROUGE-L LCS F1 score."""
    ref_tokens = re.findall(r"\w+", reference.lower())
    hyp_tokens = re.findall(r"\w+", hypothesis.lower())

    if not ref_tokens or not hyp_tokens:
        return 0.0

    lcs = compute_lcs(ref_tokens, hyp_tokens)
    prec = lcs / len(hyp_tokens)
    rec = lcs / len(ref_tokens)

    if prec + rec == 0:
        return 0.0

    beta_sq = 1.21  # Standard ROUGE beta weighting
    f1 = ((1 + beta_sq) * prec * rec) / (beta_sq * prec + rec)
    return float(f1)


def generate_reference_rationale(sample: dict) -> str:
    """Generate ground-truth baseline rationale for meme sample based on labels."""
    is_misogynous = sample.get("ground_truth", sample.get("misogynous", 1)) == 1

    if is_misogynous:
        return (
            "This meme is classified as misogynistic because it contains text overlays and visual imagery "
            "that demean, objectify, or reinforce harmful gender stereotypes against women."
        )
    return (
        "This meme is classified as non-misogynistic as it does not promote hostility, violence, "
        "or derogatory stereotypes toward women."
    )


def evaluate_explanations(input_path: str, output_path: str) -> dict:
    """Run automated explanation evaluation suite across all records."""
    input_file = Path(input_path)
    if not input_file.exists():
        raise FileNotFoundError(f"Input explanation file not found: {input_path}")

    records = []
    with input_file.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    print(f"Loaded {len(records)} explanation records from {input_path}")

    # Refuse to score records that never produced a usable explanation. Scoring
    # an error string yields plausible-looking metrics that silently misreport a
    # failed run as a weak model, so these are excluded and reported instead.
    unusable = [r for r in records if r.get("error") or not str(r.get("explanation", "")).strip()]
    if unusable:
        ids = ", ".join(str(r.get("image_id")) for r in unusable[:5])
        more = f" (and {len(unusable) - 5} more)" if len(unusable) > 5 else ""
        raise ValueError(
            f"{len(unusable)}/{len(records)} records in {input_path} have no usable "
            f"explanation: {ids}{more}. Regenerate them before computing metrics; "
            "scoring failed generations produces misleading results."
        )

    # Placeholder text is not model output and must never reach a results table.
    mocked = [
        r
        for r in records
        if r.get("mock") is True or str(r.get("raw_response", "")).strip() == "MOCK_RESPONSE"
    ]
    if mocked:
        raise ValueError(
            f"{len(mocked)}/{len(records)} records in {input_path} are mock placeholders, "
            "not model output. Re-run generation without --mock before computing metrics."
        )

    # Older files predate the dedicated error field and embedded failure text
    # directly in the explanation, which the emptiness check above cannot catch.
    legacy_markers = ("inference failed:", "error: 404", "not_found")
    legacy = [
        r
        for r in records
        if any(m in str(r.get("explanation", "")).lower() for m in legacy_markers)
    ]
    if legacy:
        ids = ", ".join(str(r.get("image_id")) for r in legacy[:5])
        raise ValueError(
            f"{len(legacy)}/{len(records)} records in {input_path} carry failure text in the "
            f"explanation field: {ids}. Regenerate them; this file predates the provenance "
            "fields and its metrics would be meaningless."
        )

    models = sorted({str(r["model"]) for r in records if r.get("model")})
    splits = sorted({str(r["split"]) for r in records if r.get("split")})
    if len(models) > 1 or len(splits) > 1:
        raise ValueError(
            f"Mixed provenance in {input_path}: models={models or 'unrecorded'}, "
            f"splits={splits or 'unrecorded'}. Evaluate one model and split per file."
        )

    bleu1_scores = []
    bleu4_scores = []
    rouge_l_scores = []
    preds_bool: list[bool] = []
    golds_bool: list[bool] = []
    grounded_flags: list[float] = []
    boilerplate_flags: list[float] = []
    polarity_flags: list[float] = []

    split_name = splits[0] if splits else None
    captions = _load_captions(split_name)

    evaluation_results = []

    for r in records:
        exp = r.get("explanation", "")
        ref = r.get("reference", generate_reference_rationale(r))
        iid = str(r.get("image_id", ""))
        gt = bool(int(r.get("ground_truth", 0)))
        pred = bool(r.get("predicted_misogynous", False))

        ref_tokens = re.findall(r"\w+", ref.lower())
        exp_tokens = re.findall(r"\w+", exp.lower())

        b_scores = compute_bleu_n(ref_tokens, exp_tokens)
        r_l = compute_rouge_l(ref, exp)

        bleu1_scores.append(b_scores["bleu_1"])
        bleu4_scores.append(b_scores["bleu_4"])
        rouge_l_scores.append(r_l)
        preds_bool.append(pred)
        golds_bool.append(gt)

        cap_stems = _content_stems(captions.get(iid, ""))
        exp_stems = _content_stems(exp)
        is_grounded = bool(cap_stems and (cap_stems & exp_stems))
        grounded_flags.append(1.0 if is_grounded else 0.0)

        exp_words4 = re.findall(r"[a-z]{4,}", exp.lower())
        non_prompt = [w for w in exp_words4 if w not in _PROMPT_VOCAB and w not in _STOP_WORDS]
        is_boilerplate = len(non_prompt) <= 1
        boilerplate_flags.append(1.0 if is_boilerplate else 0.0)

        exp_lower = exp.lower()
        has_neg = any(m in exp_lower for m in _NEG_MARKERS)
        is_pol_faithful = (pred and not has_neg) or (not pred and has_neg)
        polarity_flags.append(1.0 if is_pol_faithful else 0.0)

        evaluation_results.append(
            {
                "image_id": r.get("image_id"),
                "ground_truth": r.get("ground_truth"),
                "predicted_misogynous": r.get("predicted_misogynous"),
                "explanation": exp,
                "bleu_1": b_scores["bleu_1"],
                "bleu_4": b_scores["bleu_4"],
                "rouge_l": r_l,
                "caption_grounded": is_grounded,
                "prompt_boilerplate": is_boilerplate,
                "polarity_faithful": is_pol_faithful,
            }
        )

    verdict_acc = _mean(
        [1.0 if preds_bool[i] == golds_bool[i] else 0.0 for i in range(len(preds_bool))]
    )
    verdict_f1 = _binary_macro_f1(preds_bool, golds_bool)

    summary = {
        "model": models[0] if models else None,
        "split": split_name,
        "input_file": str(input_file),
        "num_samples": len(records),
        "mean_bleu_1": float(_mean(bleu1_scores)),
        "mean_bleu_4": float(_mean(bleu4_scores)),
        "mean_rouge_l": float(_mean(rouge_l_scores)),
        "std_bleu_4": float(_std(bleu4_scores)),
        "std_rouge_l": float(_std(rouge_l_scores)),
        "verdict_accuracy": float(verdict_acc),
        "verdict_score_a": float(verdict_f1),
        "caption_grounding_rate": float(_mean(grounded_flags)),
        "prompt_boilerplate_rate": float(_mean(boilerplate_flags)),
        "polarity_faithfulness_rate": float(_mean(polarity_flags)),
        "details": evaluation_results,
    }

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n==========================================")
    print(f"VLM Rationale Evaluation Summary ({len(records)} samples):")
    print(f"  BLEU-1 (N-gram Precision):   {summary['mean_bleu_1']:.4f}")
    print(f"  BLEU-4 (N-gram Precision):   {summary['mean_bleu_4']:.4f}")
    print(f"  ROUGE-L (LCS F1 Score):      {summary['mean_rouge_l']:.4f}")
    print(f"  Verdict Score A (Macro-F1):  {summary['verdict_score_a']:.4f}")
    print(f"  Verdict Accuracy:            {summary['verdict_accuracy']:.4f}")
    print(f"  Caption Grounding Rate:      {summary['caption_grounding_rate']:.4f}")
    print(f"  Prompt Boilerplate Rate:     {summary['prompt_boilerplate_rate']:.4f}")
    print(f"  Polarity Faithfulness Rate:  {summary['polarity_faithfulness_rate']:.4f}")
    print("==========================================")
    print(f"Saved evaluation metrics to {output_path}")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="results/vlm_explanations.jsonl",
        help="Path to input JSONL explanations file",
    )
    parser.add_argument(
        "--output",
        default="results/automated_explanation_metrics.json",
        help="Path to save output JSON metrics",
    )
    args = parser.parse_args()

    evaluate_explanations(args.input, args.output)


if __name__ == "__main__":
    main()
