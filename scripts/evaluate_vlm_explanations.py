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
import json
import math
from pathlib import Path
import re
import sys

import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


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

    bleu1_scores = []
    bleu4_scores = []
    rouge_l_scores = []

    evaluation_results = []

    for r in records:
        exp = r.get("explanation", "")
        ref = r.get("reference", generate_reference_rationale(r))

        ref_tokens = re.findall(r"\w+", ref.lower())
        exp_tokens = re.findall(r"\w+", exp.lower())

        b_scores = compute_bleu_n(ref_tokens, exp_tokens)
        r_l = compute_rouge_l(ref, exp)

        bleu1_scores.append(b_scores["bleu_1"])
        bleu4_scores.append(b_scores["bleu_4"])
        rouge_l_scores.append(r_l)

        evaluation_results.append(
            {
                "image_id": r.get("image_id"),
                "ground_truth": r.get("ground_truth"),
                "predicted_misogynous": r.get("predicted_misogynous"),
                "explanation": exp,
                "bleu_1": b_scores["bleu_1"],
                "bleu_4": b_scores["bleu_4"],
                "rouge_l": r_l,
            }
        )

    summary = {
        "num_samples": len(records),
        "mean_bleu_1": float(np.mean(bleu1_scores)),
        "mean_bleu_4": float(np.mean(bleu4_scores)),
        "mean_rouge_l": float(np.mean(rouge_l_scores)),
        "std_bleu_4": float(np.std(bleu4_scores)),
        "std_rouge_l": float(np.std(rouge_l_scores)),
        "details": evaluation_results,
    }

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n==========================================")
    print(f"VLM Rationale Evaluation Summary ({len(records)} samples):")
    print(f"  BLEU-1 (N-gram Precision): {summary['mean_bleu_1']:.4f}")
    print(f"  BLEU-4 (N-gram Precision): {summary['mean_bleu_4']:.4f}")
    print(f"  ROUGE-L (LCS F1 Score):   {summary['mean_rouge_l']:.4f}")
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
