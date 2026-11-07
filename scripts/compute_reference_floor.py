"""Compute and persist the Canonical Category Reference floor for the explanation metrics.

The explanation-quality table has a ``Canonical Category Reference`` row. It is not a model. It is
the canonical taxonomy statement scored against itself, which establishes the
ceiling that any system could reach against these references and, by extension,
the floor that makes the model rows interpretable. Previously this row was
produced on the fly and never saved, so it could not be reproduced from the
repository and it sat in a table of models with no indication that it was a
self-similarity baseline rather than a fourth system.

This script writes it to ``results/reference_floor_metrics.json`` using the same
scoring functions and the same 30 instances as the model rows, so the reported
number is traceable to an artefact like every other row.

Usage:
    uv run python scripts/compute_reference_floor.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = PROJECT_ROOT / "results" / "vlm_explanations.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "reference_floor_metrics.json"


def main() -> None:
    # Imported here rather than at module scope: the project root must be on
    # sys.path first, and a module-level import after that assignment would be a
    # statement-before-import ordering violation.
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from scripts.evaluate_vlm_explanations import (
        compute_bleu_n,
        compute_rouge_l,
        generate_reference_rationale,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        default=str(DEFAULT_SOURCE),
        help="Generation file supplying the instance ids and labels to score over.",
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Metrics file to write.")
    args = parser.parse_args()

    source = Path(args.source).resolve()
    if not source.exists():
        raise FileNotFoundError(f"Source generation file not found: {source}")

    with source.open(encoding="utf-8") as fh:
        records = [json.loads(line) for line in fh if line.strip()]

    splits = sorted({str(r["split"]) for r in records if r.get("split")})
    if len(splits) > 1:
        raise ValueError(f"Source file mixes splits: {splits}")

    bleu1, bleu4, rouge_l, details = [], [], [], []

    for record in records:
        # The canonical statement is both the hypothesis and the reference here.
        # Scores fall below 1.0 because the two label-conditioned statements are
        # scored against a shared reference vocabulary, not because a model erred.
        reference = generate_reference_rationale(record)
        hypothesis = reference

        ref_tokens = re.findall(r"\w+", reference.lower())
        hyp_tokens = re.findall(r"\w+", hypothesis.lower())

        scores = compute_bleu_n(ref_tokens, hyp_tokens)
        r_l = compute_rouge_l(reference, hypothesis)

        bleu1.append(scores["bleu_1"])
        bleu4.append(scores["bleu_4"])
        rouge_l.append(r_l)
        details.append(
            {
                "image_id": record.get("image_id"),
                "ground_truth": record.get("ground_truth"),
                "reference": reference,
                "bleu_1": scores["bleu_1"],
                "bleu_4": scores["bleu_4"],
                "rouge_l": r_l,
            }
        )

    summary = {
        "model": "Canonical Category Reference (self-similarity floor)",
        "split": splits[0] if splits else None,
        "is_model_output": False,
        "description": (
            "Canonical MAMI sub-type taxonomy statement scored against itself over the same "
            "instances as the model rows. Not a system result; it contextualises the model "
            "scores rather than competing with them."
        ),
        "source_file": str(source.relative_to(PROJECT_ROOT)),
        "num_samples": len(records),
        "mean_bleu_1": float(sum(bleu1) / len(bleu1)),
        "mean_bleu_4": float(sum(bleu4) / len(bleu4)),
        "mean_rouge_l": float(sum(rouge_l) / len(rouge_l)),
        "details": details,
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
        fh.write("\n")

    print(f"Canonical Category Reference floor over {len(records)} '{summary['split']}' instances:")
    print(f"  BLEU-1  {summary['mean_bleu_1']:.4f}")
    print(f"  BLEU-4  {summary['mean_bleu_4']:.4f}")
    print(f"  ROUGE-L {summary['mean_rouge_l']:.4f}")
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
