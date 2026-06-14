"""Exploratory label analysis for MAMI Sub-task B (Challenge 2).

Computes the multi-label distribution and class overlap for the four MAMI
sub-type labels (shaming, stereotype, objectification, violence) and writes a
machine-readable stats file that can be turned into diagrams
(class-distribution bar chart, co-occurrence heatmap, active-count histogram).

Output: results/challenge_b_label_stats.json

Usage:
    uv run python scripts/analyze_challenge_b_labels.py
"""

from __future__ import annotations

import csv
import itertools
import json
from pathlib import Path
from typing import Any

from utils.dataset import _kaggle_download

SUBTYPES = ["shaming", "stereotype", "objectification", "violence"]
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def _load_split(root: Path, split: str) -> list[dict[str, str]]:
    """Read one MAMI TSV split into a list of row dicts."""
    rows: list[dict[str, str]] = []
    with (root / f"{split}.tsv").open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            rows.append(row)
    return rows


def analyze(rows: list[dict[str, str]]) -> dict[str, Any]:
    """Compute distribution and overlap stats for a set of rows."""
    n = len(rows)
    miso = [int(r["label"]) for r in rows]
    n_miso = sum(miso)

    subtype_counts = {s: sum(int(r[s]) for r in rows) for s in SUBTYPES}

    active_counts = [sum(int(r[s]) for s in SUBTYPES) for r in rows]
    dist_all = {str(k): active_counts.count(k) for k in range(5)}

    m_active = [sum(int(r[s]) for s in SUBTYPES) for r in rows if int(r["label"]) == 1]
    dist_miso = {str(k): m_active.count(k) for k in range(5)}

    cooccurrence: dict[str, dict[str, int]] = {}
    conditional: dict[str, dict[str, float]] = {}
    for a in SUBTYPES:
        na = sum(1 for r in rows if int(r[a]))
        cooccurrence[a] = {}
        conditional[a] = {}
        for b in SUBTYPES:
            c = sum(1 for r in rows if int(r[a]) and int(r[b]))
            cooccurrence[a][b] = c
            conditional[a][b] = round(c / na, 4) if na else 0.0

    pair_cooccurrence = {
        f"{a}|{b}": sum(1 for r in rows if int(r[a]) and int(r[b]))
        for a, b in itertools.combinations(SUBTYPES, 2)
    }

    subtype_when_not_miso = sum(
        1 for r in rows if int(r["label"]) == 0 and any(int(r[s]) for s in SUBTYPES)
    )
    miso_without_subtype = sum(
        1 for r in rows if int(r["label"]) == 1 and not any(int(r[s]) for s in SUBTYPES)
    )

    return {
        "n": n,
        "misogynous": {"positive": n_miso, "negative": n - n_miso},
        "subtype_counts": subtype_counts,
        "active_count_distribution_all": dist_all,
        "active_count_distribution_misogynous": dist_miso,
        "cooccurrence": cooccurrence,
        "conditional_prob_col_given_row": conditional,
        "pair_cooccurrence": pair_cooccurrence,
        "annotation_inconsistencies": {
            "subtype_active_when_not_misogynous": subtype_when_not_miso,
            "misogynous_without_any_subtype": miso_without_subtype,
        },
    }


def main() -> None:
    """Compute stats for the eval set, train, and full dataset; write JSON."""
    root = Path(_kaggle_download())
    train = _load_split(root, "train")
    validation = _load_split(root, "validation")
    test = _load_split(root, "test")

    stats: dict[str, Any] = {
        "labels": SUBTYPES,
        "source": "MAMI 2022 (SemEval-2022 Task 5), Sub-task B",
        "datasets": {
            "test_validation": analyze(validation + test),
            "train": analyze(train),
            "full": analyze(train + validation + test),
        },
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "challenge_b_label_stats.json"
    out.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"Wrote {out}")

    ev = stats["datasets"]["test_validation"]
    print(f"eval set n={ev['n']}, misogynous={ev['misogynous']['positive']}")
    print(f"subtype_counts={ev['subtype_counts']}")


if __name__ == "__main__":
    main()
