"""
Generate a preprocessing comparison figure mirroring Qu et al. (2025) Figure 10.

For each subset (digits, hate_slangs, hate_symbols), plots exact_match_accuracy
per preprocessing filter, with one line per model. Highlights blur_histogram as
the recommended filter.

Usage:
    uv run python scripts/plot_preprocessing_comparison.py
    uv run python scripts/plot_preprocessing_comparison.py --binary
    uv run python scripts/plot_preprocessing_comparison.py --json results/vlm_classification.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
SUBSETS = ["digits", "hate_slangs", "hate_symbols"]

# Colour-blind-friendly palette (same order as Qu et al. where possible)
_MODEL_COLOURS: dict[str, str] = {
    "clip": "#1f77b4",
    "llava": "#ff7f0e",
    "llavanext": "#2ca02c",
    "qwen2vl": "#d62728",
    "gemini": "#9467bd",
    "gpt4omini": "#8c564b",
}
_HIGHLIGHT_FILTER = "blur_histogram"


def _load_results(json_path: Path, binary_only: bool) -> list[dict[str, Any]]:
    if not json_path.exists():
        raise FileNotFoundError(
            f"Results file not found: {json_path}\nRun benchmark_vlm_classification.py first."
        )
    data: list[dict[str, Any]] = json.loads(json_path.read_text(encoding="utf-8"))
    if binary_only:
        data = [e for e in data if e.get("binary") is True]
    return data


def _collect_filters(data: list[dict[str, Any]]) -> list[str]:
    """Return filters sorted so 'none' is first, then alphabetically."""
    filters = sorted({e["filter"] for e in data if "filter" in e})
    if "none" in filters:
        filters = ["none"] + [f for f in filters if f != "none"]
    return filters


def plot_comparison(
    json_path: Path,
    binary_only: bool = False,
    output_path: Path | None = None,
) -> Path:
    data = _load_results(json_path, binary_only)
    if not data:
        raise ValueError("No matching entries in results file (check --binary flag).")

    filters = _collect_filters(data)
    models = sorted({e["model"] for e in data if "model" in e})

    fig, axes = plt.subplots(1, len(SUBSETS), figsize=(16, 5), sharey=False)
    fig.patch.set_facecolor("white")

    metric_key = "exact_match_accuracy"
    y_label = "Binary accuracy (yes/no)" if binary_only else "Exact-match accuracy"
    title_suffix = " — binary mode" if binary_only else " — identification mode"

    for ax, subset in zip(axes, SUBSETS, strict=True):
        subset_data = [e for e in data if e.get("subset") == subset]

        for model in models:
            model_data = {
                e["filter"]: e.get(metric_key, 0.0) for e in subset_data if e["model"] == model
            }
            if not model_data:
                continue
            y = [model_data.get(f, np.nan) for f in filters]
            colour = _MODEL_COLOURS.get(model)
            ax.plot(
                range(len(filters)),
                y,
                marker="o",
                markersize=4,
                linewidth=1.5,
                label=model,
                color=colour,
            )

        # Highlight recommended filter
        if _HIGHLIGHT_FILTER in filters:
            idx = filters.index(_HIGHLIGHT_FILTER)
            ax.axvline(x=idx, color="gray", linestyle="--", linewidth=1.0, alpha=0.6)
            ax.text(
                idx + 0.1,
                ax.get_ylim()[0] if ax.get_ylim()[0] != ax.get_ylim()[1] else 0.0,
                _HIGHLIGHT_FILTER,
                fontsize=7,
                color="gray",
                rotation=90,
                va="bottom",
            )

        ax.set_title(subset.replace("_", " "), fontsize=11)
        ax.set_xticks(range(len(filters)))
        ax.set_xticklabels(filters, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel(y_label, fontsize=10)
        ax.set_ylim(0.0, 1.05)
        ax.yaxis.grid(True, linestyle=":", alpha=0.7)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(fontsize=8, loc="upper right")

    fig.suptitle(
        f"Preprocessing filter effect on VLM accuracy{title_suffix}\n"
        "(cf. Qu et al. 2025, Figure 10)",
        fontsize=11,
        y=1.02,
    )
    plt.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    if output_path is None:
        suffix = "_binary" if binary_only else ""
        output_path = FIGURES_DIR / f"preprocessing_comparison{suffix}.png"

    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json",
        default=str(RESULTS_DIR / "vlm_classification.json"),
        help="Path to vlm_classification.json",
    )
    parser.add_argument(
        "--binary",
        action="store_true",
        help="Filter on binary-mode rows only (task == binary)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output figure path (default: results/figures/preprocessing_comparison[_binary].png)",
    )
    args = parser.parse_args()

    out = plot_comparison(
        json_path=Path(args.json),
        binary_only=args.binary,
        output_path=Path(args.output) if args.output else None,
    )
    print(f"Figure saved to {out}")


if __name__ == "__main__":
    main()
