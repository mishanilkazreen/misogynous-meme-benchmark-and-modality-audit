"""Plot the impact of PaddleOCR transcripts on model performance.

Renders a grouped bar chart comparing with-PaddleOCR against without-OCR
performance for CLIP (fine-tuned), LLaVA-1.5-7B and Qwen2-VL-2B across the
validation and test splits. Every bar is read directly from a committed result
file under ``results/test/`` and ``results/validation/``; no value is recomputed
or fabricated. Each result file stores ``exact_match_accuracy`` and ``f1``, which
are the two metrics plotted.

Model/file pairings (base without-OCR vs with-OCR ``*_ocr_paddleocr*``):
    - CLIP (fine-tuned, multiclass):
      ``clip_<split>_finetuned_multiclass_vit_b_32_quickgelu.json`` vs
      ``clip_<split>_finetuned_multiclass_vit_b_32_quickgelu_ocr_paddleocr.json``
    - LLaVA-1.5-7B (zero-shot):
      ``llava_<split>_llava_1_5_7b_hf.json`` vs
      ``llava_<split>_llava_1_5_7b_hf_ocr_paddleocr.json``
    - Qwen2-VL-2B (zero-shot):
      ``qwen2vl_<split>_qwen2_vl_2b_instruct.json`` vs
      ``qwen2vl_<split>_qwen2_vl_2b_instruct_ocr_paddleocr.json``

Usage:
    uv run python scripts/plot_ocr_impact.py
    uv run python scripts/plot_ocr_impact.py --output results/figures/ocr_impact_comparison.png
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

SPLITS = ["validation", "test"]
METRICS = [("exact_match_accuracy", "Accuracy"), ("f1", "F1")]

# Display label -> (base file stem, with-OCR file stem), formatted per split.
_MODEL_FILES: dict[str, tuple[str, str]] = {
    "CLIP (fine-tuned)": (
        "clip_{split}_finetuned_multiclass_vit_b_32_quickgelu",
        "clip_{split}_finetuned_multiclass_vit_b_32_quickgelu_ocr_paddleocr",
    ),
    "LLaVA-1.5-7B": (
        "llava_{split}_llava_1_5_7b_hf",
        "llava_{split}_llava_1_5_7b_hf_ocr_paddleocr",
    ),
    "Qwen2-VL-2B": (
        "qwen2vl_{split}_qwen2_vl_2b_instruct",
        "qwen2vl_{split}_qwen2_vl_2b_instruct_ocr_paddleocr",
    ),
}

# Colour-blind-friendly pair (without-OCR, with-OCR).
_COLOUR_WITHOUT = "#4477aa"
_COLOUR_WITH = "#ee6677"


def _load_metrics(stem: str, split: str) -> dict[str, float]:
    """Return the metric values stored in a committed result file."""
    path = RESULTS_DIR / split / f"{stem}.json"
    if not path.exists():
        raise FileNotFoundError(f"Committed result file not found: {path}")
    data: Any = json.loads(path.read_text(encoding="utf-8"))
    record = data[0] if isinstance(data, list) else data
    metrics: dict[str, float] = {}
    for key, _label in METRICS:
        if key not in record:
            raise KeyError(f"Metric '{key}' missing from {path}")
        metrics[key] = float(record[key])
    return metrics


def _collect(split: str) -> dict[str, dict[str, dict[str, float]]]:
    """Collect without-/with-OCR metrics for every model on one split."""
    collected: dict[str, dict[str, dict[str, float]]] = {}
    for label, (base_stem, ocr_stem) in _MODEL_FILES.items():
        collected[label] = {
            "without": _load_metrics(base_stem.format(split=split), split),
            "with": _load_metrics(ocr_stem.format(split=split), split),
        }
    return collected


def plot_ocr_impact(output_path: Path | None = None) -> Path:
    """Render the grouped with-vs-without-PaddleOCR bar chart and save it."""
    models = list(_MODEL_FILES.keys())
    n_rows = len(METRICS)
    n_cols = len(SPLITS)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(12, 9),
        sharey="row",
    )
    fig.patch.set_facecolor("white")

    bar_width = 0.38
    x_positions = np.arange(len(models))

    for row_idx, (metric_key, metric_label) in enumerate(METRICS):
        for col_idx, split in enumerate(SPLITS):
            ax = axes[row_idx][col_idx]
            collected = _collect(split)

            without_values = [collected[m]["without"][metric_key] for m in models]
            with_values = [collected[m]["with"][metric_key] for m in models]

            bars_without = ax.bar(
                x_positions - bar_width / 2,
                without_values,
                bar_width,
                label="Without OCR",
                color=_COLOUR_WITHOUT,
            )
            bars_with = ax.bar(
                x_positions + bar_width / 2,
                with_values,
                bar_width,
                label="With PaddleOCR",
                color=_COLOUR_WITH,
            )

            for bars in (bars_without, bars_with):
                ax.bar_label(bars, fmt="%.3f", fontsize=7, padding=2)

            ax.set_title(f"{metric_label} ({split})", fontsize=11)
            ax.set_xticks(x_positions)
            ax.set_xticklabels(models, rotation=20, ha="right", fontsize=9)
            ax.set_ylim(0.0, 1.05)
            ax.set_ylabel(metric_label, fontsize=10)
            ax.yaxis.grid(True, linestyle=":", alpha=0.7)
            ax.set_axisbelow(True)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            if row_idx == 0 and col_idx == 0:
                ax.legend(fontsize=9, loc="upper right")

    fig.suptitle(
        "PaddleOCR transcript impact on model performance\n"
        "(with-OCR vs without-OCR; CLIP fine-tuned, LLaVA-1.5-7B, Qwen2-VL-2B)",
        fontsize=12,
        y=1.0,
    )
    plt.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    if output_path is None:
        output_path = FIGURES_DIR / "ocr_impact_comparison.png"

    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default=None,
        help="Output figure path (default: results/figures/ocr_impact_comparison.png)",
    )
    args = parser.parse_args()

    out = plot_ocr_impact(Path(args.output) if args.output else None)
    print(f"Figure saved to {out}")


if __name__ == "__main__":
    main()
