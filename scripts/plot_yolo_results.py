"""Plot YOLO benchmark results from yolo_benchmark.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

MODELS = ["yolov8n", "yolov10n", "yolo11n", "yolo12n", "yolo26n"]
SUBSETS = ["digits", "hate_slangs", "hate_symbols"]
DEFAULT_JSON = Path(__file__).resolve().parents[1] / "results" / "yolo_benchmark.json"
FIGURES_DIR = Path(__file__).resolve().parents[1] / "results" / "figures"

SUBSET_COLORS = ["#4C72B0", "#DD8452", "#55A868"]
MODEL_COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]


def load_results(json_path: Path, mode: str) -> dict[tuple[str, str], dict]:
    with json_path.open(encoding="utf-8") as f:
        data = json.load(f)
    return {(e["model"], e["subset"]): e for e in data if e["mode"] == mode}


def _get(lookup: dict, model: str, subset: str, *keys: str) -> float:
    entry = lookup.get((model, subset))
    if entry is None:
        return 0.0
    val: object = entry
    for k in keys:
        val = val.get(k, 0.0) if isinstance(val, dict) else 0.0
    return float(val)  # type: ignore[arg-type]


def _style(ax: plt.Axes) -> None:
    ax.set_facecolor("white")
    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(labelsize=11)


def _label_bars(ax: plt.Axes, bars: plt.BarContainer, fmt: str = "{:.2f}") -> None:
    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + 0.005,
            fmt.format(h),
            ha="center",
            va="bottom",
            fontsize=9,
        )


def fig1_map50_by_model_subset(lookup: dict, output_dir: Path, mode: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.set_facecolor("white")
    x = np.arange(len(MODELS))
    width = 0.22
    ns = len(SUBSETS)

    for i, (subset, color) in enumerate(zip(SUBSETS, SUBSET_COLORS, strict=False)):
        offset = (i - (ns - 1) / 2) * width
        vals = [_get(lookup, m, subset, "computed_metrics", "mAP50") for m in MODELS]
        bars = ax.bar(x + offset, vals, width, label=subset, color=color)
        _label_bars(ax, bars)

    ax.set_xticks(x)
    ax.set_xticklabels(MODELS, fontsize=11)
    ax.set_ylabel("mAP50", fontsize=11)
    ax.set_title(f"mAP50 by Model and Subset — mode: {mode}", fontsize=11)
    ax.legend(fontsize=11)
    _style(ax)
    fig.tight_layout()
    fname = "map50_by_model_subset.png"
    fig.savefig(output_dir / fname, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_dir / fname}")


def fig2_map50_by_visibility(lookup: dict, output_dir: Path, mode: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.set_facecolor("white")
    vis_levels = [1, 2, 3, 4, 5]

    for ax, subset in zip(axes, SUBSETS, strict=False):
        for model, color in zip(MODELS, MODEL_COLORS, strict=False):
            vals = [
                _get(lookup, model, subset, "visibility_metrics", str(v), "mAP50")
                for v in vis_levels
            ]
            ax.plot(vis_levels, vals, marker="o", label=model, color=color)
        ax.set_title(subset, fontsize=11)
        ax.set_xlabel("Visibility level", fontsize=11)
        ax.set_ylabel("mAP50", fontsize=11)
        ax.set_xticks(vis_levels)
        _style(ax)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", ncol=5, fontsize=11, bbox_to_anchor=(0.5, -0.05)
    )
    fig.suptitle(f"mAP50 by Visibility Level — mode: {mode}", fontsize=11)
    fig.tight_layout()
    fname = "map50_by_visibility.png"
    fig.savefig(output_dir / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_dir / fname}")


def fig3_precision_recall_f1(lookup: dict, output_dir: Path, mode: str) -> None:
    metrics = ["precision", "recall", "f1"]
    metric_colors = SUBSET_COLORS
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.set_facecolor("white")
    x = np.arange(len(MODELS))
    width = 0.22

    for ax, subset in zip(axes, SUBSETS, strict=False):
        for i, (metric, color) in enumerate(zip(metrics, metric_colors, strict=False)):
            offset = (i - 1) * width
            vals = [_get(lookup, m, subset, "computed_metrics", metric) for m in MODELS]
            bars = ax.bar(x + offset, vals, width, label=metric, color=color)
            _label_bars(ax, bars)
        ax.set_title(subset, fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(MODELS, fontsize=9, rotation=15)
        ax.set_ylabel("Score", fontsize=11)
        _style(ax)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", ncol=3, fontsize=11, bbox_to_anchor=(0.5, -0.05)
    )
    fig.suptitle(f"Precision / Recall / F1 — mode: {mode}", fontsize=11)
    fig.tight_layout()
    fname = "precision_recall_f1.png"
    fig.savefig(output_dir / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_dir / fname}")


def fig4_inference_time(lookup: dict, output_dir: Path, mode: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.set_facecolor("white")

    for ax, (subset, color) in zip(axes, zip(SUBSETS, SUBSET_COLORS, strict=False), strict=False):
        vals = [_get(lookup, m, subset, "average_inference_time_s") for m in MODELS]
        bars = ax.bar(MODELS, vals, color=color)
        _label_bars(ax, bars, fmt="{:.4f}")
        ax.set_title(subset, fontsize=11)
        ax.set_ylabel("Avg inference time (s)", fontsize=11)
        ax.tick_params(axis="x", labelsize=9, rotation=15)
        _style(ax)

    fig.suptitle(f"Average Inference Time by Model — mode: {mode}", fontsize=11)
    fig.tight_layout()
    fname = "inference_time.png"
    fig.savefig(output_dir / fname, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_dir / fname}")


def fig5_preprocessing_ablation(
    lookup_base: dict, lookup_prep: dict, output_dir: Path, mode: str
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.set_facecolor("white")
    x = np.arange(len(MODELS))
    width = 0.35

    for ax, subset in zip(axes, SUBSETS, strict=False):
        vals_base = [_get(lookup_base, m, subset, "computed_metrics", "mAP50") for m in MODELS]
        vals_prep = [_get(lookup_prep, m, subset, "computed_metrics", "mAP50") for m in MODELS]
        bars1 = ax.bar(
            x - width / 2, vals_base, width, label="without preprocessing", color="#4C72B0"
        )
        bars2 = ax.bar(
            x + width / 2, vals_prep, width, label="with blur_histogram", color="#DD8452"
        )
        _label_bars(ax, bars1)
        _label_bars(ax, bars2)
        ax.set_title(subset, fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(MODELS, fontsize=9, rotation=15)
        ax.set_ylabel("mAP50", fontsize=11)
        _style(ax)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", ncol=2, fontsize=11, bbox_to_anchor=(0.5, -0.05)
    )
    fig.suptitle(f"Preprocessing Ablation (mAP50) — mode: {mode}", fontsize=11)
    fig.tight_layout()
    fname = "preprocessing_ablation.png"
    fig.savefig(output_dir / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_dir / fname}")


def _run_for_mode(mode: str, json_path: Path, preprocessed_json: Path | None) -> None:
    output_dir = FIGURES_DIR / mode
    output_dir.mkdir(parents=True, exist_ok=True)

    lookup = load_results(json_path, mode)
    if not lookup:
        print(f"[SKIP] No data found for mode '{mode}' in {json_path}")
        return

    fig1_map50_by_model_subset(lookup, output_dir, mode)
    fig2_map50_by_visibility(lookup, output_dir, mode)
    fig3_precision_recall_f1(lookup, output_dir, mode)
    fig4_inference_time(lookup, output_dir, mode)

    if preprocessed_json is not None:
        if preprocessed_json.exists():
            lookup_prep = load_results(preprocessed_json, mode)
            if lookup_prep:
                fig5_preprocessing_ablation(lookup, lookup_prep, output_dir, mode)
            else:
                print(
                    f"[SKIP] Preprocessing ablation ({mode}): no data for this mode in {preprocessed_json}"
                )
        else:
            print(f"[SKIP] Preprocessing ablation ({mode}): file not found: {preprocessed_json}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json",
        type=Path,
        default=DEFAULT_JSON,
        dest="json",
        help="Path to yolo_benchmark.json (default: results/yolo_benchmark.json)",
    )
    parser.add_argument(
        "--mode",
        default="all",
        choices=["pretrained", "trained", "all"],
        help="Mode(s) to plot: pretrained, trained, or all (default: all)",
    )
    parser.add_argument(
        "--preprocessed-json",
        type=Path,
        default=None,
        dest="preprocessed_json",
        help="Path to yolo_benchmark_preprocessed.json for preprocessing ablation figure",
    )
    args = parser.parse_args()

    modes = ["pretrained", "trained"] if args.mode == "all" else [args.mode]
    for mode in modes:
        _run_for_mode(mode, args.json, args.preprocessed_json)


if __name__ == "__main__":
    main()
