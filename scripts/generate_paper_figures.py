#!/usr/bin/env python3
"""Generate publication-quality, visually consistent figures for Springer Nature manuscript.

Ensures strict consistency in:
- Figure dimensions, aspect ratios, and resolution (300 DPI)
- Font family, hierarchy, and font sizes (Title 12pt bold, Labels 10pt bold, Ticks 9pt)
- Harmonious color palette (Navy Blue #2B5C8F, Ochre Coral #D95F02)
- Spine styling, grid line weights, and legend formatting
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
FIGURES_DIR = ROOT_DIR / "submission" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Define unified Springer Nature publication styling
STYLE_CONFIG = {
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "axes.labelweight": "bold",
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
}

# Unified Color Palette
COLOR_PRIMARY = "#2B5C8F"  # Deep Slate Navy
COLOR_SECONDARY = "#D95F02"  # Warm Ochre Coral
COLOR_GRID = "#E5E5E5"  # Subtle Light Gray
COLOR_SPINE = "#666666"  # Muted Gray Spine


def apply_shared_axes_styling(ax: plt.Axes) -> None:
    """Apply consistent spine, grid, and background aesthetics to an axis."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(COLOR_SPINE)
    ax.spines["bottom"].set_color(COLOR_SPINE)
    ax.xaxis.grid(True, linestyle="--", linewidth=0.6, color=COLOR_GRID, alpha=0.9, zorder=0)
    ax.yaxis.grid(False)
    ax.set_axisbelow(True)


def generate_figure_1_generalisation_gap() -> None:
    """Generate Figure 1: Validation vs Test Macro-F1 Gap across Classical Classifiers."""
    task_a_path = ROOT_DIR / "results" / "task_a_evaluation_results.csv"
    if not task_a_path.exists():
        print(f"File not found: {task_a_path}")
        return

    df_a = pd.read_csv(task_a_path)
    df_trad = df_a[df_a["Origin"] == "Traditional ML"].copy()

    # Clean display names
    df_trad["Clean_Name"] = (
        df_trad["Model"].str.replace("Tabular: ", "").str.replace("Classifier", "").str.strip()
    )

    df_trad["Test_F1"] = df_trad["Test Macro-F1"].astype(float)
    df_trad["Val_F1"] = df_trad["Val Macro-F1"].astype(float)

    # Sort ascending for bottom-to-top display
    df_trad = df_trad.sort_values(by="Test_F1", ascending=True)

    fig, ax = plt.subplots(figsize=(8.0, 5.8), dpi=300)
    y_pos = np.arange(len(df_trad))
    bar_height = 0.36

    # Validation (Primary) and Test (Secondary) bars
    ax.barh(
        y_pos + bar_height / 2,
        df_trad["Val_F1"],
        bar_height,
        label="Validation Split Macro-$F_1$",
        color=COLOR_PRIMARY,
        edgecolor="none",
        zorder=3,
    )
    ax.barh(
        y_pos - bar_height / 2,
        df_trad["Test_F1"],
        bar_height,
        label="Test Split Macro-$F_1$",
        color=COLOR_SECONDARY,
        edgecolor="none",
        zorder=3,
    )

    apply_shared_axes_styling(ax)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_trad["Clean_Name"])
    ax.set_xlabel("Macro-$F_1$ Classification Score")
    ax.set_title("Validation vs. Test Generalisability Gap on Frozen CLIP Representations", pad=12)
    ax.set_xlim(0.50, 0.90)

    ax.legend(
        loc="lower right",
        frameon=True,
        facecolor="white",
        edgecolor="#D0D0D0",
        framealpha=0.95,
    )

    plt.tight_layout()
    out_file = FIGURES_DIR / "trad_ml_benchmark_comparison.png"
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Successfully generated consistent Figure 1: {out_file}")


def generate_figure_2_modality_importance() -> None:
    """Generate Figure 2: Visual vs Text Modality Reliance across Classical Classifiers."""
    importance_path = ROOT_DIR / "results" / "all_models_modality_importance.json"
    if not importance_path.exists():
        print(f"File not found: {importance_path}")
        return

    with open(importance_path) as f:
        data = json.load(f)

    # Sort ascending for bottom-to-top horizontal bar chart
    data.sort(key=lambda x: x["visual_pct"], reverse=False)

    models = [
        d["model"].replace("Classifier", "").replace("Support Vector Machine", "SVM").strip()
        for d in data
    ]
    visual_pct = [d["visual_pct"] for d in data]
    text_pct = [d["text_pct"] for d in data]

    fig, ax = plt.subplots(figsize=(8.0, 5.8), dpi=300)
    y_pos = np.arange(len(models))
    bar_height = 0.62

    # Visual Modality (Primary) and Text Modality (Secondary) stacked bars
    ax.barh(
        y_pos,
        visual_pct,
        bar_height,
        label="Visual Modality (CLIP ViT-L-14 Vision)",
        color=COLOR_PRIMARY,
        edgecolor="none",
        zorder=3,
    )
    ax.barh(
        y_pos,
        text_pct,
        bar_height,
        left=visual_pct,
        label="Text Modality (CLIP ViT-L-14 Text)",
        color=COLOR_SECONDARY,
        edgecolor="none",
        zorder=3,
    )

    # Add numeric percentage badges inside bars
    for i, (v, t) in enumerate(zip(visual_pct, text_pct, strict=False)):
        if v >= 15.0:
            ax.text(
                v / 2,
                i,
                f"{v:.1f}%",
                ha="center",
                va="center",
                color="white",
                fontweight="bold",
                fontsize=8,
                zorder=4,
            )
        if t >= 15.0:
            ax.text(
                v + t / 2,
                i,
                f"{t:.1f}%",
                ha="center",
                va="center",
                color="white",
                fontweight="bold",
                fontsize=8,
                zorder=4,
            )

    apply_shared_axes_styling(ax)

    # Add 50% parity reference line
    ax.axvline(50.0, color="#888888", linestyle=":", linewidth=1.0, alpha=0.85, zorder=2)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(models)
    ax.set_xlabel("Relative Modality Feature Attribution (%)")
    ax.set_title(
        "Modality Feature Attribution Across Classical Classifiers (MAMI Benchmark)", pad=12
    )
    ax.set_xlim(0.0, 100.0)

    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=2,
        frameon=True,
        facecolor="white",
        edgecolor="#D0D0D0",
        framealpha=0.95,
    )

    plt.tight_layout()
    out_file = FIGURES_DIR / "modality_feature_importance.png"
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Successfully generated consistent Figure 2: {out_file}")


def main() -> None:
    plt.rcParams.update(STYLE_CONFIG)
    generate_figure_1_generalisation_gap()
    generate_figure_2_modality_importance()


if __name__ == "__main__":
    main()
