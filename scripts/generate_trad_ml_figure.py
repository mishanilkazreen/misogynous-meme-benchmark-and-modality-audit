"""Generate publication-quality figure comparing all traditional ML classifiers on MAMI 2022."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT_DIR / "results"
FIGURES_DIR = ROOT_DIR / "submission" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Set publication style
plt.style.use(
    "seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default"
)
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 10


def generate_figure():
    task_a_path = RESULTS_DIR / "task_a_evaluation_results.csv"

    if not task_a_path.exists():
        print(f"File not found: {task_a_path}")
        return

    df_a = pd.read_csv(task_a_path)
    # Filter traditional ML models only
    df_trad = df_a[df_a["Origin"] == "Traditional ML"].copy()

    # Clean names
    df_trad["Model_Name"] = (
        df_trad["Model"].str.replace("Tabular: ", "").str.replace("Classifier", "").str.strip()
    )

    # Parse metrics
    df_trad["Test_F1"] = df_trad["Test Macro-F1"].astype(float)
    df_trad["Val_F1"] = df_trad["Val Macro-F1"].astype(float)

    # Sort by Test F1
    df_trad = df_trad.sort_values(by="Test_F1", ascending=True)

    _fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

    y_positions = list(range(len(df_trad)))
    height = 0.35

    # Plot Val vs Test F1 bars
    ax.barh(
        [y + height / 2 for y in y_positions],
        df_trad["Val_F1"],
        height,
        label="Validation Macro-F1",
        color="#4C72B0",
        alpha=0.9,
    )
    ax.barh(
        [y - height / 2 for y in y_positions],
        df_trad["Test_F1"],
        height,
        label="Test Macro-F1",
        color="#DD8452",
        alpha=0.9,
    )

    ax.set_yticks(y_positions)
    ax.set_yticklabels(df_trad["Model_Name"], fontsize=9)
    ax.set_xlabel("Macro-F1 Score", fontsize=11, fontweight="bold")
    ax.set_title(
        "Traditional ML Benchmark on Frozen CLIP Features (Task A)",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlim(0.5, 0.9)
    ax.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.9)

    plt.tight_layout()
    out_png = FIGURES_DIR / "trad_ml_benchmark_comparison.png"
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Successfully generated figure: {out_png}")


if __name__ == "__main__":
    generate_figure()
