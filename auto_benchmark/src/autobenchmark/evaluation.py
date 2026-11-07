"""
Evaluation and model ranking module for the autobenchmark package.
Handles sorting model results by metric, saving them to CSV, and reporting summaries.
"""

import os

import pandas as pd


def save_and_rank_results(df_results, model_cfg, output_dir):
    """
    Sort results based on the optimization metric, save to CSV, and return best model info.

    Args:
        df_results: DataFrame containing metrics for all models.
        model_cfg: Configuration dictionary for model training.
        output_dir: Directory where the CSV should be saved.

    Returns:
        dict: Summary metrics of the best performing model.
    """
    os.makedirs(output_dir, exist_ok=True)
    metric = model_cfg.get("optimize_metric", "f1")

    # Check if metric column exists in the results DataFrame
    # Let's map metric names to actual column names in DataFrame
    col_map = {
        "f1": "F1",
        "accuracy": "Accuracy",
        "precision": "Precision",
        "recall": "Recall",
        "roc_auc": "ROC_AUC",
    }
    col_name = col_map.get(metric.lower(), "F1")

    if col_name not in df_results.columns:
        # Fallback to Accuracy or first available
        col_name = "Accuracy" if "Accuracy" in df_results.columns else df_results.columns[1]

    df_sorted = df_results.sort_values(by=col_name, ascending=False).reset_index(drop=True)

    # Save to CSV
    eval_csv_path = os.path.join(output_dir, f"{model_cfg['config_name']}_evaluation.csv")
    df_sorted.to_csv(eval_csv_path, index=False)
    print(f"Saved ranked model evaluation -> {eval_csv_path}")

    # Get best model info
    best_row = df_sorted.iloc[0]
    best_model_info = {
        "Model": best_row["Model"],
        "Metric_Name": col_name,
        "Metric_Value": best_row[col_name],
        "Accuracy": best_row.get("Accuracy", None),
        "F1": best_row.get("F1", None),
        "ROC_AUC": best_row.get("ROC_AUC", None),
        "Training_Time": best_row.get("Training_Time", None),
        "Hyperparameters": best_row.get("Hyperparameters", None),
    }

    return best_model_info


def load_and_evaluate_results(eval_csv_path, metric_to_sort="F1"):
    """
    Load an existing evaluation CSV and print or return a sorted summary based on metric_to_sort.
    """
    if not os.path.exists(eval_csv_path):
        raise FileNotFoundError(f"Evaluation CSV not found at: {eval_csv_path}")

    df = pd.read_csv(eval_csv_path)

    # Ensure correct capitalization
    standard_columns = {c.lower(): c for c in df.columns}
    col_name = standard_columns.get(metric_to_sort.lower())

    if col_name is None or col_name not in df.columns:
        print(f"Warning: Metric '{metric_to_sort}' not found. Defaulting to first numeric column.")
        # Find first numeric column
        numeric_cols = df.select_dtypes(include=["number"]).columns
        col_name = numeric_cols[0] if not numeric_cols.empty else df.columns[1]

    df_sorted = df.sort_values(by=col_name, ascending=False).reset_index(drop=True)

    print("\n" + "=" * 80)
    print(f"MODEL RANKING (Sorted by {col_name} descending)")
    print("=" * 80)

    cols_to_show = ["Model", col_name, "Accuracy", "F1", "ROC_AUC", "Training_Time"]
    cols_to_show = [c for c in cols_to_show if c in df_sorted.columns]

    print(df_sorted[cols_to_show].to_string())
    print("=" * 80)

    return df_sorted


def generate_evaluation_plots(df_results, output_dir, model_cfg):
    """
    Generate evaluation plots:
    1. Bar chart comparing models by the optimize metric (and/or Accuracy/F1).
    2. Line graph of sorted confidence scores for models that support proba.
    """
    import glob

    import matplotlib.pyplot as plt
    import numpy as np

    os.makedirs(output_dir, exist_ok=True)
    config_name = model_cfg.get("config_name", "model")
    metric = model_cfg.get("optimize_metric", "f1").upper()
    col_map = {
        "F1": "F1",
        "ACCURACY": "Accuracy",
        "PRECISION": "Precision",
        "RECALL": "Recall",
        "ROC_AUC": "ROC_AUC",
    }
    metric_col = col_map.get(metric, "F1")

    # 1. Bar Chart of Model Comparison
    if not df_results.empty:
        try:
            # Check if target metric column is present
            if metric_col not in df_results.columns:
                metric_col = (
                    "Accuracy" if "Accuracy" in df_results.columns else df_results.columns[2]
                )

            df_plot = df_results.sort_values(by=metric_col, ascending=True)

            plt.figure(figsize=(10, 6))
            colors = plt.get_cmap("viridis")(np.linspace(0.2, 0.8, len(df_plot)))
            bars = plt.barh(
                df_plot["Model"], df_plot[metric_col], color=colors, edgecolor="grey", height=0.6
            )

            # Add value labels to the bars
            for bar in bars:
                width = bar.get_width()
                plt.text(
                    width + 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f"{width:.3f}",
                    va="center",
                    ha="left",
                    fontsize=8,
                    fontweight="bold",
                )

            plt.xlabel(metric_col)
            plt.title(f"Model Comparison: {metric_col}", fontsize=12, fontweight="bold", pad=15)
            plt.xlim(0, 1.1)
            plt.grid(axis="x", linestyle="--", alpha=0.7)
            plt.tight_layout()

            bar_plot_path = os.path.join(output_dir, f"{config_name}_comparison_bar.png")
            plt.savefig(bar_plot_path, dpi=150)
            plt.close()
            print(f"Saved model comparison bar chart -> {bar_plot_path}")
        except Exception as e:
            print(f"Failed to generate comparison bar chart: {e}")

    # 2. Line Graph of Confidence Scores
    try:
        uncertainty_dir = os.path.join(output_dir, "uncertainty")
        csv_files = glob.glob(os.path.join(uncertainty_dir, "*_uncertainty.csv"))

        if csv_files:
            plt.figure(figsize=(10, 6))

            for csv_path in csv_files:
                basename = os.path.basename(csv_path)
                model_part = basename[len(config_name) + 1 : -len("_uncertainty.csv")]
                model_name = model_part.replace("_", " ")

                df_unc = pd.read_csv(csv_path)
                if "Confidence_Score" in df_unc.columns:
                    conf_scores = sorted(df_unc["Confidence_Score"].values, reverse=True)
                    plt.plot(conf_scores, label=model_name, linewidth=1.5)

            plt.xlabel("Instances (Sorted by Confidence Descending)", fontsize=10)
            plt.ylabel("Confidence Score (Max Probability)", fontsize=10)
            plt.title("Prediction Confidence Curves", fontsize=12, fontweight="bold", pad=15)
            plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
            plt.grid(True, linestyle="--", alpha=0.5)
            plt.ylim(0.45, 1.05)
            plt.tight_layout()

            conf_plot_path = os.path.join(output_dir, f"{config_name}_confidence_curves.png")
            plt.savefig(conf_plot_path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"Saved confidence curves line graph -> {conf_plot_path}")
        else:
            print("No uncertainty CSV files found, skipping confidence curves plot.")
    except Exception as e:
        print(f"Failed to generate confidence curves plot: {e}")
