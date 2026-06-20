"""
Data analysis module for the autobenchmark package.
Performs target variable analysis, feature correlation, and word cloud generation.
"""

import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from wordcloud import STOPWORDS, WordCloud


def run_data_profiling(df, data_config, output_dir):
    """
    Perform target variable check, feature correlation analysis, and word cloud generation.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"Running data profiling. Saving results to: {output_dir}")

    dataset_cfg = data_config.get("dataset", {})
    target_col = dataset_cfg.get("target_column")

    # 1. Target Variable Check
    if target_col in df.columns:
        target_counts = df[target_col].value_counts(dropna=False)
        target_pct = df[target_col].value_counts(dropna=False, normalize=True) * 100
        df_target_dist = pd.DataFrame(
            {
                "Value": target_counts.index,
                "Count": target_counts.values,
                "Percentage": target_pct.values,
            }
        )
        target_dist_path = os.path.join(output_dir, "target_distribution.csv")
        df_target_dist.to_csv(target_dist_path, index=False)
        print(f"  Saved target variable distribution -> {target_dist_path}")

        # Save Target Pie Chart
        try:
            plt.figure(figsize=(6, 6))
            target_counts.plot(
                kind="pie",
                autopct="%1.1f%%",
                startangle=90,
                colors=["skyblue", "lightcoral", "lightgreen"],
            )
            plt.title(f"Target Class Distribution: {target_col}")
            plt.ylabel("")
            plt.tight_layout()
            pie_path = os.path.join(output_dir, "target_distribution_pie.png")
            plt.savefig(pie_path, dpi=150)
            plt.close()
            print(f"  Saved target pie chart -> {pie_path}")
        except Exception as e:
            print(f"  Could not generate target distribution pie chart: {e}")
    else:
        print(f"Warning: target column '{target_col}' not in dataset.")

    # 2. Numeric Feature Correlation Analysis
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if len(numeric_cols) > 1:
        corr_matrix = df[numeric_cols].corr()
        corr_path = os.path.join(output_dir, "correlation_matrix.csv")
        corr_matrix.to_csv(corr_path)
        print(f"  Saved feature correlation matrix -> {corr_path}")

        # Save Correlation Heatmap
        try:
            plt.figure(figsize=(10, 8))
            cols = [c[:15] + ".." if len(c) > 15 else c for c in corr_matrix.columns]
            plt.imshow(corr_matrix, cmap="coolwarm", interpolation="none", vmin=-1, vmax=1)
            plt.colorbar()
            plt.xticks(range(len(corr_matrix.columns)), cols, rotation=90, fontsize=6)
            plt.yticks(range(len(corr_matrix.columns)), cols, fontsize=6)
            plt.title("Feature Correlation Heatmap")
            plt.tight_layout()
            heatmap_path = os.path.join(output_dir, "correlation_heatmap.png")
            plt.savefig(heatmap_path, dpi=150)
            plt.close()
            print(f"  Saved correlation heatmap -> {heatmap_path}")
        except Exception as e:
            print(f"  Could not generate correlation heatmap: {e}")
    else:
        print("  Skipping correlation matrix: fewer than 2 numeric features found in dataset.")

    # 3. Word Cloud Generation
    analysis_cfg = data_config.get("data_analysis", {})
    wc_cfg = analysis_cfg.get("word_cloud", {})

    if wc_cfg.get("enabled", False):
        text_column = wc_cfg.get("text_column")
        target_column = wc_cfg.get("target_column", target_col)

        if text_column and text_column in df.columns:
            print("  Generating word clouds...")
            custom_stopwords = set(STOPWORDS)

            # Helper to generate and save a single word cloud
            def create_wc(texts, title, filename):
                text_block = " ".join(str(t) for t in texts if pd.notna(t))
                if text_block.strip():
                    wc = WordCloud(
                        width=800,
                        height=400,
                        background_color="white",
                        stopwords=custom_stopwords,
                        colormap="viridis",
                    ).generate(text_block)
                    plt.figure(figsize=(10, 5))
                    plt.imshow(wc, interpolation="bilinear")
                    plt.axis("off")
                    plt.title(title, fontsize=14, fontweight="bold", pad=10)
                    plt.tight_layout()
                    wc_path = os.path.join(output_dir, filename)
                    plt.savefig(wc_path, dpi=150)
                    plt.close()
                    print(f"    Saved word cloud -> {wc_path}")
                else:
                    print(f"    Skipping word cloud '{filename}': empty text content.")

            # A. Overall Word Cloud
            create_wc(df[text_column], "Overall Word Cloud", "wordcloud_overall.png")

            # B. Per-class Word Clouds
            if target_column in df.columns:
                unique_classes = df[target_column].dropna().unique()
                for cls in unique_classes:
                    cls_df = df[df[target_column] == cls]
                    create_wc(
                        cls_df[text_column],
                        f"Word Cloud for Label: {cls}",
                        f"wordcloud_class_{cls}.png",
                    )
        else:
            print(
                f"Warning: Text column '{text_column}' not found in dataset columns for word cloud."
            )

    summary = {
        "profile_path": os.path.join(output_dir, "target_distribution.csv"),
        "report_path": os.path.join(output_dir, "correlation_matrix.csv"),
    }
    return summary
