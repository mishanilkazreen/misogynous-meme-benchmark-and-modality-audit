#!/usr/bin/env python3
"""Compute and tabulate modality-level feature importance across all trained classical classifiers on frozen CLIP representations for MAMI 2022."""

import json
from pathlib import Path

import joblib
import numpy as np


def main() -> None:
    data_path = Path(
        "auto_benchmark/results/model_results/mami_tabular_model_test/mami_tabular_model_test_data.npz"
    )
    if not data_path.exists():
        print(f"Data file not found: {data_path}")
        return

    image_dim = 768  # CLIP ViT-L-14 visual dimensions

    model_dir = Path("auto_benchmark/results/model_results/mami_tabular_model_test/models")
    model_files = sorted(model_dir.glob("*.joblib"))

    results = []

    for mf in model_files:
        raw_name = mf.stem.replace("mami_tabular_model_test_", "")
        display_name = raw_name.replace("_-_", " ").replace("_", " ")

        try:
            clf = joblib.load(mf)
        except Exception as e:
            print(f"Failed to load {mf}: {e}")
            continue

        if hasattr(clf, "feature_importances_"):
            imp = clf.feature_importances_
            img_val = float(imp[:image_dim].sum())
            txt_val = float(imp[image_dim:].sum())
            method = "Gini / Split Gain"
        elif hasattr(clf, "coef_"):
            coef = np.abs(clf.coef_).ravel()
            img_val = float(coef[:image_dim].sum())
            txt_val = float(coef[image_dim:].sum())
            method = "Linear Weight Magnitude"
        else:
            continue

        tot = img_val + txt_val
        if tot > 0:
            img_pct = (img_val / tot) * 100.0
            txt_pct = (txt_val / tot) * 100.0
        else:
            img_pct, txt_pct = 50.0, 50.0

        results.append(
            {
                "model": display_name,
                "method": method,
                "visual_importance": img_val,
                "text_importance": txt_val,
                "visual_pct": img_pct,
                "text_pct": txt_pct,
            }
        )

    # Sort results by visual reliance
    results.sort(key=lambda x: x["visual_pct"], reverse=True)

    print(
        f"\n{'Model Architecture':35s} | {'Attribution Method':24s} | {'Visual Reliance':16s} | {'Text Reliance':16s}"
    )
    print("-" * 98)
    for r in results:
        print(
            f"{r['model']:35s} | {r['method']:24s} | {r['visual_pct']:13.2f}% | {r['text_pct']:13.2f}%"
        )

    out_csv = Path("results/all_models_modality_importance.csv")
    out_json = Path("results/all_models_modality_importance.json")

    with open(out_csv, "w") as f:
        f.write(
            "model,attribution_method,visual_importance,text_importance,visual_percentage,text_percentage\n"
        )
        for r in results:
            f.write(
                f"{r['model']},{r['method']},{r['visual_importance']:.4f},{r['text_importance']:.4f},{r['visual_pct']:.2f},{r['text_pct']:.2f}\n"
            )

    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved full results to {out_csv} and {out_json}")


if __name__ == "__main__":
    main()
