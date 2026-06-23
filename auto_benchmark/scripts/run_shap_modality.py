#!/usr/bin/env python3
"""Modality-Level SHAP Attribution (Global) for the XGBoost Fusion classifier.

Implements Gist task ID 12 / GitHub issue #89.

The XGBoost Fusion model is trained on a concatenated CLIP ViT-L-14 + PaddleOCR
feature vector produced by ``scripts/train_classifier.py`` with fusion mode
``concat`` = ``np.concatenate([img_emb, txt_emb], axis=1)``. With 768-dim CLIP
towers this gives a fixed column layout:

    columns 0   .. 767   -> image (CLIP visual) features
    columns 768 .. 1535  -> text  (PaddleOCR -> CLIP text) features

This script loads the trained binary model, computes SHAP values on the test
split with ``shap.TreeExplainer``, and sums the absolute SHAP values within each
modality block to produce a global image-vs-text reliance split.

Outputs (paths relative to the parent content-moderation repo):
    results/figures/shap_modality_importance.png
    results/shap_modality_importance.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pickle
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Repo root = parent of auto_benchmark/. Outputs go to the parent repo's results/.
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = (
    REPO_ROOT
    / "results"
    / "models"
    / "xgboost_singleclass_concat_vit_l_14_quickgelu_ocr_paddleocr.pkl"
)
DEFAULT_EMB = REPO_ROOT / "results" / "embeddings" / "test_vit_l_14_quickgelu_ocr_paddleocr.npz"


def _to_class1_2d(shap_values: Any, n_features: int) -> np.ndarray:
    """Normalise TreeExplainer output to a 2D (n_samples, n_features) array.

    Handles the several shapes shap returns across versions:
      - list[array] (one per class) -> take the positive class (index 1)
      - 3D array (n, features, classes) -> take the last class slice
      - 2D array (n, features) -> use as-is
    """
    if isinstance(shap_values, list):
        arr = np.asarray(shap_values[1] if len(shap_values) > 1 else shap_values[0])
    else:
        arr = np.asarray(shap_values)
        if arr.ndim == 3:
            arr = arr[:, :, -1]
    if arr.ndim != 2 or arr.shape[1] != n_features:
        raise ValueError(
            f"Unexpected SHAP value shape {arr.shape}; expected (n_samples, {n_features})."
        )
    return arr


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL))
    parser.add_argument("--embeddings", default=str(DEFAULT_EMB))
    parser.add_argument(
        "--image-dim",
        type=int,
        default=768,
        help="Number of image-feature columns at the start of the fused vector",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=1000,
        help="Cap on test samples used for SHAP (TreeExplainer is exact but O(n)).",
    )
    args = parser.parse_args()

    import shap  # imported here: heavy (numba/llvmlite) and only needed at run time

    model_path = Path(args.model_path)
    emb_path = Path(args.embeddings)
    if not model_path.exists():
        raise SystemExit(f"Model not found: {model_path}")
    if not emb_path.exists():
        raise SystemExit(f"Embeddings not found: {emb_path}")

    print(f"Loading model: {model_path}")
    with model_path.open("rb") as f:
        model = pickle.load(f)

    print(f"Loading embeddings: {emb_path}")
    data = np.load(emb_path, allow_pickle=True)
    img_emb = data["image_embeddings"]
    txt_emb = data["text_embeddings"]
    x = np.concatenate([img_emb, txt_emb], axis=1)
    n_features = x.shape[1]
    image_dim = args.image_dim
    if image_dim >= n_features:
        raise SystemExit(f"--image-dim {image_dim} >= total features {n_features}")
    if args.max_samples and x.shape[0] > args.max_samples:
        x = x[: args.max_samples]
    print(
        f"Feature matrix: {x.shape} (image cols 0:{image_dim}, text cols {image_dim}:{n_features})"
    )

    explainer = shap.TreeExplainer(model)
    shap_values = _to_class1_2d(explainer.shap_values(x), n_features)

    abs_shap = np.abs(shap_values)
    image_importance = float(abs_shap[:, :image_dim].sum())
    text_importance = float(abs_shap[:, image_dim:].sum())
    total = image_importance + text_importance
    image_pct = 100.0 * image_importance / total if total > 0 else 0.0
    text_pct = 100.0 * text_importance / total if total > 0 else 0.0

    print(f"Image modality importance: {image_importance:.4f} ({image_pct:.1f}%)")
    print(f"Text  modality importance: {text_importance:.4f} ({text_pct:.1f}%)")

    figures_dir = REPO_ROOT / "results" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    csv_path = REPO_ROOT / "results" / "shap_modality_importance.csv"

    with csv_path.open("w", encoding="utf-8") as f:
        f.write("modality,sum_abs_shap,percentage\n")
        f.write(f"image,{image_importance:.6f},{image_pct:.4f}\n")
        f.write(f"text,{text_importance:.6f},{text_pct:.4f}\n")
    print(f"Saved CSV: {csv_path}")

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(
        ["Image (CLIP visual)", "Text (PaddleOCR->CLIP)"],
        [image_pct, text_pct],
        color=["#4C72B0", "#DD8452"],
    )
    ax.set_ylabel("Global modality reliance (% of total |SHAP|)")
    ax.set_title("XGBoost Fusion: Image vs Text Modality Importance (SHAP)")
    ax.set_ylim(0, 100)
    for b, pct in zip(bars, [image_pct, text_pct], strict=True):
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + 1,
            f"{pct:.1f}%",
            ha="center",
            va="bottom",
        )
    fig.tight_layout()
    png_path = figures_dir / "shap_modality_importance.png"
    fig.savefig(png_path, dpi=150)
    print(f"Saved plot: {png_path}")


if __name__ == "__main__":
    main()
