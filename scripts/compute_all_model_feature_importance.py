#!/usr/bin/env python3
"""Compute modality-level feature attribution across all classical classifier families.

Evaluates global reliance on visual (CLIP visual) vs textual (CLIP text) features
across tree-based models (XGBoost, Random Forest, Extra Trees) and regularised linear
models (Logistic Regression, Ridge, Linear SVC, LDA).
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
import xgboost as xgb


def main() -> None:
    data_path = Path("auto_benchmark/results/model_results/mami_tabular_model_test/mami_tabular_model_test_data.npz")
    if not data_path.exists():
        raise FileNotFoundError(f"Missing test data: {data_path}")

    data = np.load(data_path, allow_pickle=True)
    x_train, _ = data["X_train"], data["X_test"]
    y_train, _ = data["y_train"], data["y_test"]

    image_dim = 768

    models = {
        "XGBoost": (xgb.XGBClassifier(n_estimators=100, n_jobs=8, random_state=42, eval_metric="logloss"), "Tree Gain"),
        "Random Forest": (RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=8), "Gini Importance"),
        "Extra Trees": (ExtraTreesClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=8), "Gini Importance"),
        "Logistic Regression": (LogisticRegression(max_iter=500, random_state=42, n_jobs=8), "|Weight Coef|"),
        "Ridge Classifier": (RidgeClassifier(random_state=42), "|Weight Coef|"),
        "Linear SVC (SGD)": (SGDClassifier(loss="hinge", random_state=42), "|Weight Coef|"),
        "Linear Discriminant Analysis": (LinearDiscriminantAnalysis(), "Scaling Coef"),
    }

    out_csv = Path("results/all_model_modality_importance.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    rows = ["model,method,visual_importance,text_importance,visual_pct,text_pct\n"]

    for name, (clf, method) in models.items():
        clf.fit(x_train, y_train)
        if hasattr(clf, "feature_importances_"):
            imp = clf.feature_importances_
            img_imp = float(imp[:image_dim].sum())
            txt_imp = float(imp[image_dim:].sum())
        elif hasattr(clf, "coef_"):
            coef = np.abs(clf.coef_).ravel()
            img_imp = float(coef[:image_dim].sum())
            txt_imp = float(coef[image_dim:].sum())
        elif hasattr(clf, "scalings_"):
            scal = np.abs(clf.scalings_).ravel()
            img_imp = float(scal[:image_dim].sum())
            txt_imp = float(scal[image_dim:].sum())
        else:
            continue

        tot = img_imp + txt_imp
        img_pct = 100.0 * img_imp / tot if tot > 0 else 0.0
        txt_pct = 100.0 * txt_imp / tot if tot > 0 else 0.0
        rows.append(f"{name},{method},{img_imp:.6f},{txt_imp:.6f},{img_pct:.2f},{txt_pct:.2f}\n")

    out_csv.write_text("".join(rows))
    print(f"Saved: {out_csv}")


if __name__ == "__main__":
    main()
