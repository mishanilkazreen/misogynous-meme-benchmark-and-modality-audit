"""Threshold calibration and validation sweep across classical ML classifiers.

Sweeps decision thresholds tau in [0.05, 0.95] (step 0.01) on the validation split
to maximize validation Macro-F1, then evaluates transfer to the test split.
"""

from __future__ import annotations

import csv
import glob
import json
import os
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

ROOT_DIR = Path(__file__).resolve().parent.parent
AB_DIR = ROOT_DIR / "auto_benchmark" / "results" / "model_results"
RESULTS_DIR = ROOT_DIR / "results"


def load_uncertainty_data(csv_path: str) -> tuple[np.ndarray, np.ndarray]:
    y_list: list[int] = []
    p_list: list[float] = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            y_list.append(int(float(row["True_Label"])))
            p_list.append(float(row["Probability_Class_1"]))
    return np.array(y_list), np.array(p_list)


def run_threshold_calibration() -> list[dict[str, float | str]]:
    val_files = sorted(
        glob.glob(str(AB_DIR / "mami_tabular_model_provided" / "uncertainty" / "*_uncertainty.csv"))
    )
    if not val_files:
        raise FileNotFoundError("No uncertainty CSV files found.")

    records: list[dict[str, float | str]] = []
    thresholds = np.linspace(0.05, 0.95, 91)

    for vf in val_files:
        tf = vf.replace("mami_tabular_model_provided", "mami_tabular_model_provided_test")
        if not os.path.exists(tf):
            continue
        model_name = (
            os.path.basename(vf)
            .replace("mami_tabular_model_provided_", "")
            .replace("_uncertainty.csv", "")
            .replace("_", " ")
        )

        y_val, p_val = load_uncertainty_data(vf)
        y_test, p_test = load_uncertainty_data(tf)

        # Default tau = 0.50
        val_f1_default = float(f1_score(y_val, (p_val >= 0.50).astype(int), average="macro"))
        val_acc_default = float(accuracy_score(y_val, (p_val >= 0.50).astype(int)))
        test_f1_default = float(f1_score(y_test, (p_test >= 0.50).astype(int), average="macro"))
        test_acc_default = float(accuracy_score(y_test, (p_test >= 0.50).astype(int)))

        # Sweep on validation split
        best_tau = 0.50
        best_val_f1 = -1.0
        for tau in thresholds:
            f1 = float(f1_score(y_val, (p_val >= tau).astype(int), average="macro"))
            if f1 > best_val_f1:
                best_val_f1 = f1
                best_tau = round(float(tau), 2)

        # Apply best_tau to validation and test
        val_acc_calibrated = float(accuracy_score(y_val, (p_val >= best_tau).astype(int)))
        test_f1_calibrated = float(
            f1_score(y_test, (p_test >= best_tau).astype(int), average="macro")
        )
        test_acc_calibrated = float(accuracy_score(y_test, (p_test >= best_tau).astype(int)))

        records.append(
            {
                "model": model_name,
                "val_default_acc": round(val_acc_default, 4),
                "val_default_macro_f1": round(val_f1_default, 4),
                "test_default_acc": round(test_acc_default, 4),
                "test_default_macro_f1": round(test_f1_default, 4),
                "calibrated_tau": best_tau,
                "val_calibrated_acc": round(val_acc_calibrated, 4),
                "val_calibrated_macro_f1": round(best_val_f1, 4),
                "test_calibrated_acc": round(test_acc_calibrated, 4),
                "test_calibrated_macro_f1": round(test_f1_calibrated, 4),
                "test_f1_delta": round(test_f1_calibrated - test_f1_default, 4),
            }
        )

    out_csv = RESULTS_DIR / "threshold_calibration_results.csv"
    out_json = RESULTS_DIR / "threshold_calibration_results.json"

    if records:
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
            writer.writeheader()
            writer.writerows(records)

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
        f.write("\n")

    print(f"Calibration completed for {len(records)} models.")
    print(f"Saved: {out_csv} and {out_json}")
    return records


if __name__ == "__main__":
    run_threshold_calibration()
