"""
Uncertainty estimation module for the autobenchmark package.
Calculates Shannon Entropy and prediction confidence scores for classifiers
supporting probabilistic predictions.
"""

import os
import re

import numpy as np
import pandas as pd


def _safe_filename(name):
    return re.sub(r"[^a-zA-Z0-9_-]", "_", name)


def calculate_prediction_uncertainty(model, model_name, X_test, y_test, output_dir, model_cfg):
    """
    Calculate Shannon Entropy, confidence score, and margin for test set predictions.
    Saves outputs in a CSV file.

    Args:
        model: Trained model instance.
        model_name: Name of the model.
        X_test: Preprocessed test set features.
        y_test: True targets for test set.
        output_dir: Directory where the output CSV should be written.
        model_cfg: Configuration dictionary.

    Returns:
        pd.DataFrame: Uncertainty summary table.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"Calculating prediction uncertainty for: {model_name}")

    if not hasattr(model, "predict_proba"):
        print(
            f"  Skipping uncertainty analysis: Model '{model_name}' does not support predict_proba."
        )
        return None

    # Get predictions and probabilities
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)

    # Shannon Entropy calculation: H(X) = -sum(p_i * log2(p_i))
    # Clip probabilities to prevent log(0)
    probs_clipped = np.clip(probs, 1e-15, 1.0)
    entropy = -np.sum(probs_clipped * np.log2(probs_clipped), axis=1)

    # Confidence Score (maximum probability)
    confidence = np.max(probs, axis=1)

    # Confidence Margin
    # If binary, margin is absolute difference between probabilities
    # If multiclass, margin is difference between top 1 and top 2 probabilities
    if probs.shape[1] == 2:
        margin = np.abs(probs[:, 0] - probs[:, 1])
    else:
        # Sort probabilities per row
        sorted_probs = np.sort(probs, axis=1)
        margin = sorted_probs[:, -1] - sorted_probs[:, -2]

    # Construct uncertainty summary DataFrame
    df_uncertainty = pd.DataFrame(
        {
            "Instance_Index": np.arange(len(y_test)),
            "True_Label": np.array(y_test),
            "Predicted_Label": preds,
            "Correct": (np.array(y_test) == preds).astype(int),
            "Confidence_Score": confidence,
            "Confidence_Margin": margin,
            "Shannon_Entropy": entropy,
        }
    )

    # Add probability of each class
    for i in range(probs.shape[1]):
        df_uncertainty[f"Probability_Class_{i}"] = probs[:, i]

    # Save to CSV
    safe_name = _safe_filename(model_name)
    uncertainty_csv_path = os.path.join(
        output_dir, f"{model_cfg['config_name']}_{safe_name}_uncertainty.csv"
    )
    df_uncertainty.to_csv(uncertainty_csv_path, index=False)
    print(f"  Saved prediction uncertainty table -> {uncertainty_csv_path}")

    # Log summary statistics
    print(f"    Avg Shannon Entropy: {entropy.mean():.4f}")
    print(f"    Avg Confidence Score: {confidence.mean():.4f}")

    return df_uncertainty
