"""
YOLO object detection training and benchmarking module for the autobenchmark package.

Trains multiple YOLO model variants (v5, v8, v11, v12, v26) on a dataset in YOLO format,
collects detection metrics (mAP50, mAP50-95, precision, recall, F1), and ranks them.
"""

from datetime import datetime
import json
import os
from pathlib import Path
import re
import time

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _safe_filename(name):
    """Convert a model name to a safe directory/file name."""
    return re.sub(r"[^a-zA-Z0-9_-]", "_", name)


def _compute_f1(precision, recall):
    """Compute F1 score from precision and recall."""
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def resolve_data_yaml(data_yaml_path, init_config=None):
    """
    Resolve the path to the YOLO data.yaml file.

    Args:
        data_yaml_path: Path specified in the config.
        init_config: Optional init config dict for base_dir resolution.

    Returns:
        Resolved absolute path string.
    """
    if os.path.isabs(data_yaml_path):
        return data_yaml_path

    if init_config:
        base_dir = init_config.get("system", {}).get("base_dir", "")
        if base_dir:
            full_path = os.path.join(base_dir, data_yaml_path)
            if os.path.exists(full_path):
                return full_path

    if os.path.exists(data_yaml_path):
        return data_yaml_path

    return data_yaml_path


def train_single_yolo_model(model_spec, data_cfg, output_dir, model_config_name):
    """
    Train a single YOLO model and collect evaluation metrics.

    Args:
        model_spec: Dict with 'name' and 'weights' keys.
        data_cfg: Data configuration dict (from image_data1.yaml).
        output_dir: Base output directory for this model's results.
        model_config_name: Name of the model config (for project naming).

    Returns:
        Dict with model name, metrics, training time, and parameter count.
    """
    from ultralytics import YOLO

    model_name = model_spec["name"]
    weights = model_spec["weights"]
    safe_name = _safe_filename(model_name)

    # Extract training parameters
    train_cfg = data_cfg.get("training", {})
    aug_cfg = data_cfg.get("augmentation", {})
    dataset_cfg = data_cfg.get("dataset", {})

    epochs = train_cfg.get("epochs", 100)
    patience = train_cfg.get("patience", 10)
    image_size = train_cfg.get("image_size", 640)
    batch_size = train_cfg.get("batch_size", 16)
    optimizer = train_cfg.get("optimizer", "AdamW")
    lr0 = train_cfg.get("lr0", 0.001)
    lrf = train_cfg.get("lrf", 0.1)
    import torch

    device_setting = train_cfg.get("device", "auto")
    if device_setting == "auto":
        device = "0" if torch.cuda.is_available() else "cpu"
    else:
        device = str(device_setting)
    workers = train_cfg.get("workers", 0)

    data_yaml_path = dataset_cfg.get("data_yaml_path", "")

    # Model output directory
    model_dir = os.path.join(output_dir, safe_name)
    os.makedirs(model_dir, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"Training: {model_name}")
    print(f"  Weights: {weights}")
    print(f"  Epochs: {epochs}, Patience: {patience}")
    print(f"  Image size: {image_size}, Batch: {batch_size}")
    print(f"  Optimizer: {optimizer}, LR: {lr0}")
    print(f"  Device: {device}")
    print(f"{'=' * 60}\n")

    # Initialise model
    try:
        model = YOLO(weights)
    except Exception as e:
        print(f"  ERROR: Failed to load model {weights}: {e}")
        return {
            "Model": model_name,
            "Weights": weights,
            "Status": f"FAILED: {e}",
            "mAP50": 0.0,
            "mAP50-95": 0.0,
            "Precision": 0.0,
            "Recall": 0.0,
            "F1": 0.0,
            "Parameters": 0,
            "Training_Time": 0.0,
        }

    # Train
    train_start = time.time()
    try:
        model.train(
            data=data_yaml_path,
            epochs=epochs,
            imgsz=image_size,
            patience=patience,
            batch=batch_size,
            optimizer=optimizer,
            lr0=lr0,
            lrf=lrf,
            device=device,
            project=model_dir,
            name="train",
            exist_ok=True,
            verbose=True,
            workers=workers,
            # Augmentation
            mosaic=aug_cfg.get("mosaic", 1.0),
            scale=aug_cfg.get("scale", 0.5),
            fliplr=aug_cfg.get("fliplr", 0.5),
            hsv_h=aug_cfg.get("hsv_h", 0.015),
            hsv_s=aug_cfg.get("hsv_s", 0.7),
            hsv_v=aug_cfg.get("hsv_v", 0.4),
        )
    except Exception as e:
        training_time = time.time() - train_start
        print(f"  ERROR: Training failed for {model_name}: {e}")
        return {
            "Model": model_name,
            "Weights": weights,
            "Status": f"TRAIN_FAILED: {e}",
            "mAP50": 0.0,
            "mAP50-95": 0.0,
            "Precision": 0.0,
            "Recall": 0.0,
            "F1": 0.0,
            "Parameters": 0,
            "Training_Time": training_time,
        }

    training_time = time.time() - train_start
    print(f"  Training completed in {training_time:.1f}s")

    # Find best checkpoint
    best_pt = Path(model_dir) / "train" / "weights" / "best.pt"
    if not best_pt.exists():
        # Fallback: look for last.pt
        best_pt = Path(model_dir) / "train" / "weights" / "last.pt"

    if not best_pt.exists():
        print(f"  WARNING: No checkpoint found for {model_name}")
        return {
            "Model": model_name,
            "Weights": weights,
            "Status": "NO_CHECKPOINT",
            "mAP50": 0.0,
            "mAP50-95": 0.0,
            "Precision": 0.0,
            "Recall": 0.0,
            "F1": 0.0,
            "Parameters": 0,
            "Training_Time": training_time,
        }

    # Evaluate using the best checkpoint
    print(f"  Evaluating {model_name} (best checkpoint)...")
    eval_model = YOLO(str(best_pt))

    try:
        results = eval_model.val(data=data_yaml_path, verbose=False, plots=False, workers=workers)

        map50 = float(results.box.map50) if hasattr(results.box, "map50") else 0.0
        map50_95 = float(results.box.map) if hasattr(results.box, "map") else 0.0
        precision = float(results.box.mp) if hasattr(results.box, "mp") else 0.0
        recall = float(results.box.mr) if hasattr(results.box, "mr") else 0.0
        f1 = _compute_f1(precision, recall)
    except Exception as e:
        print(f"  WARNING: Evaluation failed for {model_name}: {e}")
        map50 = map50_95 = precision = recall = f1 = 0.0

    # Get model info
    try:
        total_params = sum(p.numel() for p in eval_model.model.parameters())
    except Exception:
        total_params = 0

    # Get GFLOPs if available
    gflops = None
    try:
        if hasattr(eval_model.model, "info"):
            model_info = eval_model.model.info(verbose=False)
            if isinstance(model_info, dict) and "GFLOPs" in model_info:
                gflops = float(model_info["GFLOPs"])
    except Exception:
        pass

    # Extract per-class metrics if available
    per_class_metrics = {}
    try:
        if hasattr(results.box, "maps"):
            per_class_metrics = {
                "mAP50_per_class": [float(x) for x in results.box.ap50],
                "mAP50-95_per_class": [float(x) for x in results.box.ap],
            }
            if hasattr(results.box, "p"):
                per_class_metrics["precision_per_class"] = [float(x) for x in results.box.p]
            if hasattr(results.box, "r"):
                per_class_metrics["recall_per_class"] = [float(x) for x in results.box.r]
    except Exception:
        pass

    # Save detailed metrics JSON
    detailed_metrics = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_name": model_name,
        "weights": weights,
        "checkpoint_path": str(best_pt),
        "training_time_seconds": training_time,
        "training_time_minutes": training_time / 60.0,
        "parameters": total_params,
        "metrics": {
            "mAP50": map50,
            "mAP50-95": map50_95,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        },
        "training_config": {
            "epochs": epochs,
            "patience": patience,
            "image_size": image_size,
            "batch_size": batch_size,
            "optimizer": optimizer,
            "lr0": lr0,
            "lrf": lrf,
        },
    }
    if gflops is not None:
        detailed_metrics["GFLOPs"] = gflops
    if per_class_metrics:
        detailed_metrics["per_class_metrics"] = per_class_metrics

    metrics_path = os.path.join(model_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(detailed_metrics, f, indent=2)
        f.write("\n")
    print(f"  Saved detailed metrics -> {metrics_path}")

    # Build result row
    result = {
        "Model": model_name,
        "Weights": weights,
        "Status": "OK",
        "mAP50": map50,
        "mAP50-95": map50_95,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Parameters": total_params,
        "Training_Time": training_time,
    }
    if gflops is not None:
        result["GFLOPs"] = gflops

    print(
        f"  {model_name}: mAP50={map50:.4f}, mAP50-95={map50_95:.4f}, "
        f"P={precision:.4f}, R={recall:.4f}, F1={f1:.4f}"
    )

    return result


def train_yolo_benchmark(models_list, data_cfg, model_cfg, output_dir, init_config=None):
    """
    Run the full YOLO benchmark: train and evaluate all specified models sequentially.

    Args:
        models_list: List of dicts, each with 'name' and 'weights'.
        data_cfg: Data configuration dict.
        model_cfg: Model configuration dict.
        output_dir: Base output directory.
        init_config: Optional init config for path resolution.

    Returns:
        pd.DataFrame with ranked results.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Resolve data.yaml path
    raw_data_yaml = data_cfg.get("dataset", {}).get("data_yaml_path", "")
    resolved_yaml = resolve_data_yaml(raw_data_yaml, init_config)

    # Update the config so train functions use the resolved path
    data_cfg.setdefault("dataset", {})["data_yaml_path"] = resolved_yaml

    if not os.path.exists(resolved_yaml):
        print(f"ERROR: YOLO data.yaml not found at: {resolved_yaml}")
        print("Run 'python scripts/download_coco8.py' first to download a test dataset.")
        return pd.DataFrame()

    print(f"YOLO data config: {resolved_yaml}")
    print(f"Models to benchmark: {len(models_list)}")
    print(f"Output directory: {output_dir}")

    model_config_name = model_cfg.get("config_name", "yolo_benchmark")
    optimize_metric = model_cfg.get("optimize_metric", "mAP50")

    # Train each model sequentially
    all_results = []
    for i, model_spec in enumerate(models_list, 1):
        print(f"\n[{i}/{len(models_list)}] Benchmarking {model_spec['name']}...")
        result = train_single_yolo_model(model_spec, data_cfg, output_dir, model_config_name)
        all_results.append(result)

    if not all_results:
        print("No models were trained successfully.")
        return pd.DataFrame()

    # Build results DataFrame
    df_results = pd.DataFrame(all_results)

    # Sort by optimize metric (descending)
    metric_col = _normalize_metric_column(optimize_metric, df_results.columns)
    if metric_col and metric_col in df_results.columns:
        df_results = df_results.sort_values(by=metric_col, ascending=False).reset_index(drop=True)

    # Save evaluation CSV
    eval_csv_path = os.path.join(output_dir, f"{model_config_name}_evaluation.csv")
    df_results.to_csv(eval_csv_path, index=False)
    print(f"\nSaved ranked model evaluation -> {eval_csv_path}")

    # Print ranked table
    print(f"\n{'=' * 80}")
    print(f"MODEL RANKING (Sorted by {optimize_metric} descending)")
    print(f"{'=' * 80}")
    display_cols = [
        "Model",
        "mAP50",
        "mAP50-95",
        "Precision",
        "Recall",
        "F1",
        "Parameters",
        "Training_Time",
        "Status",
    ]
    display_cols = [c for c in display_cols if c in df_results.columns]
    print(df_results[display_cols].to_string())
    print(f"{'=' * 80}")

    # Generate comparison bar chart
    _generate_comparison_chart(df_results, optimize_metric, output_dir, model_config_name)

    # Identify best model
    successful = df_results[df_results["Status"] == "OK"]
    if not successful.empty and metric_col in successful.columns:
        best_row = successful.iloc[0]
        print(f"\nBest Model: {best_row['Model']} ({optimize_metric} = {best_row[metric_col]:.4f})")

    return df_results


def _normalize_metric_column(metric_name, columns):
    """Map user-facing metric name to actual DataFrame column name."""
    mapping = {
        "mAP50": "mAP50",
        "map50": "mAP50",
        "mAP50-95": "mAP50-95",
        "map50-95": "mAP50-95",
        "precision": "Precision",
        "recall": "Recall",
        "f1": "F1",
        "F1": "F1",
    }
    col = mapping.get(metric_name, metric_name)
    if col in columns:
        return col
    # Case-insensitive fallback
    for c in columns:
        if c.lower() == metric_name.lower():
            return c
    return metric_name


def _generate_comparison_chart(df_results, metric_name, output_dir, config_name):
    """Generate a horizontal bar chart comparing models on the specified metric."""
    metric_col = _normalize_metric_column(metric_name, df_results.columns)
    if metric_col not in df_results.columns:
        print(f"  Warning: Cannot generate chart — column '{metric_col}' not in results.")
        return

    successful = df_results[df_results["Status"] == "OK"].copy()
    if successful.empty:
        return

    # Sort ascending for horizontal bar chart (best on top)
    successful = successful.sort_values(by=metric_col, ascending=True)

    _fig, ax = plt.subplots(figsize=(10, max(4, len(successful) * 0.6)))

    colors = plt.get_cmap("viridis")(np.linspace(0.2, 0.8, len(successful)))
    bars = ax.barh(successful["Model"], successful[metric_col], color=colors)

    # Add value labels
    for bar, val in zip(bars, successful[metric_col], strict=False):
        ax.text(
            bar.get_width() + 0.005,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}",
            va="center",
            fontsize=9,
        )

    ax.set_xlabel(metric_col, fontsize=12)
    ax.set_title(f"YOLO Model Comparison — {metric_col}", fontsize=14, fontweight="bold")
    ax.set_xlim(0, min(1.0, successful[metric_col].max() * 1.15))
    plt.tight_layout()

    chart_path = os.path.join(output_dir, f"{config_name}_comparison_bar.png")
    plt.savefig(chart_path, dpi=150)
    plt.close()
    print(f"Saved model comparison bar chart -> {chart_path}")
