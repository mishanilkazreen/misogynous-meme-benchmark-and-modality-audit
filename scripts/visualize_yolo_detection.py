"""
YOLO Classification Pipeline Visualization.

This script demonstrates the YOLO classification pipeline on the HatefulIllusion dataset:
1. Loads samples with varying visibility levels (prioritizing harder instances)
2. Shows original, preprocessed, and ground truth (hidden digit) side by side
3. Computes classification metrics: accuracy, precision, recall, F1

The HatefulIllusion dataset contains AI-generated images with hidden digits (0-9).
The model classifies whether an image contains hidden content and predicts the digit.

Usage:
    python scripts/visualize_yolo_detection.py [--samples N] [--output PATH]
"""
# pylint: disable=no-member,wrong-import-position

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
import torch
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from PIL import Image

from models.yolo import YOLOClassifier, ClassificationResult
from utils.preprocessing import PreprocessingPipeline


def download_image(image_path: str) -> np.ndarray:
    """Download an image from the HatefulIllusion dataset."""
    filename = f"digits/{image_path}"
    local_path = hf_hub_download(
        repo_id="yiting/HatefulIllusion_Dataset",
        filename=filename,
        repo_type="dataset"
    )
    pil_img = Image.open(local_path)
    return np.array(pil_img.convert("RGB"))


def download_condition_image(condition_path: str) -> np.ndarray:
    """Download the condition/ground truth image (the hidden digit)."""
    filename = f"digits/{condition_path}"
    local_path = hf_hub_download(
        repo_id="yiting/HatefulIllusion_Dataset",
        filename=filename,
        repo_type="dataset"
    )
    pil_img = Image.open(local_path)
    return np.array(pil_img.convert("RGB"))


def load_samples_by_visibility(  # pylint: disable=too-many-locals
    num_samples: int = 10
) -> List[Dict]:
    """Load samples prioritizing lower visibility (harder instances)."""
    print(f"Loading {num_samples} samples (prioritizing low visibility)...")
    ds = load_dataset("yiting/HatefulIllusion_Dataset", "digits", split="train")

    by_visibility: Dict[int, List[int]] = {i: [] for i in range(6)}
    for idx, item in enumerate(ds):
        by_visibility[item["visibility"]].append(idx)

    print("  Visibility distribution:")
    for vis, indices in sorted(by_visibility.items()):
        print(f"    Level {vis}: {len(indices)} images")

    selected: List[int] = []
    for vis in range(6):
        available = by_visibility[vis]
        needed = num_samples - len(selected)
        if needed <= 0:
            break
        step = max(1, len(available) // min(needed, len(available)))
        for i in range(0, len(available), step):
            if len(selected) >= num_samples:
                break
            selected.append(available[i])

    samples: List[Dict] = []
    for idx in selected:
        item = ds[idx]
        print(f"  Loading idx={idx}, digit={item['message']}, vis={item['visibility']}")
        samples.append({
            "index": idx,
            "image": download_image(item["image"]),
            "condition_image": download_condition_image(item["condition_image"]),
            "message": int(item["message"]),
            "visibility": item["visibility"],
            "visibility_level": "low" if item["visibility"] <= 2 else "high",
        })
    return samples


def run_classification(
    model: YOLOClassifier,
    image: np.ndarray,
    preprocessor: Optional[PreprocessingPipeline] = None,
) -> Tuple[ClassificationResult, float, np.ndarray]:
    """Run YOLO classification and return result, time, and processed image."""
    input_size = (416, 416)
    img_resized = cv2.resize(image, input_size)

    if preprocessor is not None:
        img_resized = preprocessor.preprocess(img_resized)

    tensor = torch.from_numpy(img_resized.transpose(2, 0, 1)).float() / 255.0
    tensor = tensor.unsqueeze(0)

    start = time.perf_counter()
    result = model.predict(tensor)
    elapsed_ms = (time.perf_counter() - start) * 1000

    return result, elapsed_ms, img_resized


def compute_metrics(  # pylint: disable=too-many-locals
    samples: List[Dict],
    model: YOLOClassifier,
    preprocessor: Optional[PreprocessingPipeline],
) -> Dict:
    """Compute accuracy, precision, recall, F1 metrics."""
    correct = 0
    total = len(samples)
    total_time = 0.0
    tp, fp, fn = 0, 0, 0

    # Track predictions vs ground truth for detailed output
    predictions: List[int] = []
    ground_truth: List[int] = []

    for sample in samples:
        result, elapsed, _ = run_classification(model, sample["image"], preprocessor)
        total_time += elapsed

        predicted_digit = result.predicted_class
        actual_digit = sample["message"]

        predictions.append(predicted_digit)
        ground_truth.append(actual_digit)

        if predicted_digit == actual_digit:
            correct += 1
            tp += 1
        else:
            fn += 1  # Missed the correct digit

    accuracy = correct / max(1, total)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-6, precision + recall)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "avg_inference_ms": total_time / max(1, total),
        "total_samples": total,
        "correct": correct,
        "predictions": predictions,
        "ground_truth": ground_truth,
    }


def create_row_visualization(  # pylint: disable=too-many-locals
    sample: Dict,
    model: YOLOClassifier,
    preprocessor: PreprocessingPipeline,
    target_size: Tuple[int, int] = (350, 350),
) -> np.ndarray:
    """Create a 3-column row: Original + Preprocessed + Ground Truth."""
    res_orig, time_orig, img_orig = run_classification(model, sample["image"], None)
    res_prep, time_prep, img_prep = run_classification(
        model, sample["image"], preprocessor)

    img_orig = cv2.resize(img_orig, target_size)
    img_prep = cv2.resize(img_prep, target_size)
    condition = cv2.resize(sample["condition_image"], target_size)

    # Add prediction info to images with color coding
    def add_label(img: np.ndarray, text: str, is_correct: bool) -> np.ndarray:
        result = img.copy()
        # Color: Green if correct, Red if wrong
        color = (0, 150, 0) if is_correct else (150, 0, 0)
        cv2.rectangle(result, (0, 0), (result.shape[1], 22), color, -1)
        cv2.putText(result, text, (5, 16), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (255, 255, 255), 1)
        return result

    actual = sample["message"]
    correct_orig = res_orig.predicted_class == actual
    correct_prep = res_prep.predicted_class == actual

    vis_orig = add_label(
        img_orig,
        f"Pred: {res_orig.predicted_class} ({time_orig:.0f}ms)",
        correct_orig
    )
    vis_prep = add_label(
        img_prep,
        f"Pred: {res_prep.predicted_class} ({time_prep:.0f}ms)",
        correct_prep
    )

    # Ground truth with neutral color
    gt_img = condition.copy()
    cv2.rectangle(gt_img, (0, 0), (gt_img.shape[1], 22), (50, 50, 50), -1)
    cv2.putText(gt_img, f"Ground Truth: {actual}",
                (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    # Info bar with prediction accuracy
    status_orig = "✓" if correct_orig else "✗"
    status_prep = "✓" if correct_prep else "✗"
    info = f"Sample {sample['index']} | GT: {actual} | Vis: {sample['visibility']} | "
    info += f"Orig: {res_orig.predicted_class}{status_orig} | "
    info += f"Prep: {res_prep.predicted_class}{status_prep}"

    info_bar = np.zeros((28, target_size[0] * 3, 3), dtype=np.uint8)
    cv2.putText(info_bar, info, (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (255, 255, 255), 1)

    row = np.hstack([vis_orig, vis_prep, gt_img])
    return np.vstack([info_bar, row])


def print_metrics_table(metrics: Dict, title: str) -> None:
    """Print metrics in formatted table with prediction details."""
    print(f"\n{'=' * 60}")
    print(f" {title}")
    print(f"{'=' * 60}")
    print(f"{'Metric':<25} {'Value':>20}")
    print("-" * 60)
    print(f"{'Accuracy':<25} {metrics['accuracy']:>20.4f}")
    print(f"{'Precision':<25} {metrics['precision']:>20.4f}")
    print(f"{'Recall':<25} {metrics['recall']:>20.4f}")
    print(f"{'F1 Score':<25} {metrics['f1']:>20.4f}")
    print("-" * 60)
    print(f"{'Avg Inference (ms)':<25} {metrics['avg_inference_ms']:>20.1f}")
    print(f"{'Correct / Total':<25} "
          f"{metrics['correct']:>10} / {metrics['total_samples']:<7}")

    # Show prediction vs ground truth comparison
    if 'predictions' in metrics and 'ground_truth' in metrics:
        print("-" * 60)
        print("Predictions vs Ground Truth:")
        preds = metrics['predictions']
        gt = metrics['ground_truth']
        for i, (pred, actual) in enumerate(zip(preds, gt)):
            status = "✓" if pred == actual else "✗"
            print(f"  Sample {i+1}: Predicted {pred}, Actual {actual} {status}")

    print("=" * 60)


def build_visualization(
    samples: List[Dict],
    model: YOLOClassifier,
    preprocessor: PreprocessingPipeline,
) -> np.ndarray:
    """Build the full visualization grid."""
    rows = []
    for i, sample in enumerate(samples):
        print(f"  Processing {i+1}/{len(samples)}: "
              f"digit={sample['message']}, vis={sample['visibility']}")
        rows.append(create_row_visualization(sample, model, preprocessor))

    title_width = rows[0].shape[1]
    title_bar = np.zeros((45, title_width, 3), dtype=np.uint8)
    cv2.putText(title_bar, "HatefulIllusion: Hidden Digit Classification",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    col_width = title_width // 3
    header = np.zeros((25, title_width, 3), dtype=np.uint8)
    cv2.putText(header, "Original", (col_width // 2 - 30, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
    cv2.putText(header, "Preprocessed", (col_width + col_width // 2 - 50, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
    cv2.putText(header, "Ground Truth", (col_width * 2 + col_width // 2 - 50, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

    # Add legend
    legend = np.zeros((30, title_width, 3), dtype=np.uint8)
    cv2.putText(legend, "Legend: Green=Correct Prediction, Red=Wrong Prediction",
                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    return np.vstack([title_bar, header, legend] + rows)


def main() -> None:  # pylint: disable=too-many-locals
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Visualize YOLO classification on HatefulIllusion dataset")
    parser.add_argument("--samples", "-n", type=int, default=10)
    parser.add_argument("--output", "-o", default="yolo_detection_results.png")
    parser.add_argument("--conf", type=float, default=0.3)
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print(" YOLO Classification - HatefulIllusion Dataset")
    print("=" * 60)

    print("\nInitializing YOLO classifier (untrained weights)...")
    model = YOLOClassifier(num_classes=10, conf_threshold=args.conf)

    print("Initializing preprocessor (blur + histogram eq)...")
    preprocessor = PreprocessingPipeline(apply_blur=True, apply_equalization=True)

    samples = load_samples_by_visibility(args.samples)

    print("\nComputing classification metrics...")
    print("(Note: Model has random weights - metrics show baseline)")

    metrics_orig = compute_metrics(samples, model, None)
    print_metrics_table(metrics_orig, "Metrics WITHOUT Preprocessing")

    metrics_prep = compute_metrics(samples, model, preprocessor)
    print_metrics_table(metrics_prep, "Metrics WITH Preprocessing")

    print(f"\nCreating visualization for {len(samples)} samples...")
    grid = build_visualization(samples, model, preprocessor)

    output_path = Path(args.output)
    cv2.imwrite(str(output_path), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    print(f"\nVisualization saved to: {output_path.absolute()}")

    print("\n" + "=" * 60)
    print(" Summary")
    print("=" * 60)
    print(f"{'Metric':<20} {'No Prep':>12} {'With Prep':>12}")
    print("-" * 60)
    for key in ["accuracy", "precision", "recall", "f1"]:
        print(f"{key.capitalize():<20} {metrics_orig[key]:>12.4f} "
              f"{metrics_prep[key]:>12.4f}")

    print("\n" + "=" * 60)
    print(" VISUALIZATION GUIDE")
    print("=" * 60)
    print("• 3 columns: Original image | Preprocessed | Ground Truth digit")
    print("• Green headers: Correct prediction")
    print("• Red headers: Wrong prediction")
    print("• Info bar shows: Sample ID, Ground Truth, Visibility, Predictions")
    print("• ✓ = Correct, ✗ = Wrong")
    print("• Model has untrained weights (random predictions)")


if __name__ == "__main__":
    main()
