"""
YOLO Detection Model Evaluation and Visualization.

This script evaluates a trained YOLO detection model on unseen test samples:
1. Loads the trained model from checkpoint
2. Selects random unseen samples from the dataset
3. Runs inference and compares predictions vs ground truth
4. Generates a visualization grid showing results

Usage:
    python scripts/visualize_yolo_detection.py [--samples N] [--output PATH]
"""
# pylint: disable=no-member,wrong-import-position

import argparse
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
from datasets import load_dataset
from huggingface_hub import hf_hub_download
import numpy as np
from PIL import Image
import torch
from torch import nn
from torchvision import models  # type: ignore[import-untyped]


class ResNetBackbone(nn.Module):
    """Pretrained ResNet backbone for feature extraction."""

    def __init__(self, pretrained: bool = True):
        """Initialize ResNet18 backbone."""
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        resnet = models.resnet18(weights=weights)
        self.features = nn.Sequential(*list(resnet.children())[:-2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.feature_dim = 512

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.features(x)
        x = self.avgpool(x)
        return x.flatten(1)


class YOLODetector(nn.Module):
    """YOLO-style detector with pretrained backbone."""

    def __init__(self, num_classes: int = 10, pretrained: bool = True):
        """Initialize detector."""
        super().__init__()
        self.backbone = ResNetBackbone(pretrained=pretrained)
        self.classifier = nn.Sequential(
            nn.Linear(self.backbone.feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )
        self.bbox_head = nn.Sequential(
            nn.Linear(self.backbone.feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 4),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass."""
        features = self.backbone(x)
        return self.classifier(features), self.bbox_head(features)


def extract_bbox_from_condition(condition_path: str, subset: str) -> tuple[int, int, int, int]:
    """Extract bounding box from condition image using OpenCV."""
    local_path = hf_hub_download(
        repo_id="yiting/HatefulIllusion_Dataset",
        filename=f"{subset}/{condition_path}",
        repo_type="dataset",
    )
    img = np.array(Image.open(local_path).convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        return x * 2, y * 2, w * 2, h * 2
    return 0, 0, 512, 512


def load_test_samples(  # pylint: disable=too-many-locals
    num_samples: int,
    subsets: list[str],
    seed: int = 42,
) -> list[dict]:
    """Load random unseen test samples from specified subsets."""
    random.seed(seed)
    all_samples: list[dict] = []

    for subset in subsets:
        ds = load_dataset("yiting/HatefulIllusion_Dataset", subset, split="train")
        total = len(ds)
        test_start = int(total * 0.8)
        test_indices = list(range(test_start, total))

        for idx in test_indices:
            item = ds[idx]
            all_samples.append(
                {
                    "subset": subset,
                    "index": idx,
                    "image_path": item["image"],
                    "condition_path": item["condition_image"],
                    "message": item["message"],
                    "visibility": item["visibility"],
                }
            )

    selected = random.sample(all_samples, min(num_samples, len(all_samples)))
    print(f"Selected {len(selected)} test samples from {len(all_samples)} available")
    return selected


def load_and_preprocess_image(
    image_path: str,
    subset: str,
    input_size: tuple[int, int] = (224, 224),
) -> tuple[torch.Tensor, np.ndarray]:
    """Load image and return tensor + original for visualization."""
    local_path = hf_hub_download(
        repo_id="yiting/HatefulIllusion_Dataset",
        filename=f"{subset}/{image_path}",
        repo_type="dataset",
    )
    pil_image = Image.open(local_path).convert("RGB")
    img_resized = pil_image.resize(input_size, Image.Resampling.BILINEAR)
    img_np = np.array(img_resized)

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_norm = (img_np / 255.0 - mean) / std
    tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).float()

    return tensor, img_np


def load_condition_image(condition_path: str, subset: str) -> np.ndarray:
    """Load condition/ground truth image."""
    local_path = hf_hub_download(
        repo_id="yiting/HatefulIllusion_Dataset",
        filename=f"{subset}/{condition_path}",
        repo_type="dataset",
    )
    return np.array(Image.open(local_path).convert("RGB"))


def calculate_iou(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Calculate IoU between predicted and target bbox."""
    pred_x1 = float(pred[0] - pred[2] / 2)
    pred_y1 = float(pred[1] - pred[3] / 2)
    pred_x2 = float(pred[0] + pred[2] / 2)
    pred_y2 = float(pred[1] + pred[3] / 2)

    target_x1 = float(target[0] - target[2] / 2)
    target_y1 = float(target[1] - target[3] / 2)
    target_x2 = float(target[0] + target[2] / 2)
    target_y2 = float(target[1] + target[3] / 2)

    inter_x1 = max(pred_x1, target_x1)
    inter_y1 = max(pred_y1, target_y1)
    inter_x2 = min(pred_x2, target_x2)
    inter_y2 = min(pred_y2, target_y2)

    inter_area = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
    pred_area = float(pred[2] * pred[3])
    target_area = float(target[2] * target[3])
    union_area = pred_area + target_area - inter_area

    return inter_area / (union_area + 1e-6)


def draw_bbox(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    img: np.ndarray,
    bbox: tuple[float, float, float, float],
    color: tuple[int, int, int],
    label: str,
    thickness: int = 2,
) -> np.ndarray:
    """Draw bounding box on image."""
    h, w = img.shape[:2]
    cx, cy, bw, bh = bbox
    x1 = int((cx - bw / 2) * w)
    y1 = int((cy - bh / 2) * h)
    x2 = int((cx + bw / 2) * w)
    y2 = int((cy + bh / 2) * h)

    result = img.copy()
    cv2.rectangle(result, (x1, y1), (x2, y2), color, thickness)
    cv2.putText(result, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    return result


def evaluate_and_visualize(  # pylint: disable=too-many-locals,too-many-statements
    model: YOLODetector,
    samples: list[dict],
    device: torch.device,
    label_to_idx: dict[str, int],
    idx_to_label: dict[int, str],
    target_size: tuple[int, int] = (300, 300),
) -> tuple[np.ndarray, dict]:
    """Evaluate model and create visualization grid."""
    model.eval()
    rows: list[np.ndarray] = []
    correct = 0
    total_iou = 0.0

    with torch.no_grad():
        for i, sample in enumerate(samples):
            tensor, img_np = load_and_preprocess_image(sample["image_path"], sample["subset"])
            condition_img = load_condition_image(sample["condition_path"], sample["subset"])

            x, y, w, h = extract_bbox_from_condition(sample["condition_path"], sample["subset"])
            orig_size = 1024
            gt_bbox = (
                (x + w / 2) / orig_size,
                (y + h / 2) / orig_size,
                w / orig_size,
                h / orig_size,
            )
            gt_bbox_tensor = torch.tensor(gt_bbox)

            tensor = tensor.unsqueeze(0).to(device)
            class_logits, bbox_pred = model(tensor)
            pred_idx = class_logits.argmax(dim=1).item()
            pred_label = idx_to_label.get(pred_idx, str(pred_idx))
            pred_bbox = bbox_pred[0].cpu()
            confidence = torch.softmax(class_logits, dim=1).max().item()

            actual_label = str(sample["message"])
            actual_idx = label_to_idx.get(actual_label, -1)
            is_correct = pred_idx == actual_idx
            iou = calculate_iou(pred_bbox, gt_bbox_tensor)

            if is_correct:
                correct += 1
            total_iou += iou

            img_vis = cv2.resize(img_np, target_size)
            condition_vis = cv2.resize(condition_img, target_size)

            # Truncate long labels for display
            pred_short = pred_label[:12] + ".." if len(pred_label) > 14 else pred_label
            actual_short = actual_label[:12] + ".." if len(actual_label) > 14 else actual_label

            img_with_pred = draw_bbox(
                img_vis, tuple(pred_bbox.tolist()), (0, 0, 255), f"P:{pred_short}"
            )
            img_with_both = draw_bbox(img_with_pred, gt_bbox, (0, 255, 0), "GT")

            header_color = (0, 150, 0) if is_correct else (150, 0, 0)
            header = np.zeros((30, target_size[0] * 3, 3), dtype=np.uint8)
            header[:, :] = header_color

            status = "CORRECT" if is_correct else "WRONG"
            text = (
                f"#{i + 1} | Pred: {pred_short} ({confidence:.2f}) | "
                f"GT: {actual_short} | IoU: {iou:.2f} | {status}"
            )
            cv2.putText(header, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

            row_images = np.hstack([img_vis, img_with_both, condition_vis])
            row = np.vstack([header, row_images])
            rows.append(row)

            print(
                f"  Sample {i + 1}: Pred={pred_label}, GT={actual_label}, IoU={iou:.2f}, {status}"
            )

    width = rows[0].shape[1]
    title = np.zeros((50, width, 3), dtype=np.uint8)
    cv2.putText(
        title,
        "YOLO Detection Model - Test Set Evaluation",
        (10, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
    )

    col_header = np.zeros((25, width, 3), dtype=np.uint8)
    col_w = width // 3
    cv2.putText(
        col_header,
        "Original Image",
        (col_w // 2 - 50, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (180, 180, 180),
        1,
    )
    cv2.putText(
        col_header,
        "Pred (Blue) vs GT (Green)",
        (col_w + col_w // 2 - 80, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (180, 180, 180),
        1,
    )
    cv2.putText(
        col_header,
        "Ground Truth",
        (col_w * 2 + col_w // 2 - 50, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (180, 180, 180),
        1,
    )

    accuracy = correct / len(samples)
    avg_iou = total_iou / len(samples)
    metrics = {"accuracy": accuracy, "avg_iou": avg_iou, "correct": correct, "total": len(samples)}

    summary = np.zeros((40, width, 3), dtype=np.uint8)
    summary_text = (
        f"RESULTS: Accuracy={accuracy:.1%} ({correct}/{len(samples)}) | Avg IoU={avg_iou:.2f}"
    )
    cv2.putText(summary, summary_text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    grid = np.vstack([title, col_header, *rows, summary])
    return grid, metrics


def main() -> None:  # pylint: disable=too-many-locals,too-many-statements
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Evaluate trained YOLO detector on test samples")
    parser.add_argument(
        "--samples", "-n", type=int, default=10, help="Number of test samples to evaluate"
    )
    parser.add_argument(
        "--output", "-o", default="yolo_test_results.png", help="Output visualization path"
    )
    parser.add_argument(
        "--checkpoint",
        "-c",
        default="checkpoints/best_detector.pt",
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--subsets",
        type=str,
        default="digits,hate_slangs,hate_symbols",
        help="Dataset subsets to test on",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed for sample selection (None=random)"
    )
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print(" YOLO Detection Model - Test Evaluation")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint not found at {checkpoint_path}")
        print("Please train the model first:")
        print("  python scripts/train_yolo_detection.py")
        sys.exit(1)

    print(f"\nLoading model from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    num_classes = checkpoint.get("num_classes", 10)
    label_to_idx = checkpoint.get("label_to_idx", {str(i): i for i in range(10)})
    idx_to_label = checkpoint.get("idx_to_label", {i: str(i) for i in range(10)})

    model = YOLODetector(num_classes=num_classes, pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"  Checkpoint accuracy: {checkpoint.get('accuracy', 'N/A')}")
    print(f"  Checkpoint IoU: {checkpoint.get('iou', 'N/A')}")
    print(f"  Number of classes: {num_classes}")

    subsets = [s.strip() for s in args.subsets.split(",")]
    seed = args.seed if args.seed is not None else random.randint(0, 10000)
    print(f"\nLoading test samples (seed={seed})...")
    samples = load_test_samples(args.samples, subsets, seed)

    print("\nRunning evaluation...")
    grid, metrics = evaluate_and_visualize(model, samples, device, label_to_idx, idx_to_label)

    output_path = Path(args.output)
    cv2.imwrite(str(output_path), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    print(f"\nVisualization saved to: {output_path.absolute()}")

    print("\n" + "=" * 60)
    print(" EVALUATION SUMMARY")
    print("=" * 60)
    print(f"  Test samples: {metrics['total']}")
    print(f"  Correct predictions: {metrics['correct']}")
    print(f"  Accuracy: {metrics['accuracy']:.1%}")
    print(f"  Average IoU: {metrics['avg_iou']:.2f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
