"""
Train YOLO model with bounding box detection using transfer learning.

Extracts bounding boxes from condition images and trains a proper detection model
with a pretrained ResNet backbone.

Usage:
    python scripts/train_yolo_detection.py [--epochs N] [--batch-size N]
"""
# pylint: disable=no-member,wrong-import-position

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models  # type: ignore[import-untyped]
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from PIL import Image


def extract_bbox_from_condition(condition_path: str, subset: str) -> Tuple[int, int, int, int]:
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
        # Scale from 512x512 to 1024x1024 (main image size)
        return x * 2, y * 2, w * 2, h * 2
    return 0, 0, 512, 512  # Default to center if no contour found


class ResNetBackbone(nn.Module):
    """Pretrained ResNet backbone for feature extraction."""

    def __init__(self, pretrained: bool = True):
        """Initialize ResNet18 backbone with optional pretrained weights."""
        super().__init__()
        # Use smaller ResNet18 to reduce overfitting
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        self.features = nn.Sequential(*list(resnet.children())[:-2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.feature_dim = 512  # ResNet18 has 512 features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through backbone."""
        x = self.features(x)
        x = self.avgpool(x)
        return x.flatten(1)


class YOLODetector(nn.Module):
    """YOLO-style detector with pretrained backbone for digit detection."""

    def __init__(self, num_classes: int = 10, pretrained: bool = True):
        """Initialize detector with classification and bbox heads."""
        super().__init__()
        self.backbone = ResNetBackbone(pretrained=pretrained)

        # Classification head with stronger regularization
        self.classifier = nn.Sequential(
            nn.Linear(self.backbone.feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

        # Bounding box regression head
        self.bbox_head = nn.Sequential(
            nn.Linear(self.backbone.feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 4),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning class logits and bbox predictions."""
        features = self.backbone(x)
        class_logits = self.classifier(features)
        bbox = self.bbox_head(features)
        return class_logits, bbox


class HatefulIllusionDetectionDataset(Dataset):  # pylint: disable=too-many-instance-attributes
    """Dataset with bounding box annotations extracted from condition images."""

    def __init__(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        split: str = "train",
        input_size: Tuple[int, int] = (224, 224),
        indices: Optional[List[int]] = None,
        augment: bool = False,
        subsets: Optional[List[str]] = None,
        label_to_idx: Optional[Dict[str, int]] = None,
    ):
        self.input_size = input_size
        self.augment = augment and split == "train"
        self.samples: List[Dict] = []

        # Load specified subsets (default: all subsets)
        if subsets is None:
            subsets = ["digits", "hate_slangs", "hate_symbols"]

        for subset in subsets:
            ds = load_dataset("yiting/HatefulIllusion_Dataset", subset, split="train")
            for item in ds:
                self.samples.append({"subset": subset, **item})

        # Build label mapping from all unique messages
        if label_to_idx is None:
            unique_labels = sorted(set(str(s["message"]) for s in self.samples))
            self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        else:
            self.label_to_idx = label_to_idx

        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        self.num_classes = len(self.label_to_idx)

        total = len(self.samples)
        if indices is not None:
            self.indices = indices
        elif split == "val":
            val_start = int(total * 0.8)
            self.indices = list(range(val_start, total))
        else:
            val_start = int(total * 0.8)
            self.indices = list(range(val_start))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict:  # pylint: disable=too-many-locals
        real_idx = self.indices[idx]
        item = self.samples[real_idx]
        subset = item["subset"]

        # Load main image
        image_path = item["image"]
        local_path = hf_hub_download(
            repo_id="yiting/HatefulIllusion_Dataset",
            filename=f"{subset}/{image_path}",
            repo_type="dataset",
        )
        pil_image = Image.open(local_path).convert("RGB")
        orig_w, orig_h = pil_image.size

        # Extract bbox from condition image
        x, y, w, h = extract_bbox_from_condition(item["condition_image"], subset)

        # Normalize bbox to [0, 1]
        bbox_norm = torch.tensor([
            (x + w / 2) / orig_w,  # center_x
            (y + h / 2) / orig_h,  # center_y
            w / orig_w,            # width
            h / orig_h,            # height
        ], dtype=torch.float32)

        # Resize image
        img_resized = pil_image.resize(self.input_size, Image.Resampling.BILINEAR)
        img_np = np.array(img_resized)

        # Data augmentation (stronger for better generalization)
        if self.augment:
            # Random horizontal flip
            if np.random.random() > 0.5:
                img_np = np.fliplr(img_np).copy()
                bbox_norm[0] = 1.0 - bbox_norm[0]  # Flip center_x

            # Random brightness/contrast
            if np.random.random() > 0.5:
                factor = np.random.uniform(0.7, 1.3)
                img_np = np.clip(img_np * factor, 0, 255).astype(np.uint8)

            # Color jitter (hue/saturation shift)
            if np.random.random() > 0.5:
                hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV).astype(np.float32)
                hsv[:, :, 0] = (hsv[:, :, 0] + np.random.uniform(-10, 10)) % 180
                hsv[:, :, 1] = np.clip(hsv[:, :, 1] * np.random.uniform(0.8, 1.2), 0, 255)
                img_np = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

            # Add Gaussian noise
            if np.random.random() > 0.6:
                noise = np.random.normal(0, 15, img_np.shape).astype(np.int16)
                img_np = np.clip(img_np.astype(np.int16) + noise, 0, 255).astype(np.uint8)

            # Random rotation (small angle)
            if np.random.random() > 0.7:
                angle = np.random.uniform(-15, 15)
                h, w = img_np.shape[:2]
                matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
                img_np = cv2.warpAffine(img_np, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)

            # Random blur
            if np.random.random() > 0.8:
                ksize = np.random.choice([3, 5])
                img_np = cv2.GaussianBlur(img_np, (ksize, ksize), 0)

        # Normalize for pretrained model
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_norm = (img_np / 255.0 - mean) / std

        tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).float()
        label_str = str(item["message"])
        label_idx = self.label_to_idx[label_str]

        return {
            "image": tensor,
            "label": torch.tensor(label_idx, dtype=torch.long),
            "bbox": bbox_norm,
            "visibility": item["visibility"],
            "label_str": label_str,
        }


def train_epoch(  # pylint: disable=too-many-locals
    model: YOLODetector,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float, float]:
    """Train for one epoch, return losses."""
    model.train()
    total_cls_loss = 0.0
    total_bbox_loss = 0.0
    num_batches = 0

    cls_criterion = nn.CrossEntropyLoss()
    bbox_criterion = nn.SmoothL1Loss()

    for batch in dataloader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        bboxes = batch["bbox"].to(device)

        optimizer.zero_grad()
        class_logits, bbox_pred = model(images)

        cls_loss = cls_criterion(class_logits, labels)
        bbox_loss = bbox_criterion(bbox_pred, bboxes)

        # Combined loss (classification weighted higher)
        loss = 2.0 * cls_loss + bbox_loss
        loss.backward()
        optimizer.step()

        total_cls_loss += cls_loss.item()
        total_bbox_loss += bbox_loss.item()
        num_batches += 1

    return (
        total_cls_loss / max(1, num_batches),
        total_bbox_loss / max(1, num_batches),
        (total_cls_loss + total_bbox_loss) / max(1, num_batches),
    )


def validate(  # pylint: disable=too-many-locals
    model: YOLODetector,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[float, float, float, float]:
    """Validate and return metrics."""
    model.eval()
    correct = 0
    total = 0
    total_cls_loss = 0.0
    total_bbox_loss = 0.0
    total_iou = 0.0
    num_batches = 0

    cls_criterion = nn.CrossEntropyLoss()
    bbox_criterion = nn.SmoothL1Loss()

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            bboxes = batch["bbox"].to(device)

            class_logits, bbox_pred = model(images)

            cls_loss = cls_criterion(class_logits, labels)
            bbox_loss = bbox_criterion(bbox_pred, bboxes)

            total_cls_loss += cls_loss.item()
            total_bbox_loss += bbox_loss.item()

            _, predicted = class_logits.max(dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            # Calculate IoU
            iou = calculate_iou(bbox_pred, bboxes)
            total_iou += iou.mean().item()
            num_batches += 1

    accuracy = correct / max(1, total)
    avg_iou = total_iou / max(1, num_batches)

    return (
        total_cls_loss / max(1, num_batches),
        total_bbox_loss / max(1, num_batches),
        accuracy,
        avg_iou,
    )


def calculate_iou(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:  # pylint: disable=too-many-locals
    """Calculate IoU between predicted and target bboxes (center format)."""
    # Convert from center format to corner format
    pred_x1 = pred[:, 0] - pred[:, 2] / 2
    pred_y1 = pred[:, 1] - pred[:, 3] / 2
    pred_x2 = pred[:, 0] + pred[:, 2] / 2
    pred_y2 = pred[:, 1] + pred[:, 3] / 2

    target_x1 = target[:, 0] - target[:, 2] / 2
    target_y1 = target[:, 1] - target[:, 3] / 2
    target_x2 = target[:, 0] + target[:, 2] / 2
    target_y2 = target[:, 1] + target[:, 3] / 2

    # Intersection
    inter_x1 = torch.max(pred_x1, target_x1)
    inter_y1 = torch.max(pred_y1, target_y1)
    inter_x2 = torch.min(pred_x2, target_x2)
    inter_y2 = torch.min(pred_y2, target_y2)

    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)

    # Union
    pred_area = pred[:, 2] * pred[:, 3]
    target_area = target[:, 2] * target[:, 3]
    union_area = pred_area + target_area - inter_area

    return inter_area / (union_area + 1e-6)


def main() -> None:  # pylint: disable=too-many-locals,too-many-statements
    """Main training entry point."""
    parser = argparse.ArgumentParser(description="Train YOLO detector with transfer learning")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--subsets", type=str, default="digits,hate_slangs,hate_symbols",
                        help="Dataset subsets (comma-separated): digits,hate_slangs,hate_symbols")
    args = parser.parse_args()

    # Parse subsets
    subsets = [s.strip() for s in args.subsets.split(",")]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'=' * 60}")
    print(" YOLO Detection Training (Transfer Learning)")
    print(f"{'=' * 60}")
    print(f"Device: {device}")

    print("\nLoading datasets...")
    train_dataset = HatefulIllusionDetectionDataset(split="train", augment=True, subsets=subsets)
    # Share label mapping with validation dataset
    val_dataset = HatefulIllusionDetectionDataset(
        split="val", augment=False, subsets=subsets, label_to_idx=train_dataset.label_to_idx
    )
    num_classes = train_dataset.num_classes
    print(f"  Subsets: {subsets}")
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")
    print(f"  Number of classes: {num_classes}")

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    print("\nInitializing model with pretrained ResNet18 backbone...")
    model = YOLODetector(num_classes=num_classes, pretrained=True).to(device)

    # Freeze backbone initially for first few epochs
    for param in model.backbone.parameters():
        param.requires_grad = False

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr * 10,  # Higher LR for heads only
        weight_decay=0.01
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    print("\nTraining configuration:")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Early stopping patience: {args.patience}")

    best_accuracy = 0.0
    best_iou = 0.0
    epochs_without_improvement = 0
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)

    print(f"\n{'-' * 60}")
    print("Starting training...")
    print(f"{'-' * 60}")

    for epoch in range(args.epochs):
        # Unfreeze backbone after 5 epochs
        if epoch == 5:
            print("Unfreezing backbone...")
            for param in model.backbone.parameters():
                param.requires_grad = True
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.epochs - epoch
            )

        cls_loss, bbox_loss, _ = train_epoch(model, train_loader, optimizer, device)
        val_cls, val_bbox, accuracy, iou = validate(model, val_loader, device)
        scheduler.step()

        print(f"Epoch {epoch:3d} | "
              f"Train: cls={cls_loss:.4f} bbox={bbox_loss:.4f} | "
              f"Val: cls={val_cls:.4f} bbox={val_bbox:.4f} | "
              f"Acc={accuracy:.4f} IoU={iou:.4f}")

        # Save best model based on accuracy
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_iou = iou
            epochs_without_improvement = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "accuracy": accuracy,
                "iou": iou,
                "epoch": epoch,
                "num_classes": num_classes,
                "label_to_idx": train_dataset.label_to_idx,
                "idx_to_label": train_dataset.idx_to_label,
            }, checkpoint_dir / "best_detector.pt")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= args.patience:
            print(f"Early stopping at epoch {epoch}")
            break

    print(f"\n{'=' * 60}")
    print(" Training Complete")
    print(f"{'=' * 60}")
    print(f"Best accuracy: {best_accuracy:.4f}")
    print(f"Best IoU: {best_iou:.4f}")
    print(f"Model saved to: {checkpoint_dir / 'best_detector.pt'}")


if __name__ == "__main__":
    main()
