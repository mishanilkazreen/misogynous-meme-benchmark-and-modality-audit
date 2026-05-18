"""
Train YOLO models on HatefulIllusion dataset.

This script fine-tunes pre-trained YOLO models on the HatefulIllusion
dataset for content moderation tasks.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import tempfile
from typing import Any

import yaml

from models.yolo.wrapper import UltralyticsYOLO
from utils.dataset import DatasetManager


def prepare_yolo_training_data(samples: list[dict[str, Any]], output_dir: Path) -> str:
    """
    Prepare HatefulIllusion samples in YOLO training format.

    Creates:
    - images/ directory with symlinks to images
    - labels/ directory with YOLO format annotations
    - data.yaml with dataset configuration
    """
    images_dir = output_dir / "images"
    labels_dir = output_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    # Get unique classes from samples
    classes = set()
    for sample in samples:
        if "objects" in sample:
            for obj in sample["objects"]:
                classes.add(obj["category"])
    class_list = sorted(classes)
    class_to_id = {cls: i for i, cls in enumerate(class_list)}

    # Write labels
    for sample in samples:
        image_id = sample["image_id"]
        image_path = Path(sample["image_path"])
        label_path = labels_dir / f"{image_id}.txt"

        # Create symlink to image
        target_image = images_dir / f"{image_id}{image_path.suffix}"
        if not target_image.exists():
            target_image.symlink_to(image_path)

        # Write YOLO annotations
        with label_path.open("w") as f:
            if "objects" in sample:
                for obj in sample["objects"]:
                    category = obj["category"]
                    bbox = obj["bbox"]  # Assuming normalized [x_min, y_min, x_max, y_max]
                    class_id = class_to_id[category]
                    # Convert to YOLO format: class_id x_center y_center width height (normalized)
                    x_center = (bbox[0] + bbox[2]) / 2
                    y_center = (bbox[1] + bbox[3]) / 2
                    width = bbox[2] - bbox[0]
                    height = bbox[3] - bbox[1]
                    f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

    # Create data.yaml
    data_yaml = {
        "path": str(output_dir),
        "train": "images",
        "val": "images",  # Using same for now, can split later
        "names": class_list,
    }

    data_yaml_path = output_dir / "data.yaml"
    with data_yaml_path.open("w") as f:
        yaml.dump(data_yaml, f)

    return str(data_yaml_path)


def train_yolo_model(
    checkpoint: str,
    data_yaml: str,
    epochs: int = 50,
    batch_size: int = 16,
    device: str = "cpu",
    output_dir: Path | None = None,
) -> str:
    """
    Fine-tune a YOLO model on the prepared dataset.
    """
    model = UltralyticsYOLO(checkpoint=checkpoint, device=device, verbose=True)

    model.model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch_size,
        device=device,
        project=str(output_dir) if output_dir else None,
        name=f"train_{Path(checkpoint).stem}",
        save=True,
    )

    # Return path to best weights
    if output_dir:
        return str(output_dir / f"train_{Path(checkpoint).stem}" / "weights" / "best.pt")
    else:
        # Default Ultralytics save location
        return f"runs/train/train_{Path(checkpoint).stem}/weights/best.pt"


def main():
    parser = argparse.ArgumentParser(description="Train YOLO models on HatefulIllusion")
    parser.add_argument("--model", required=True, help="YOLO checkpoint to fine-tune")
    parser.add_argument(
        "--subset",
        default="digits",
        choices=["digits", "hate_slangs", "hate_symbols"],
        help="Dataset subset",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--device", default="cpu", help="Device to train on")
    parser.add_argument("--output-dir", type=Path, help="Output directory for results")

    args = parser.parse_args()

    # Load dataset
    manager = DatasetManager()
    dataset = manager.load_dataset("HatefulIllusion", subset=args.subset)
    samples = list(dataset)

    # Prepare data
    with tempfile.TemporaryDirectory(prefix="yolo_train_") as temp_dir:
        temp_path = Path(temp_dir)
        data_yaml = prepare_yolo_training_data(samples, temp_path)

        # Train model
        best_weights = train_yolo_model(
            checkpoint=args.model,
            data_yaml=data_yaml,
            epochs=args.epochs,
            batch_size=args.batch_size,
            device=args.device,
            output_dir=args.output_dir,
        )

        print(f"Training completed. Best weights saved to: {best_weights}")


if __name__ == "__main__":
    main()
