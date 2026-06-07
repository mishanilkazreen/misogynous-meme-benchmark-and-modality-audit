"""Fine-tune YOLO models on the HatefulIllusion dataset."""

from __future__ import annotations

import argparse
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
from PIL import Image as PilImage
import yaml

from models.yolo.wrapper import UltralyticsYOLO
from utils.dataset import DatasetManager


def prepare_yolo_training_data(samples: list[dict[str, Any]], output_dir: Path) -> str:
    """Prepare HatefulIllusion samples in YOLO training format."""
    images_dir = output_dir / "images"
    labels_dir = output_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    # HatefulIllusion is image-level only; treat every image as a single full-image box.
    class_list = ["embedded_hateful_content"]

    for sample in samples:
        image_id = sample["image_id"]
        label_path = labels_dir / f"{image_id}.txt"

        target_image = images_dir / f"{image_id}.png"
        if not target_image.exists():
            img = sample["image"]
            if hasattr(img, "detach"):
                img_np = img.permute(1, 2, 0).detach().cpu().numpy()
                img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
            else:
                img_np = np.asarray(img)
            PilImage.fromarray(img_np).save(target_image)

        # Full-image proxy box in YOLO normalised format: class cx cy w h
        label_path.write_text("0 0.5 0.5 1.0 1.0\n", encoding="utf-8")

    data_yaml = {
        "path": str(output_dir),
        "train": "images",
        "val": "images",
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
    """Fine-tune a YOLO model on the prepared dataset."""
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

    if output_dir:
        return str(output_dir / f"train_{Path(checkpoint).stem}" / "weights" / "best.pt")
    return f"runs/train/train_{Path(checkpoint).stem}/weights/best.pt"


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune YOLO models on HatefulIllusion")
    parser.add_argument("--model", required=True, help="YOLO checkpoint to fine-tune")
    parser.add_argument(
        "--subset",
        default="digits",
        choices=["digits", "hate_slangs", "hate_symbols"],
        help="Dataset subset",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    import torch

    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    parser.add_argument("--device", default=default_device, help="Device to train on")
    default_output_dir = (
        Path(__file__).resolve().parents[2] / "runs" / "detect" / "results" / "trained_models"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir,
        help="Output directory for trained weights (default matches benchmark expected path)",
    )

    args = parser.parse_args()

    manager = DatasetManager()
    dataset = manager.load_dataset(split="train", subset=args.subset)
    samples = [dataset[i] for i in range(len(dataset))]

    with tempfile.TemporaryDirectory(prefix="yolo_train_") as temp_dir:
        temp_path = Path(temp_dir)
        data_yaml = prepare_yolo_training_data(samples, temp_path)

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
