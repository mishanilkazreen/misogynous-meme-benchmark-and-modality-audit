"""Visualize YOLO detections on a custom image folder.

Runs inference on up to --max-images images and produces a grid PNG
showing each image with detected bounding boxes overlaid in green.
Images where nothing was detected have a red border.

Usage:
    # Explicit weights file
    uv run python scripts/visualize_detections.py \\
        --dataset /path/to/ww2 \\
        --weights results/trained_models/train_yolov8n_hate_symbols/weights/best.pt

    # Pretrained checkpoint
    uv run python scripts/visualize_detections.py \\
        --dataset /path/to/ww2 --model yolov8n.pt --mode pretrained

    # Trained weights (picks hate_symbols by default)
    uv run python scripts/visualize_detections.py \\
        --dataset /path/to/ww2 --model yolov8n.pt --mode trained --trained-on hate_symbols
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

from models.yolo.wrapper import UltralyticsYOLO

MODEL_CHECKPOINTS = ["yolov8n.pt", "yolov10n.pt", "yolo11n.pt", "yolo12n.pt", "yolo26n.pt"]
SUBSET_NAMES = ["digits", "hate_slangs", "hate_symbols"]
SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tiff", ".tif"}
DEFAULT_TRAINED_RESULTS_DIR = Path(__file__).resolve().parents[1] / "results" / "trained_models"
FIGURES_DIR = Path(__file__).resolve().parents[1] / "results" / "figures"

DETECTION_COLOR = "#00C853"  # green
NO_DETECTION_COLOR = "#D50000"  # red border when nothing detected
BOX_LINE_WIDTH = 3
FONT_SIZE = 9


def _resolve_weights(
    model: str,
    mode: str,
    trained_on: str,
    weights: str | None,
    trained_dir: Path,
) -> str:
    if weights:
        return weights
    if mode == "pretrained":
        return model
    weights_path = trained_dir / f"train_{Path(model).stem}_{trained_on}" / "weights" / "best.pt"
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Trained weights not found: {weights_path}\n"
            f"  → Run: uv run python scripts/benchmark_yolo.py --mode trained --model {model} --subset {trained_on}\n"
            f"  → Or use --mode pretrained"
        )
    return str(weights_path)


def _load_images(folder: Path, max_images: int) -> list[tuple[Path, np.ndarray]]:
    paths = sorted(p for p in folder.iterdir() if p.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS)
    paths = paths[:max_images]
    if not paths:
        raise ValueError(f"No supported images found in {folder}")
    result = []
    for p in paths:
        pil = Image.open(p).convert("RGB")
        result.append((p, np.array(pil)))
    return result


def _draw_boxes(
    image_rgb: np.ndarray,
    boxes_xyxy: np.ndarray,
    confs: np.ndarray,
    class_ids: np.ndarray,
    class_names: dict[int, str],
) -> np.ndarray:
    """Draw green bounding boxes with class labels on a copy of the image."""
    pil = Image.fromarray(image_rgb)
    draw = ImageDraw.Draw(pil)
    for (x1, y1, x2, y2), conf, cls_id in zip(boxes_xyxy, confs, class_ids, strict=False):
        draw.rectangle([x1, y1, x2, y2], outline=DETECTION_COLOR, width=BOX_LINE_WIDTH)
        class_name = class_names.get(int(cls_id), str(int(cls_id)))
        label = f"{class_name} {conf:.2f}"
        draw.text((x1 + 4, y1 + 2), label, fill=DETECTION_COLOR)
    return np.array(pil)


def visualize(
    folder: Path,
    checkpoint: str,
    max_images: int,
    conf_threshold: float,
    cols: int,
    output_path: Path,
) -> None:
    print(f"Loading model: {checkpoint}")
    model = UltralyticsYOLO(checkpoint=checkpoint, device="cpu", verbose=False)

    images = _load_images(folder, max_images)
    print(f"Running inference on {len(images)} images (conf ≥ {conf_threshold})…")

    rows = (len(images) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3.5))
    fig.set_facecolor("#1a1a1a")

    # Flatten axes grid for easy indexing
    axes_flat: list[plt.Axes] = (
        np.array(axes).flatten().tolist() if rows > 1 or cols > 1 else [axes]
    )

    for i, (img_path, img_rgb) in enumerate(images):
        results = model.predict(source=img_rgb, conf=conf_threshold, save=False, verbose=False)
        result = results[0]

        n_detections = 0
        detected_labels: list[str] = []
        annotated = img_rgb.copy()

        if result.boxes is not None and len(result.boxes) > 0:
            xyxy = result.boxes.xyxy
            xyxy = xyxy.cpu().numpy() if hasattr(xyxy, "cpu") else np.asarray(xyxy)
            conf = result.boxes.conf
            conf = conf.cpu().numpy() if hasattr(conf, "cpu") else np.asarray(conf)
            cls_ids = result.boxes.cls
            cls_ids = cls_ids.cpu().numpy() if hasattr(cls_ids, "cpu") else np.asarray(cls_ids)
            class_names: dict[int, str] = result.names if result.names else {}
            n_detections = len(xyxy)
            annotated = _draw_boxes(img_rgb, xyxy, conf, cls_ids, class_names)
            # Deduplicated list preserving order of first occurrence
            seen: set[str] = set()
            for cls_id in cls_ids:
                name = class_names.get(int(cls_id), str(int(cls_id)))
                if name not in seen:
                    detected_labels.append(name)
                    seen.add(name)

        labels_str = ", ".join(detected_labels) if detected_labels else "—"
        ax = axes_flat[i]
        ax.imshow(annotated)
        ax.set_title(
            f"{img_path.name}\n{n_detections} detection(s) flag(s): {labels_str}",
            fontsize=FONT_SIZE,
            color="white",
            pad=3,
        )
        ax.axis("off")

        # Red border when nothing detected
        border_color = NO_DETECTION_COLOR if n_detections == 0 else DETECTION_COLOR
        for spine in ax.spines.values():
            spine.set_edgecolor(border_color)
            spine.set_linewidth(2)
            spine.set_visible(True)

    # Hide unused subplot cells
    for j in range(len(images), len(axes_flat)):
        axes_flat[j].set_visible(False)

    model_label = Path(checkpoint).stem
    legend_handles = [
        mpatches.Patch(color=DETECTION_COLOR, label="Detection"),
        mpatches.Patch(color=NO_DETECTION_COLOR, label="No detection"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=2,
        fontsize=10,
        facecolor="#1a1a1a",
        labelcolor="white",
        bbox_to_anchor=(0.5, 0.01),
    )
    fig.suptitle(
        f"Detections — model: {model_label}  |  dataset: {folder.name}  |  conf ≥ {conf_threshold}",
        fontsize=11,
        color="white",
        y=1.01,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Path to the image folder to evaluate.",
    )
    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        choices=MODEL_CHECKPOINTS,
        help="YOLO checkpoint name (default: yolov8n.pt)",
    )
    parser.add_argument(
        "--mode",
        default="trained",
        choices=["pretrained", "trained"],
        help="pretrained = COCO weights, trained = HatefulIllusion fine-tuned (default: trained)",
    )
    parser.add_argument(
        "--trained-on",
        default="hate_symbols",
        choices=SUBSET_NAMES,
        dest="trained_on",
        help="Which HatefulIllusion subset the trained weights come from (default: hate_symbols)",
    )
    parser.add_argument(
        "--weights",
        default=None,
        help="Explicit path to a .pt weights file (overrides --model/--mode/--trained-on).",
    )
    parser.add_argument(
        "--trained-dir",
        type=Path,
        default=DEFAULT_TRAINED_RESULTS_DIR,
        dest="trained_dir",
        help="Base directory of trained model folders.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold for detections (default: 0.25)",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=16,
        dest="max_images",
        help="Maximum number of images to display in the grid (default: 16)",
    )
    parser.add_argument(
        "--cols",
        type=int,
        default=4,
        help="Number of columns in the image grid (default: 4)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output PNG path. Default: "
            "results/figures/yolo_detections_{dataset_name}_{model_stem}.png"
        ),
    )
    args = parser.parse_args()

    checkpoint = _resolve_weights(
        model=args.model,
        mode=args.mode,
        trained_on=args.trained_on,
        weights=args.weights,
        trained_dir=args.trained_dir,
    )

    dataset_name = args.dataset.resolve().name
    model_stem = Path(checkpoint).stem
    output_path = args.output or FIGURES_DIR / f"yolo_detections_{dataset_name}_{model_stem}.png"

    visualize(
        folder=args.dataset.resolve(),
        checkpoint=checkpoint,
        max_images=args.max_images,
        conf_threshold=args.conf,
        cols=args.cols,
        output_path=output_path,
    )


if __name__ == "__main__":
    main()
