"""
Manual visualization script for image transformations.

This script loads a random image from the HatefulIllusion dataset and applies
all available transformations, saving the results as a grid visualization.

Usage:
    python scripts/visualize_transformations.py [--output OUTPUT_PATH] [--index IMAGE_INDEX]

Example:
    python scripts/visualize_transformations.py --output transformations.png --index 42
"""
# pylint: disable=no-member

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.preprocessing import PreprocessingPipeline


def download_image_from_huggingface(image_path: str) -> np.ndarray:
    """
    Download an image from the HatefulIllusion HuggingFace dataset.

    Args:
        image_path: Relative path from dataset (e.g., 'images/0.png')

    Returns:
        Image as numpy array (H, W, C) in RGB format
    """
    from huggingface_hub import hf_hub_download

    # The dataset stores images under digits/ folder
    filename = f"digits/{image_path}"
    print(f"Downloading: {filename}")

    local_path = hf_hub_download(
        repo_id="yiting/HatefulIllusion_Dataset",
        filename=filename,
        repo_type="dataset"
    )

    # Load image
    img = Image.open(local_path)
    img = img.convert("RGB")

    return np.array(img)


def load_dataset_image(index: int = None):
    """
    Load an image from the HatefulIllusion dataset.

    Downloads the actual image from the GitHub repository.

    Args:
        index: Specific image index, or None for random

    Returns:
        Tuple of (image as numpy array, metadata dict)
    """
    from datasets import load_dataset

    print("Loading HatefulIllusion dataset metadata...")
    ds = load_dataset("yiting/HatefulIllusion_Dataset", "digits", split="train")

    if index is None:
        index = np.random.randint(0, len(ds))

    item = ds[index]

    # Download actual image from GitHub
    image_path = item["image"]
    print(f"Selected sample {index}: {image_path}")

    image = download_image_from_huggingface(image_path)

    metadata = {
        "index": index,
        "message": item["message"],
        "visibility": item["visibility"],
        "prompt": item["prompt"],
    }

    print(f"Loaded image {index}:")
    print(f"  Hidden message: {metadata['message']}")
    print(f"  Visibility: {metadata['visibility']}")
    print(f"  Prompt: {metadata['prompt'][:50]}...")

    return image, metadata


def create_visualization_grid(results: dict, metadata: dict) -> np.ndarray:
    """
    Create a grid visualization of all transformations.

    Args:
        results: Dictionary of transformation name -> image
        metadata: Image metadata

    Returns:
        Grid image as numpy array
    """
    # Get all images and names
    names = ["original"] + PreprocessingPipeline.TRANSFORMATIONS
    images = [results[name] for name in names]

    # Calculate grid dimensions (4 columns)
    n_cols = 4
    n_rows = (len(images) + n_cols - 1) // n_cols

    # Resize all images to same size for grid
    target_size = (256, 256)
    resized = []
    for img in images:
        if img.shape[:2] != target_size:
            img = cv2.resize(img, target_size)
        resized.append(img)

    # Create grid
    rows = []
    for i in range(n_rows):
        row_images = []
        for j in range(n_cols):
            idx = i * n_cols + j
            if idx < len(resized):
                # Add label to image
                img = resized[idx].copy()
                label = names[idx].replace("_", " ").title()
                cv2.putText(
                    img, label, (5, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 2
                )
                cv2.putText(
                    img, label, (5, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 0), 1
                )
                row_images.append(img)
            else:
                # Pad with black
                row_images.append(
                    np.zeros((target_size[0], target_size[1], 3), dtype=np.uint8)
                )
        rows.append(np.hstack(row_images))

    grid = np.vstack(rows)

    # Add title bar
    title_height = 60
    title_bar = np.zeros((title_height, grid.shape[1], 3), dtype=np.uint8)
    title = f"Message: '{metadata['message']}', Visibility: {metadata['visibility']}"
    cv2.putText(
        title_bar, title, (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
        0.7, (255, 255, 255), 2
    )

    return np.vstack([title_bar, grid])


def main():
    """Main entry point for visualization script."""
    parser = argparse.ArgumentParser(
        description="Visualize image transformations on HatefulIllusion dataset"
    )
    parser.add_argument(
        "--output", "-o",
        default="transformation_visualization.png",
        help="Output file path (default: transformation_visualization.png)"
    )
    parser.add_argument(
        "--index", "-i",
        type=int,
        default=None,
        help="Image index to use (default: random)"
    )
    args = parser.parse_args()

    # Load image
    image, metadata = load_dataset_image(args.index)

    # Create pipeline and apply all transformations
    pipeline = PreprocessingPipeline()
    print("\nApplying transformations...")
    results = pipeline.apply_all_transformations(image)

    # Create visualization
    print("Creating visualization grid...")
    grid = create_visualization_grid(results, metadata)

    # Save result (convert RGB to BGR for OpenCV)
    output_path = Path(args.output)
    cv2.imwrite(str(output_path), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    print(f"\nVisualization saved to: {output_path.absolute()}")

    # Print transformation list
    print("\nTransformations applied:")
    for i, name in enumerate(["original"] + PreprocessingPipeline.TRANSFORMATIONS, 1):
        print(f"  {i:2d}. {name}")


if __name__ == "__main__":
    main()
