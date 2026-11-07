import argparse
import os

import numpy as np
from PIL import Image


def generate_synthetic_dataset(output_dir, num_classes=2, images_per_class=20):
    """
    Generate synthetic color-dominant images in class subdirectories.
    Class 0: Red-dominant random noise.
    Class 1: Blue-dominant random noise.
    Others: Green-dominant or random noise.
    This ensures that the feature extractor and classifiers can easily learn the task.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"Generating synthetic classification dataset at: {output_dir}")
    print(f"  Classes: {num_classes}, Images per class: {images_per_class}")

    np.random.seed(42)

    for class_idx in range(num_classes):
        class_name = f"class_{class_idx}"
        class_path = os.path.join(output_dir, class_name)
        os.makedirs(class_path, exist_ok=True)

        for img_idx in range(images_per_class):
            # Create a 224x224 random color image
            if class_idx == 0:
                # Red dominant
                r = np.random.randint(150, 256, (224, 224), dtype=np.uint8)
                g = np.random.randint(0, 100, (224, 224), dtype=np.uint8)
                b = np.random.randint(0, 100, (224, 224), dtype=np.uint8)
            elif class_idx == 1:
                # Blue dominant
                r = np.random.randint(0, 100, (224, 224), dtype=np.uint8)
                g = np.random.randint(0, 100, (224, 224), dtype=np.uint8)
                b = np.random.randint(150, 256, (224, 224), dtype=np.uint8)
            else:
                # Green/other dominant
                r = np.random.randint(0, 100, (224, 224), dtype=np.uint8)
                g = np.random.randint(150, 256, (224, 224), dtype=np.uint8)
                b = np.random.randint(0, 100, (224, 224), dtype=np.uint8)

            img_arr = np.stack([r, g, b], axis=2)
            img = Image.fromarray(img_arr)

            img_name = f"img_{img_idx:03d}.png"
            img.save(os.path.join(class_path, img_name))

    print("Synthetic dataset generation complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic classification images")
    parser.add_argument(
        "--output",
        type=str,
        default="data_files/synthetic_class_data",
        help="Output directory path",
    )
    parser.add_argument("--classes", type=int, default=2, help="Number of classes to generate")
    parser.add_argument("--images", type=int, default=20, help="Images per class")

    args = parser.parse_args()
    generate_synthetic_dataset(args.output, args.classes, args.images)
