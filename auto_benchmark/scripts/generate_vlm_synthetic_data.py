import argparse
import os

import numpy as np
import pandas as pd
from PIL import Image


def generate_vlm_synthetic_dataset(output_dir):
    """
    Generate synthetic solid color images and a mapping labels.csv.
    Generates 2 red, 2 blue, 2 green images.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"Generating synthetic VLM dataset at: {output_dir}")

    labels_list = []
    colors_info = [
        ("red_01.png", "red", [255, 0, 0]),
        ("red_02.png", "red", [200, 20, 20]),
        ("blue_01.png", "blue", [0, 0, 255]),
        ("blue_02.png", "blue", [10, 10, 200]),
        ("green_01.png", "green", [0, 255, 0]),
        ("green_02.png", "green", [30, 180, 30]),
    ]

    for filename, color_name, rgb in colors_info:
        img_arr = np.zeros((224, 224, 3), dtype=np.uint8)
        img_arr[:, :] = rgb
        img = Image.fromarray(img_arr)
        img_path = os.path.join(output_dir, filename)
        img.save(img_path)

        labels_list.append({"filename": filename, "label": color_name})

    df_labels = pd.DataFrame(labels_list)
    csv_path = os.path.join(output_dir, "labels.csv")
    df_labels.to_csv(csv_path, index=False)
    print(f"Synthetic dataset complete! Saved labels sheet -> {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic VLM classification data")
    parser.add_argument(
        "--output", type=str, default="data_files/vlm_synthetic_data", help="Output directory path"
    )

    args = parser.parse_args()
    generate_vlm_synthetic_dataset(args.output)
