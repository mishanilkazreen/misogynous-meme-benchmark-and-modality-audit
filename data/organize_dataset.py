#!/usr/bin/env python3
"""
Helper script to organize downloaded dataset into train/val/test splits.

Usage:
    python organize_dataset.py --source <path_to_downloaded_images> --split 0.7 0.15 0.15
"""
import argparse
import shutil
from pathlib import Path
import random
import json


def split_dataset(source_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Split images from source directory into train/val/test.
    
    Args:
        source_dir: Path to directory containing all images
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set
        test_ratio: Proportion for test set
        seed: Random seed for reproducibility
    """
    random.seed(seed)
    
    source_path = Path(source_dir)
    if not source_path.exists():
        raise ValueError(f"Source directory not found: {source_dir}")
    
    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    images = [f for f in source_path.iterdir() 
              if f.suffix.lower() in image_extensions]
    
    if not images:
        raise ValueError(f"No images found in {source_dir}")
    
    print(f"Found {len(images)} images")
    
    # Shuffle
    random.shuffle(images)
    
    # Calculate splits
    total = len(images)
    train_size = int(train_ratio * total)
    val_size = int(val_ratio * total)
    
    train_imgs = images[:train_size]
    val_imgs = images[train_size:train_size + val_size]
    test_imgs = images[train_size + val_size:]
    
    # Create directories
    base_dir = Path('raw')
    base_dir.mkdir(exist_ok=True)
    
    for split_name in ['train', 'val', 'test']:
        (base_dir / split_name).mkdir(exist_ok=True)
    
    # Copy files
    print("\nCopying files...")
    for img in train_imgs:
        shutil.copy2(img, base_dir / 'train' / img.name)
    print(f"✓ Copied {len(train_imgs)} images to train/")
    
    for img in val_imgs:
        shutil.copy2(img, base_dir / 'val' / img.name)
    print(f"✓ Copied {len(val_imgs)} images to val/")
    
    for img in test_imgs:
        shutil.copy2(img, base_dir / 'test' / img.name)
    print(f"✓ Copied {len(test_imgs)} images to test/")
    
    print(f"\n✓ Dataset split complete!")
    print(f"  Train: {len(train_imgs)} ({train_ratio*100:.0f}%)")
    print(f"  Val:   {len(val_imgs)} ({val_ratio*100:.0f}%)")
    print(f"  Test:  {len(test_imgs)} ({test_ratio*100:.0f}%)")
    
    return train_imgs, val_imgs, test_imgs


def create_sample_annotations(image_list, split_name):
    """
    Create sample VLM annotation file for a split.
    
    Args:
        image_list: List of image paths
        split_name: 'train', 'val', or 'test'
    """
    annotations = []
    
    for img_path in image_list:
        annotation = {
            "image_id": img_path.name,
            "image_path": f"raw/{split_name}/{img_path.name}",
            "has_hateful_content": False,  # TODO: Update with actual labels
            "message_type": "textual",      # TODO: Update with actual labels
            "visibility_level": "high",     # TODO: Update with actual labels
            "extracted_text": "",
            "annotator_agreement": 1.0
        }
        annotations.append(annotation)
    
    # Save to JSON
    output_path = Path('annotations/vlm') / f'{split_name}.json'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(annotations, f, indent=2)
    
    print(f"✓ Created sample annotations: {output_path}")
    print(f"  ⚠️  Remember to update with actual labels!")


def main():
    parser = argparse.ArgumentParser(
        description='Organize dataset into train/val/test splits'
    )
    parser.add_argument(
        '--source',
        required=True,
        help='Path to directory containing all images'
    )
    parser.add_argument(
        '--split',
        nargs=3,
        type=float,
        default=[0.7, 0.15, 0.15],
        help='Train/val/test split ratios (default: 0.7 0.15 0.15)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    parser.add_argument(
        '--create-sample-annotations',
        action='store_true',
        help='Create sample VLM annotation files (need manual updating)'
    )
    
    args = parser.parse_args()
    
    # Validate split ratios
    if abs(sum(args.split) - 1.0) > 0.01:
        raise ValueError("Split ratios must sum to 1.0")
    
    # Split dataset
    train_imgs, val_imgs, test_imgs = split_dataset(
        args.source,
        train_ratio=args.split[0],
        val_ratio=args.split[1],
        test_ratio=args.split[2],
        seed=args.seed
    )
    
    # Create sample annotations if requested
    if args.create_sample_annotations:
        print("\nCreating sample annotation files...")
        create_sample_annotations(train_imgs, 'train')
        create_sample_annotations(val_imgs, 'val')
        create_sample_annotations(test_imgs, 'test')


if __name__ == '__main__':
    main()
