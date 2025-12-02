#!/bin/bash
# Script to create the data directory structure

echo "Creating data directory structure..."

# Create raw data directories
mkdir -p raw/train
mkdir -p raw/val
mkdir -p raw/test

# Create annotation directories
mkdir -p annotations/yolo/train
mkdir -p annotations/yolo/val
mkdir -p annotations/yolo/test
mkdir -p annotations/vlm

# Create processed data directories
mkdir -p processed/blur_equalized
mkdir -p processed/augmented

echo "✓ Directory structure created successfully!"
echo ""
echo "Next steps:"
echo "1. Download your dataset (see DATASETS.md for recommendations)"
echo "2. Place images in raw/train/, raw/val/, raw/test/"
echo "3. Add annotations in annotations/yolo/ and annotations/vlm/"
echo "4. Run validation: python -c 'from utils.dataset import DatasetManager; ...'"
