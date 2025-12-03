"""
DatasetManager for loading and validating hateful content datasets.
Supports the MMHS150K dataset format.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


@dataclass
class Annotation:
    """Represents an annotation for a single image."""
    image_id: str
    labels: List[int]  # 3 annotator labels [0-5]
    labels_str: List[str]
    tweet_text: str
    ocr_text: Optional[str] = None
    bbox: Optional[Tuple[int, int, int, int]] = None  # For YOLO: (x, y, w, h)
    
    @property
    def majority_label(self) -> int:
        """Get majority vote label from annotators."""
        from collections import Counter
        return Counter(self.labels).most_common(1)[0][0]
    
    @property
    def is_hate(self) -> bool:
        """Check if majority label indicates hate content."""
        return self.majority_label != 0
    
    @property
    def message_type(self) -> str:
        """Classify as textual or symbolic based on label."""
        # Labels 1-2 (Racist, Sexist) are typically textual
        # Labels 3-5 (Homophobe, Religion, OtherHate) can be symbolic
        label = self.majority_label
        if label in [1, 2]:
            return "textual"
        elif label in [3, 4, 5]:
            return "symbolic"
        return "none"
    
    @property
    def visibility_level(self) -> str:
        """Determine visibility level (placeholder - requires image analysis)."""
        # Default to high visibility; actual determination requires image processing
        return "high"


class MMHS150KDataset(Dataset):
    """PyTorch Dataset for MMHS150K hateful content dataset."""
    
    def __init__(
        self,
        data_path: str,
        split: str = "train",
        transform=None,
        include_ocr: bool = True
    ):
        """
        Initialize the dataset.
        
        Args:
            data_path: Path to the MMHS150K dataset root
            split: One of 'train', 'val', 'test'
            transform: Optional torchvision transforms
            include_ocr: Whether to load OCR text
        """
        self.data_path = Path(data_path)
        self.split = split
        self.transform = transform
        self.include_ocr = include_ocr
        
        self.annotations: Dict[str, Annotation] = {}
        self.image_ids: List[str] = []
        
        self._load_dataset()

    def _load_dataset(self) -> None:
        """Load dataset from disk."""
        # Load ground truth annotations
        gt_path = self.data_path / "MMHS150K_GT.json"
        if not gt_path.exists():
            raise FileNotFoundError(f"Ground truth file not found: {gt_path}")
        
        with open(gt_path, "r", encoding="utf-8") as f:
            gt_data = json.load(f)
        
        # Load split IDs
        split_path = self.data_path / "splits" / f"{self.split}_ids.txt"
        if not split_path.exists():
            raise FileNotFoundError(f"Split file not found: {split_path}")
        
        with open(split_path, "r", encoding="utf-8") as f:
            split_ids = set(line.strip() for line in f if line.strip())
        
        # Load OCR text if available
        ocr_data = {}
        if self.include_ocr:
            ocr_dir = self.data_path / "img_txt"
            if ocr_dir.exists():
                for ocr_file in ocr_dir.glob("*.txt"):
                    img_id = ocr_file.stem
                    with open(ocr_file, "r", encoding="utf-8", errors="ignore") as f:
                        ocr_data[img_id] = f.read().strip()
        
        # Build annotations for this split
        for img_id, data in gt_data.items():
            if img_id not in split_ids:
                continue
            
            # Check if image exists
            img_path = self._get_image_path(img_id)
            if img_path is None:
                continue
            
            annotation = Annotation(
                image_id=img_id,
                labels=data.get("labels", [0, 0, 0]),
                labels_str=data.get("labels_str", ["NotHate", "NotHate", "NotHate"]),
                tweet_text=data.get("tweet_text", ""),
                ocr_text=ocr_data.get(img_id)
            )
            
            self.annotations[img_id] = annotation
            self.image_ids.append(img_id)
    
    def _get_image_path(self, image_id: str) -> Optional[Path]:
        """Get path to image file."""
        img_dir = self.data_path / "img_resized"
        
        # Try common extensions
        for ext in [".jpg", ".jpeg", ".png", ".gif"]:
            path = img_dir / f"{image_id}{ext}"
            if path.exists():
                return path
        
        return None
    
    def __len__(self) -> int:
        return len(self.image_ids)
    
    def __getitem__(self, idx: int) -> Dict:
        """Get a single sample."""
        img_id = self.image_ids[idx]
        annotation = self.annotations[img_id]
        
        # Load image
        img_path = self._get_image_path(img_id)
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        else:
            # Default: convert to tensor
            image = torch.from_numpy(
                np.array(image).transpose(2, 0, 1)
            ).float() / 255.0
        
        return {
            "image": image,
            "image_id": img_id,
            "label": annotation.majority_label,
            "is_hate": annotation.is_hate,
            "message_type": annotation.message_type,
            "visibility_level": annotation.visibility_level,
            "tweet_text": annotation.tweet_text,
            "ocr_text": annotation.ocr_text or "",
            "annotator_labels": annotation.labels
        }


class DatasetManager:
    """
    Manager for loading, validating, and preparing datasets for training.
    Supports both YOLO (bounding box) and VLM (image-level) annotations.
    """
    
    def __init__(self, data_path: str):
        """
        Initialize the DatasetManager.
        
        Args:
            data_path: Path to the dataset root directory
        """
        self.data_path = Path(data_path)
        self._datasets: Dict[str, MMHS150KDataset] = {}
    
    def load_dataset(
        self,
        split: str = "train",
        transform=None,
        include_ocr: bool = True
    ) -> MMHS150KDataset:
        """
        Load a dataset split.
        
        Args:
            split: One of 'train', 'val', 'test'
            transform: Optional torchvision transforms
            include_ocr: Whether to include OCR text
            
        Returns:
            PyTorch Dataset instance
        """
        cache_key = f"{split}_{include_ocr}"
        
        if cache_key not in self._datasets:
            self._datasets[cache_key] = MMHS150KDataset(
                data_path=str(self.data_path),
                split=split,
                transform=transform,
                include_ocr=include_ocr
            )
        
        return self._datasets[cache_key]
    
    def validate_annotations(
        self,
        annotations: Optional[List[Annotation]] = None,
        split: str = "train"
    ) -> float:
        """
        Calculate Fleiss Kappa for annotation quality validation.
        
        Args:
            annotations: List of annotations to validate, or None to use loaded dataset
            split: Dataset split to use if annotations not provided
            
        Returns:
            Fleiss Kappa score (should be >= 0.783 per requirements)
        """
        if annotations is None:
            dataset = self.load_dataset(split=split)
            annotations = list(dataset.annotations.values())
        
        if len(annotations) == 0:
            return 0.0
        
        return self._calculate_fleiss_kappa(annotations)
    
    def _calculate_fleiss_kappa(self, annotations: List[Annotation]) -> float:
        """
        Calculate Fleiss Kappa inter-annotator agreement.
        
        Fleiss Kappa measures agreement among multiple raters.
        Formula: κ = (P̄ - P̄e) / (1 - P̄e)
        
        Args:
            annotations: List of annotations with multiple rater labels
            
        Returns:
            Fleiss Kappa score in range [-1, 1]
        """
        n_subjects = len(annotations)
        n_raters = 3  # MMHS150K has 3 annotators
        n_categories = 6  # Labels 0-5
        
        if n_subjects == 0:
            return 0.0
        
        # Build rating matrix: n_subjects x n_categories
        # Each cell contains count of raters who assigned that category
        rating_matrix = np.zeros((n_subjects, n_categories), dtype=int)
        
        for i, ann in enumerate(annotations):
            for label in ann.labels:
                if 0 <= label < n_categories:
                    rating_matrix[i, label] += 1
        
        # Calculate P_i for each subject (proportion of agreeing pairs)
        p_i = np.zeros(n_subjects)
        for i in range(n_subjects):
            sum_squared = np.sum(rating_matrix[i] ** 2)
            p_i[i] = (sum_squared - n_raters) / (n_raters * (n_raters - 1))
        
        # Calculate P̄ (mean of P_i)
        p_bar = np.mean(p_i)
        
        # Calculate p_j (proportion of all ratings in category j)
        p_j = np.sum(rating_matrix, axis=0) / (n_subjects * n_raters)
        
        # Calculate P̄e (expected agreement by chance)
        p_e_bar = np.sum(p_j ** 2)
        
        # Calculate Fleiss Kappa
        if p_e_bar == 1.0:
            return 1.0  # Perfect agreement
        
        kappa = (p_bar - p_e_bar) / (1 - p_e_bar)
        
        return float(kappa)
    
    def get_dataset_stats(self, split: str = "train") -> Dict:
        """
        Get statistics about the dataset.
        
        Args:
            split: Dataset split to analyze
            
        Returns:
            Dictionary with dataset statistics
        """
        dataset = self.load_dataset(split=split)
        annotations = list(dataset.annotations.values())
        
        # Count by label
        label_counts = {i: 0 for i in range(6)}
        hate_count = 0
        textual_count = 0
        symbolic_count = 0
        
        for ann in annotations:
            label_counts[ann.majority_label] += 1
            if ann.is_hate:
                hate_count += 1
            if ann.message_type == "textual":
                textual_count += 1
            elif ann.message_type == "symbolic":
                symbolic_count += 1
        
        return {
            "total_images": len(annotations),
            "hate_images": hate_count,
            "non_hate_images": len(annotations) - hate_count,
            "textual_count": textual_count,
            "symbolic_count": symbolic_count,
            "label_distribution": label_counts,
            "fleiss_kappa": self.validate_annotations(annotations)
        }
    
    def check_composition_completeness(self, split: str = "train") -> Dict[str, bool]:
        """
        Check if dataset contains all required content type combinations.
        
        Per Requirements 1.2: Dataset should include both high/low visibility
        content with both textual hate speech and visual hate symbols.
        
        Returns:
            Dictionary indicating presence of each combination
        """
        dataset = self.load_dataset(split=split)
        
        combinations = {
            "high_visibility_textual": False,
            "high_visibility_symbolic": False,
            "low_visibility_textual": False,
            "low_visibility_symbolic": False
        }
        
        for ann in dataset.annotations.values():
            if not ann.is_hate:
                continue
            
            key = f"{ann.visibility_level}_visibility_{ann.message_type}"
            if key in combinations:
                combinations[key] = True
        
        return combinations
    
    def supports_minimum_size(self, min_size: int = 5000) -> bool:
        """
        Check if dataset supports minimum required size.
        
        Per Requirements 1.1: Support datasets with at least 5000 images.
        
        Args:
            min_size: Minimum required dataset size
            
        Returns:
            True if dataset meets minimum size requirement
        """
        total = 0
        for split in ["train", "val", "test"]:
            try:
                dataset = self.load_dataset(split=split)
                total += len(dataset)
            except FileNotFoundError:
                continue
        
        return total >= min_size


def download_mmhs150k_dataset() -> str:
    """
    Download the MMHS150K dataset using kagglehub.
    
    Note: kagglehub must be installed globally (pip install kagglehub)
    
    Returns:
        Path to the downloaded dataset
    """
    try:
        import kagglehub  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "kagglehub is required to download the dataset. "
            "Install it globally with: pip install kagglehub"
        ) from exc
    
    # Download using kagglehub
    path = kagglehub.dataset_download("victorcallejasf/multimodal-hate-speech")
    
    print(f"Dataset downloaded to: {path}")
    return path
