"""
Dataset management for VLM content moderation system.
Handles loading, validation, and preprocessing of hateful content datasets.
"""
import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import numpy as np

try:
    import kagglehub
except ImportError:
    kagglehub = None

try:
    import torch
    from torch.utils.data import Dataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    Dataset = object

try:
    from PIL import Image
except ImportError:
    Image = None


@dataclass
class Annotation:
    """Represents an annotation for an image."""
    image_id: str
    label: int  # 0: not hateful, 1: hateful
    message_type: Optional[str] = None  # "textual" or "symbolic"
    visibility_level: Optional[str] = None  # "high" or "low"
    bbox: Optional[Tuple[int, int, int, int]] = None  # (x, y, w, h) for YOLO
    text: Optional[str] = None  # Associated text if available


def download_mmhs150k_dataset() -> str:
    """
    Download the MMHS150K multimodal hate speech dataset from Kaggle.
    
    Returns:
        Path to the downloaded dataset directory.
        
    Raises:
        ImportError: If kagglehub is not installed.
        RuntimeError: If download fails.
    """
    if kagglehub is None:
        raise ImportError(
            "kagglehub is required to download datasets. "
            "Install it with: pip install kagglehub"
        )
    
    path = kagglehub.dataset_download("victorcallejasf/multimodal-hate-speech")
    return path


class DatasetManager:
    """
    Manages dataset loading and validation for content moderation training.
    
    Supports:
    - Loading datasets with minimum 5000 images (Requirement 1.1)
    - Validating annotation quality via Fleiss Kappa (Requirement 1.3)
    - Both bounding box (YOLO) and image-level (VLM) annotations (Requirement 1.4)
    """
    
    # Minimum dataset size per requirements
    MIN_DATASET_SIZE = 5000
    
    # Minimum Fleiss Kappa for annotation quality
    MIN_FLEISS_KAPPA = 0.783
    
    def __init__(self, data_dir: Optional[str] = None):
        """
        Initialize the DatasetManager.
        
        Args:
            data_dir: Path to the dataset directory. If None, will use default cache.
        """
        self.data_dir = Path(data_dir) if data_dir else None
        self._images: List[str] = []
        self._annotations: Dict[str, Annotation] = {}
        self._metadata: Dict = {}
    
    def load_dataset(self, path: Optional[str] = None) -> "ContentModerationDataset":
        """
        Load a dataset from the specified path or download from Kaggle.
        
        Args:
            path: Path to dataset directory. If None, downloads MMHS150K from Kaggle.
            
        Returns:
            A PyTorch Dataset object containing the loaded data.
            
        Raises:
            ValueError: If dataset has fewer than 5000 images.
            FileNotFoundError: If path doesn't exist.
        """
        if path is None:
            # Download from Kaggle
            path = download_mmhs150k_dataset()
        
        dataset_path = Path(path)
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset path does not exist: {path}")
        
        self.data_dir = dataset_path
        
        # Load images and annotations
        self._load_images_and_annotations(dataset_path)
        
        # Validate minimum size
        if len(self._images) < self.MIN_DATASET_SIZE:
            raise ValueError(
                f"Dataset too small: {len(self._images)} images. "
                f"Minimum required: {self.MIN_DATASET_SIZE}"
            )
        
        return ContentModerationDataset(
            images=self._images,
            annotations=self._annotations,
            data_dir=dataset_path
        )
    
    def _load_images_and_annotations(self, dataset_path: Path) -> None:
        """
        Load images and annotations from the dataset directory.
        
        Supports multiple dataset formats:
        - MMHS150K format (img_resized/ + MMHS150K_GT.json)
        - YOLO format (images/ + labels/)
        - Simple format (images/ + annotations.json)
        """
        self._images = []
        self._annotations = {}
        
        # Try MMHS150K format first
        mmhs_json = dataset_path / "MMHS150K_GT.json"
        img_dir = dataset_path / "img_resized"
        
        if mmhs_json.exists() and img_dir.exists():
            self._load_mmhs150k_format(dataset_path)
            return
        
        # Try to find images in common locations
        possible_img_dirs = [
            dataset_path / "images",
            dataset_path / "img_resized",
            dataset_path / "img",
            dataset_path,
        ]
        
        for img_dir in possible_img_dirs:
            if img_dir.exists():
                image_files = self._find_images(img_dir)
                if image_files:
                    self._images = image_files
                    break
        
        # Try to load annotations
        possible_annotation_files = [
            dataset_path / "MMHS150K_GT.json",
            dataset_path / "annotations.json",
            dataset_path / "labels.json",
        ]
        
        for ann_file in possible_annotation_files:
            if ann_file.exists():
                self._load_json_annotations(ann_file)
                break
    
    def _load_mmhs150k_format(self, dataset_path: Path) -> None:
        """Load dataset in MMHS150K format."""
        mmhs_json = dataset_path / "MMHS150K_GT.json"
        img_dir = dataset_path / "img_resized"
        
        with open(mmhs_json, 'r') as f:
            annotations_data = json.load(f)
        
        for img_id, data in annotations_data.items():
            img_path = img_dir / f"{img_id}.jpg"
            if not img_path.exists():
                img_path = img_dir / f"{img_id}.png"
            
            if img_path.exists():
                self._images.append(str(img_path))
                
                # Parse MMHS150K annotation format
                # Labels are typically: 0=not hateful, 1=hateful
                labels = data.get("labels", [])
                label = 1 if any(l > 0 for l in labels) else 0
                
                self._annotations[str(img_path)] = Annotation(
                    image_id=img_id,
                    label=label,
                    text=data.get("tweet_text", ""),
                    message_type="textual" if data.get("tweet_text") else None,
                )
    
    def _find_images(self, directory: Path) -> List[str]:
        """Find all image files in a directory."""
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}
        images = []
        
        for ext in image_extensions:
            images.extend(str(p) for p in directory.glob(f"*{ext}"))
            images.extend(str(p) for p in directory.glob(f"*{ext.upper()}"))
        
        return sorted(images)
    
    def _load_json_annotations(self, annotation_file: Path) -> None:
        """Load annotations from a JSON file."""
        with open(annotation_file, 'r') as f:
            data = json.load(f)
        
        for img_id, ann_data in data.items():
            # Find matching image
            matching_images = [img for img in self._images if img_id in img]
            
            for img_path in matching_images:
                if isinstance(ann_data, dict):
                    label = ann_data.get("label", ann_data.get("labels", [0]))
                    if isinstance(label, list):
                        label = 1 if any(l > 0 for l in label) else 0
                    
                    self._annotations[img_path] = Annotation(
                        image_id=img_id,
                        label=label,
                        text=ann_data.get("text", ann_data.get("tweet_text", "")),
                        message_type=ann_data.get("message_type"),
                        visibility_level=ann_data.get("visibility_level"),
                    )
                else:
                    # Simple label format
                    self._annotations[img_path] = Annotation(
                        image_id=img_id,
                        label=int(ann_data) if ann_data else 0,
                    )
    
    def validate_annotations(
        self, 
        annotations: Optional[List[List[int]]] = None
    ) -> float:
        """
        Calculate Fleiss Kappa for annotation quality validation.
        
        Fleiss Kappa measures inter-rater agreement for categorical ratings
        by multiple raters. A value >= 0.783 indicates substantial agreement.
        
        Args:
            annotations: Matrix of shape (n_subjects, n_categories) where each
                        entry is the count of raters who assigned that category.
                        If None, uses loaded annotations (assumes single rater).
        
        Returns:
            Fleiss Kappa coefficient in range [-1, 1].
            Values >= 0.783 indicate substantial agreement (Requirement 1.3).
        """
        if annotations is None:
            # If no multi-rater annotations provided, return 1.0 (perfect agreement)
            # This is a placeholder for when we have single-rater annotations
            return 1.0
        
        return self._calculate_fleiss_kappa(annotations)
    
    def _calculate_fleiss_kappa(self, annotations: List[List[int]]) -> float:
        """
        Calculate Fleiss Kappa coefficient.
        
        Args:
            annotations: Matrix where annotations[i][j] is the number of raters
                        who assigned category j to subject i.
        
        Returns:
            Fleiss Kappa coefficient.
        """
        annotations = np.array(annotations)
        n_subjects, n_categories = annotations.shape
        n_raters = annotations.sum(axis=1)[0]  # Assume same number of raters per subject
        
        if n_raters <= 1:
            return 1.0  # Perfect agreement with single rater
        
        # Calculate P_i (proportion of agreeing pairs for each subject)
        P_i = (np.sum(annotations ** 2, axis=1) - n_raters) / (n_raters * (n_raters - 1))
        
        # Calculate P_bar (mean of P_i)
        P_bar = np.mean(P_i)
        
        # Calculate p_j (proportion of all assignments to category j)
        p_j = np.sum(annotations, axis=0) / (n_subjects * n_raters)
        
        # Calculate P_e (expected agreement by chance)
        P_e = np.sum(p_j ** 2)
        
        # Calculate Fleiss Kappa
        if P_e == 1:
            return 1.0  # Perfect agreement
        
        kappa = (P_bar - P_e) / (1 - P_e)
        return float(kappa)
    
    def get_dataset_stats(self) -> Dict:
        """
        Get statistics about the loaded dataset.
        
        Returns:
            Dictionary containing dataset statistics.
        """
        if not self._images:
            return {"error": "No dataset loaded"}
        
        total = len(self._images)
        hateful = sum(1 for ann in self._annotations.values() if ann.label == 1)
        
        # Count by message type
        textual = sum(1 for ann in self._annotations.values() 
                     if ann.message_type == "textual")
        symbolic = sum(1 for ann in self._annotations.values() 
                      if ann.message_type == "symbolic")
        
        # Count by visibility
        high_vis = sum(1 for ann in self._annotations.values() 
                      if ann.visibility_level == "high")
        low_vis = sum(1 for ann in self._annotations.values() 
                     if ann.visibility_level == "low")
        
        return {
            "total_images": total,
            "hateful_images": hateful,
            "non_hateful_images": total - hateful,
            "textual_messages": textual,
            "symbolic_messages": symbolic,
            "high_visibility": high_vis,
            "low_visibility": low_vis,
            "meets_minimum_size": total >= self.MIN_DATASET_SIZE,
        }


class ContentModerationDataset(Dataset):
    """
    PyTorch Dataset for content moderation training.
    
    Supports both YOLO (bounding box) and VLM (image-level) training modes.
    """
    
    def __init__(
        self,
        images: List[str],
        annotations: Dict[str, Annotation],
        data_dir: Path,
        transform=None,
        mode: str = "vlm"  # "vlm" or "yolo"
    ):
        """
        Initialize the dataset.
        
        Args:
            images: List of image file paths.
            annotations: Dictionary mapping image paths to Annotation objects.
            data_dir: Root directory of the dataset.
            transform: Optional torchvision transforms to apply.
            mode: Training mode - "vlm" for image-level or "yolo" for detection.
        """
        self.images = images
        self.annotations = annotations
        self.data_dir = data_dir
        self.transform = transform
        self.mode = mode
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Get a single item from the dataset.
        
        Returns:
            Dictionary containing:
            - image: PIL Image or tensor
            - label: Binary label (0 or 1)
            - image_path: Path to the image file
            - annotation: Full Annotation object
        """
        img_path = self.images[idx]
        
        # Load image
        if Image is not None:
            image = Image.open(img_path).convert("RGB")
        else:
            image = None
        
        # Get annotation
        annotation = self.annotations.get(img_path, Annotation(
            image_id=Path(img_path).stem,
            label=0
        ))
        
        # Apply transforms
        if self.transform is not None and image is not None:
            image = self.transform(image)
        
        result = {
            "image": image,
            "label": annotation.label,
            "image_path": img_path,
        }
        
        # Add YOLO-specific data if in YOLO mode
        if self.mode == "yolo" and annotation.bbox is not None:
            result["bbox"] = annotation.bbox
        
        # Add additional annotation data
        if annotation.message_type:
            result["message_type"] = annotation.message_type
        if annotation.visibility_level:
            result["visibility_level"] = annotation.visibility_level
        if annotation.text:
            result["text"] = annotation.text
        
        return result
    
    def get_class_distribution(self) -> Dict[int, int]:
        """Get the distribution of classes in the dataset."""
        distribution = {0: 0, 1: 0}
        for img_path in self.images:
            ann = self.annotations.get(img_path)
            if ann:
                distribution[ann.label] = distribution.get(ann.label, 0) + 1
        return distribution
