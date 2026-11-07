"""
Utils package for VLM content moderation system.
Contains preprocessing, data loading, and helper functions.
"""

from utils.augmentation import BalancedSampler, DataAugmentation
from utils.dataset import (
    DatasetManager,
    HatefulIllusionDataset,  # deprecated alias for MamiDataset
    MamiDataset,
    download_hateful_illusion_dataset,  # deprecated alias for download_mami_dataset
    download_mami_dataset,
)
from utils.ocr import OCRPipeline
from utils.preprocessing import ImageTransformations, PreprocessingPipeline

__all__ = [
    "BalancedSampler",
    "DataAugmentation",
    "DatasetManager",
    "HatefulIllusionDataset",
    "ImageTransformations",
    "MamiDataset",
    "OCRPipeline",
    "PreprocessingPipeline",
    "download_hateful_illusion_dataset",
    "download_mami_dataset",
]
