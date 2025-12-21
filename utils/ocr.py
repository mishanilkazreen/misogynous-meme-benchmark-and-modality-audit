"""
OCR Pipeline for text extraction from images.
Integrates EasyOCR for text extraction and provides normalization utilities.
"""

import re
import unicodedata
from typing import Dict, List, Union

import numpy as np
import torch
from PIL import Image


class OCRPipeline:
    """
    OCR pipeline for extracting and normalizing text from images.

    Uses EasyOCR for text detection and recognition.
    Provides text normalization for downstream VLM processing.
    """

    def __init__(
        self,
        languages: List[str] = None,
        gpu: bool = False,
        confidence_threshold: float = 0.3,
    ):
        """
        Initialize the OCR pipeline.

        Args:
            languages: List of language codes (default: ['en'])
            gpu: Whether to use GPU acceleration
            confidence_threshold: Minimum confidence for text detection
        """
        self.languages = languages or ["en"]
        self.gpu = gpu
        self.confidence_threshold = confidence_threshold
        self._reader = None

    @property
    def reader(self):
        """Lazy initialization of EasyOCR reader."""
        if self._reader is None:
            import easyocr
            self._reader = easyocr.Reader(self.languages, gpu=self.gpu)
        return self._reader

    def extract_text(
        self, image: Union[np.ndarray, Image.Image, torch.Tensor]
    ) -> str:
        """
        Extract text from an image.

        Args:
            image: Input image (numpy array, PIL Image, or torch Tensor)

        Returns:
            Extracted text as a single string
        """
        img = self._to_numpy(image)

        # Run OCR
        results = self.reader.readtext(img)

        # Filter by confidence and extract text
        texts = []
        for _, text, confidence in results:
            if confidence >= self.confidence_threshold:
                texts.append(text)

        return " ".join(texts)

    def extract_text_with_boxes(
        self, image: Union[np.ndarray, Image.Image, torch.Tensor]
    ) -> List[Dict]:
        """
        Extract text with bounding box information.

        Args:
            image: Input image

        Returns:
            List of dicts with 'text', 'bbox', and 'confidence' keys
        """
        img = self._to_numpy(image)

        results = self.reader.readtext(img)

        detections = []
        for box, text, confidence in results:
            if confidence >= self.confidence_threshold:
                # Convert bbox to (x, y, w, h) format
                x_coords = [p[0] for p in box]
                y_coords = [p[1] for p in box]
                x, y = min(x_coords), min(y_coords)
                w, h = max(x_coords) - x, max(y_coords) - y

                detections.append({
                    "text": text,
                    "bbox": (int(x), int(y), int(w), int(h)),
                    "confidence": confidence,
                })

        return detections

    def normalize_text(self, text: str) -> str:
        """
        Clean and normalize extracted text.

        Normalization steps:
        1. Unicode normalization (NFKC)
        2. Lowercase conversion
        3. Remove extra whitespace
        4. Remove special characters (keep alphanumeric and basic punctuation)

        Args:
            text: Raw extracted text

        Returns:
            Normalized text
        """
        if not text:
            return ""

        # Unicode normalization
        text = unicodedata.normalize("NFKC", text)

        # Lowercase
        text = text.lower()

        # Remove special characters, keep alphanumeric, spaces, and basic punctuation
        text = re.sub(r"[^\w\s.,!?'-]", "", text)

        # Normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def extract_and_normalize(
        self, image: Union[np.ndarray, Image.Image, torch.Tensor]
    ) -> str:
        """
        Extract text from image and normalize it.

        Args:
            image: Input image

        Returns:
            Normalized extracted text
        """
        raw_text = self.extract_text(image)
        return self.normalize_text(raw_text)

    def _to_numpy(
        self, image: Union[np.ndarray, Image.Image, torch.Tensor]
    ) -> np.ndarray:
        """Convert input to numpy array in (H, W, C) format."""
        if isinstance(image, np.ndarray):
            img = image.copy()
        elif isinstance(image, Image.Image):
            img = np.array(image)
        elif isinstance(image, torch.Tensor):
            # Handle (C, H, W) tensor format
            if image.dim() == 3 and image.shape[0] in [1, 3]:
                img = image.permute(1, 2, 0).numpy()
            else:
                img = image.numpy()
            # Scale from [0, 1] to [0, 255] if needed
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

        # Ensure uint8 format
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)

        return img

    def get_config(self) -> Dict:
        """Return current configuration."""
        return {
            "languages": self.languages,
            "gpu": self.gpu,
            "confidence_threshold": self.confidence_threshold,
        }
