"""
PreprocessingPipeline for image preprocessing.
Implements Gaussian blur and histogram equalization for Pathway B.
"""

from typing import Optional, Tuple, Union

import cv2
import numpy as np
import torch
from PIL import Image


class PreprocessingPipeline:
    """
    Preprocessing pipeline for content moderation images.
    
    Applies Gaussian blur followed by histogram equalization.
    This order is critical for Pathway B in the VLM architecture.
    """

    def __init__(
        self,
        blur_kernel_size: int = 5,
        blur_sigma: float = 0.0,
        apply_blur: bool = True,
        apply_equalization: bool = True,
    ):
        """
        Initialize the preprocessing pipeline.

        Args:
            blur_kernel_size: Size of Gaussian blur kernel (must be odd)
            blur_sigma: Gaussian blur sigma (0 = auto-calculate from kernel size)
            apply_blur: Whether to apply Gaussian blur
            apply_equalization: Whether to apply histogram equalization
        """
        if blur_kernel_size % 2 == 0:
            raise ValueError("blur_kernel_size must be odd")
        
        self.blur_kernel_size = blur_kernel_size
        self.blur_sigma = blur_sigma
        self.apply_blur = apply_blur
        self.apply_equalization = apply_equalization

    def preprocess(
        self, image: Union[np.ndarray, Image.Image, torch.Tensor]
    ) -> np.ndarray:
        """
        Apply preprocessing to a single image.
        
        Order: Gaussian blur -> Histogram equalization
        This order is required by the VLM Pathway B specification.

        Args:
            image: Input image (numpy array, PIL Image, or torch Tensor)

        Returns:
            Preprocessed image as numpy array (H, W, C) in uint8 format
        """
        # Convert to numpy array
        img = self._to_numpy(image)
        original_shape = img.shape
        
        # Step 1: Gaussian blur (if enabled)
        if self.apply_blur:
            img = self._apply_gaussian_blur(img)
        
        # Step 2: Histogram equalization (if enabled)
        if self.apply_equalization:
            img = self._apply_histogram_equalization(img)
        
        # Verify dimensions are preserved
        assert img.shape == original_shape, "Preprocessing changed image dimensions"
        
        return img

    def preprocess_batch(
        self, images: list
    ) -> list:
        """
        Apply preprocessing to a batch of images.

        Args:
            images: List of images

        Returns:
            List of preprocessed images
        """
        return [self.preprocess(img) for img in images]

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

    def _apply_gaussian_blur(self, image: np.ndarray) -> np.ndarray:
        """Apply Gaussian blur to image."""
        return cv2.GaussianBlur(
            image,
            (self.blur_kernel_size, self.blur_kernel_size),
            self.blur_sigma,
        )

    def _apply_histogram_equalization(self, image: np.ndarray) -> np.ndarray:
        """
        Apply histogram equalization to image.
        
        For color images, converts to YUV, equalizes Y channel, converts back.
        """
        if len(image.shape) == 2:
            # Grayscale
            return cv2.equalizeHist(image)
        elif image.shape[2] == 3:
            # Color image - equalize in YUV space
            yuv = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
            return cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)
        else:
            # Handle other channel counts
            return image

    def to_tensor(self, image: np.ndarray) -> torch.Tensor:
        """Convert preprocessed image to PyTorch tensor."""
        # (H, W, C) -> (C, H, W)
        tensor = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        return tensor

    def get_config(self) -> dict:
        """Return current configuration."""
        return {
            "blur_kernel_size": self.blur_kernel_size,
            "blur_sigma": self.blur_sigma,
            "apply_blur": self.apply_blur,
            "apply_equalization": self.apply_equalization,
        }
