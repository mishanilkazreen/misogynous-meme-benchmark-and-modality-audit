"""
PreprocessingPipeline for image preprocessing.
Implements transformations from the HatefulIllusion paper for Pathway B.

Transformations include:
- Gaussian Blur: Obscures fine details to reveal hidden patterns
- Downscale: Reduces resolution to simulate viewing from distance
- Grid Repetition: Repeats image in grid layout
- Gradient Magnitude: Emphasizes edges and patterns
- Canny Edges: Edge detection
- Grayscale: Removes color information
- Histogram Equalization: Enhances contrast
- Gamma Correction: Adjusts brightness/contrast
- Combined transformations: Histogram+Blur, Gamma+Blur, Blur+Gradient, Blur+Histogram
"""
# pylint: disable=no-member, too-many-arguments, too-many-positional-arguments
# pylint: disable=too-many-instance-attributes, duplicate-code

from collections.abc import Callable

import cv2
import numpy as np
from PIL import Image
import torch


class ImageTransformations:
    """
    Individual image transformations for hateful illusion detection.
    Based on the HatefulIllusion paper's mitigation strategies.
    """

    @staticmethod
    def gaussian_blur(
        image: np.ndarray,
        kernel_size: int = 21,
        sigma: float = 0.0,
    ) -> np.ndarray:
        """
        Apply Gaussian blur to obscure fine details.

        Args:
            image: Input image (H, W, C) or (H, W)
            kernel_size: Size of blur kernel (must be odd)
            sigma: Gaussian sigma (0 = auto-calculate)

        Returns:
            Blurred image
        """
        if kernel_size % 2 == 0:
            kernel_size += 1
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)

    @staticmethod
    def downscale(
        image: np.ndarray,
        scale_factor: float = 0.25,
        restore_size: bool = True,
    ) -> np.ndarray:
        """
        Downscale image to simulate viewing from distance.

        Args:
            image: Input image (H, W, C) or (H, W)
            scale_factor: Factor to scale down (0.25 = 1/4 size)
            restore_size: If True, upscale back to original size

        Returns:
            Downscaled (and optionally restored) image
        """
        h, w = image.shape[:2]
        new_h, new_w = int(h * scale_factor), int(w * scale_factor)

        # Downscale
        downscaled = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

        if restore_size:
            # Upscale back to original size (creates pixelation effect)
            return cv2.resize(downscaled, (w, h), interpolation=cv2.INTER_NEAREST)
        return downscaled

    @staticmethod
    def grid_repetition(
        image: np.ndarray,
        grid_size: int = 3,
    ) -> np.ndarray:
        """
        Repeat image in a grid layout.

        Args:
            image: Input image (H, W, C) or (H, W)
            grid_size: Number of repetitions in each dimension

        Returns:
            Grid of repeated images (same size as original)
        """
        h, w = image.shape[:2]

        # Shrink image to fit in grid
        small_h, small_w = h // grid_size, w // grid_size
        small = cv2.resize(image, (small_w, small_h), interpolation=cv2.INTER_AREA)

        # Tile the small image
        if len(image.shape) == 3:
            tiled = np.tile(small, (grid_size, grid_size, 1))
        else:
            tiled = np.tile(small, (grid_size, grid_size))

        # Resize back to original size to ensure exact dimensions
        return cv2.resize(tiled, (w, h), interpolation=cv2.INTER_NEAREST)

    @staticmethod
    def gradient_magnitude(image: np.ndarray) -> np.ndarray:
        """
        Calculate gradient magnitude to emphasize edges and patterns.

        Args:
            image: Input image (H, W, C) or (H, W)

        Returns:
            Gradient magnitude image (grayscale, 3-channel)
        """
        # Convert to grayscale if needed
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image

        # Calculate gradients
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

        # Magnitude
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        magnitude = np.clip(magnitude, 0, 255).astype(np.uint8)

        # Convert back to 3-channel
        return cv2.cvtColor(magnitude, cv2.COLOR_GRAY2RGB)

    @staticmethod
    def canny_edges(
        image: np.ndarray,
        low_threshold: int = 50,
        high_threshold: int = 150,
    ) -> np.ndarray:
        """
        Apply Canny edge detection.

        Args:
            image: Input image (H, W, C) or (H, W)
            low_threshold: Lower threshold for edge detection
            high_threshold: Upper threshold for edge detection

        Returns:
            Edge image (3-channel)
        """
        # Convert to grayscale if needed
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image

        edges = cv2.Canny(gray, low_threshold, high_threshold)
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

    @staticmethod
    def grayscale(image: np.ndarray) -> np.ndarray:
        """
        Convert image to grayscale.

        Args:
            image: Input image (H, W, C)

        Returns:
            Grayscale image (3-channel for consistency)
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    @staticmethod
    def histogram_equalization(image: np.ndarray) -> np.ndarray:
        """
        Apply histogram equalization to enhance contrast.

        For color images, converts to YUV, equalizes Y channel, converts back.

        Args:
            image: Input image (H, W, C) or (H, W)

        Returns:
            Contrast-enhanced image
        """
        if len(image.shape) == 2:
            return cv2.equalizeHist(image)

        if image.shape[2] == 3:
            yuv = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
            return cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)
        return image

    @staticmethod
    def gamma_correction(
        image: np.ndarray,
        gamma: float = 2.2,
    ) -> np.ndarray:
        """
        Apply gamma correction to adjust brightness/contrast.

        Args:
            image: Input image (H, W, C) or (H, W)
            gamma: Gamma value (>1 darkens, <1 brightens)

        Returns:
            Gamma-corrected image
        """
        # Build lookup table
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype(np.uint8)

        return cv2.LUT(image, table)


class PreprocessingPipeline:
    """
    Preprocessing pipeline for content moderation images.

    Supports individual transformations and combinations.
    The default pipeline applies Gaussian blur followed by histogram equalization,
    which is the most effective combination according to the HatefulIllusion paper.
    """

    # Available transformation names
    TRANSFORMATIONS = [
        "blur",
        "downscale",
        "grid",
        "gradient",
        "canny",
        "grayscale",
        "histogram",
        "gamma",
        "histogram_blur",
        "gamma_blur",
        "blur_gradient",
        "blur_histogram",
    ]

    def __init__(
        self,
        blur_kernel_size: int = 21,
        blur_sigma: float = 0.0,
        downscale_factor: float = 0.25,
        grid_size: int = 3,
        canny_low: int = 50,
        canny_high: int = 150,
        gamma: float = 2.2,
        apply_blur: bool = True,
        apply_equalization: bool = True,
    ):
        """
        Initialize the preprocessing pipeline.

        Args:
            blur_kernel_size: Size of Gaussian blur kernel (must be odd)
            blur_sigma: Gaussian blur sigma (0 = auto-calculate from kernel size)
            downscale_factor: Factor for downscaling (0.25 = 1/4 size)
            grid_size: Number of repetitions for grid transformation
            canny_low: Lower threshold for Canny edge detection
            canny_high: Upper threshold for Canny edge detection
            gamma: Gamma value for gamma correction
            apply_blur: Whether to apply Gaussian blur (default pipeline)
            apply_equalization: Whether to apply histogram equalization (default pipeline)
        """
        if blur_kernel_size % 2 == 0:
            raise ValueError("blur_kernel_size must be odd")

        self.blur_kernel_size = blur_kernel_size
        self.blur_sigma = blur_sigma
        self.downscale_factor = downscale_factor
        self.grid_size = grid_size
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.gamma = gamma
        self.apply_blur = apply_blur
        self.apply_equalization = apply_equalization

        self._transforms = ImageTransformations()

    def preprocess(self, image: np.ndarray | Image.Image | torch.Tensor) -> np.ndarray:
        """
        Apply default preprocessing (blur -> histogram equalization).

        This is the most effective combination for detecting hateful illusions
        according to the paper.

        Args:
            image: Input image (numpy array, PIL Image, or torch Tensor)

        Returns:
            Preprocessed image as numpy array (H, W, C) in uint8 format
        """
        img = self._to_numpy(image)
        original_shape = img.shape

        # Step 1: Gaussian blur (if enabled)
        if self.apply_blur:
            img = self._transforms.gaussian_blur(img, self.blur_kernel_size, self.blur_sigma)

        # Step 2: Histogram equalization (if enabled)
        if self.apply_equalization:
            img = self._transforms.histogram_equalization(img)

        assert img.shape == original_shape, "Preprocessing changed image dimensions"
        return img

    def _build_dispatch(self) -> dict[str, Callable[[np.ndarray], np.ndarray]]:
        """Build transformation name -> callable dispatch table."""
        t = self._transforms
        return {
            "blur": lambda img: t.gaussian_blur(img, self.blur_kernel_size, self.blur_sigma),
            "downscale": lambda img: t.downscale(img, self.downscale_factor),
            "grid": lambda img: t.grid_repetition(img, self.grid_size),
            "gradient": t.gradient_magnitude,
            "canny": lambda img: t.canny_edges(img, self.canny_low, self.canny_high),
            "grayscale": t.grayscale,
            "histogram": t.histogram_equalization,
            "gamma": lambda img: t.gamma_correction(img, self.gamma),
            "histogram_blur": lambda img: t.gaussian_blur(
                t.histogram_equalization(img), self.blur_kernel_size, self.blur_sigma
            ),
            "gamma_blur": lambda img: t.gaussian_blur(
                t.gamma_correction(img, self.gamma), self.blur_kernel_size, self.blur_sigma
            ),
            "blur_gradient": lambda img: t.gradient_magnitude(
                t.gaussian_blur(img, self.blur_kernel_size, self.blur_sigma)
            ),
            "blur_histogram": lambda img: t.histogram_equalization(
                t.gaussian_blur(img, self.blur_kernel_size, self.blur_sigma)
            ),
        }

    def apply_transformation(
        self,
        image: np.ndarray | Image.Image | torch.Tensor,
        transformation: str,
    ) -> np.ndarray:
        """
        Apply a specific transformation to an image.

        Args:
            image: Input image
            transformation: Name of transformation to apply

        Returns:
            Transformed image
        """
        img = self._to_numpy(image)
        dispatch = self._build_dispatch()
        if transformation not in dispatch:
            raise ValueError(f"Unknown transformation: {transformation}")
        return dispatch[transformation](img)

    def apply_all_transformations(
        self,
        image: np.ndarray | Image.Image | torch.Tensor,
    ) -> dict:
        """
        Apply all available transformations and return results.

        Args:
            image: Input image

        Returns:
            Dictionary mapping transformation names to results
        """
        img = self._to_numpy(image)
        results = {"original": img.copy()}

        for name in self.TRANSFORMATIONS:
            results[name] = self.apply_transformation(img, name)

        return results

    def preprocess_batch(self, images: list) -> list:
        """
        Apply default preprocessing to a batch of images.

        Args:
            images: List of images

        Returns:
            List of preprocessed images
        """
        return [self.preprocess(img) for img in images]

    def _to_numpy(self, image: np.ndarray | Image.Image | torch.Tensor) -> np.ndarray:
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

    def to_tensor(self, image: np.ndarray) -> torch.Tensor:
        """Convert preprocessed image to PyTorch tensor."""
        # (H, W, C) -> (C, H, W)
        if len(image.shape) == 2:
            tensor = torch.from_numpy(image).unsqueeze(0).float() / 255.0
        else:
            tensor = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        return tensor

    def get_config(self) -> dict:
        """Return current configuration."""
        return {
            "blur_kernel_size": self.blur_kernel_size,
            "blur_sigma": self.blur_sigma,
            "downscale_factor": self.downscale_factor,
            "grid_size": self.grid_size,
            "canny_low": self.canny_low,
            "canny_high": self.canny_high,
            "gamma": self.gamma,
            "apply_blur": self.apply_blur,
            "apply_equalization": self.apply_equalization,
        }
