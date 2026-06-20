"""
Data augmentation utilities for content moderation training.
Supports rotation, scaling, brightness adjustments.
"""
# pylint: disable=no-member, too-many-arguments, too-many-positional-arguments, too-few-public-methods

import random

import cv2
import numpy as np


class DataAugmentation:
    """
    Data augmentation for training content moderation models.

    Supports rotation, scaling, and brightness adjustments while
    preserving the embedded content visibility characteristics.
    """

    def __init__(
        self,
        rotation_range: tuple[float, float] = (-15.0, 15.0),
        scale_range: tuple[float, float] = (0.9, 1.1),
        brightness_range: tuple[float, float] = (0.8, 1.2),
        horizontal_flip: bool = True,
        vertical_flip: bool = False,
        probability: float = 0.5,
    ):
        """
        Initialize augmentation settings.

        Args:
            rotation_range: Min/max rotation in degrees
            scale_range: Min/max scale factor
            brightness_range: Min/max brightness multiplier
            horizontal_flip: Enable horizontal flipping
            vertical_flip: Enable vertical flipping
            probability: Probability of applying each augmentation
        """
        self.rotation_range = rotation_range
        self.scale_range = scale_range
        self.brightness_range = brightness_range
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.probability = probability

    def augment(self, image: np.ndarray) -> np.ndarray:
        """
        Apply random augmentations to an image.

        Args:
            image: Input image as numpy array (H, W, C)

        Returns:
            Augmented image
        """
        img = image.copy()

        # Rotation
        if random.random() < self.probability:
            angle = random.uniform(*self.rotation_range)
            img = self._rotate(img, angle)

        # Scaling
        if random.random() < self.probability:
            scale = random.uniform(*self.scale_range)
            img = self._scale(img, scale)

        # Brightness
        if random.random() < self.probability:
            factor = random.uniform(*self.brightness_range)
            img = self._adjust_brightness(img, factor)

        # Horizontal flip
        if self.horizontal_flip and random.random() < self.probability:
            img = self._flip_horizontal(img)

        # Vertical flip
        if self.vertical_flip and random.random() < self.probability:
            img = self._flip_vertical(img)

        return img

    def _rotate(self, image: np.ndarray, angle: float) -> np.ndarray:
        """Rotate image by angle degrees."""
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)

    def _scale(self, image: np.ndarray, scale: float) -> np.ndarray:
        """Scale image and crop/pad to original size."""
        h, w = image.shape[:2]
        new_h, new_w = int(h * scale), int(w * scale)

        # Resize
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Crop or pad to original size
        if scale > 1.0:
            # Crop center
            start_y = (new_h - h) // 2
            start_x = (new_w - w) // 2
            return resized[start_y : start_y + h, start_x : start_x + w]

        # Pad with reflection
        pad_y = (h - new_h) // 2
        pad_x = (w - new_w) // 2
        return cv2.copyMakeBorder(
            resized, pad_y, h - new_h - pad_y, pad_x, w - new_w - pad_x, cv2.BORDER_REFLECT
        )

    def _adjust_brightness(self, image: np.ndarray, factor: float) -> np.ndarray:
        """Adjust image brightness."""
        return np.clip(image.astype(np.float32) * factor, 0, 255).astype(np.uint8)

    def _flip_horizontal(self, image: np.ndarray) -> np.ndarray:
        """Flip image horizontally."""
        return cv2.flip(image, 1)

    def _flip_vertical(self, image: np.ndarray) -> np.ndarray:
        """Flip image vertically."""
        return cv2.flip(image, 0)


class BalancedSampler:
    """
    Sampler that ensures balanced distribution across content types.

    Balances by visibility level (high/low) and message type (textual/symbolic).
    """

    def __init__(self, annotations: list[dict]):
        """
        Initialize with dataset annotations.

        Args:
            annotations: List of annotation dictionaries with visibility_level and message_type
        """
        self.annotations = annotations
        self._build_indices()

    def _build_indices(self) -> None:
        """Build indices for each category combination."""
        self.category_indices: dict[str, list[int]] = {
            "high_visibility_textual": [],
            "high_visibility_symbolic": [],
            "low_visibility_textual": [],
            "low_visibility_symbolic": [],
        }

        for idx, ann in enumerate(self.annotations):
            vis = ann.get("visibility_level", "high")
            msg_type = ann.get("message_type", "textual")
            key = f"{vis}_visibility_{msg_type}"
            if key in self.category_indices:
                self.category_indices[key].append(idx)

    def get_balanced_indices(self, num_samples: int) -> list[int]:
        """
        Get balanced sample indices.

        Args:
            num_samples: Total number of samples to return

        Returns:
            List of indices balanced across categories
        """
        # Get non-empty categories
        non_empty = {k: v for k, v in self.category_indices.items() if v}

        if not non_empty:
            return []

        samples_per_category = num_samples // len(non_empty)
        remainder = num_samples % len(non_empty)

        indices = []
        for i, (_, cat_indices) in enumerate(non_empty.items()):
            n = samples_per_category + (1 if i < remainder else 0)
            # Sample with replacement if needed
            if len(cat_indices) >= n:
                indices.extend(random.sample(cat_indices, n))
            else:
                indices.extend(random.choices(cat_indices, k=n))

        random.shuffle(indices)
        return indices

    def get_category_counts(self) -> dict[str, int]:
        """Get count of samples in each category."""
        return {k: len(v) for k, v in self.category_indices.items()}

    def check_composition_completeness(self) -> dict[str, bool]:
        """Check if all category combinations are present."""
        return {k: len(v) > 0 for k, v in self.category_indices.items()}
