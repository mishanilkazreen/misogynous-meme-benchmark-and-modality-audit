"""Unit tests for DataAugmentation and BalancedSampler."""

import numpy as np

from utils.augmentation import BalancedSampler, DataAugmentation


class TestDataAugmentation:
    """Tests for DataAugmentation class."""

    def test_augment_preserves_shape(self):
        """Augmentation should preserve image dimensions."""
        aug = DataAugmentation(probability=1.0)
        image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

        result = aug.augment(image)

        assert result.shape == image.shape
        assert result.dtype == np.uint8

    def test_augment_with_zero_probability(self):
        """With probability=0, image should be unchanged."""
        aug = DataAugmentation(probability=0.0)
        image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

        result = aug.augment(image)

        np.testing.assert_array_equal(result, image)

    def test_rotation_range(self):
        """Test rotation within specified range."""
        aug = DataAugmentation(
            rotation_range=(-45.0, 45.0),
            probability=1.0,
            horizontal_flip=False,
            vertical_flip=False,
        )
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        image[40:60, 40:60] = 255  # White square in center

        result = aug.augment(image)

        # Shape should be preserved
        assert result.shape == image.shape

    def test_scale_range(self):
        """Test scaling within specified range."""
        aug = DataAugmentation(
            scale_range=(0.8, 1.2),
            probability=1.0,
            rotation_range=(0, 0),
            horizontal_flip=False,
            vertical_flip=False,
        )
        image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

        result = aug.augment(image)

        assert result.shape == image.shape

    def test_brightness_adjustment(self):
        """Test brightness adjustment."""
        aug = DataAugmentation(
            brightness_range=(0.5, 0.5),  # Fixed to 0.5
            probability=1.0,
            rotation_range=(0, 0),
            scale_range=(1.0, 1.0),
            horizontal_flip=False,
            vertical_flip=False,
        )
        image = np.full((100, 100, 3), 200, dtype=np.uint8)

        result = aug.augment(image)

        # Should be darker
        assert result.mean() < image.mean()

    def test_horizontal_flip(self):
        """Test horizontal flip."""
        aug = DataAugmentation(
            horizontal_flip=True,
            vertical_flip=False,
            probability=1.0,
            rotation_range=(0, 0),
            scale_range=(1.0, 1.0),
            brightness_range=(1.0, 1.0),
        )
        # Create asymmetric image
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        image[:, :50] = 255  # Left half white

        result = aug.augment(image)

        # Right half should now be white
        assert result[:, 50:].mean() > result[:, :50].mean()

    def test_vertical_flip(self):
        """Test vertical flip."""
        aug = DataAugmentation(
            horizontal_flip=False,
            vertical_flip=True,
            probability=1.0,
            rotation_range=(0, 0),
            scale_range=(1.0, 1.0),
            brightness_range=(1.0, 1.0),
        )
        # Create asymmetric image
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        image[:50, :] = 255  # Top half white

        result = aug.augment(image)

        # Bottom half should now be white
        assert result[50:, :].mean() > result[:50, :].mean()


class TestBalancedSampler:
    """Tests for BalancedSampler class."""

    def test_get_balanced_indices(self):
        """Test getting balanced sample indices."""
        annotations = [
            {"visibility_level": "high", "message_type": "textual"},
            {"visibility_level": "high", "message_type": "symbolic"},
            {"visibility_level": "low", "message_type": "textual"},
            {"visibility_level": "low", "message_type": "symbolic"},
        ]

        sampler = BalancedSampler(annotations)
        indices = sampler.get_balanced_indices(4)

        assert len(indices) == 4
        assert all(0 <= idx < 4 for idx in indices)

    def test_category_counts(self):
        """Test category counting."""
        annotations = [
            {"visibility_level": "high", "message_type": "textual"},
            {"visibility_level": "high", "message_type": "textual"},
            {"visibility_level": "low", "message_type": "symbolic"},
        ]

        sampler = BalancedSampler(annotations)
        counts = sampler.get_category_counts()

        assert counts["high_visibility_textual"] == 2
        assert counts["low_visibility_symbolic"] == 1
        assert counts["high_visibility_symbolic"] == 0
        assert counts["low_visibility_textual"] == 0

    def test_composition_completeness_complete(self):
        """Test completeness check with all categories."""
        annotations = [
            {"visibility_level": "high", "message_type": "textual"},
            {"visibility_level": "high", "message_type": "symbolic"},
            {"visibility_level": "low", "message_type": "textual"},
            {"visibility_level": "low", "message_type": "symbolic"},
        ]

        sampler = BalancedSampler(annotations)
        completeness = sampler.check_composition_completeness()

        assert all(completeness.values())

    def test_composition_completeness_incomplete(self):
        """Test completeness check with missing categories."""
        annotations = [
            {"visibility_level": "high", "message_type": "textual"},
        ]

        sampler = BalancedSampler(annotations)
        completeness = sampler.check_composition_completeness()

        assert completeness["high_visibility_textual"] is True
        assert completeness["high_visibility_symbolic"] is False
        assert completeness["low_visibility_textual"] is False
        assert completeness["low_visibility_symbolic"] is False

    def test_empty_annotations(self):
        """Test with empty annotations."""
        sampler = BalancedSampler([])

        indices = sampler.get_balanced_indices(10)
        counts = sampler.get_category_counts()

        assert indices == []
        assert all(c == 0 for c in counts.values())

    def test_sampling_with_replacement(self):
        """Test sampling when category has fewer items than requested."""
        annotations = [
            {"visibility_level": "high", "message_type": "textual"},
        ]

        sampler = BalancedSampler(annotations)
        indices = sampler.get_balanced_indices(5)

        # Should sample with replacement
        assert len(indices) == 5
        assert all(idx == 0 for idx in indices)
