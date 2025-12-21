"""
Unit tests for PreprocessingPipeline and ImageTransformations.
"""

import numpy as np
import pytest
import torch
from PIL import Image

from utils.preprocessing import ImageTransformations, PreprocessingPipeline


class TestImageTransformations:
    """Tests for individual image transformations."""

    @pytest.fixture
    def sample_image(self):
        """Create a sample RGB image."""
        return np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

    @pytest.fixture
    def grayscale_image(self):
        """Create a sample grayscale image."""
        return np.random.randint(0, 256, (100, 100), dtype=np.uint8)

    def test_gaussian_blur(self, sample_image):
        """Test Gaussian blur transformation."""
        result = ImageTransformations.gaussian_blur(sample_image, kernel_size=5)
        assert result.shape == sample_image.shape
        assert result.dtype == sample_image.dtype

    def test_gaussian_blur_even_kernel_corrected(self, sample_image):
        """Test that even kernel size is corrected to odd."""
        result = ImageTransformations.gaussian_blur(sample_image, kernel_size=4)
        assert result.shape == sample_image.shape

    def test_downscale_with_restore(self, sample_image):
        """Test downscale with size restoration."""
        result = ImageTransformations.downscale(sample_image, scale_factor=0.25, restore_size=True)
        assert result.shape == sample_image.shape

    def test_downscale_without_restore(self, sample_image):
        """Test downscale without size restoration."""
        result = ImageTransformations.downscale(sample_image, scale_factor=0.5, restore_size=False)
        assert result.shape[0] == sample_image.shape[0] // 2
        assert result.shape[1] == sample_image.shape[1] // 2

    def test_grid_repetition(self, sample_image):
        """Test grid repetition transformation."""
        result = ImageTransformations.grid_repetition(sample_image, grid_size=3)
        assert result.shape == sample_image.shape

    def test_gradient_magnitude(self, sample_image):
        """Test gradient magnitude transformation."""
        result = ImageTransformations.gradient_magnitude(sample_image)
        assert result.shape == sample_image.shape
        assert result.dtype == np.uint8

    def test_gradient_magnitude_grayscale(self, grayscale_image):
        """Test gradient magnitude on grayscale image."""
        result = ImageTransformations.gradient_magnitude(grayscale_image)
        assert result.shape == (grayscale_image.shape[0], grayscale_image.shape[1], 3)

    def test_canny_edges(self, sample_image):
        """Test Canny edge detection."""
        result = ImageTransformations.canny_edges(sample_image)
        assert result.shape == sample_image.shape
        assert result.dtype == np.uint8

    def test_grayscale(self, sample_image):
        """Test grayscale conversion."""
        result = ImageTransformations.grayscale(sample_image)
        assert result.shape == sample_image.shape
        # All channels should be equal (grayscale)
        assert np.allclose(result[:, :, 0], result[:, :, 1])
        assert np.allclose(result[:, :, 1], result[:, :, 2])

    def test_histogram_equalization(self, sample_image):
        """Test histogram equalization."""
        result = ImageTransformations.histogram_equalization(sample_image)
        assert result.shape == sample_image.shape
        assert result.dtype == np.uint8

    def test_histogram_equalization_grayscale(self, grayscale_image):
        """Test histogram equalization on grayscale."""
        result = ImageTransformations.histogram_equalization(grayscale_image)
        assert result.shape == grayscale_image.shape

    def test_gamma_correction(self, sample_image):
        """Test gamma correction."""
        result = ImageTransformations.gamma_correction(sample_image, gamma=2.2)
        assert result.shape == sample_image.shape
        assert result.dtype == np.uint8

    def test_gamma_correction_brightens(self):
        """Test gamma correction changes brightness."""
        # Create a mid-tone image for consistent testing
        mid_image = np.full((100, 100, 3), 128, dtype=np.uint8)

        # With inv_gamma = 1/gamma:
        # gamma > 1 -> inv_gamma < 1 -> brightens (values increase)
        result_bright = ImageTransformations.gamma_correction(mid_image, gamma=2.2)
        assert result_bright.mean() > mid_image.mean()

        # gamma < 1 -> inv_gamma > 1 -> darkens (values decrease)
        result_dark = ImageTransformations.gamma_correction(mid_image, gamma=0.45)
        assert result_dark.mean() < mid_image.mean()


class TestPreprocessingPipeline:
    """Tests for PreprocessingPipeline class."""

    @pytest.fixture
    def pipeline(self):
        """Create a default pipeline."""
        return PreprocessingPipeline()

    @pytest.fixture
    def sample_image(self):
        """Create a sample RGB image."""
        return np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

    def test_init_default(self):
        """Test default initialization."""
        pipeline = PreprocessingPipeline()
        assert pipeline.blur_kernel_size == 21
        assert pipeline.apply_blur is True
        assert pipeline.apply_equalization is True

    def test_init_custom(self):
        """Test custom initialization."""
        pipeline = PreprocessingPipeline(
            blur_kernel_size=7,
            gamma=1.5,
            apply_blur=False,
        )
        assert pipeline.blur_kernel_size == 7
        assert pipeline.gamma == 1.5
        assert pipeline.apply_blur is False

    def test_init_even_kernel_raises(self):
        """Test that even kernel size raises ValueError."""
        with pytest.raises(ValueError, match="odd"):
            PreprocessingPipeline(blur_kernel_size=4)

    def test_preprocess_numpy(self, pipeline, sample_image):
        """Test preprocessing numpy array."""
        result = pipeline.preprocess(sample_image)
        assert result.shape == sample_image.shape
        assert result.dtype == np.uint8

    def test_preprocess_pil(self, pipeline):
        """Test preprocessing PIL Image."""
        pil_image = Image.fromarray(np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8))
        result = pipeline.preprocess(pil_image)
        assert result.shape == (100, 100, 3)
        assert result.dtype == np.uint8

    def test_preprocess_tensor(self, pipeline):
        """Test preprocessing PyTorch tensor."""
        tensor = torch.rand(3, 100, 100)
        result = pipeline.preprocess(tensor)
        assert result.shape == (100, 100, 3)
        assert result.dtype == np.uint8

    def test_preprocess_batch(self, pipeline, sample_image):
        """Test batch preprocessing."""
        images = [sample_image, sample_image.copy()]
        results = pipeline.preprocess_batch(images)
        assert len(results) == 2
        for result in results:
            assert result.shape == sample_image.shape

    def test_apply_transformation_blur(self, pipeline, sample_image):
        """Test applying blur transformation."""
        result = pipeline.apply_transformation(sample_image, "blur")
        assert result.shape == sample_image.shape

    def test_apply_transformation_downscale(self, pipeline, sample_image):
        """Test applying downscale transformation."""
        result = pipeline.apply_transformation(sample_image, "downscale")
        assert result.shape == sample_image.shape

    def test_apply_transformation_grid(self, pipeline, sample_image):
        """Test applying grid transformation."""
        result = pipeline.apply_transformation(sample_image, "grid")
        assert result.shape == sample_image.shape

    def test_apply_transformation_gradient(self, pipeline, sample_image):
        """Test applying gradient transformation."""
        result = pipeline.apply_transformation(sample_image, "gradient")
        assert result.shape == sample_image.shape

    def test_apply_transformation_canny(self, pipeline, sample_image):
        """Test applying canny transformation."""
        result = pipeline.apply_transformation(sample_image, "canny")
        assert result.shape == sample_image.shape

    def test_apply_transformation_grayscale(self, pipeline, sample_image):
        """Test applying grayscale transformation."""
        result = pipeline.apply_transformation(sample_image, "grayscale")
        assert result.shape == sample_image.shape

    def test_apply_transformation_histogram(self, pipeline, sample_image):
        """Test applying histogram transformation."""
        result = pipeline.apply_transformation(sample_image, "histogram")
        assert result.shape == sample_image.shape

    def test_apply_transformation_gamma(self, pipeline, sample_image):
        """Test applying gamma transformation."""
        result = pipeline.apply_transformation(sample_image, "gamma")
        assert result.shape == sample_image.shape

    def test_apply_transformation_histogram_blur(self, pipeline, sample_image):
        """Test applying histogram_blur transformation."""
        result = pipeline.apply_transformation(sample_image, "histogram_blur")
        assert result.shape == sample_image.shape

    def test_apply_transformation_gamma_blur(self, pipeline, sample_image):
        """Test applying gamma_blur transformation."""
        result = pipeline.apply_transformation(sample_image, "gamma_blur")
        assert result.shape == sample_image.shape

    def test_apply_transformation_blur_gradient(self, pipeline, sample_image):
        """Test applying blur_gradient transformation."""
        result = pipeline.apply_transformation(sample_image, "blur_gradient")
        assert result.shape == sample_image.shape

    def test_apply_transformation_blur_histogram(self, pipeline, sample_image):
        """Test applying blur_histogram transformation."""
        result = pipeline.apply_transformation(sample_image, "blur_histogram")
        assert result.shape == sample_image.shape

    def test_apply_transformation_unknown_raises(self, pipeline, sample_image):
        """Test that unknown transformation raises ValueError."""
        with pytest.raises(ValueError, match="Unknown transformation"):
            pipeline.apply_transformation(sample_image, "unknown")

    def test_apply_all_transformations(self, pipeline, sample_image):
        """Test applying all transformations."""
        results = pipeline.apply_all_transformations(sample_image)

        assert "original" in results
        for name in PreprocessingPipeline.TRANSFORMATIONS:
            assert name in results
            assert results[name].shape == sample_image.shape

    def test_to_tensor(self, pipeline, sample_image):
        """Test converting to tensor."""
        tensor = pipeline.to_tensor(sample_image)
        assert tensor.shape == (3, 100, 100)
        assert tensor.dtype == torch.float32
        assert tensor.max() <= 1.0
        assert tensor.min() >= 0.0

    def test_get_config(self, pipeline):
        """Test getting configuration."""
        config = pipeline.get_config()
        assert "blur_kernel_size" in config
        assert "downscale_factor" in config
        assert "grid_size" in config
        assert "gamma" in config
        assert "apply_blur" in config
        assert "apply_equalization" in config

    def test_transformations_list(self):
        """Test that TRANSFORMATIONS list is complete."""
        expected = [
            "blur", "downscale", "grid", "gradient", "canny",
            "grayscale", "histogram", "gamma",
            "histogram_blur", "gamma_blur", "blur_gradient", "blur_histogram",
        ]
        assert PreprocessingPipeline.TRANSFORMATIONS == expected
