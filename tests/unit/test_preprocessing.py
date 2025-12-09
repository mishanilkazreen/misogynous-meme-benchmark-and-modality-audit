"""Unit tests for PreprocessingPipeline."""

import numpy as np
import torch
from PIL import Image

from utils.preprocessing import PreprocessingPipeline


class TestPreprocessingPipeline:
    """Tests for PreprocessingPipeline class."""

    def test_preprocess_numpy_array(self):
        """Test preprocessing with numpy array input."""
        pipeline = PreprocessingPipeline()
        image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        
        result = pipeline.preprocess(image)
        
        assert result.shape == image.shape
        assert result.dtype == np.uint8

    def test_preprocess_pil_image(self):
        """Test preprocessing with PIL Image input."""
        pipeline = PreprocessingPipeline()
        image = Image.new("RGB", (100, 100), color=(128, 128, 128))
        
        result = pipeline.preprocess(image)
        
        assert result.shape == (100, 100, 3)
        assert result.dtype == np.uint8

    def test_preprocess_torch_tensor(self):
        """Test preprocessing with PyTorch tensor input."""
        pipeline = PreprocessingPipeline()
        # (C, H, W) format, normalized to [0, 1]
        tensor = torch.rand(3, 100, 100)
        
        result = pipeline.preprocess(tensor)
        
        assert result.shape == (100, 100, 3)
        assert result.dtype == np.uint8

    def test_blur_only(self):
        """Test with only blur enabled."""
        pipeline = PreprocessingPipeline(
            apply_blur=True,
            apply_equalization=False,
        )
        image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        
        result = pipeline.preprocess(image)
        
        assert result.shape == image.shape
        # Blurred image should be smoother (lower variance)
        assert np.var(result) <= np.var(image)

    def test_equalization_only(self):
        """Test with only equalization enabled."""
        pipeline = PreprocessingPipeline(
            apply_blur=False,
            apply_equalization=True,
        )
        # Low contrast image
        image = np.random.randint(100, 150, (100, 100, 3), dtype=np.uint8)
        
        result = pipeline.preprocess(image)
        
        assert result.shape == image.shape
        # Equalized image should have wider range
        assert (result.max() - result.min()) >= (image.max() - image.min())

    def test_batch_preprocessing(self):
        """Test batch preprocessing."""
        pipeline = PreprocessingPipeline()
        images = [
            np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
            for _ in range(5)
        ]
        
        results = pipeline.preprocess_batch(images)
        
        assert len(results) == 5
        for result in results:
            assert result.shape == (100, 100, 3)

    def test_to_tensor_conversion(self):
        """Test conversion to PyTorch tensor."""
        pipeline = PreprocessingPipeline()
        image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        
        preprocessed = pipeline.preprocess(image)
        tensor = pipeline.to_tensor(preprocessed)
        
        assert tensor.shape == (3, 100, 100)
        assert tensor.dtype == torch.float32
        assert tensor.min() >= 0.0
        assert tensor.max() <= 1.0

    def test_different_kernel_sizes(self):
        """Test with different blur kernel sizes."""
        image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        
        for kernel_size in [3, 5, 7, 9]:
            pipeline = PreprocessingPipeline(blur_kernel_size=kernel_size)
            result = pipeline.preprocess(image)
            assert result.shape == image.shape

    def test_grayscale_image(self):
        """Test preprocessing grayscale image."""
        pipeline = PreprocessingPipeline()
        image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        
        result = pipeline.preprocess(image)
        
        assert result.shape == image.shape
