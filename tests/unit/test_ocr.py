"""Unit tests for OCRPipeline."""

from unittest.mock import MagicMock

import numpy as np
from PIL import Image
import pytest

from utils.ocr import OCRPipeline


class TestOCRPipeline:
    """Tests for OCRPipeline class."""

    def test_init_default_config(self):
        """Test default initialization."""
        pipeline = OCRPipeline()
        config = pipeline.get_config()

        assert config["languages"] == ["en"]
        assert config["gpu"] is False
        assert config["confidence_threshold"] == 0.3

    def test_init_custom_config(self):
        """Test custom initialization."""
        pipeline = OCRPipeline(
            languages=["en", "fr"],
            gpu=True,
            confidence_threshold=0.5,
        )
        config = pipeline.get_config()

        assert config["languages"] == ["en", "fr"]
        assert config["gpu"] is True
        assert config["confidence_threshold"] == 0.5

    def test_normalize_text_basic(self):
        """Test basic text normalization."""
        pipeline = OCRPipeline()

        # Test lowercase
        assert pipeline.normalize_text("HELLO WORLD") == "hello world"

        # Test whitespace normalization
        assert pipeline.normalize_text("hello   world") == "hello world"
        assert pipeline.normalize_text("  hello  ") == "hello"

        # Test empty string
        assert pipeline.normalize_text("") == ""
        assert pipeline.normalize_text(None) == ""

    def test_normalize_text_special_chars(self):
        """Test normalization removes special characters."""
        pipeline = OCRPipeline()

        # Keep basic punctuation
        assert "hello" in pipeline.normalize_text("hello!")
        assert "world" in pipeline.normalize_text("hello, world")

        # Remove special unicode
        result = pipeline.normalize_text("hello™ world®")
        assert "™" not in result
        assert "®" not in result

    def test_normalize_text_unicode(self):
        """Test Unicode normalization."""
        pipeline = OCRPipeline()

        # NFKC normalization converts full-width to ASCII
        result = pipeline.normalize_text("ＨＥＬＬＯ")
        assert result == "hello"

    def test_extract_text_filters_by_confidence(self):
        """Test that low confidence results are filtered."""
        pipeline = OCRPipeline(confidence_threshold=0.5)

        mock_reader = MagicMock()
        mock_reader.readtext.return_value = [
            ([[0, 0], [10, 0], [10, 10], [0, 10]], "high", 0.9),
            ([[0, 0], [10, 0], [10, 10], [0, 10]], "low", 0.1),
        ]
        pipeline._reader = mock_reader

        img = np.zeros((100, 100, 3), dtype=np.uint8)
        result = pipeline.extract_text(img)

        assert "high" in result
        assert "low" not in result

    def test_extract_text_with_boxes(self):
        """Test extraction with bounding boxes."""
        pipeline = OCRPipeline()

        mock_reader = MagicMock()
        mock_reader.readtext.return_value = [
            ([[10, 20], [50, 20], [50, 40], [10, 40]], "test", 0.95),
        ]
        pipeline._reader = mock_reader

        img = np.zeros((100, 100, 3), dtype=np.uint8)
        results = pipeline.extract_text_with_boxes(img)

        assert len(results) == 1
        assert results[0]["text"] == "test"
        assert results[0]["confidence"] == 0.95
        assert results[0]["bbox"] == (10, 20, 40, 20)  # x, y, w, h

    def test_extract_and_normalize(self):
        """Test combined extraction and normalization."""
        pipeline = OCRPipeline()

        mock_reader = MagicMock()
        mock_reader.readtext.return_value = [
            ([[0, 0], [10, 0], [10, 10], [0, 10]], "HELLO", 0.9),
            ([[0, 0], [10, 0], [10, 10], [0, 10]], "WORLD", 0.9),
        ]
        pipeline._reader = mock_reader

        img = np.zeros((100, 100, 3), dtype=np.uint8)
        result = pipeline.extract_and_normalize(img)

        assert result == "hello world"

    def test_to_numpy_from_numpy(self):
        """Test conversion from numpy array."""
        pipeline = OCRPipeline()

        img = np.zeros((100, 100, 3), dtype=np.uint8)
        result = pipeline._to_numpy(img)

        assert result.shape == (100, 100, 3)
        assert result.dtype == np.uint8

    def test_to_numpy_from_pil(self):
        """Test conversion from PIL Image."""
        pipeline = OCRPipeline()

        img = Image.new("RGB", (100, 100), color=(128, 128, 128))
        result = pipeline._to_numpy(img)

        assert result.shape == (100, 100, 3)
        assert result.dtype == np.uint8

    def test_to_numpy_from_tensor(self):
        """Test conversion from PyTorch tensor."""
        import torch

        pipeline = OCRPipeline()

        # CHW format tensor
        tensor = torch.zeros((3, 100, 100))
        result = pipeline._to_numpy(tensor)

        assert result.shape == (100, 100, 3)
        assert result.dtype == np.uint8

    def test_to_numpy_invalid_type(self):
        """Test that invalid types raise TypeError."""
        pipeline = OCRPipeline()

        with pytest.raises(TypeError):
            pipeline._to_numpy("invalid")

    def test_empty_image_returns_empty_string(self):
        """Test that empty image returns empty string."""
        pipeline = OCRPipeline()

        mock_reader = MagicMock()
        mock_reader.readtext.return_value = []
        pipeline._reader = mock_reader

        img = np.zeros((100, 100, 3), dtype=np.uint8)
        result = pipeline.extract_text(img)

        assert result == ""
