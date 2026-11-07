"""
Property-based tests for OCR pipeline.

**Feature: vlm-content-moderation, Property 5a: OCR text extraction**
**Validates: Requirements 1.6**
"""

from unittest.mock import patch

from hypothesis import given, settings
from hypothesis import strategies as st
import numpy as np

from utils.ocr import OCRPipeline


class TestOCRTextExtraction:
    """
    Property tests for OCR text extraction.

    **Feature: vlm-content-moderation, Property 5a: OCR text extraction**
    *For any* image processed for VLM training, OCR should extract and normalize text.
    **Validates: Requirements 1.6**
    """

    @given(
        st.text(
            min_size=0,
            max_size=100,
            alphabet=st.characters(whitelist_categories=("L", "N", "P", "Z")),
        )
    )
    @settings(max_examples=100)
    def test_normalize_text_idempotent(self, text):
        """
        Property: Normalizing already normalized text should be idempotent.

        *For any* text, normalize(normalize(text)) == normalize(text)
        """
        pipeline = OCRPipeline()

        normalized_once = pipeline.normalize_text(text)
        normalized_twice = pipeline.normalize_text(normalized_once)

        assert normalized_once == normalized_twice

    @given(
        st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=("L", "N")))
    )
    @settings(max_examples=100)
    def test_normalize_text_lowercase(self, text):
        """
        Property: Normalized text should be lowercase.

        *For any* alphanumeric text, normalize(text) should be lowercase.
        """
        pipeline = OCRPipeline()

        normalized = pipeline.normalize_text(text)

        # All alphabetic characters should be lowercase
        assert normalized == normalized.lower()

    @given(st.text(min_size=1, max_size=50))
    @settings(max_examples=100)
    def test_normalize_text_no_leading_trailing_whitespace(self, text):
        """
        Property: Normalized text should have no leading/trailing whitespace.

        *For any* text, normalize(text) should have no leading/trailing spaces.
        """
        pipeline = OCRPipeline()

        normalized = pipeline.normalize_text(text)

        assert normalized == normalized.strip()

    @given(st.integers(min_value=32, max_value=512), st.integers(min_value=32, max_value=512))
    @settings(max_examples=50)
    def test_extract_text_accepts_various_image_sizes(self, width, height):
        """
        Property: OCR should accept images of various sizes.

        *For any* valid image dimensions, extract_text should not raise.
        """
        pipeline = OCRPipeline()

        # Create a blank image
        img = np.zeros((height, width, 3), dtype=np.uint8)

        # Mock the reader to avoid actual OCR
        with patch.object(pipeline, "_reader") as mock_reader:
            mock_reader.readtext.return_value = []

            # Should not raise
            result = pipeline.extract_text(img)
            assert isinstance(result, str)

    @given(st.floats(min_value=0.0, max_value=1.0))
    @settings(max_examples=50)
    def test_confidence_threshold_filters_correctly(self, threshold):
        """
        Property: Results below confidence threshold should be filtered.

        *For any* confidence threshold, only results >= threshold should be included.
        """
        pipeline = OCRPipeline(confidence_threshold=threshold)

        # Create mock results with various confidences
        mock_results = [
            ([[0, 0], [10, 0], [10, 10], [0, 10]], "text1", 0.1),
            ([[0, 0], [10, 0], [10, 10], [0, 10]], "text2", 0.5),
            ([[0, 0], [10, 0], [10, 10], [0, 10]], "text3", 0.9),
        ]

        with patch.object(pipeline, "_reader") as mock_reader:
            mock_reader.readtext.return_value = mock_results

            img = np.zeros((100, 100, 3), dtype=np.uint8)
            result = pipeline.extract_text(img)

            # Check that only texts with confidence >= threshold are included
            if threshold <= 0.1:
                assert "text1" in result
            else:
                assert "text1" not in result

            if threshold <= 0.5:
                assert "text2" in result
            else:
                assert "text2" not in result

            if threshold <= 0.9:
                assert "text3" in result
            else:
                assert "text3" not in result

    @given(
        st.lists(
            st.text(min_size=1, max_size=20, alphabet="abcdefghijklmnopqrstuvwxyz"),
            min_size=0,
            max_size=5,
        )
    )
    @settings(max_examples=50)
    def test_extract_text_joins_multiple_detections(self, texts):
        """
        Property: Multiple text detections should be joined with spaces.

        *For any* list of detected texts, they should be joined into a single string.
        """
        pipeline = OCRPipeline()

        # Create mock results
        mock_results = [([[0, 0], [10, 0], [10, 10], [0, 10]], text, 0.9) for text in texts]

        with patch.object(pipeline, "_reader") as mock_reader:
            mock_reader.readtext.return_value = mock_results

            img = np.zeros((100, 100, 3), dtype=np.uint8)
            result = pipeline.extract_text(img)

            # All texts should be in the result
            for text in texts:
                assert text in result

    def test_extract_text_with_boxes_returns_valid_format(self):
        """
        Property: extract_text_with_boxes should return valid bbox format.

        *For any* detection, bbox should be (x, y, w, h) with non-negative values.
        """
        pipeline = OCRPipeline()

        mock_results = [
            ([[10, 20], [50, 20], [50, 60], [10, 60]], "test", 0.9),
        ]

        with patch.object(pipeline, "_reader") as mock_reader:
            mock_reader.readtext.return_value = mock_results

            img = np.zeros((100, 100, 3), dtype=np.uint8)
            results = pipeline.extract_text_with_boxes(img)

            for detection in results:
                assert "text" in detection
                assert "bbox" in detection
                assert "confidence" in detection

                x, y, w, h = detection["bbox"]
                assert x >= 0
                assert y >= 0
                assert w >= 0
                assert h >= 0
                assert 0 <= detection["confidence"] <= 1
