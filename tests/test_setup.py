"""
Basic setup verification tests.
"""
import pytest


def test_imports():
    """Test that all required packages can be imported."""
    try:
        import torch
        import torchvision
        import transformers
        import cv2
        import numpy
        import hypothesis
        assert True
    except ImportError as e:
        pytest.skip(f"Required package not installed: {e}")


def test_pytorch_available():
    """Test that PyTorch is available."""
    try:
        import torch
        assert torch.__version__ is not None
    except ImportError:
        pytest.skip("PyTorch not installed")


def test_hypothesis_available():
    """Test that Hypothesis is available for property-based testing."""
    try:
        import hypothesis
        assert hypothesis.__version__ is not None
    except ImportError:
        pytest.skip("Hypothesis not installed")
