"""Unit tests for DatasetManager and MMHS150KDataset."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from utils.dataset import Annotation, DatasetManager, MMHS150KDataset


class TestAnnotation:
    """Tests for the Annotation dataclass."""
    
    def test_majority_label_unanimous(self):
        """Test majority label with unanimous agreement."""
        ann = Annotation(
            image_id="123",
            labels=[1, 1, 1],
            labels_str=["Racist", "Racist", "Racist"],
            tweet_text="test"
        )
        assert ann.majority_label == 1
    
    def test_majority_label_split(self):
        """Test majority label with 2-1 split."""
        ann = Annotation(
            image_id="123",
            labels=[0, 1, 1],
            labels_str=["NotHate", "Racist", "Racist"],
            tweet_text="test"
        )
        assert ann.majority_label == 1
    
    def test_is_hate_true(self):
        """Test is_hate returns True for hate labels."""
        ann = Annotation(
            image_id="123",
            labels=[1, 1, 2],
            labels_str=["Racist", "Racist", "Sexist"],
            tweet_text="test"
        )
        assert ann.is_hate is True
    
    def test_is_hate_false(self):
        """Test is_hate returns False for non-hate labels."""
        ann = Annotation(
            image_id="123",
            labels=[0, 0, 0],
            labels_str=["NotHate", "NotHate", "NotHate"],
            tweet_text="test"
        )
        assert ann.is_hate is False
    
    def test_message_type_textual(self):
        """Test message_type for textual hate (Racist, Sexist)."""
        ann = Annotation(
            image_id="123",
            labels=[1, 1, 1],
            labels_str=["Racist", "Racist", "Racist"],
            tweet_text="test"
        )
        assert ann.message_type == "textual"
    
    def test_message_type_symbolic(self):
        """Test message_type for symbolic hate (Homophobe, Religion, OtherHate)."""
        ann = Annotation(
            image_id="123",
            labels=[3, 3, 4],
            labels_str=["Homophobe", "Homophobe", "Religion"],
            tweet_text="test"
        )
        assert ann.message_type == "symbolic"


@pytest.fixture
def mock_dataset_dir():
    """Create a mock MMHS150K dataset directory structure."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create directory structure
        (tmpdir / "img_resized").mkdir()
        (tmpdir / "img_txt").mkdir()
        (tmpdir / "splits").mkdir()
        
        # Create sample images
        for i in range(10):
            img = Image.new("RGB", (100, 100), color=(i * 25, i * 25, i * 25))
            img.save(tmpdir / "img_resized" / f"{i}.jpg")
        
        # Create ground truth JSON
        gt_data = {}
        for i in range(10):
            # Vary labels to test different scenarios
            if i < 3:
                labels = [0, 0, 0]  # NotHate
            elif i < 6:
                labels = [1, 1, 2]  # Textual hate
            else:
                labels = [3, 4, 3]  # Symbolic hate
            
            gt_data[str(i)] = {
                "tweet_url": f"http://example.com/{i}",
                "labels": labels,
                "img_url": f"http://example.com/img/{i}",
                "tweet_text": f"Sample tweet {i}",
                "labels_str": ["NotHate" if l == 0 else "Hate" for l in labels]
            }
        
        with open(tmpdir / "MMHS150K_GT.json", "w") as f:
            json.dump(gt_data, f)
        
        # Create split files
        train_ids = [str(i) for i in range(7)]
        val_ids = [str(i) for i in range(7, 9)]
        test_ids = ["9"]
        
        (tmpdir / "splits" / "train_ids.txt").write_text("\n".join(train_ids))
        (tmpdir / "splits" / "val_ids.txt").write_text("\n".join(val_ids))
        (tmpdir / "splits" / "test_ids.txt").write_text("\n".join(test_ids))
        
        # Create OCR text files
        for i in range(10):
            (tmpdir / "img_txt" / f"{i}.txt").write_text(f"OCR text for image {i}")
        
        yield tmpdir


class TestMMHS150KDataset:
    """Tests for MMHS150KDataset."""
    
    def test_load_train_split(self, mock_dataset_dir):
        """Test loading train split."""
        dataset = MMHS150KDataset(str(mock_dataset_dir), split="train")
        assert len(dataset) == 7
    
    def test_load_val_split(self, mock_dataset_dir):
        """Test loading validation split."""
        dataset = MMHS150KDataset(str(mock_dataset_dir), split="val")
        assert len(dataset) == 2
    
    def test_load_test_split(self, mock_dataset_dir):
        """Test loading test split."""
        dataset = MMHS150KDataset(str(mock_dataset_dir), split="test")
        assert len(dataset) == 1
    
    def test_getitem_returns_dict(self, mock_dataset_dir):
        """Test __getitem__ returns expected dictionary structure."""
        dataset = MMHS150KDataset(str(mock_dataset_dir), split="train")
        sample = dataset[0]
        
        assert "image" in sample
        assert "image_id" in sample
        assert "label" in sample
        assert "is_hate" in sample
        assert "message_type" in sample
        assert "visibility_level" in sample
        assert "tweet_text" in sample
        assert "ocr_text" in sample
        assert "annotator_labels" in sample
    
    def test_image_tensor_shape(self, mock_dataset_dir):
        """Test image is converted to proper tensor shape."""
        dataset = MMHS150KDataset(str(mock_dataset_dir), split="train")
        sample = dataset[0]
        
        # Should be (C, H, W) format
        assert len(sample["image"].shape) == 3
        assert sample["image"].shape[0] == 3  # RGB channels
    
    def test_ocr_text_loaded(self, mock_dataset_dir):
        """Test OCR text is loaded when include_ocr=True."""
        dataset = MMHS150KDataset(str(mock_dataset_dir), split="train", include_ocr=True)
        sample = dataset[0]
        
        assert sample["ocr_text"] != ""
        assert "OCR text" in sample["ocr_text"]


class TestDatasetManager:
    """Tests for DatasetManager."""
    
    def test_load_dataset(self, mock_dataset_dir):
        """Test loading dataset through manager."""
        manager = DatasetManager(str(mock_dataset_dir))
        dataset = manager.load_dataset(split="train")
        
        assert len(dataset) == 7
    
    def test_dataset_caching(self, mock_dataset_dir):
        """Test that datasets are cached."""
        manager = DatasetManager(str(mock_dataset_dir))
        
        dataset1 = manager.load_dataset(split="train")
        dataset2 = manager.load_dataset(split="train")
        
        assert dataset1 is dataset2
    
    def test_validate_annotations_returns_float(self, mock_dataset_dir):
        """Test validate_annotations returns a float."""
        manager = DatasetManager(str(mock_dataset_dir))
        kappa = manager.validate_annotations(split="train")
        
        assert isinstance(kappa, float)
        assert -1.0 <= kappa <= 1.0
    
    def test_get_dataset_stats(self, mock_dataset_dir):
        """Test get_dataset_stats returns expected structure."""
        manager = DatasetManager(str(mock_dataset_dir))
        stats = manager.get_dataset_stats(split="train")
        
        assert "total_images" in stats
        assert "hate_images" in stats
        assert "non_hate_images" in stats
        assert "textual_count" in stats
        assert "symbolic_count" in stats
        assert "label_distribution" in stats
        assert "fleiss_kappa" in stats
    
    def test_check_composition_completeness(self, mock_dataset_dir):
        """Test composition completeness check."""
        manager = DatasetManager(str(mock_dataset_dir))
        composition = manager.check_composition_completeness(split="train")
        
        assert "high_visibility_textual" in composition
        assert "high_visibility_symbolic" in composition
        assert "low_visibility_textual" in composition
        assert "low_visibility_symbolic" in composition
    
    def test_supports_minimum_size(self, mock_dataset_dir):
        """Test minimum size check."""
        manager = DatasetManager(str(mock_dataset_dir))
        
        # Our mock has 10 images total
        assert manager.supports_minimum_size(min_size=10) is True
        assert manager.supports_minimum_size(min_size=100) is False


class TestFleissKappa:
    """Tests for Fleiss Kappa calculation."""
    
    def test_perfect_agreement(self, mock_dataset_dir):
        """Test Fleiss Kappa with perfect agreement."""
        manager = DatasetManager(str(mock_dataset_dir))
        
        # Create annotations with perfect agreement
        annotations = [
            Annotation("1", [0, 0, 0], ["NotHate"] * 3, ""),
            Annotation("2", [0, 0, 0], ["NotHate"] * 3, ""),
            Annotation("3", [0, 0, 0], ["NotHate"] * 3, ""),
        ]
        
        kappa = manager._calculate_fleiss_kappa(annotations)
        assert kappa == 1.0
    
    def test_no_agreement(self, mock_dataset_dir):
        """Test Fleiss Kappa with no agreement beyond chance."""
        manager = DatasetManager(str(mock_dataset_dir))
        
        # Create annotations with varied disagreement
        annotations = [
            Annotation("1", [0, 1, 2], ["NotHate", "Racist", "Sexist"], ""),
            Annotation("2", [3, 4, 5], ["Homophobe", "Religion", "OtherHate"], ""),
            Annotation("3", [0, 2, 4], ["NotHate", "Sexist", "Religion"], ""),
        ]
        
        kappa = manager._calculate_fleiss_kappa(annotations)
        # Should be low or negative
        assert kappa < 0.5
    
    def test_empty_annotations(self, mock_dataset_dir):
        """Test Fleiss Kappa with empty annotations."""
        manager = DatasetManager(str(mock_dataset_dir))
        kappa = manager._calculate_fleiss_kappa([])
        assert kappa == 0.0
