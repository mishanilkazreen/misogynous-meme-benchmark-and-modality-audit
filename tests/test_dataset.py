"""
Tests for the DatasetManager class.
"""
import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

from utils.dataset import (
    DatasetManager,
    ContentModerationDataset,
    Annotation,
    download_mmhs150k_dataset,
)


class TestDatasetManager:
    """Tests for DatasetManager class."""
    
    def test_init_default(self):
        """Test default initialization."""
        manager = DatasetManager()
        assert manager.data_dir is None
        assert manager._images == []
        assert manager._annotations == {}
    
    def test_init_with_path(self):
        """Test initialization with custom path."""
        manager = DatasetManager(data_dir="/tmp/test")
        assert manager.data_dir == Path("/tmp/test")
    
    def test_min_dataset_size_constant(self):
        """Test minimum dataset size is 5000 per requirements."""
        assert DatasetManager.MIN_DATASET_SIZE == 5000
    
    def test_min_fleiss_kappa_constant(self):
        """Test minimum Fleiss Kappa is 0.783 per requirements."""
        assert DatasetManager.MIN_FLEISS_KAPPA == 0.783
    
    def test_load_dataset_path_not_found(self):
        """Test loading from non-existent path raises error."""
        manager = DatasetManager()
        with pytest.raises(FileNotFoundError):
            manager.load_dataset("/nonexistent/path")
    
    def test_load_dataset_too_small(self):
        """Test loading dataset with fewer than 5000 images raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a small dataset (less than 5000 images)
            img_dir = Path(tmpdir) / "images"
            img_dir.mkdir()
            
            # Create only 10 dummy images
            for i in range(10):
                (img_dir / f"img_{i}.jpg").touch()
            
            manager = DatasetManager()
            with pytest.raises(ValueError, match="Dataset too small"):
                manager.load_dataset(tmpdir)


class TestFleissKappa:
    """Tests for Fleiss Kappa calculation."""
    
    def test_perfect_agreement(self):
        """Test Fleiss Kappa with perfect agreement."""
        manager = DatasetManager()
        # All 3 raters agree on all 5 subjects
        annotations = [
            [3, 0],  # All 3 raters chose category 0
            [3, 0],
            [0, 3],  # All 3 raters chose category 1
            [3, 0],
            [0, 3],
        ]
        kappa = manager.validate_annotations(annotations)
        assert kappa == pytest.approx(1.0, abs=0.01)
    
    def test_no_agreement(self):
        """Test Fleiss Kappa with random agreement."""
        manager = DatasetManager()
        # Raters split evenly - no agreement beyond chance
        # With 4 raters split 2-2, kappa is negative (worse than chance)
        annotations = [
            [2, 2],  # 2 raters chose 0, 2 chose 1
            [2, 2],
            [2, 2],
            [2, 2],
        ]
        kappa = manager.validate_annotations(annotations)
        # With even split, kappa is -1/3 (worse than chance)
        assert kappa == pytest.approx(-0.333, abs=0.1)
    
    def test_single_rater_returns_one(self):
        """Test that single rater annotations return 1.0."""
        manager = DatasetManager()
        # Single rater
        annotations = [
            [1, 0],
            [0, 1],
            [1, 0],
        ]
        kappa = manager.validate_annotations(annotations)
        assert kappa == 1.0
    
    def test_no_annotations_returns_one(self):
        """Test that no annotations returns 1.0."""
        manager = DatasetManager()
        kappa = manager.validate_annotations(None)
        assert kappa == 1.0


class TestAnnotation:
    """Tests for Annotation dataclass."""
    
    def test_annotation_creation(self):
        """Test creating an Annotation."""
        ann = Annotation(
            image_id="test_001",
            label=1,
            message_type="textual",
            visibility_level="high",
        )
        assert ann.image_id == "test_001"
        assert ann.label == 1
        assert ann.message_type == "textual"
        assert ann.visibility_level == "high"
    
    def test_annotation_defaults(self):
        """Test Annotation default values."""
        ann = Annotation(image_id="test", label=0)
        assert ann.message_type is None
        assert ann.visibility_level is None
        assert ann.bbox is None
        assert ann.text is None


class TestContentModerationDataset:
    """Tests for ContentModerationDataset class."""
    
    def test_dataset_length(self):
        """Test dataset length."""
        images = ["img1.jpg", "img2.jpg", "img3.jpg"]
        annotations = {
            "img1.jpg": Annotation(image_id="1", label=0),
            "img2.jpg": Annotation(image_id="2", label=1),
            "img3.jpg": Annotation(image_id="3", label=0),
        }
        dataset = ContentModerationDataset(
            images=images,
            annotations=annotations,
            data_dir=Path("/tmp"),
        )
        assert len(dataset) == 3
    
    def test_class_distribution(self):
        """Test class distribution calculation."""
        images = ["img1.jpg", "img2.jpg", "img3.jpg", "img4.jpg"]
        annotations = {
            "img1.jpg": Annotation(image_id="1", label=0),
            "img2.jpg": Annotation(image_id="2", label=1),
            "img3.jpg": Annotation(image_id="3", label=0),
            "img4.jpg": Annotation(image_id="4", label=1),
        }
        dataset = ContentModerationDataset(
            images=images,
            annotations=annotations,
            data_dir=Path("/tmp"),
        )
        dist = dataset.get_class_distribution()
        assert dist[0] == 2
        assert dist[1] == 2


class TestDatasetStats:
    """Tests for dataset statistics."""
    
    def test_stats_no_dataset(self):
        """Test stats when no dataset is loaded."""
        manager = DatasetManager()
        stats = manager.get_dataset_stats()
        assert "error" in stats
    
    def test_stats_with_dataset(self):
        """Test stats with loaded dataset."""
        manager = DatasetManager()
        manager._images = ["img1.jpg", "img2.jpg", "img3.jpg"]
        manager._annotations = {
            "img1.jpg": Annotation(image_id="1", label=0),
            "img2.jpg": Annotation(image_id="2", label=1),
            "img3.jpg": Annotation(image_id="3", label=1),
        }
        
        stats = manager.get_dataset_stats()
        assert stats["total_images"] == 3
        assert stats["hateful_images"] == 2
        assert stats["non_hateful_images"] == 1
        assert stats["meets_minimum_size"] is False  # 3 < 5000
