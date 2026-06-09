"""Unit tests for DatasetManager and HatefulIllusionDataset."""

from unittest.mock import MagicMock, patch

from utils.dataset import Annotation, DatasetManager, HatefulIllusionDataset


class TestAnnotation:
    """Tests for the Annotation dataclass."""

    def test_is_hate_always_true(self):
        """Test is_hate returns True for all annotations."""
        ann = Annotation(image_id="0", message="5", prompt="test", visibility=3)
        assert ann.is_hate is True

    def test_message_type_textual_for_digits(self):
        """Test message_type returns textual for digit messages."""
        ann = Annotation(image_id="0", message="5", prompt="test", visibility=3)
        assert ann.message_type == "textual"

    def test_message_type_symbolic_for_non_digits(self):
        """Test message_type returns symbolic for non-digit messages."""
        ann = Annotation(image_id="0", message="symbol", prompt="test", visibility=3)
        assert ann.message_type == "symbolic"

    def test_visibility_level_low(self):
        """Test visibility_level returns low for scores 1-2."""
        ann1 = Annotation(image_id="0", message="5", prompt="test", visibility=1)
        ann2 = Annotation(image_id="1", message="5", prompt="test", visibility=2)
        assert ann1.visibility_level == "low"
        assert ann2.visibility_level == "low"

    def test_visibility_level_high(self):
        """Test visibility_level returns high for scores 3+."""
        ann3 = Annotation(image_id="0", message="5", prompt="test", visibility=3)
        ann4 = Annotation(image_id="1", message="5", prompt="test", visibility=4)
        ann5 = Annotation(image_id="2", message="5", prompt="test", visibility=5)
        assert ann3.visibility_level == "high"
        assert ann4.visibility_level == "high"
        assert ann5.visibility_level == "high"


def create_mock_hf_dataset(num_samples=10):
    """Create a mock HuggingFace dataset."""
    mock_data = [
        {
            "message": str(i % 10),
            "prompt": f"Scene {i}",
            "visibility": (i % 5) + 1,
            "image": f"img{i}.png",
        }
        for i in range(num_samples)
    ]

    mock_dataset = MagicMock()
    mock_dataset.__iter__ = lambda _: iter(mock_data)
    mock_dataset.__len__ = lambda _: len(mock_data)
    mock_dataset.__getitem__ = lambda _, idx: mock_data[idx]

    return {"train": mock_dataset}


class TestHatefulIllusionDataset:
    """Tests for HatefulIllusionDataset."""

    @patch("utils.dataset._hf_load_dataset")
    def test_load_dataset(self, mock_load):
        """Test loading dataset."""
        mock_load.return_value = create_mock_hf_dataset(5)

        dataset = HatefulIllusionDataset(split="train")

        assert len(dataset) == 5
        assert len(dataset.annotations) == 5

    @patch("utils.dataset.hf_hub_download")
    @patch("utils.dataset._hf_load_dataset")
    def test_getitem_returns_dict(self, mock_load, mock_hf_download):
        """Test __getitem__ returns expected dictionary structure."""
        import tempfile

        from PIL import Image

        mock_load.return_value = create_mock_hf_dataset(5)

        # Create a temporary test image
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            img = Image.new("RGB", (256, 256), color=(128, 128, 128))
            img.save(f.name)
            mock_hf_download.return_value = f.name

        dataset = HatefulIllusionDataset(split="train")
        sample = dataset[0]

        assert "image" in sample
        assert "image_id" in sample
        assert "message" in sample
        assert "is_hate" in sample
        assert "message_type" in sample
        assert "visibility_level" in sample
        assert "visibility_score" in sample
        assert "prompt" in sample

    @patch("utils.dataset.hf_hub_download")
    @patch("utils.dataset._hf_load_dataset")
    def test_image_tensor_shape(self, mock_load, mock_hf_download):
        """Test image is converted to proper tensor shape."""
        import tempfile

        from PIL import Image

        mock_load.return_value = create_mock_hf_dataset(5)

        # Create a temporary test image
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            img = Image.new("RGB", (256, 256), color=(128, 128, 128))
            img.save(f.name)
            mock_hf_download.return_value = f.name

        dataset = HatefulIllusionDataset(split="train")
        sample = dataset[0]

        assert len(sample["image"].shape) == 3
        assert sample["image"].shape[0] == 3  # RGB channels

    @patch("utils.dataset.hf_hub_download")
    @patch("utils.dataset._hf_load_dataset")
    def test_getitem_uses_subset_specific_image_path(self, mock_load, mock_hf_download):
        """Test __getitem__ downloads a subset-specific image path."""
        import tempfile

        from PIL import Image

        mock_data = [
            {
                "message": "5",
                "prompt": "Scene 0",
                "visibility": 3,
                "image": "images/0.png",
            }
        ]

        mock_dataset = MagicMock()
        mock_dataset.__iter__ = lambda _: iter(mock_data)
        mock_dataset.__len__ = lambda _: len(mock_data)
        mock_dataset.__getitem__ = lambda _, idx: mock_data[idx]
        mock_load.return_value = {"train": mock_dataset}

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            img = Image.new("RGB", (128, 128), color=(128, 128, 128))
            img.save(f.name)
            mock_hf_download.return_value = f.name

        dataset = HatefulIllusionDataset(split="train", subset="hate_slangs")
        _ = dataset[0]

        mock_hf_download.assert_called_once_with(
            repo_id="yiting/HatefulIllusion_Dataset",
            filename="hate_slangs/images/0.png",
            repo_type="dataset",
            cache_dir=None,
        )


class TestDatasetManager:
    """Tests for DatasetManager."""

    @patch("utils.dataset._hf_load_dataset")
    def test_load_dataset(self, mock_load):
        """Test loading dataset through manager."""
        mock_load.return_value = create_mock_hf_dataset(10)

        manager = DatasetManager()
        dataset = manager.load_dataset(split="train")

        assert len(dataset) == 10

    @patch("utils.dataset._hf_load_dataset")
    def test_dataset_caching(self, mock_load):
        """Test that datasets are cached."""
        mock_load.return_value = create_mock_hf_dataset(10)

        manager = DatasetManager()
        dataset1 = manager.load_dataset(split="train")
        dataset2 = manager.load_dataset(split="train")

        assert dataset1 is dataset2

    @patch("utils.dataset._hf_load_dataset")
    def test_get_dataset_stats(self, mock_load):
        """Test get_dataset_stats returns expected structure."""
        mock_load.return_value = create_mock_hf_dataset(10)

        manager = DatasetManager()
        stats = manager.get_dataset_stats(split="train")

        assert "total_images" in stats
        assert "high_visibility" in stats
        assert "low_visibility" in stats
        assert "textual_count" in stats
        assert "symbolic_count" in stats

    @patch("utils.dataset._hf_load_dataset")
    def test_check_composition_completeness(self, mock_load):
        """Test composition completeness check."""
        mock_load.return_value = create_mock_hf_dataset(10)

        manager = DatasetManager()
        composition = manager.check_composition_completeness(split="train")

        assert "high_visibility_textual" in composition
        assert "high_visibility_symbolic" in composition
        assert "low_visibility_textual" in composition
        assert "low_visibility_symbolic" in composition

    @patch("utils.dataset._hf_load_dataset")
    def test_supports_minimum_size(self, mock_load):
        """Test minimum size check."""
        mock_load.return_value = create_mock_hf_dataset(10)

        manager = DatasetManager()

        assert manager.supports_minimum_size(min_size=10) is True
        assert manager.supports_minimum_size(min_size=100) is False
