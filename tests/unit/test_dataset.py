"""Unit tests for DatasetManager and MamiDataset (MAMI 2022)."""

from __future__ import annotations

import csv
from pathlib import Path
from unittest.mock import patch

from PIL import Image
import pytest

from utils.dataset import DatasetManager, MamiDataset

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fake_dataset_root(
    tmp_path: Path,
    split: str = "train",
    num_samples: int = 5,
    include_images: bool = True,
) -> str:
    """Create a minimal MAMI-like directory structure for offline testing."""
    tsv_name = f"{split}.tsv"
    img_subdir = "training_images" if split in ("train", "validation") else "test_images"

    images_dir = tmp_path / "MAMI_2022_images" / img_subdir
    images_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for i in range(num_samples):
        file_name = f"{1000 + i}.jpg"
        rows.append(
            {
                "file_name": file_name,
                "label": str(i % 2),
                "shaming": str(i % 2),
                "stereotype": str((i + 1) % 2),
                "objectification": "0",
                "violence": "0",
                "text": f"Sample meme text {i}",
            }
        )
        if include_images:
            img = Image.new("RGB", (64, 64), color=(i * 20, 100, 150))
            img.save(images_dir / file_name)

    tsv_path = tmp_path / tsv_name
    with tsv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    return str(tmp_path)


# ---------------------------------------------------------------------------
# MamiDataset
# ---------------------------------------------------------------------------


class TestMamiDataset:
    """Tests for MamiDataset."""

    def test_len(self, tmp_path: Path) -> None:
        """Dataset length matches the number of TSV rows."""
        root = _make_fake_dataset_root(tmp_path, num_samples=7)
        ds = MamiDataset(dataset_path=root, split="train")
        assert len(ds) == 7

    def test_getitem_keys(self, tmp_path: Path) -> None:
        """__getitem__ returns the required keys."""
        root = _make_fake_dataset_root(tmp_path, num_samples=3)
        ds = MamiDataset(dataset_path=root, split="train")
        sample = ds[0]
        assert "image" in sample
        assert "image_id" in sample
        assert "text" in sample
        assert "misogynous" in sample
        assert "shaming" in sample
        assert "stereotype" in sample
        assert "objectification" in sample
        assert "violence" in sample

    def test_image_tensor_shape(self, tmp_path: Path) -> None:
        """Image is returned as a (3, H, W) float tensor."""
        root = _make_fake_dataset_root(tmp_path, num_samples=3)
        ds = MamiDataset(dataset_path=root, split="train")
        sample = ds[0]
        img = sample["image"]
        assert len(img.shape) == 3
        assert img.shape[0] == 3  # RGB

    def test_image_values_normalized(self, tmp_path: Path) -> None:
        """Pixel values are in [0, 1]."""
        root = _make_fake_dataset_root(tmp_path, num_samples=3)
        ds = MamiDataset(dataset_path=root, split="train")
        sample = ds[0]
        assert float(sample["image"].min()) >= 0.0
        assert float(sample["image"].max()) <= 1.0

    def test_label_types(self, tmp_path: Path) -> None:
        """All label fields are Python ints."""
        root = _make_fake_dataset_root(tmp_path, num_samples=4)
        ds = MamiDataset(dataset_path=root, split="train")
        sample = ds[0]
        for key in ("misogynous", "shaming", "stereotype", "objectification", "violence"):
            assert isinstance(sample[key], int), f"{key} should be int"

    def test_text_is_string(self, tmp_path: Path) -> None:
        """Text field is a string."""
        root = _make_fake_dataset_root(tmp_path, num_samples=3)
        ds = MamiDataset(dataset_path=root, split="train")
        assert isinstance(ds[0]["text"], str)

    def test_image_id_is_stem(self, tmp_path: Path) -> None:
        """image_id is the filename stem (no extension)."""
        root = _make_fake_dataset_root(tmp_path, num_samples=3)
        ds = MamiDataset(dataset_path=root, split="train")
        sample = ds[0]
        assert "." not in sample["image_id"], "image_id should not contain file extension"

    def test_transform_applied(self, tmp_path: Path) -> None:
        """Custom transform is called with a PIL image."""
        import torch

        root = _make_fake_dataset_root(tmp_path, num_samples=3)

        def to_ones(pil: Image.Image) -> torch.Tensor:
            return torch.ones(3, 8, 8)

        ds = MamiDataset(dataset_path=root, split="train", transform=to_ones)
        sample = ds[0]
        assert sample["image"].shape == (3, 8, 8)
        assert float(sample["image"].sum()) == 3 * 8 * 8

    def test_invalid_split_raises(self, tmp_path: Path) -> None:
        """Passing an unknown split raises ValueError."""
        with pytest.raises(ValueError, match="Invalid split"):
            MamiDataset(dataset_path=str(tmp_path), split="bogus")

    def test_missing_tsv_raises(self, tmp_path: Path) -> None:
        """FileNotFoundError raised when TSV is absent."""
        with pytest.raises(FileNotFoundError):
            MamiDataset(dataset_path=str(tmp_path), split="train")

    def test_validation_split(self, tmp_path: Path) -> None:
        """Validation split loads correctly (uses training_images dir)."""
        root = _make_fake_dataset_root(tmp_path, split="validation", num_samples=4)
        ds = MamiDataset(dataset_path=root, split="validation")
        assert len(ds) == 4

    def test_test_split(self, tmp_path: Path) -> None:
        """Test split loads correctly (uses test_images dir)."""
        root = _make_fake_dataset_root(tmp_path, split="test", num_samples=3)
        ds = MamiDataset(dataset_path=root, split="test")
        assert len(ds) == 3


# ---------------------------------------------------------------------------
# DatasetManager
# ---------------------------------------------------------------------------


class TestDatasetManager:
    """Tests for DatasetManager."""

    def test_load_dataset(self, tmp_path: Path) -> None:
        """load_dataset returns a MamiDataset of correct length."""
        root = _make_fake_dataset_root(tmp_path, num_samples=6)
        manager = DatasetManager(dataset_path=root)
        ds = manager.load_dataset(split="train")
        assert isinstance(ds, MamiDataset)
        assert len(ds) == 6

    def test_dataset_caching(self, tmp_path: Path) -> None:
        """Successive calls with the same split return the same object."""
        root = _make_fake_dataset_root(tmp_path, num_samples=4)
        manager = DatasetManager(dataset_path=root)
        ds1 = manager.load_dataset(split="train")
        ds2 = manager.load_dataset(split="train")
        assert ds1 is ds2

    def test_subset_kwarg_ignored(self, tmp_path: Path) -> None:
        """Passing subset= does not raise (backward compat)."""
        root = _make_fake_dataset_root(tmp_path, num_samples=4)
        manager = DatasetManager(dataset_path=root)
        # Should not raise even though subset is ignored
        ds = manager.load_dataset(split="train", subset="hate_symbols")
        assert len(ds) == 4

    def test_get_dataset_stats_structure(self, tmp_path: Path) -> None:
        """get_dataset_stats returns the expected keys."""
        root = _make_fake_dataset_root(tmp_path, num_samples=10)
        manager = DatasetManager(dataset_path=root)
        stats = manager.get_dataset_stats(split="train")
        expected_keys = {
            "total_images",
            "misogynous_count",
            "non_misogynous_count",
            "shaming_count",
            "stereotype_count",
            "objectification_count",
            "violence_count",
        }
        assert expected_keys.issubset(stats.keys())

    def test_get_dataset_stats_counts(self, tmp_path: Path) -> None:
        """misogynous_count + non_misogynous_count == total_images."""
        root = _make_fake_dataset_root(tmp_path, num_samples=8)
        manager = DatasetManager(dataset_path=root)
        stats = manager.get_dataset_stats(split="train")
        assert stats["misogynous_count"] + stats["non_misogynous_count"] == stats["total_images"]

    def test_check_composition_completeness(self, tmp_path: Path) -> None:
        """Composition check returns bool dict with both classes present."""
        root = _make_fake_dataset_root(tmp_path, num_samples=10)
        manager = DatasetManager(dataset_path=root)
        result = manager.check_composition_completeness(split="train")
        assert "has_misogynous" in result
        assert "has_non_misogynous" in result
        # Our fake dataset alternates 0/1 so both classes appear for >=2 samples
        assert result["has_misogynous"] is True
        assert result["has_non_misogynous"] is True

    def test_supports_minimum_size_true(self, tmp_path: Path) -> None:
        """Returns True when dataset has enough samples."""
        root = _make_fake_dataset_root(tmp_path, num_samples=10)
        manager = DatasetManager(dataset_path=root)
        assert manager.supports_minimum_size(min_size=5) is True

    def test_supports_minimum_size_false(self, tmp_path: Path) -> None:
        """Returns False when dataset is smaller than the minimum."""
        root = _make_fake_dataset_root(tmp_path, num_samples=3)
        manager = DatasetManager(dataset_path=root)
        assert manager.supports_minimum_size(min_size=100) is False

    def test_supports_minimum_size_missing_dataset(self, tmp_path: Path) -> None:
        """Returns False gracefully when the dataset path is missing."""
        manager = DatasetManager(dataset_path=str(tmp_path / "nonexistent"))
        assert manager.supports_minimum_size(min_size=5) is False

    def test_resolves_path_via_kaggle(self, tmp_path: Path) -> None:
        """When dataset_path is None, _resolve_path calls _kaggle_download."""
        root = _make_fake_dataset_root(tmp_path, num_samples=4)
        manager = DatasetManager()  # no dataset_path

        with patch("utils.dataset._kaggle_download", return_value=root) as mock_dl:
            ds = manager.load_dataset(split="train")

        mock_dl.assert_called_once()
        assert len(ds) == 4


# ---------------------------------------------------------------------------
# Backward-compat aliases
# ---------------------------------------------------------------------------


class TestBackwardCompatAliases:
    """Ensure old import names still resolve."""

    def test_hateful_illusion_alias(self) -> None:
        """HatefulIllusionDataset is importable and is MamiDataset."""
        from utils.dataset import HatefulIllusionDataset, MamiDataset

        assert HatefulIllusionDataset is MamiDataset

    def test_download_alias(self) -> None:
        """download_hateful_illusion_dataset is importable."""
        from utils.dataset import download_hateful_illusion_dataset, download_mami_dataset

        assert download_hateful_illusion_dataset is download_mami_dataset
