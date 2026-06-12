"""
DatasetManager for loading and validating misogyny detection datasets.
Supports the MAMI 2022 (Multimodal Misogyny Detection) dataset from Kaggle.

Dataset: chukwuebukaanulunko/multimodal-misogyny-detection-mami-2022
Splits : train (9,000), validation (1,000), test (1,000)
Labels : misogynous (binary), plus sub-tasks shaming / stereotype /
         objectification / violence (all binary).
"""

from __future__ import annotations

from collections.abc import Callable
import csv
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATASET_SLUG = "chukwuebukaanulunko/multimodal-misogyny-detection-mami-2022"

# Map split name -> (TSV filename, image subdirectory inside MAMI_2022_images/)
_SPLIT_META: dict[str, tuple[str, str]] = {
    "train": ("train.tsv", "training_images"),
    "validation": ("validation.tsv", "training_images"),
    "test": ("test.tsv", "test_images"),
}


# ---------------------------------------------------------------------------
# MamiDataset
# ---------------------------------------------------------------------------


class MamiDataset(Dataset):
    """PyTorch Dataset for the MAMI 2022 misogyny detection dataset.

    Each sample contains:
    - ``image``          : float32 RGB tensor, shape (3, H, W), values in [0, 1].
    - ``image_id``       : filename stem (e.g. ``"8716"``).
    - ``text``           : meme text transcription (str).
    - ``misogynous``     : binary int label (0 = not misogynous, 1 = misogynous).
    - ``shaming``        : binary int sub-task label.
    - ``stereotype``     : binary int sub-task label.
    - ``objectification``: binary int sub-task label.
    - ``violence``       : binary int sub-task label.

    Args:
        dataset_path: Root path returned by ``kagglehub.dataset_download()`` or
            by ``download_mami_dataset()``.  Must contain the TSV files and the
            ``MAMI_2022_images/`` folder.
        split: One of ``"train"``, ``"validation"``, or ``"test"``.
        transform: Optional callable applied to the ``PIL.Image`` before
            conversion to a tensor.  Receives a PIL Image and must return
            either another PIL Image or a torch.Tensor.
        cache_dir: Unused by this class (kept for API compatibility with
            ``DatasetManager``).  Pass ``None`` or omit.
    """

    def __init__(
        self,
        dataset_path: str,
        split: str = "train",
        transform: Callable | None = None,
        cache_dir: str | None = None,  # kept for API compat, unused
    ) -> None:
        if split not in _SPLIT_META:
            raise ValueError(f"Invalid split '{split}'. Choose from {list(_SPLIT_META.keys())}.")

        self.dataset_path = Path(dataset_path)
        self.split = split
        self.transform = transform
        self.cache_dir = cache_dir

        tsv_name, img_subdir = _SPLIT_META[split]
        self._tsv_path = self.dataset_path / tsv_name
        self._images_dir = self.dataset_path / "MAMI_2022_images" / img_subdir

        self._records: list[dict[str, str]] = []
        self._load_records()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_records(self) -> None:
        """Read the TSV for this split into ``self._records``."""
        if not self._tsv_path.exists():
            raise FileNotFoundError(
                f"TSV file not found: {self._tsv_path}. Have you run download_mami_dataset() first?"
            )
        with self._tsv_path.open(encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh, delimiter="\t")
            for row in reader:
                self._records.append(dict(row))

    def _load_image(self, file_name: str) -> Image.Image:
        """Load a PIL image by its filename, converting to RGB."""
        image_path = self._images_dir / file_name
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        pil_image: Image.Image = Image.open(image_path)
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")
        return pil_image

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int) -> dict:
        row = self._records[idx]

        file_name: str = row["file_name"]
        pil_image = self._load_image(file_name)

        if self.transform is not None:
            image = self.transform(pil_image)
        else:
            image = torch.from_numpy(np.array(pil_image).transpose(2, 0, 1)).float() / 255.0

        return {
            "image": image,
            "image_id": Path(file_name).stem,
            "text": row.get("text", ""),
            "misogynous": int(row.get("label", 0)),
            "shaming": int(row.get("shaming", 0)),
            "stereotype": int(row.get("stereotype", 0)),
            "objectification": int(row.get("objectification", 0)),
            "violence": int(row.get("violence", 0)),
        }


# ---------------------------------------------------------------------------
# DatasetManager
# ---------------------------------------------------------------------------


class DatasetManager:
    """Manager for loading, validating, and caching MAMI 2022 dataset splits.

    Args:
        cache_dir: Optional directory used as the ``cache_dir`` argument when
            calling ``download_mami_dataset()``.  If the dataset has already
            been downloaded, pass the root path via ``dataset_path`` instead.
        dataset_path: Explicit root path of the downloaded dataset.  When
            omitted, the manager will call ``kagglehub.dataset_download()``
            to resolve the path on first use.
    """

    def __init__(
        self,
        cache_dir: str | None = None,
        dataset_path: str | None = None,
    ) -> None:
        self.cache_dir = cache_dir
        self._dataset_path: str | None = dataset_path
        self._datasets: dict[str, MamiDataset] = {}

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _resolve_path(self) -> str:
        """Return the dataset root, downloading if necessary."""
        if self._dataset_path is not None:
            return self._dataset_path
        self._dataset_path = _kaggle_download()
        return self._dataset_path

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_dataset(
        self,
        split: str = "train",
        transform: Callable | None = None,
        # kept for backward-compat with callers that pass subset=; ignored
        subset: str | None = None,
    ) -> MamiDataset:
        """Load (and cache) a dataset split.

        Args:
            split: ``"train"``, ``"validation"``, or ``"test"``.
            transform: Optional image transform (see :class:`MamiDataset`).
            subset: Ignored. Present for backward compatibility only.

        Returns:
            A :class:`MamiDataset` instance.
        """
        cache_key = split
        if cache_key not in self._datasets:
            root = self._resolve_path()
            self._datasets[cache_key] = MamiDataset(
                dataset_path=root,
                split=split,
                transform=transform,
                cache_dir=self.cache_dir,
            )
        return self._datasets[cache_key]

    def get_dataset_stats(self, split: str = "train") -> dict:
        """Return basic label statistics for a split.

        Returns:
            Dict with keys: ``total_images``, ``misogynous_count``,
            ``non_misogynous_count``, ``shaming_count``, ``stereotype_count``,
            ``objectification_count``, ``violence_count``.
        """
        dataset = self.load_dataset(split=split)
        records = dataset._records

        misogynous = sum(int(r.get("label", 0)) for r in records)
        shaming = sum(int(r.get("shaming", 0)) for r in records)
        stereotype = sum(int(r.get("stereotype", 0)) for r in records)
        objectification = sum(int(r.get("objectification", 0)) for r in records)
        violence = sum(int(r.get("violence", 0)) for r in records)

        return {
            "total_images": len(records),
            "misogynous_count": misogynous,
            "non_misogynous_count": len(records) - misogynous,
            "shaming_count": shaming,
            "stereotype_count": stereotype,
            "objectification_count": objectification,
            "violence_count": violence,
        }

    def check_composition_completeness(self, split: str = "train") -> dict[str, bool]:
        """Check that both binary classes are present in a split."""
        dataset = self.load_dataset(split=split)
        records = dataset._records
        labels = {int(r.get("label", 0)) for r in records}
        return {
            "has_misogynous": 1 in labels,
            "has_non_misogynous": 0 in labels,
        }

    def supports_minimum_size(self, min_size: int = 5000) -> bool:
        """Return True if the training split has at least *min_size* samples."""
        try:
            dataset = self.load_dataset(split="train")
            return len(dataset) >= min_size
        except (FileNotFoundError, ValueError, KeyError):
            return False


# ---------------------------------------------------------------------------
# Convenience download function
# ---------------------------------------------------------------------------


def _kaggle_download() -> str:
    """Internal helper: download via kagglehub and return the dataset root path."""
    import kagglehub  # type: ignore[import-untyped]  # no stubs available

    path: str = kagglehub.dataset_download(DATASET_SLUG)
    return path


def download_mami_dataset() -> str:
    """Download the MAMI 2022 dataset from Kaggle via kagglehub.

    Reads ``KAGGLE_USERNAME`` and ``KAGGLE_KEY`` from the environment (loaded
    from ``.env`` by ``python-dotenv``).  Call this once before running
    benchmarks; subsequent calls return immediately from the local cache.
    kagglehub manages its own cache at ``~/.cache/kagglehub``.

    Returns:
        The local path to the downloaded dataset root directory.
    """
    from dotenv import load_dotenv

    # Load .env from the project root (two levels above this file).
    env_path = Path(__file__).resolve().parents[1] / ".env"
    load_dotenv(env_path)

    root = _kaggle_download()
    print(f"MAMI 2022 dataset path: {root}")
    return root


# ---------------------------------------------------------------------------
# Backward-compat shim — deprecated aliases kept so old import sites compile.
# Use MamiDataset / download_mami_dataset in new code.
# ---------------------------------------------------------------------------

HatefulIllusionDataset = MamiDataset  # deprecated alias
download_hateful_illusion_dataset = download_mami_dataset  # deprecated alias
