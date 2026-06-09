"""
DatasetManager for loading and validating hateful content datasets.
Supports the HatefulIllusion dataset format from Hugging Face.
"""

from dataclasses import dataclass

from datasets import load_dataset as _hf_load_dataset  # must precede torch to avoid OpenMP segfault
from huggingface_hub import hf_hub_download
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


@dataclass
class Annotation:
    """Represents an annotation for a single image."""

    image_id: str
    message: str
    prompt: str
    visibility: int
    labels: list[int] | None = None
    bbox: tuple[int, int, int, int] | None = None

    @property
    def is_hate(self) -> bool:
        return True

    @property
    def message_type(self) -> str:
        if self.message.isdigit():
            return "textual"
        return "symbolic"

    @property
    def visibility_level(self) -> str:
        return "low" if self.visibility <= 2 else "high"


class HatefulIllusionDataset(Dataset):
    """PyTorch Dataset for HatefulIllusion dataset from Hugging Face."""

    def __init__(
        self,
        split: str = "train",
        subset: str = "digits",
        transform=None,
        cache_dir: str | None = None,
    ):
        self.split = split
        self.subset = subset
        self.transform = transform
        self.cache_dir = cache_dir
        self.annotations: dict[str, Annotation] = {}
        self.image_ids: list[str] = []
        self._hf_dataset = None
        self._load_dataset()

    def _load_dataset(self) -> None:
        self._hf_dataset = _hf_load_dataset(
            "yiting/HatefulIllusion_Dataset",
            self.subset,
            cache_dir=self.cache_dir,
        )[self.split]

        for idx, item in enumerate(self._hf_dataset):  # type: ignore[arg-type, var-annotated]
            img_id = str(idx)
            annotation = Annotation(
                image_id=img_id,
                message=item["message"],  # type: ignore[index]
                prompt=item["prompt"],  # type: ignore[index]
                visibility=item["visibility"],  # type: ignore[index]
            )
            self.annotations[img_id] = annotation
            self.image_ids.append(img_id)

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> dict:
        img_id = self.image_ids[idx]
        annotation = self.annotations[img_id]

        # Load actual image from HuggingFace dataset
        hf_item = self._hf_dataset[idx]  # type: ignore[index]
        image_path = hf_item["image"]  # e.g., "images/0.png"

        # Download image from HuggingFace Hub
        local_path = hf_hub_download(
            repo_id="yiting/HatefulIllusion_Dataset",
            filename=f"{self.subset}/{image_path}",
            repo_type="dataset",
            cache_dir=self.cache_dir,
        )
        pil_image: Image.Image = Image.open(local_path)

        # Ensure RGB format
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")

        if self.transform:
            image = self.transform(pil_image)
        else:
            image = torch.from_numpy(np.array(pil_image).transpose(2, 0, 1)).float() / 255.0

        return {
            "image": image,
            "image_id": img_id,
            "message": annotation.message,
            "is_hate": annotation.is_hate,
            "message_type": annotation.message_type,
            "visibility_level": annotation.visibility_level,
            "visibility_score": annotation.visibility,
            "prompt": annotation.prompt,
        }


class DatasetManager:
    """Manager for loading, validating, and preparing datasets."""

    def __init__(self, cache_dir: str | None = None):
        self.cache_dir = cache_dir
        self._datasets: dict[str, HatefulIllusionDataset] = {}

    def load_dataset(
        self,
        split: str = "train",
        subset: str = "digits",
        transform=None,
    ) -> HatefulIllusionDataset:
        cache_key = f"{split}:{subset}"
        if cache_key not in self._datasets:
            self._datasets[cache_key] = HatefulIllusionDataset(
                split=split,
                subset=subset,
                transform=transform,
                cache_dir=self.cache_dir,
            )
        return self._datasets[cache_key]

    def get_dataset_stats(self, split: str = "train") -> dict:
        """Get statistics about the dataset."""
        dataset = self.load_dataset(split=split)
        annotations = list(dataset.annotations.values())

        high_vis = sum(1 for a in annotations if a.visibility_level == "high")
        low_vis = sum(1 for a in annotations if a.visibility_level == "low")
        textual = sum(1 for a in annotations if a.message_type == "textual")
        symbolic = sum(1 for a in annotations if a.message_type == "symbolic")

        return {
            "total_images": len(annotations),
            "high_visibility": high_vis,
            "low_visibility": low_vis,
            "textual_count": textual,
            "symbolic_count": symbolic,
        }

    def check_composition_completeness(self, split: str = "train") -> dict[str, bool]:
        dataset = self.load_dataset(split=split)
        combinations = {
            "high_visibility_textual": False,
            "high_visibility_symbolic": False,
            "low_visibility_textual": False,
            "low_visibility_symbolic": False,
        }
        for ann in dataset.annotations.values():
            key = f"{ann.visibility_level}_visibility_{ann.message_type}"
            if key in combinations:
                combinations[key] = True
        return combinations

    def supports_minimum_size(self, min_size: int = 5000) -> bool:
        try:
            dataset = self.load_dataset(split="train")
            return len(dataset) >= min_size
        except (FileNotFoundError, ValueError, KeyError):
            return False


def download_hateful_illusion_dataset(cache_dir: str | None = None) -> str:
    """Download the HatefulIllusion dataset from Hugging Face."""
    from datasets import load_dataset

    ds = load_dataset("yiting/HatefulIllusion_Dataset", "digits", cache_dir=cache_dir)
    print(f"Dataset downloaded: {len(ds['train'])} samples")
    return cache_dir or "~/.cache/huggingface/datasets"
