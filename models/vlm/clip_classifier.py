"""CLIP zero-shot image classifier wrapping open_clip."""

from __future__ import annotations

import time

import numpy as np
import open_clip  # type: ignore[import-untyped]
from PIL import Image
import torch


class CLIPClassifier:
    """Zero-shot classifier using CLIP cosine similarity between image and text embeddings."""

    def __init__(
        self,
        model_name: str = "ViT-L-14",
        pretrained: str = "openai",
        device: str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.model = self.model.to(self.device)
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self._labels: list[str] = []
        self._text_embeddings: torch.Tensor | None = None

    def set_classes(self, labels: list[str]) -> None:
        """Precompute and cache normalised text embeddings for candidate labels."""
        self._labels = labels
        tokens = self.tokenizer(labels).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        self._text_embeddings = text_features

    def _embed_image(self, image: np.ndarray) -> torch.Tensor:
        pil = Image.fromarray(image)
        tensor = self.preprocess(pil).unsqueeze(0).to(self.device)  # type: ignore[operator]
        with torch.no_grad():
            features = self.model.encode_image(tensor)
            features = features / features.norm(dim=-1, keepdim=True)
        return features

    def predict(self, image: np.ndarray) -> tuple[str, float]:
        """Return (predicted_label, confidence) for a single image."""
        if self._text_embeddings is None or not self._labels:
            raise RuntimeError("Call set_classes() before predict().")
        image_features = self._embed_image(image)
        logits = (image_features @ self._text_embeddings.T).squeeze(0)
        probs = logits.softmax(dim=-1)
        idx = int(probs.argmax().item())
        return self._labels[idx], float(probs[idx].item())

    def predict_batch(
        self, images: list[np.ndarray], chunk_size: int = 32
    ) -> list[tuple[str, float]]:
        """Return (predicted_label, confidence) for each image, processed in chunks."""
        if self._text_embeddings is None or not self._labels:
            raise RuntimeError("Call set_classes() before predict_batch().")
        results: list[tuple[str, float]] = []
        for start in range(0, len(images), chunk_size):
            chunk = images[start : start + chunk_size]
            pil_tensors = [self.preprocess(Image.fromarray(img)) for img in chunk]  # type: ignore[operator]
            batch = torch.stack(pil_tensors).to(self.device)
            with torch.no_grad():
                image_features = self.model.encode_image(batch)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            logits = image_features @ self._text_embeddings.T
            probs = logits.softmax(dim=-1)
            for row in probs:
                idx = int(row.argmax().item())
                results.append((self._labels[idx], float(row[idx].item())))
        return results

    def timed_predict_batch(
        self, images: list[np.ndarray], chunk_size: int = 32
    ) -> tuple[list[tuple[str, float]], float]:
        """Return predictions and total elapsed wall-clock seconds."""
        start = time.perf_counter()
        preds = self.predict_batch(images, chunk_size=chunk_size)
        elapsed = time.perf_counter() - start
        return preds, elapsed
