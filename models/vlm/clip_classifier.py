"""CLIP zero-shot image classifier wrapping open_clip."""

from __future__ import annotations

import time

import numpy as np
import open_clip  # type: ignore[import-untyped]
from PIL import Image
import torch

from models.vlm.classifier import BaseVLMClassifier, ClassificationResult


class CLIPClassifier(BaseVLMClassifier):
    """Zero-shot classifier using CLIP cosine similarity between image and text embeddings."""

    def __init__(
        self,
        model_name: str = "ViT-L-14",
        pretrained: str = "openai",
        device: str = "cpu",
        model_path: str | None = None,
    ) -> None:
        self.device = torch.device(device)
        if model_path:
            path_lower = model_path.lower()
            if "vit_b_32_quickgelu" in path_lower or "vit-b-32-quickgelu" in path_lower:
                model_name = "ViT-B-32-quickgelu"
            elif "vit_l_14_quickgelu" in path_lower or "vit-l-14-quickgelu" in path_lower:
                model_name = "ViT-L-14-quickgelu"
            elif "vit_b_32" in path_lower or "vit-b-32" in path_lower:
                model_name = "ViT-B-32"
            elif "vit_l_14" in path_lower or "vit-l-14" in path_lower:
                model_name = "ViT-L-14"

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.model = self.model.to(self.device)
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self._labels: list[str] = []
        self._text_embeddings: torch.Tensor | None = None
        self.is_classification = False
        self.classification_head: nn.Module | None = None

        if model_path:
            import logging

            logger = logging.getLogger(__name__)
            logger.info("Loading fine-tuned CLIP weights from %s", model_path)
            state_dict = torch.load(model_path, map_location=self.device)
            if "classifier.weight" in state_dict:
                from torch import nn

                embed_dim = (
                    self.model.text_projection.shape[1]
                    if hasattr(self.model, "text_projection")
                    else 512
                )
                num_classes = state_dict["classifier.bias"].shape[0]

                class CLIPClassifierHead(nn.Module):
                    def __init__(
                        self, clip_model: nn.Module, embed_dim: int, num_classes: int
                    ) -> None:
                        super().__init__()
                        self.clip = clip_model
                        self.classifier = nn.Linear(embed_dim * 2, num_classes)

                    def forward(
                        self, images: torch.Tensor, text_tokens: torch.Tensor
                    ) -> torch.Tensor:
                        image_features = self.clip.encode_image(images)
                        text_features = self.clip.encode_text(text_tokens)

                        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

                        fused = torch.cat([image_features, text_features], dim=-1)
                        return self.classifier(fused)

                self.classification_head = CLIPClassifierHead(
                    self.model, embed_dim, num_classes
                ).to(self.device)
                self.classification_head.load_state_dict(state_dict)
                self.is_classification = True
                logger.info("Loaded CLIP classification head checkpoint (%d classes)", num_classes)
            else:
                self.model.load_state_dict(state_dict)
                logger.info("Loaded CLIP contrastive weights checkpoint")

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

    def predict(self, image: np.ndarray, text: str | None = None) -> tuple[str, float]:
        """Return (predicted_label, confidence) for a single image."""
        if self.is_classification:
            res = self.predict_batch([image], texts=[text or ""])
            return res[0]
        if self._text_embeddings is None or not self._labels:
            raise RuntimeError("Call set_classes() before predict().")
        image_features = self._embed_image(image)
        logits = (image_features @ self._text_embeddings.T).squeeze(0)
        probs = logits.softmax(dim=-1)
        idx = int(probs.argmax().item())
        return self._labels[idx], float(probs[idx].item())

    def classify(
        self, image: np.ndarray, labels: list[str], text: str | None = None
    ) -> ClassificationResult:
        """Implement BaseVLMClassifier interface for a single image."""
        if labels != self._labels:
            self.set_classes(labels)
        start = time.perf_counter()
        label, confidence = self.predict(image, text=text)
        elapsed = time.perf_counter() - start
        return ClassificationResult(
            prediction=label,
            confidence=confidence,
            latency_s=elapsed,
            refusal=False,
        )

    def predict_batch(
        self, images: list[np.ndarray], chunk_size: int = 32, texts: list[str] | None = None
    ) -> list[tuple[str, float]]:
        """Return (predicted_label, confidence) for each image, processed in chunks."""
        if not self.is_classification and (self._text_embeddings is None or not self._labels):
            raise RuntimeError("Call set_classes() before predict_batch().")
        results: list[tuple[str, float]] = []
        for start in range(0, len(images), chunk_size):
            chunk = images[start : start + chunk_size]
            pil_tensors = [self.preprocess(Image.fromarray(img)) for img in chunk]  # type: ignore[operator]
            batch = torch.stack(pil_tensors).to(self.device)

            if self.is_classification:
                chunk_texts = texts[start : start + chunk_size] if texts else [""] * len(chunk)
                clean_texts = []
                for t in chunk_texts:
                    words = t.strip().split()
                    clean_text = " ".join(words[:60]) if len(words) > 60 else t.strip()
                    clean_texts.append(clean_text or "empty text")
                tokens = self.tokenizer(clean_texts).to(self.device)

                with torch.no_grad():
                    assert self.classification_head is not None
                    logits = self.classification_head(batch, tokens)

                if logits.shape[1] == 2:
                    probs = logits.softmax(dim=-1)
                    for row in probs:
                        idx = int(row.argmax().item())
                        if len(self._labels) == 2:
                            label_idx = 0 if idx == 1 else 1
                            results.append((self._labels[label_idx], float(row[idx].item())))
                        else:
                            lbl = "yes" if idx == 1 else "no"
                            results.append((lbl, float(row[idx].item())))
                else:
                    probs = logits.sigmoid()
                    from models.vlm.classifier import CLIP_SUBTYPE_LABELS, SUBTYPE_LABELS

                    category_idx = 0
                    for c_idx, cat in enumerate(SUBTYPE_LABELS):
                        pos, _ = CLIP_SUBTYPE_LABELS[cat]
                        if pos == self._labels[0]:
                            category_idx = c_idx
                            break
                    for row in probs:
                        prob_pos = float(row[category_idx].item())
                        if prob_pos >= 0.5:
                            results.append((self._labels[0], prob_pos))
                        else:
                            results.append((self._labels[1], 1.0 - prob_pos))
            else:
                assert self._text_embeddings is not None
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
        self, images: list[np.ndarray], chunk_size: int = 32, texts: list[str] | None = None
    ) -> tuple[list[tuple[str, float]], float]:
        """Return predictions and total elapsed wall-clock seconds."""
        t0 = time.perf_counter()
        preds = self.predict_batch(images, chunk_size=chunk_size, texts=texts)
        elapsed = time.perf_counter() - t0
        return preds, elapsed
