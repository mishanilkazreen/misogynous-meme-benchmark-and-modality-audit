"""CLIP zero-shot image classifier wrapping open_clip."""
# pylint: disable=too-many-instance-attributes, too-many-locals, too-many-branches
# pylint: disable=import-outside-toplevel

from __future__ import annotations

import time

import numpy as np
import open_clip  # type: ignore[import-untyped]
from PIL import Image
import torch
from torch import nn

from models.vlm.classifier import BaseVLMClassifier, ClassificationResult


def _infer_mlp_hidden_dim(state_dict: dict) -> int:
    """Infer the hidden dim of an MLP classification head from its state dict.

    The training-time MLP head (see ``scripts.train_clip.CLIPClassifierHead``
    with ``hidden_dim > 0``) has the structure ``LayerNorm -> Linear(in,
    hidden) -> GELU -> Dropout -> Linear(hidden, out)``. When serialised,
    the two linear layers become ``classifier.1.weight`` (shape
    ``[hidden, in]``) and ``classifier.4.weight`` (shape
    ``[out, hidden]``). We read ``classifier.1.weight``'s first dimension
    to recover ``hidden_dim``.

    Returns ``-1`` if the state dict does not contain an MLP head at all
    (the caller must fall back to another loading strategy).
    """
    key = "classifier.1.weight"
    if key not in state_dict:
        return -1
    return int(state_dict[key].shape[0])


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

            from scripts.train_clip import CLIPClassifierHead as TrainCLIPClassifierHead

            logger = logging.getLogger(__name__)
            logger.info("Loading fine-tuned CLIP weights from %s", model_path)
            state_dict = torch.load(model_path, map_location=self.device)
            # Detect head architecture from state_dict keys. The old linear
            # head serialises as ``classifier.weight`` and ``classifier.bias``;
            # the new MLP head from docs/CODE_REVIEW_ISSUES.md §2.1
            # serialises as ``classifier.0.weight``, ``classifier.1.weight``,
            # etc. We inspect the keys and reconstruct the matching head so
            # both old and new checkpoints load cleanly.
            has_linear_head = "classifier.weight" in state_dict
            mlp_hidden_dim = _infer_mlp_hidden_dim(state_dict) if not has_linear_head else 0

            if has_linear_head or mlp_hidden_dim >= 0:
                embed_dim = (
                    self.model.text_projection.shape[1]
                    if hasattr(self.model, "text_projection")
                    else 512
                )
                if has_linear_head:
                    num_classes = state_dict["classifier.bias"].shape[0]
                    hidden_dim = 0
                else:
                    # For the MLP head, the last Linear is at classifier.4.
                    num_classes = state_dict["classifier.4.bias"].shape[0]
                    hidden_dim = mlp_hidden_dim

                self.classification_head = TrainCLIPClassifierHead(
                    self.model,
                    embed_dim,
                    num_classes,
                    hidden_dim=hidden_dim,
                ).to(self.device)
                self.classification_head.load_state_dict(state_dict)
                self.is_classification = True
                logger.info(
                    "Loaded CLIP classification head checkpoint (num_classes=%d, "
                    "head=%s hidden_dim=%d)",
                    num_classes,
                    "linear" if hidden_dim == 0 else "MLP",
                    hidden_dim,
                )
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

    def set_classes_ensemble(self, prompts_per_class: dict[str, list[str]]) -> None:
        """Precompute per-class text embeddings by averaging multiple prompts.

        Zero-shot CLIP is sensitive to the exact wording of the class prompts;
        averaging L2-normalised embeddings across a bank of phrase variants
        (5-8 per class) gives a more robust text representation and typically
        buys 2-4 F1 points on the CLIP zero-shot rows without any training.
        See docs/CODE_REVIEW_ISSUES.md §7.1.

        Args:
            prompts_per_class: Mapping from class label (as it will appear in
                :attr:`_labels`) to the list of prompts to average for that
                class. Keys become the returned labels; order is preserved.
        """
        self._labels = list(prompts_per_class.keys())
        class_embeddings: list[torch.Tensor] = []
        with torch.no_grad():
            for prompts in prompts_per_class.values():
                if not prompts:
                    raise ValueError("Every class needs at least one prompt for the ensemble.")
                tokens = self.tokenizer(prompts).to(self.device)
                feats = self.model.encode_text(tokens)
                feats = feats / feats.norm(dim=-1, keepdim=True)
                mean_feat = feats.mean(dim=0)
                mean_feat = mean_feat / mean_feat.norm()
                class_embeddings.append(mean_feat)
        self._text_embeddings = torch.stack(class_embeddings)

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
        self,
        images: list[np.ndarray],
        chunk_size: int = 32,
        texts: list[str] | None = None,
        tta: bool = False,
    ) -> list[tuple[str, float]]:
        """Return (predicted_label, confidence) for each image, processed in chunks.

        Args:
            images: Batch of images as HxWxC uint8 arrays.
            chunk_size: Forward-pass batch size for the CLIP tower.
            texts: Optional per-image texts used only by the fine-tuned head.
            tta: Enable test-time augmentation by also running each image
                horizontally flipped and averaging softmax outputs. Applies
                to the zero-shot path only; the fine-tuned classification
                head ignores this flag (its predictions are not softmax-
                aggregable in a meaningful way). See
                docs/CODE_REVIEW_ISSUES.md §7.2.
        """
        if not self.is_classification and (self._text_embeddings is None or not self._labels):
            raise RuntimeError("Call set_classes() before predict_batch().")
        results: list[tuple[str, float]] = []
        for start in range(0, len(images), chunk_size):
            chunk = images[start : start + chunk_size]
            pil_tensors = [
                self.preprocess(Image.fromarray(img))  # type: ignore[operator]
                for img in chunk
            ]
            batch = torch.stack(pil_tensors).to(self.device)

            if self.is_classification:
                chunk_texts = texts[start : start + chunk_size] if texts else [""] * len(chunk)
                # open_clip's tokenizer truncates to 77 tokens; enforce non-empty only.
                clean_texts = [t.strip() or "empty text" for t in chunk_texts]
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
                    if tta:
                        # Horizontal flip along the last spatial dim.
                        flipped = torch.flip(batch, dims=[-1])
                        image_features_flip = self.model.encode_image(flipped)
                        image_features_flip = image_features_flip / image_features_flip.norm(
                            dim=-1, keepdim=True
                        )
                probs_original = (image_features @ self._text_embeddings.T).softmax(dim=-1)
                if tta:
                    probs_flip = (image_features_flip @ self._text_embeddings.T).softmax(dim=-1)
                    probs = 0.5 * (probs_original + probs_flip)
                else:
                    probs = probs_original
                for row in probs:
                    idx = int(row.argmax().item())
                    results.append((self._labels[idx], float(row[idx].item())))
        return results

    def timed_predict_batch(
        self,
        images: list[np.ndarray],
        chunk_size: int = 32,
        texts: list[str] | None = None,
        tta: bool = False,
    ) -> tuple[list[tuple[str, float]], float]:
        """Return predictions and total elapsed wall-clock seconds."""
        t0 = time.perf_counter()
        preds = self.predict_batch(images, chunk_size=chunk_size, texts=texts, tta=tta)
        elapsed = time.perf_counter() - t0
        return preds, elapsed
