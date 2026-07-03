"""
Script to fine-tune CLIP on the MAMI dataset.
Supports contrastive (InfoNCE) alignment and classification fine-tuning.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import time
from typing import Any

import numpy as np
import open_clip  # type: ignore[import-untyped]
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils.dataset import DatasetManager
from utils.text_source import load_text_source_transcripts, resolve_text_source

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).resolve().parents[1] / "results" / "models"


class CLIPClassifierHead(nn.Module):
    """CLIP model combined with a classification projection head."""

    def __init__(self, clip_model: nn.Module, embed_dim: int, num_classes: int) -> None:
        super().__init__()
        self.clip = clip_model
        # We concatenate image and text embeddings, so input is 2 * embed_dim
        self.classifier = nn.Linear(embed_dim * 2, num_classes)

    def forward(self, images: torch.Tensor, text_tokens: torch.Tensor) -> torch.Tensor:
        image_features = self.clip.encode_image(images)  # type: ignore[operator]
        text_features = self.clip.encode_text(text_tokens)  # type: ignore[operator]

        # Normalize features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)  # type: ignore[operator]
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)  # type: ignore[operator]

        # Concat
        fused = torch.cat([image_features, text_features], dim=-1)
        return self.classifier(fused)


def train_contrastive(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    tokenizer: Any,
    device: torch.device,
    ocr_map: dict[str, str] | None = None,
) -> float:
    model.train()
    total_loss = 0.0

    for batch in dataloader:
        images = batch["image"].to(device)
        if ocr_map:
            texts = [ocr_map.get(str(img_id), "") for img_id in batch["image_id"]]
        else:
            texts = batch["text"]

        # Tokenize text. open_clip's tokenizer truncates to 77 tokens natively;
        # we only enforce the non-empty invariant.
        clean_texts = [t.strip() or "empty text" for t in texts]

        tokens = tokenizer(clean_texts).to(device)

        optimizer.zero_grad()

        # Extract features
        image_features = model.encode_image(images)  # type: ignore[operator]
        text_features = model.encode_text(tokens)  # type: ignore[operator]

        # Normalize features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)  # type: ignore[operator]
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)  # type: ignore[operator]

        # Compute symmetric InfoNCE loss
        # Use logit scale if available, otherwise default to a reasonable scale (e.g. 100)
        logit_scale = model.logit_scale.exp() if hasattr(model, "logit_scale") else 100.0  # type: ignore[operator]
        logits_per_image = logit_scale * image_features @ text_features.t()  # type: ignore[operator]
        logits_per_text = logits_per_image.t()

        labels = torch.arange(len(images), device=device)
        loss = (
            F.cross_entropy(logits_per_image, labels) + F.cross_entropy(logits_per_text, labels)
        ) / 2

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


MULTILABEL_ORDER: list[str] = ["shaming", "stereotype", "objectification", "violence"]


def compute_multilabel_pos_weight(
    records: list[dict[str, Any]],
    labels: list[str] = MULTILABEL_ORDER,
) -> torch.Tensor:
    """Return per-label ``pos_weight = N_neg / N_pos`` for BCE-with-logits.

    Used to rebalance the Task B loss on MAMI's imbalanced sub-type
    positives (shaming ~14 %, stereotype ~31 %, objectification ~29 %,
    violence ~13 %). Without this rebalancing the loss-minimising
    solution for the two rare classes is "always predict 0", which is
    exactly what the pre-fix runs showed (see docs/CODE_REVIEW_ISSUES.md
    §1.2).

    Args:
        records: The list of raw label dicts from the training split
            (``MamiDataset._records``). Each record must contain integer
            fields matching ``labels``.
        labels: Ordered list of sub-type label names.

    Returns:
        1-D float tensor of shape ``(len(labels),)`` with per-label
        ``N_neg / N_pos``. Falls back to 1.0 for any label with no
        positives (no rebalancing rather than a divide-by-zero).
    """
    n = len(records)
    weights: list[float] = []
    for lbl in labels:
        n_pos = sum(int(r.get(lbl, 0)) for r in records)
        n_neg = n - n_pos
        weights.append(n_neg / n_pos if n_pos > 0 else 1.0)
    return torch.tensor(weights, dtype=torch.float32)


def train_classification(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    tokenizer: Any,
    device: torch.device,
    task: str,
    ocr_map: dict[str, str] | None = None,
    pos_weight: torch.Tensor | None = None,
) -> float:
    """Run one epoch of classification training.

    Args:
        model, dataloader, optimizer, tokenizer, device: standard training
            components.
        task: ``"singleclass"`` (Task A, binary) or ``"multiclass"``
            (Task B, 4-way multi-label).
        ocr_map: optional mapping of image_id -> OCR transcript.
        pos_weight: only used when ``task == "multiclass"``. Per-label
            ``N_neg / N_pos`` tensor of shape ``(4,)``. Passed to
            ``binary_cross_entropy_with_logits`` as its ``pos_weight``
            argument to rebalance rare sub-types (shaming, violence).
            If ``None`` in the multi-label branch the loss is plain BCE,
            which reproduces the (broken) pre-fix behaviour and is kept
            for backward compatibility.
    """
    model.train()
    total_loss = 0.0

    if pos_weight is not None:
        pos_weight = pos_weight.to(device)

    for batch in dataloader:
        images = batch["image"].to(device)
        if ocr_map:
            texts = [ocr_map.get(str(img_id), "") for img_id in batch["image_id"]]
        else:
            texts = batch["text"]

        # Parse labels based on task
        if task == "singleclass":
            targets = batch["misogynous"].to(device)
        else:
            # shaming, stereotype, objectification, violence — MUST match
            # the order that `pos_weight` was computed in.
            targets = (
                torch.stack(
                    [
                        batch["shaming"],
                        batch["stereotype"],
                        batch["objectification"],
                        batch["violence"],
                    ],
                    dim=1,
                )
                .float()
                .to(device)
            )

        # open_clip's tokenizer truncates to 77 tokens; we only enforce non-empty.
        clean_texts = [t.strip() or "empty text" for t in texts]

        tokens = tokenizer(clean_texts).to(device)

        optimizer.zero_grad()
        logits = model(images, tokens)

        if task == "singleclass":
            loss = F.cross_entropy(logits, targets)
        else:
            loss = F.binary_cross_entropy_with_logits(
                logits, targets, pos_weight=pos_weight
            )

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def load_ocr_transcripts(split: str, ocr_engine: str, embeddings_dir: Path) -> dict[str, str]:
    """Deprecated shim. Kept so external callers (e.g. old notebooks) still import.

    Prefer :func:`utils.text_source.load_text_source_transcripts`; this
    wrapper is equivalent to calling it with ``text_source="ocr"``.
    """
    return load_text_source_transcripts(split, "ocr", ocr_engine, embeddings_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune CLIP on MAMI dataset")
    parser.add_argument("--model", default="ViT-L-14-quickgelu", help="CLIP model identifier")
    parser.add_argument("--pretrained", default="openai", help="Pretrained weights source")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs to train")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument(
        "--loss-mode",
        default="contrastive",
        choices=["contrastive", "classification"],
        help="Loss mode to train CLIP with",
    )
    parser.add_argument(
        "--task",
        default="singleclass",
        choices=["singleclass", "multiclass"],
        help="Task when --loss-mode is classification",
    )
    parser.add_argument("--freeze-image", action="store_true", help="Freeze CLIP image encoder")
    parser.add_argument("--freeze-text", action="store_true", help="Freeze CLIP text encoder")
    parser.add_argument("--device", default="cuda", help="Target device")
    parser.add_argument(
        "--limit", type=int, default=None, help="Cap split samples for smoke testing"
    )
    parser.add_argument(
        "--text-source",
        default=None,
        choices=["provided", "ocr", "combined"],
        help=(
            "Where the text modality comes from. 'provided' uses MAMI's "
            "text-transcription column (default). 'ocr' or 'combined' loads a "
            "pre-extracted NPZ produced by scripts/extract_embeddings.py."
        ),
    )
    parser.add_argument(
        "--use-ocr",
        action="store_true",
        help=(
            "Deprecated alias: equivalent to --text-source ocr. Kept for "
            "backward compatibility."
        ),
    )
    parser.add_argument(
        "--ocr-engine",
        default="easyocr",
        choices=["easyocr", "paddleocr"],
        help="OCR engine that produced the pre-extracted transcripts",
    )
    args = parser.parse_args()

    device = torch.device(
        args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    )
    logger.info("Using device: %s", device)

    # 1. Create model and preprocessing transforms
    logger.info("Loading CLIP model %s...", args.model)
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        args.model, pretrained=args.pretrained
    )
    tokenizer = open_clip.get_tokenizer(args.model)

    # Freeze encoders based on flags
    for name, param in clip_model.named_parameters():
        if ("visual" in name and args.freeze_image) or ("visual" not in name and args.freeze_text):
            param.requires_grad = False

    # Retrieve embedding dimension
    embed_dim = (
        clip_model.text_projection.shape[1] if hasattr(clip_model, "text_projection") else 512
    )

    # Instantiate overall model wrapper
    if args.loss_mode == "classification":
        num_classes = 2 if args.task == "singleclass" else 4
        model: nn.Module = CLIPClassifierHead(clip_model, embed_dim, num_classes).to(device)
    else:
        model = clip_model.to(device)

    # 2. Datasets
    logger.info("Loading dataset splits...")
    manager = DatasetManager()
    train_dataset = manager.load_dataset(split="train", transform=preprocess)

    if args.limit:
        train_dataset._records = train_dataset._records[: args.limit]
        logger.info("Capped training samples to %d", args.limit)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )

    # 3. Setup optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)

    # 4. Training Loop
    logger.info("Starting fine-tuning loop in %s mode...", args.loss_mode)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    text_source = resolve_text_source(args.text_source, args.use_ocr)
    logger.info("Text source for training: %s", text_source)
    ocr_map: dict[str, str] | None = None
    if text_source != "provided":
        ocr_map = load_text_source_transcripts(
            "train", text_source, args.ocr_engine, Path("results/embeddings")
        )
        if not ocr_map:
            logger.warning(
                "Text-source NPZ not found; falling back to dataset transcripts."
            )
            ocr_map = None

    # Compute per-label pos_weight ONCE from the training set for Task B BCE.
    # Cheap: it's a single pass over the label dicts, not over the images.
    pos_weight: torch.Tensor | None = None
    if args.loss_mode == "classification" and args.task == "multiclass":
        pos_weight = compute_multilabel_pos_weight(
            train_dataset._records, labels=MULTILABEL_ORDER  # pylint: disable=protected-access
        )
        logger.info(
            "Task B pos_weight (shaming, stereotype, objectification, violence): %s",
            pos_weight.tolist(),
        )

    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        if args.loss_mode == "contrastive":
            avg_loss = train_contrastive(
                model, train_loader, optimizer, tokenizer, device, ocr_map=ocr_map
            )
        else:
            avg_loss = train_classification(
                model,
                train_loader,
                optimizer,
                tokenizer,
                device,
                args.task,
                ocr_map=ocr_map,
                pos_weight=pos_weight,
            )

        elapsed = time.perf_counter() - t0
        logger.info(
            "Epoch %d/%d completed | Avg Loss: %.4f | Time: %.2f seconds",
            epoch,
            args.epochs,
            avg_loss,
            elapsed,
        )

    # 5. Save model checkpoint
    model_name_clean = args.model.lower().replace("-", "_")
    task_suffix = f"_{args.task}" if args.loss_mode == "classification" else ""
    # Encode the resolved text source in the checkpoint filename so
    # provided-text, PaddleOCR, and combined runs never overwrite each other.
    from utils.text_source import filename_suffix_for_source

    ocr_suffix = filename_suffix_for_source(text_source, args.ocr_engine)
    checkpoint_name = (
        f"finetuned_clip_{args.loss_mode}{task_suffix}_{model_name_clean}{ocr_suffix}.pth"
    )
    save_path = MODELS_DIR / checkpoint_name

    torch.save(model.state_dict(), save_path)
    logger.info("Successfully saved fine-tuned CLIP checkpoint to %s", save_path)


if __name__ == "__main__":
    main()
