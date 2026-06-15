"""
Script to fine-tune CLIP on the MAMI dataset.
Supports contrastive (InfoNCE) alignment and classification fine-tuning.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import time

import numpy as np
import open_clip  # type: ignore[import-untyped]
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.dataset import DatasetManager

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
        image_features = self.clip.encode_image(images)
        text_features = self.clip.encode_text(text_tokens)

        # Normalize features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

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

        # Tokenize text (max length in CLIP is 77)
        clean_texts = []
        for t in texts:
            words = t.strip().split()
            clean_text = " ".join(words[:60]) if len(words) > 60 else t.strip()
            clean_texts.append(clean_text or "empty text")

        tokens = tokenizer(clean_texts).to(device)

        optimizer.zero_grad()

        # Extract features
        image_features = model.encode_image(images)
        text_features = model.encode_text(tokens)

        # Normalize features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # Compute symmetric InfoNCE loss
        # Use logit scale if available, otherwise default to a reasonable scale (e.g. 100)
        logit_scale = model.logit_scale.exp() if hasattr(model, "logit_scale") else 100.0
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        labels = torch.arange(len(images), device=device)
        loss = (F.cross_entropy(logits_per_image, labels) + F.cross_entropy(logits_per_text, labels)) / 2

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def train_classification(
    model: CLIPClassifierHead,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    tokenizer: Any,
    device: torch.device,
    task: str,
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

        # Parse labels based on task
        if task == "singleclass":
            targets = batch["misogynous"].to(device)
        else:
            # shaming, stereotype, objectification, violence
            targets = torch.stack([
                batch["shaming"],
                batch["stereotype"],
                batch["objectification"],
                batch["violence"]
            ], dim=1).float().to(device)

        clean_texts = []
        for t in texts:
            words = t.strip().split()
            clean_text = " ".join(words[:60]) if len(words) > 60 else t.strip()
            clean_texts.append(clean_text or "empty text")

        tokens = tokenizer(clean_texts).to(device)

        optimizer.zero_grad()
        logits = model(images, tokens)

        if task == "singleclass":
            loss = F.cross_entropy(logits, targets)
        else:
            loss = F.binary_cross_entropy_with_logits(logits, targets)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def load_ocr_transcripts(split: str, ocr_engine: str, embeddings_dir: Path) -> dict[str, str]:
    """Load OCR-extracted texts from any pre-existing NPZ file for the split and engine."""
    import glob
    pattern = str(embeddings_dir / f"{split}_*_{ocr_engine}.npz")
    files = glob.glob(pattern)
    if not files:
        pattern = str(embeddings_dir / f"{split}_*_ocr_{ocr_engine}.npz")
        files = glob.glob(pattern)
        
    if not files:
        logger.warning(
            f"No pre-extracted OCR NPZ file found for split '{split}' and engine '{ocr_engine}' in {embeddings_dir}. "
            "Please run scripts/extract_embeddings.py with --use-ocr --ocr-engine {ocr_engine} first."
        )
        return {}
        
    file_path = Path(files[0])
    logger.info("Loading OCR transcripts from %s...", file_path)
    data = np.load(file_path, allow_pickle=True)
    image_ids = data["image_ids"]
    raw_texts = data["raw_texts"]
    return {str(img_id): str(txt) for img_id, txt in zip(image_ids, raw_texts)}


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
    parser.add_argument("--limit", type=int, default=None, help="Cap split samples for smoke testing")
    parser.add_argument(
        "--use-ocr",
        action="store_true",
        help="Use OCR-extracted text instead of dataset transcripts",
    )
    parser.add_argument(
        "--ocr-engine",
        default="easyocr",
        choices=["easyocr", "paddleocr"],
        help="OCR engine to load transcripts for",
    )
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    logger.info("Using device: %s", device)

    # 1. Create model and preprocessing transforms
    logger.info("Loading CLIP model %s...", args.model)
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        args.model, pretrained=args.pretrained
    )
    tokenizer = open_clip.get_tokenizer(args.model)

    # Freeze encoders based on flags
    for name, param in clip_model.named_parameters():
        if "visual" in name and args.freeze_image:
            param.requires_grad = False
        elif "visual" not in name and args.freeze_text:
            param.requires_grad = False

    # Retrieve embedding dimension
    embed_dim = clip_model.text_projection.shape[1] if hasattr(clip_model, "text_projection") else 512

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
        train_dataset._records = train_dataset._records[:args.limit]
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

    ocr_map = None
    if args.use_ocr:
        ocr_map = load_ocr_transcripts("train", args.ocr_engine, Path("results/embeddings"))

    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        if args.loss_mode == "contrastive":
            avg_loss = train_contrastive(model, train_loader, optimizer, tokenizer, device, ocr_map=ocr_map)
        else:
            avg_loss = train_classification(model, train_loader, optimizer, tokenizer, device, args.task, ocr_map=ocr_map)
        
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
    checkpoint_name = f"finetuned_clip_{args.loss_mode}{task_suffix}_{model_name_clean}.pth"
    save_path = MODELS_DIR / checkpoint_name

    torch.save(model.state_dict(), save_path)
    logger.info("Successfully saved fine-tuned CLIP checkpoint to %s", save_path)


if __name__ == "__main__":
    main()
