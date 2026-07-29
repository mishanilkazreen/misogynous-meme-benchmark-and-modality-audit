"""
Script to fine-tune CLIP on the MAMI dataset.
Supports contrastive (InfoNCE) alignment and classification fine-tuning.
"""

from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path
import time
from typing import Any

import open_clip  # type: ignore[import-untyped]
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torchvision import transforms as T  # type: ignore[import-untyped]

from models.vlm.metrics_multilabel import compute_mami_score_b
from utils.dataset import DatasetManager
from utils.seed import set_seed
from utils.task_names import TASK_CHOICES, canonical_task
from utils.text_source import load_text_source_transcripts, resolve_text_source

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).resolve().parents[1] / "results" / "models"


class CLIPClassifierHead(nn.Module):
    """CLIP tower + a small MLP classification head on the concat of image and text features.

    Two architectures selectable at construction time:

    * ``hidden_dim == 0``: single ``Linear(2 * embed_dim, num_classes)``
      layer (the pre-fix behaviour, kept for backwards-compatible
      checkpoint loading).
    * ``hidden_dim > 0`` (default 512): a 2-layer MLP with LayerNorm,
      ReLU, and dropout. This is the recommended head when the CLIP
      towers are frozen; it gives the classifier enough capacity to
      recover the extra decision boundary that used to come from
      end-to-end tower fine-tuning, without any of the overfitting risk.

    Fix reference: docs/CODE_REVIEW_ISSUES.md §2.1.
    """

    def __init__(
        self,
        clip_model: nn.Module,
        embed_dim: int,
        num_classes: int,
        hidden_dim: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.clip = clip_model
        self.hidden_dim = hidden_dim
        # Input is 2 * embed_dim because image and text features are concatenated.
        in_features = 2 * embed_dim
        if hidden_dim > 0:
            self.classifier = nn.Sequential(
                nn.LayerNorm(in_features),
                nn.Linear(in_features, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )
        else:
            self.classifier = nn.Linear(in_features, num_classes)

    def forward(self, images: torch.Tensor, text_tokens: torch.Tensor) -> torch.Tensor:
        image_features = self.clip.encode_image(images)  # type: ignore[operator]
        text_features = self.clip.encode_text(text_tokens)  # type: ignore[operator]

        # L2-normalise so the head sees unit-scale features.
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)  # type: ignore[operator]
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)  # type: ignore[operator]

        # Concat the two towers' features along the last dim.
        fused = torch.cat([image_features, text_features], dim=-1)
        return self.classifier(fused)


def train_contrastive(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    tokenizer: Any,
    device: torch.device,
    ocr_map: dict[str, str] | None = None,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
    grad_clip_norm: float | None = 1.0,
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
        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                (p for p in model.parameters() if p.requires_grad), grad_clip_norm
            )
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

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
    scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
    grad_clip_norm: float | None = 1.0,
    label_smoothing: float = 0.0,
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
            loss = F.cross_entropy(logits, targets, label_smoothing=label_smoothing)
        else:
            # For multi-label BCE, apply label smoothing by mixing a small
            # uniform prior into the {0, 1} target vector: 0 -> smoothing/2,
            # 1 -> 1 - smoothing/2. Composes cleanly with pos_weight from
            # docs/CODE_REVIEW_ISSUES.md §1.2.
            if label_smoothing > 0.0:
                smoothed_targets = targets * (1.0 - label_smoothing) + label_smoothing / 2.0
            else:
                smoothed_targets = targets
            loss = F.binary_cross_entropy_with_logits(
                logits, smoothed_targets, pos_weight=pos_weight
            )

        loss.backward()
        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                (p for p in model.parameters() if p.requires_grad), grad_clip_norm
            )
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def load_ocr_transcripts(split: str, ocr_engine: str, embeddings_dir: Path) -> dict[str, str]:
    """Deprecated shim. Kept so external callers (e.g. old notebooks) still import.

    Prefer :func:`utils.text_source.load_text_source_transcripts`; this
    wrapper is equivalent to calling it with ``text_source="ocr"``.
    """
    return load_text_source_transcripts(split, "ocr", ocr_engine, embeddings_dir)


def build_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_steps: int,
) -> LambdaLR:
    """Linear warmup for ``warmup_steps`` steps, then cosine decay to 0.

    Standard open_clip / transformers recipe. Prevents the destabilising
    "hot start" of AdamW at a fixed LR on a randomly-initialised head
    while still exploring the LR range aggressively during the middle of
    training. See docs/CODE_REVIEW_ISSUES.md §2.3.

    Args:
        optimizer: The optimiser whose LR the scheduler modulates.
        total_steps: Total number of optimizer steps across the whole run
            (``epochs * len(train_loader)``).
        warmup_steps: Number of initial steps to linearly ramp the LR
            from 0 to the optimiser's configured LR. Set to 0 to skip
            warmup and go straight to cosine decay.

    Returns:
        A ``LambdaLR`` that scales the LR by a value in ``[0, 1]``.
    """
    if warmup_steps < 0:
        raise ValueError(f"warmup_steps must be >= 0, got {warmup_steps}")
    if total_steps <= 0:
        raise ValueError(f"total_steps must be > 0, got {total_steps}")

    def _lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        # Clamp progress to [0, 1] so any extra step beyond the schedule
        # gives LR=0 rather than a negative multiplier.
        progress = min(1.0, max(0.0, progress))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda=_lr_lambda)


@torch.no_grad()
def evaluate_classification(
    model: nn.Module,
    dataloader: DataLoader,
    tokenizer: Any,
    device: torch.device,
    task: str,
    ocr_map: dict[str, str] | None = None,
) -> float:
    """Compute the primary validation metric for the CLIP classification head.

    * Task A (``singleclass``): macro F1 across ``{misogynous, not misogynous}``.
    * Task B (``multiclass``): MAMI 2022 official Sub-task B score
      (positive-support weighted average of per-sub-type binary-macro F1;
      see ``models.vlm.metrics_multilabel.compute_mami_score_b``).

    Both metrics land in ``[0, 1]`` and are the numbers the paper reports,
    so tracking them directly lets ``main()`` save the best-val checkpoint
    without any post-hoc conversion (docs/CODE_REVIEW_ISSUES.md §2.4).
    """
    from sklearn.metrics import f1_score

    was_training = model.training
    model.eval()
    all_preds: list[int] = []
    all_gts: list[int] = []
    all_pred_dicts: list[dict[str, int]] = []
    all_gt_dicts: list[dict[str, int]] = []
    for batch in dataloader:
        images = batch["image"].to(device)
        if ocr_map:
            texts = [ocr_map.get(str(img_id), "") for img_id in batch["image_id"]]
        else:
            texts = batch["text"]
        clean_texts = [t.strip() or "empty text" for t in texts]
        tokens = tokenizer(clean_texts).to(device)
        logits = model(images, tokens)

        if task == "singleclass":
            preds = logits.argmax(dim=-1).detach().cpu().tolist()
            gts = batch["misogynous"].detach().cpu().tolist()
            all_preds.extend(int(p) for p in preds)
            all_gts.extend(int(g) for g in gts)
        else:
            probs = torch.sigmoid(logits).detach().cpu()
            preds_bin = (probs >= 0.5).int().tolist()
            for i in range(len(preds_bin)):
                all_pred_dicts.append(
                    {lbl: int(preds_bin[i][j]) for j, lbl in enumerate(MULTILABEL_ORDER)}
                )
                all_gt_dicts.append({lbl: int(batch[lbl][i].item()) for lbl in MULTILABEL_ORDER})

    if was_training:
        model.train()

    if task == "singleclass":
        if not all_gts:
            return 0.0
        return float(f1_score(all_gts, all_preds, average="macro", zero_division=0))
    mami = compute_mami_score_b(all_pred_dicts, all_gt_dicts, MULTILABEL_ORDER)
    return float(mami["mami_score_b"])


def build_train_preprocess(eval_preprocess: T.Compose) -> T.Compose:
    """Derive an augmented training preprocess from the model's eval preprocess.

    Applies RandomResizedCrop, RandomHorizontalFlip, and ColorJitter before
    ToTensor and the CLIP-specific Normalize. Values match SRCB's
    SemEval 2022 augmentation set (docs/CODE_REVIEW_ISSUES.md §2.2).

    Crop size and Normalize statistics are inferred from ``eval_preprocess``
    so this helper works for any open_clip variant (ViT-B-32, ViT-L-14,
    ViT-L-14-336, etc.) without hardcoding.

    Vertical flip is disabled by default: meme overlay text is read
    left-to-right and flipping vertically would make the text unreadable
    to the CLIP tower.
    """
    crop_size: int = 224
    normalize: T.Normalize | None = None
    for stage in eval_preprocess.transforms:
        if isinstance(stage, T.CenterCrop):
            size = stage.size
            crop_size = size[0] if isinstance(size, (tuple, list)) else int(size)
        elif isinstance(stage, T.Normalize):
            normalize = stage
    if normalize is None:
        raise ValueError(
            "Could not find a Normalize transform inside the eval preprocess. "
            "The training-time preprocess needs the CLIP-specific mean/std."
        )
    return T.Compose(
        [
            T.RandomResizedCrop(crop_size, scale=(0.85, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            T.ToTensor(),
            normalize,
        ]
    )


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
        type=canonical_task,
        choices=TASK_CHOICES,
        help=(
            "Task when --loss-mode is classification. Accepts 'binary' or "
            "'singleclass' for Task A (Task A binary misogyny), 'multilabel' "
            "or 'multiclass' for Task B (multi-label sub-types). Legacy "
            "names kept for backward compatibility (docs/CODE_REVIEW_ISSUES.md §4.1)."
        ),
    )
    parser.add_argument(
        "--freeze-image",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Freeze the CLIP image tower during training. Default: True (matches "
            "Recipe A in docs/CODE_REVIEW_ISSUES.md §2.1). Pass --no-freeze-image "
            "to train the visual tower end-to-end (Recipe B; needs LR warmup and "
            "cosine schedule to converge)."
        ),
    )
    parser.add_argument(
        "--freeze-text",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Freeze the CLIP text tower during training. Default: True. "
            "Pass --no-freeze-text to unfreeze."
        ),
    )
    parser.add_argument(
        "--head-hidden-dim",
        type=int,
        default=512,
        help=(
            "Hidden dimension of the classification head MLP. Set to 0 to use "
            "a single linear layer (matches the pre-fix architecture; kept only "
            "for backward-compatible checkpoint loading). Default: 512."
        ),
    )
    parser.add_argument(
        "--augment",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Apply training-time image augmentation (RandomResizedCrop, "
            "horizontal flip, ColorJitter). Default: on. Pass --no-augment for an "
            "ablation that reproduces the pre-fix deterministic-preprocess path."
        ),
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=100,
        help=(
            "Number of linear-warmup steps before the cosine decay kicks in. "
            "Set to 0 to disable warmup. See docs/CODE_REVIEW_ISSUES.md §2.3."
        ),
    )
    parser.add_argument(
        "--lr-schedule",
        default="cosine",
        choices=["cosine", "none"],
        help=(
            "LR schedule after warmup. 'cosine' decays from the base LR to 0 "
            "over the remaining steps; 'none' holds the LR constant."
        ),
    )
    parser.add_argument(
        "--grad-clip-norm",
        type=float,
        default=1.0,
        help=("Maximum L2 norm for gradient clipping. Set to 0 to disable clipping. Default: 1.0."),
    )
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.1,
        help=(
            "Label smoothing factor for both Task A cross-entropy and Task B "
            "BCE. 0.1 is a mild regulariser that also mitigates MAMI's known "
            "annotation ambiguity (see docs/CODE_REVIEW_ISSUES.md §7.5). "
            "Set to 0 to disable."
        ),
    )
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
            "Deprecated alias: equivalent to --text-source ocr. Kept for backward compatibility."
        ),
    )
    parser.add_argument(
        "--ocr-engine",
        default="easyocr",
        choices=["easyocr", "paddleocr"],
        help="OCR engine that produced the pre-extracted transcripts",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help=(
            "RNG seed used to make training reproducible across runs. Applied to "
            "Python's random module, NumPy, and PyTorch (CPU + CUDA). See "
            "docs/CODE_REVIEW_ISSUES.md §3.1."
        ),
    )
    args = parser.parse_args()

    # Seed every RNG before touching the dataset or the model.
    set_seed(args.seed)

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
        model: nn.Module = CLIPClassifierHead(
            clip_model,
            embed_dim,
            num_classes,
            hidden_dim=args.head_hidden_dim,
        ).to(device)
        logger.info(
            "Classification head: %s (hidden_dim=%d)",
            "MLP" if args.head_hidden_dim > 0 else "Linear",
            args.head_hidden_dim,
        )
    else:
        model = clip_model.to(device)

    # Resolve the text source ONCE, up front. Downstream code (train and val
    # OCR-map loading, checkpoint filename suffix) all consumes this.
    text_source = resolve_text_source(args.text_source, args.use_ocr)
    logger.info("Text source for training and validation: %s", text_source)

    # 2. Datasets. The eval preprocess is deterministic (open_clip's default),
    # used at validation and inference time. For training we build an augmented
    # variant on top of it so the CLIP tower does not see the same 224x224 crop
    # every epoch (docs/CODE_REVIEW_ISSUES.md §2.2). Set --no-augment to fall
    # back to the deterministic preprocess for an ablation.
    logger.info("Loading dataset splits...")
    train_preprocess = build_train_preprocess(preprocess) if args.augment else preprocess
    if args.augment:
        logger.info("Training with augmentation: RandomResizedCrop, HorizontalFlip, ColorJitter.")
    else:
        logger.info("Training with deterministic preprocess (augmentation disabled).")
    manager = DatasetManager()
    train_dataset = manager.load_dataset(split="train", transform=train_preprocess)

    if args.limit:
        train_dataset._records = train_dataset._records[: args.limit]
        logger.info("Capped training samples to %d", args.limit)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )

    # Validation loader: only needed for classification runs where the
    # best-val checkpoint selection is meaningful. The validation split uses
    # the deterministic eval preprocess (no augmentation) so val numbers are
    # stable across epochs. See docs/CODE_REVIEW_ISSUES.md §2.4.
    val_loader: DataLoader | None = None
    val_ocr_map: dict[str, str] | None = None
    if args.loss_mode == "classification":
        val_dataset = manager.load_dataset(split="validation", transform=preprocess)
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
        )
        if text_source != "provided":
            val_ocr_map = (
                load_text_source_transcripts(
                    "validation", text_source, args.ocr_engine, Path("results/embeddings")
                )
                or None
            )

    # 3. Setup optimizer + LR scheduler + gradient clipping settings.
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)

    total_steps = args.epochs * len(train_loader)
    scheduler: torch.optim.lr_scheduler._LRScheduler | None = None
    if args.lr_schedule == "cosine":
        scheduler = build_warmup_cosine_scheduler(
            optimizer, total_steps=total_steps, warmup_steps=args.warmup_steps
        )
        logger.info(
            "LR schedule: linear warmup for %d steps then cosine over %d total steps",
            args.warmup_steps,
            total_steps,
        )
    else:
        logger.info("LR schedule disabled (constant LR).")

    grad_clip_norm: float | None = args.grad_clip_norm if args.grad_clip_norm > 0 else None
    if grad_clip_norm is not None:
        logger.info("Gradient clipping enabled (max_norm=%.2f).", grad_clip_norm)
    else:
        logger.info("Gradient clipping disabled.")

    # 4. Training Loop
    logger.info("Starting fine-tuning loop in %s mode...", args.loss_mode)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    ocr_map: dict[str, str] | None = None
    if text_source != "provided":
        ocr_map = load_text_source_transcripts(
            "train", text_source, args.ocr_engine, Path("results/embeddings")
        )
        if not ocr_map:
            logger.warning("Text-source NPZ not found; falling back to dataset transcripts.")
            ocr_map = None

    # Compute per-label pos_weight ONCE from the training set for Task B BCE.
    # Cheap: it's a single pass over the label dicts, not over the images.
    pos_weight: torch.Tensor | None = None
    if args.loss_mode == "classification" and args.task == "multiclass":
        pos_weight = compute_multilabel_pos_weight(
            train_dataset._records,
            labels=MULTILABEL_ORDER,  # pylint: disable=protected-access
        )
        logger.info(
            "Task B pos_weight (shaming, stereotype, objectification, violence): %s",
            pos_weight.tolist(),
        )

    best_val_metric = -1.0
    best_state: dict[str, torch.Tensor] | None = None

    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        if args.loss_mode == "contrastive":
            avg_loss = train_contrastive(
                model,
                train_loader,
                optimizer,
                tokenizer,
                device,
                ocr_map=ocr_map,
                scheduler=scheduler,
                grad_clip_norm=grad_clip_norm,
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
                scheduler=scheduler,
                grad_clip_norm=grad_clip_norm,
                label_smoothing=args.label_smoothing,
            )

        elapsed = time.perf_counter() - t0

        # Best-val checkpoint selection (classification runs only). Contrastive
        # runs have no defined "val metric" here so they still fall through to
        # the last-epoch save at the end.
        val_metric_str = ""
        if args.loss_mode == "classification" and val_loader is not None:
            val_metric = evaluate_classification(
                model,
                val_loader,
                tokenizer,
                device,
                args.task,
                ocr_map=val_ocr_map,
            )
            val_metric_str = f" | Val metric: {val_metric:.4f}"
            if val_metric > best_val_metric:
                best_val_metric = val_metric
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                val_metric_str += " (new best)"

        logger.info(
            "Epoch %d/%d completed | Avg Loss: %.4f | Time: %.2f seconds%s",
            epoch,
            args.epochs,
            avg_loss,
            elapsed,
            val_metric_str,
        )

    # Load the best checkpoint back into the model before saving, so the
    # persisted weights are the ones that scored highest on validation
    # rather than whatever the last epoch happened to produce.
    if best_state is not None:
        logger.info(
            "Restoring best-val checkpoint (val metric: %.4f) before saving.",
            best_val_metric,
        )
        model.load_state_dict(best_state)

    # 5. Save model checkpoint
    model_name_clean = args.model.lower().replace("-", "_")
    task_suffix = f"_{args.task}" if args.loss_mode == "classification" else ""
    # Encode the resolved text source in the checkpoint filename so
    # provided-text, PaddleOCR, and combined runs never overwrite each other.
    from utils.text_source import filename_suffix_for_source

    ocr_suffix = filename_suffix_for_source(text_source, args.ocr_engine)
    checkpoint_name = f"finetuned_clip_{args.loss_mode}{task_suffix}_{model_name_clean}{ocr_suffix}_seed{args.seed}.pth"
    save_path = MODELS_DIR / checkpoint_name

    torch.save(model.state_dict(), save_path)
    logger.info("Successfully saved fine-tuned CLIP checkpoint to %s", save_path)


if __name__ == "__main__":
    main()
