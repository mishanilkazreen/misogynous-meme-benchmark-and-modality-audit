"""
YOLO Training module for hateful content classification.

Implements training loop with early stopping, checkpointing, and optional preprocessing.
Uses cross-entropy loss for multi-class classification (digits 0-9).
"""
# pylint: disable=too-many-instance-attributes

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from models.yolo.detector import YOLOClassifier
from utils.preprocessing import PreprocessingPipeline


@dataclass
class YOLOTrainingConfig:
    """Configuration for YOLO training."""
    batch_size: int = 16
    epochs: int = 100
    learning_rate: float = 0.001
    weight_decay: float = 0.0005
    momentum: float = 0.9
    preprocessing: bool = False
    early_stopping_patience: int = 10
    checkpoint_dir: str = "checkpoints"
    save_best_only: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    input_size: Tuple[int, int] = (416, 416)


@dataclass
class TrainingMetrics:
    """Metrics collected during training."""
    epoch: int
    train_loss: float
    val_loss: Optional[float] = None
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1: Optional[float] = None


class YOLOTrainer:
    """
    Trainer for YOLO classification model.

    Supports:
    - Multi-class classification (digits 0-9)
    - Optional preprocessing (blur + equalization)
    - Early stopping and checkpointing
    """

    def __init__(self, model: YOLOClassifier, config: YOLOTrainingConfig):
        self.model = model
        self.config = config
        self.device = torch.device(config.device)
        self.model.to(self.device)

        self.preprocessor: Optional[PreprocessingPipeline] = None
        if config.preprocessing:
            self.preprocessor = PreprocessingPipeline(
                apply_blur=True, apply_equalization=True
            )

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(
            model.parameters(),
            lr=config.learning_rate,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.epochs
        )

        self.best_accuracy = 0.0
        self.epochs_without_improvement = 0
        self.training_history: List[TrainingMetrics] = []

        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def preprocess_batch(self, images: torch.Tensor) -> torch.Tensor:
        """Apply preprocessing to a batch of images if enabled."""
        if self.preprocessor is None:
            return images

        processed = []
        for img in images:
            img_np = img.permute(1, 2, 0).cpu().numpy()
            img_np = (img_np * 255).astype("uint8")
            preprocessed = self.preprocessor.preprocess(img_np)
            tensor = torch.from_numpy(preprocessed).permute(2, 0, 1).float() / 255.0
            processed.append(tensor)

        return torch.stack(processed).to(self.device)

    def train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in dataloader:
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)

            if self.config.preprocessing:
                images = self.preprocess_batch(images)

            self.optimizer.zero_grad()
            logits = self.model(images)
            loss = self.criterion(logits, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / max(1, num_batches)

    def validate(  # pylint: disable=too-many-locals
        self, dataloader: DataLoader
    ) -> Tuple[float, Dict[str, float]]:
        """Validate the model and compute metrics."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        correct = 0
        total = 0
        tp, fp, fn = 0, 0, 0

        with torch.no_grad():
            for batch in dataloader:
                images = batch["image"].to(self.device)
                labels = batch["label"].to(self.device)

                if self.config.preprocessing:
                    images = self.preprocess_batch(images)

                logits = self.model(images)
                loss = self.criterion(logits, labels)

                total_loss += loss.item()
                num_batches += 1

                _, predicted = logits.max(dim=1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

                # Binary metrics (hateful = any digit detected vs benign)
                pred_positive = predicted >= 0  # All predictions are "positive"
                label_positive = labels >= 0
                tp += ((pred_positive) & (label_positive)).sum().item()

        val_loss = total_loss / max(1, num_batches)
        accuracy = correct / max(1, total)
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(1e-6, precision + recall)

        return val_loss, {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        callbacks: Optional[List[Callable]] = None,
    ) -> List[TrainingMetrics]:
        """Full training loop with early stopping and checkpointing."""
        callbacks = callbacks or []

        for epoch in range(self.config.epochs):
            train_loss = self.train_epoch(train_loader)

            val_loss = None
            metrics_dict: Dict[str, float] = {}
            if val_loader is not None:
                val_loss, metrics_dict = self.validate(val_loader)

            self.scheduler.step()

            metrics = TrainingMetrics(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                accuracy=metrics_dict.get("accuracy"),
                precision=metrics_dict.get("precision"),
                recall=metrics_dict.get("recall"),
                f1=metrics_dict.get("f1"),
            )
            self.training_history.append(metrics)

            current_acc = metrics_dict.get("accuracy", 0.0)
            if current_acc > self.best_accuracy:
                self.best_accuracy = current_acc
                self.epochs_without_improvement = 0
                if self.config.save_best_only:
                    self.save_checkpoint("best_model.pt")
            else:
                self.epochs_without_improvement += 1

            if not self.config.save_best_only:
                self.save_checkpoint(f"checkpoint_epoch_{epoch}.pt")

            if self.epochs_without_improvement >= self.config.early_stopping_patience:
                print(f"Early stopping at epoch {epoch}")
                break

            for callback in callbacks:
                callback(metrics)

        return self.training_history

    def save_checkpoint(self, filename: str) -> None:
        """Save model checkpoint."""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_accuracy": self.best_accuracy,
            "training_history": self.training_history,
        }
        torch.save(checkpoint, self.checkpoint_dir / filename)

    def load_checkpoint(self, filename: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(
            self.checkpoint_dir / filename, weights_only=False
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.best_accuracy = checkpoint.get("best_accuracy", 0.0)
        self.training_history = checkpoint.get("training_history", [])
