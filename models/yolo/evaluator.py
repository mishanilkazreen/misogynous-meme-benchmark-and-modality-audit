"""
YOLO Evaluator for classification metrics.

Computes accuracy, precision, recall, and F1 score for the YOLO classifier.
Supports stratification by visibility level (high/low).
"""

from dataclasses import dataclass, field

import torch

from models.yolo.detector import ClassificationResult, YOLOClassifier


@dataclass
class EvaluationMetrics:  # pylint: disable=too-many-instance-attributes
    """Container for evaluation metrics."""

    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    # Stratified metrics
    accuracy_high_vis: float = 0.0
    accuracy_low_vis: float = 0.0
    # Counts
    total_samples: int = 0
    true_positives: int = 0
    false_positives: int = 0
    true_negatives: int = 0
    false_negatives: int = 0


@dataclass
class StratifiedCounts:
    """Counts for stratified evaluation."""

    high_vis: dict[str, int] = field(default_factory=lambda: {"tp": 0, "fp": 0, "tn": 0, "fn": 0})
    low_vis: dict[str, int] = field(default_factory=lambda: {"tp": 0, "fp": 0, "tn": 0, "fn": 0})


class YOLOEvaluator:
    """
    Evaluator for YOLO classification model.

    Computes standard classification metrics:
    - Accuracy: (TP + TN) / Total
    - Precision: TP / (TP + FP)
    - Recall: TP / (TP + FN)
    - F1: 2 * (Precision * Recall) / (Precision + Recall)

    Also provides metrics stratified by visibility level.
    """

    def __init__(self, model: YOLOClassifier):
        self.model = model

    def evaluate(
        self,
        images: list[torch.Tensor],
        labels: list[bool],
        visibility_levels: list[str] | None = None,
    ) -> EvaluationMetrics:
        """
        Evaluate model on a dataset.

        Args:
            images: List of image tensors
            labels: List of ground truth labels (True = hateful)
            visibility_levels: Optional list of "high" or "low" per sample

        Returns:
            EvaluationMetrics with all computed metrics
        """
        if visibility_levels is None:
            visibility_levels = ["low"] * len(images)

        counts = StratifiedCounts()
        total_counts = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}

        for image, label, vis_level in zip(images, labels, visibility_levels, strict=False):
            result = self.model.predict(image)
            predicted = result.is_hateful

            # Update counts
            if label and predicted:
                key = "tp"
            elif label and not predicted:
                key = "fn"
            elif not label and predicted:
                key = "fp"
            else:
                key = "tn"

            total_counts[key] += 1
            if vis_level == "high":
                counts.high_vis[key] += 1
            else:
                counts.low_vis[key] += 1

        return self._compute_metrics(total_counts, counts)

    def evaluate_batch(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        visibility_levels: list[str] | None = None,
    ) -> EvaluationMetrics:
        """
        Evaluate model on a batch of images.

        Args:
            images: Batch tensor of shape (B, 3, H, W)
            labels: Tensor of ground truth labels (1 = hateful, 0 = benign)
            visibility_levels: Optional list of visibility levels

        Returns:
            EvaluationMetrics with all computed metrics
        """
        batch_size = images.shape[0]
        if visibility_levels is None:
            visibility_levels = ["low"] * batch_size

        results = self.model.predict_batch(images)
        labels_list = labels.bool().tolist()

        counts = StratifiedCounts()
        total_counts = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}

        for result, label, vis_level in zip(results, labels_list, visibility_levels, strict=False):
            predicted = result.is_hateful

            if label and predicted:
                key = "tp"
            elif label and not predicted:
                key = "fn"
            elif not label and predicted:
                key = "fp"
            else:
                key = "tn"

            total_counts[key] += 1
            if vis_level == "high":
                counts.high_vis[key] += 1
            else:
                counts.low_vis[key] += 1

        return self._compute_metrics(total_counts, counts)

    def _compute_metrics(  # pylint: disable=too-many-locals
        self,
        total: dict[str, int],
        stratified: StratifiedCounts,
    ) -> EvaluationMetrics:
        """Compute all metrics from counts."""
        tp, fp, tn, fn = total["tp"], total["fp"], total["tn"], total["fn"]
        total_samples = tp + fp + tn + fn

        # Overall metrics
        accuracy = (tp + tn) / max(1, total_samples)
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(1e-6, precision + recall)

        # Stratified accuracy
        high = stratified.high_vis
        low = stratified.low_vis
        high_total = high["tp"] + high["fp"] + high["tn"] + high["fn"]
        low_total = low["tp"] + low["fp"] + low["tn"] + low["fn"]

        accuracy_high = (high["tp"] + high["tn"]) / max(1, high_total)
        accuracy_low = (low["tp"] + low["tn"]) / max(1, low_total)

        return EvaluationMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1=f1,
            accuracy_high_vis=accuracy_high,
            accuracy_low_vis=accuracy_low,
            total_samples=total_samples,
            true_positives=tp,
            false_positives=fp,
            true_negatives=tn,
            false_negatives=fn,
        )

    def evaluate_from_results(
        self,
        results: list[ClassificationResult],
        labels: list[bool],
        visibility_levels: list[str] | None = None,
    ) -> EvaluationMetrics:
        """
        Compute metrics from pre-computed results.

        Args:
            results: List of ClassificationResult from model
            labels: Ground truth labels
            visibility_levels: Optional visibility levels

        Returns:
            EvaluationMetrics
        """
        if visibility_levels is None:
            visibility_levels = ["low"] * len(results)

        counts = StratifiedCounts()
        total_counts = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}

        for result, label, vis_level in zip(results, labels, visibility_levels, strict=False):
            predicted = result.is_hateful

            if label and predicted:
                key = "tp"
            elif label and not predicted:
                key = "fn"
            elif not label and predicted:
                key = "fp"
            else:
                key = "tn"

            total_counts[key] += 1
            if vis_level == "high":
                counts.high_vis[key] += 1
            else:
                counts.low_vis[key] += 1

        return self._compute_metrics(total_counts, counts)
