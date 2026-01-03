"""
YOLO classification models for hateful content detection.

Note: This is a classification model (not detection) since the HatefulIllusion
dataset does not provide bounding box annotations.
"""

from models.yolo.detector import (
    ClassificationResult,
    YOLOBackbone,
    YOLOClassifier,
    YOLODetector,  # Backward compatibility alias
)
from models.yolo.evaluator import (
    EvaluationMetrics,
    YOLOEvaluator,
)
from models.yolo.trainer import (
    TrainingMetrics,
    YOLOTrainer,
    YOLOTrainingConfig,
)

__all__ = [
    "ClassificationResult",
    "EvaluationMetrics",
    "TrainingMetrics",
    "YOLOBackbone",
    "YOLOClassifier",
    "YOLODetector",
    "YOLOEvaluator",
    "YOLOTrainer",
    "YOLOTrainingConfig",
]
