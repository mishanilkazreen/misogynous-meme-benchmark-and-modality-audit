"""
YOLO classification models for hateful content detection.

Note: This is a classification model (not detection) since the HatefulIllusion
dataset does not provide bounding box annotations.
"""

from models.yolo.detector import (
    ClassificationResult,
    YOLOClassifier,
    YOLOBackbone,
    YOLODetector,  # Backward compatibility alias
)
from models.yolo.trainer import (
    YOLOTrainingConfig,
    TrainingMetrics,
    YOLOTrainer,
)
from models.yolo.evaluator import (
    EvaluationMetrics,
    YOLOEvaluator,
)

__all__ = [
    "ClassificationResult",
    "YOLOClassifier",
    "YOLOBackbone",
    "YOLODetector",
    "YOLOTrainingConfig",
    "TrainingMetrics",
    "YOLOTrainer",
    "EvaluationMetrics",
    "YOLOEvaluator",
]
