"""
YOLOClassifier: PyTorch implementation for hateful content classification.

This is a classification model (not detection) since the HatefulIllusion dataset
does not provide bounding box annotations. The model classifies whether an image
contains hidden hateful content (digits 0-9 embedded in AI-generated images).
"""
# pylint: disable=too-many-instance-attributes

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class ClassificationResult:
    """Represents a single classification result."""

    is_hateful: bool
    confidence: float
    predicted_class: int  # 0-9 for digit classification
    visibility_level: str  # "high" or "low" based on confidence


class ConvBlock(nn.Module):
    """Convolutional block with BatchNorm and LeakyReLU."""

    def __init__(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
    ):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through conv-bn-activation."""
        return self.act(self.bn(self.conv(x)))


class ResidualBlock(nn.Module):
    """Residual block for feature extraction."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = ConvBlock(channels, channels // 2, kernel_size=1, padding=0)
        self.conv2 = ConvBlock(channels // 2, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection."""
        return x + self.conv2(self.conv1(x))


class YOLOBackbone(nn.Module):
    """YOLO backbone for feature extraction."""

    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, 32, kernel_size=3, stride=1)
        self.stage1 = self._make_stage(32, 64, num_blocks=1)
        self.stage2 = self._make_stage(64, 128, num_blocks=2)
        self.stage3 = self._make_stage(128, 256, num_blocks=8)
        self.stage4 = self._make_stage(256, 512, num_blocks=8)
        self.stage5 = self._make_stage(512, 1024, num_blocks=4)

    def _make_stage(self, in_channels: int, out_channels: int, num_blocks: int) -> nn.Sequential:
        """Create a downsampling stage with residual blocks."""
        layers: list[nn.Module] = [ConvBlock(in_channels, out_channels, kernel_size=3, stride=2)]
        for _ in range(num_blocks):
            layers.append(ResidualBlock(out_channels))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning final feature map."""
        x = self.conv1(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        return x


class YOLOClassifier(nn.Module):
    """
    YOLO-based classifier for hateful content detection.

    This model classifies images as containing hidden hateful content or not.
    It does NOT produce bounding boxes since the HatefulIllusion dataset
    only provides classification labels (digit 0-9, visibility level).

    The model can operate in two modes:
    - Binary classification: hateful vs benign
    - Multi-class: predict which digit (0-9) is hidden
    """

    def __init__(
        self,
        num_classes: int = 10,  # 0-9 digits
        input_size: tuple[int, int] = (416, 416),
        conf_threshold: float = 0.5,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.input_size = input_size
        self.conf_threshold = conf_threshold

        self.backbone = YOLOBackbone()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (B, 3, H, W)

        Returns:
            Logits tensor of shape (B, num_classes)
        """
        features = self.backbone(x)
        pooled = self.global_pool(features)
        logits = self.classifier(pooled)
        return logits

    def predict(
        self,
        image: torch.Tensor,
        conf_threshold: float | None = None,
    ) -> ClassificationResult:
        """
        Run inference on a single image.

        Args:
            image: Input tensor of shape (3, H, W) or (1, 3, H, W)
            conf_threshold: Confidence threshold for positive classification

        Returns:
            ClassificationResult with prediction details
        """
        self.eval()
        threshold = conf_threshold or self.conf_threshold

        with torch.no_grad():
            if image.dim() == 3:
                image = image.unsqueeze(0)

            logits = self.forward(image)
            probs = F.softmax(logits, dim=1)
            confidence, predicted_class = probs.max(dim=1)

            conf_val = float(confidence.item())
            pred_class = int(predicted_class.item())

            return ClassificationResult(
                is_hateful=conf_val >= threshold,
                confidence=conf_val,
                predicted_class=pred_class,
                visibility_level="high" if conf_val > 0.7 else "low",
            )

    def predict_batch(
        self,
        images: torch.Tensor,
        conf_threshold: float | None = None,
    ) -> list[ClassificationResult]:
        """
        Run inference on a batch of images.

        Args:
            images: Input tensor of shape (B, 3, H, W)
            conf_threshold: Confidence threshold

        Returns:
            List of ClassificationResult per image
        """
        self.eval()
        threshold = conf_threshold or self.conf_threshold

        with torch.no_grad():
            logits = self.forward(images)
            probs = F.softmax(logits, dim=1)
            confidences, predicted_classes = probs.max(dim=1)

            results = []
            for conf, pred_cls in zip(confidences, predicted_classes, strict=False):
                conf_val = float(conf.item())
                results.append(
                    ClassificationResult(
                        is_hateful=conf_val >= threshold,
                        confidence=conf_val,
                        predicted_class=int(pred_cls.item()),
                        visibility_level="high" if conf_val > 0.7 else "low",
                    )
                )
            return results

    def get_config(self) -> dict:
        """Return model configuration."""
        return {
            "num_classes": self.num_classes,
            "input_size": self.input_size,
            "conf_threshold": self.conf_threshold,
        }

    @classmethod
    def from_config(cls, config: dict) -> "YOLOClassifier":
        """Create model from configuration dictionary."""
        return cls(
            num_classes=config.get("num_classes", 10),
            input_size=config.get("input_size", (416, 416)),
            conf_threshold=config.get("conf_threshold", 0.5),
        )


# Backward compatibility alias
YOLODetector = YOLOClassifier
