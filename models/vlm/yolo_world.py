"""YOLO-World text-prompted detector wrapper."""

from __future__ import annotations

import time
from typing import List, Optional, Tuple

import numpy as np
from ultralytics import YOLOWorld as _YOLOWorld  # type: ignore[import-untyped]


class YOLOWorldWrapper:
    """Thin wrapper around ultralytics.YOLOWorld for text-prompted detection.

    Call set_classes() once before the benchmark loop — the text encoder runs
    only once, not per-image.
    """

    def __init__(self, checkpoint: str = "yolov8s-worldv2.pt", device: str = "cpu") -> None:
        self.model = _YOLOWorld(checkpoint)
        self.model.to(device)
        self._classes: List[str] = []

    def set_classes(self, classes: List[str]) -> None:
        """Encode text prompts and cache them on the model."""
        self._classes = classes
        self.model.set_classes(classes)

    def predict(self, image: np.ndarray, conf: float = 0.25) -> bool:
        """Return True if any detection fires above conf threshold."""
        results = self.model.predict(source=image, conf=conf, verbose=False)
        boxes = results[0].boxes
        return len(boxes) > 0 if boxes is not None else False

    def timed_predict(self, image: np.ndarray, conf: float = 0.25) -> Tuple[bool, float]:
        """Return (fired, elapsed_seconds) for a single image."""
        start = time.perf_counter()
        fired = self.predict(image, conf=conf)
        elapsed = time.perf_counter() - start
        return fired, elapsed

    def predict_batch(
        self, images: List[np.ndarray], conf: float = 0.25
    ) -> Tuple[List[bool], float]:
        """Return (list of fired bools, total elapsed seconds) for all images."""
        results_list: List[bool] = []
        start = time.perf_counter()
        for image in images:
            results_list.append(self.predict(image, conf=conf))
        elapsed = time.perf_counter() - start
        return results_list, elapsed
