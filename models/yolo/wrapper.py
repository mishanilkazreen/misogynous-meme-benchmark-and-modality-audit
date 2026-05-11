from __future__ import annotations

from pathlib import Path
import time
from typing import Any

from ultralytics import YOLO
from ultralytics.engine.results import Results


class UltralyticsYOLO:
    """Thin wrapper around official Ultralytics YOLO checkpoints."""

    def __init__(self, checkpoint: str, device: str | None = "cpu", verbose: bool = False):
        self.checkpoint = checkpoint
        self.device = device or "cpu"
        self.verbose = verbose
        self.model = YOLO(checkpoint)

    def predict(self, source: Any, **kwargs: Any) -> list[Results]:
        kwargs.setdefault("device", self.device)
        kwargs.setdefault("verbose", self.verbose)
        return self.model.predict(source=source, **kwargs)

    def timed_predict(self, source: Any, **kwargs: Any) -> tuple[list[Results], float]:
        start = time.perf_counter()
        results = self.predict(source=source, **kwargs)
        elapsed = time.perf_counter() - start
        return results, elapsed

    def val(self, data: str | Path, **kwargs: Any) -> Any:
        kwargs.setdefault("device", self.device)
        kwargs.setdefault("verbose", self.verbose)
        return self.model.val(data=str(data), **kwargs)
