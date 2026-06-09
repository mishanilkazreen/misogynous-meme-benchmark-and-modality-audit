"""LLaVA classifier wrapper for closed-set image classification."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from PIL import Image

from models.vlm.classifier import BaseVLMClassifier, ClassificationResult, extract_label

try:
    from transformers import (  # type: ignore[import-untyped]
        AutoProcessor,
        LlavaForConditionalGeneration,
    )

    _TRANSFORMERS_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
    _TRANSFORMERS_AVAILABLE = False

DEFAULT_MODEL_ID = "llava-hf/llava-1.5-7b-hf"
MAX_NEW_TOKENS = 20


class LLaVAClassifier(BaseVLMClassifier):
    """Closed-set classifier using LLaVA via HuggingFace transformers."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device: str = "cuda",
    ) -> None:
        if not _TRANSFORMERS_AVAILABLE:
            raise RuntimeError("transformers not available. Install: uv sync --group vlm-gpu")
        import torch

        self._device = device
        self._model_id = model_id
        self._processor: Any = AutoProcessor.from_pretrained(model_id, use_fast=True)

        self._model: Any = LlavaForConditionalGeneration.from_pretrained(
            model_id,
            dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map=device,
        )
        self._model.eval()

    def classify(self, image: np.ndarray, labels: list[str]) -> ClassificationResult:
        import torch

        from models.vlm.classifier import build_prompt

        pil = Image.fromarray(image)
        prompt_text = build_prompt("digits", labels)
        # LLaVA expects the <image> token in the conversation template
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]
        prompt = self._processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = self._processor(images=pil, text=prompt, return_tensors="pt").to(
            self._model.device
        )

        t0 = time.perf_counter()
        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False
            )
        elapsed = time.perf_counter() - t0

        trimmed = output_ids[:, inputs["input_ids"].shape[1] :]
        response = self._processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()

        matched = extract_label(response, labels)
        return ClassificationResult(
            prediction=matched,
            confidence=1.0 if matched else 0.0,
            latency_s=elapsed,
            refusal=False,
        )

    def classify_batch(
        self, images: list[np.ndarray], labels: list[str], subset: str = "digits"
    ) -> list[ClassificationResult]:
        return [self.classify(img, labels) for img in images]
