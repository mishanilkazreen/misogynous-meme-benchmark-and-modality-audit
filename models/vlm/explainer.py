"""VLM Explainer for content moderation decisions.

Prompts local generative VLMs (LLaVA, Qwen2-VL) to output natural-language
rationales justifying classification decisions.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any

import numpy as np
from PIL import Image
import torch

from models.vlm.prompt_templates import build_explainability_prompt

try:
    from transformers import (  # type: ignore[import-untyped]
        AutoProcessor,
        BitsAndBytesConfig,
        LlavaForConditionalGeneration,
        Qwen2VLForConditionalGeneration,
    )

    _TRANSFORMERS_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
    _TRANSFORMERS_AVAILABLE = False


class VLMExplainer:
    """Explains misogyny classification decisions using a generative VLM."""

    def __init__(
        self,
        model_type: str = "llava",
        model_id: str | None = None,
        device: str | None = None,
        quantize: str = "none",
    ) -> None:
        if not _TRANSFORMERS_AVAILABLE:
            raise RuntimeError("transformers not available. Install: uv sync --group vlm-gpu")

        self.model_type = model_type.lower()
        if self.model_type not in ("llava", "qwen"):
            raise ValueError(f"Unsupported model type: {model_type}. Choose 'llava' or 'qwen'.")

        # Determine default model IDs
        if model_id is None:
            if self.model_type == "llava":
                self.model_id = "llava-hf/llava-1.5-7b-hf"
            else:
                self.model_id = "Qwen/Qwen2-VL-7B-Instruct"
        else:
            self.model_id = model_id

        # Determine device
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        print(f"Initializing VLMExplainer using model={self.model_id} on device={self.device}")

        # Set up load kwargs
        load_kwargs: dict[str, Any] = {}
        if self.device == "cuda":
            load_kwargs["device_map"] = "auto"
            if quantize == "4bit":
                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
            elif quantize == "8bit":
                load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            else:
                load_kwargs["torch_dtype"] = torch.float16
        elif self.device == "mps":
            # bitsandbytes is CUDA-only; load in float16 for MPS
            load_kwargs["torch_dtype"] = torch.float16
        else:
            load_kwargs["torch_dtype"] = torch.float32

        # Load processor & model
        if self.model_type == "llava":
            self.processor = AutoProcessor.from_pretrained(self.model_id, use_fast=True)
            self.model = LlavaForConditionalGeneration.from_pretrained(self.model_id, **load_kwargs)
        else:
            self.processor = AutoProcessor.from_pretrained(self.model_id, max_pixels=512 * 28 * 28)
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_id, **load_kwargs
            )

        if self.device != "cuda" and not hasattr(self.model, "hf_device_map"):
            # For non-cuda devices (e.g. cpu/mps), map model to device manually if not using device_map
            self.model.to(self.device)

        self.model.eval()

    def explain(
        self,
        image: np.ndarray | Image.Image,
        ocr_transcript: str | None = None,
    ) -> dict[str, Any]:
        """Generate classification and explanation for a single image.

        Args:
            image: NumPy array or PIL Image.
            ocr_transcript: Extracted text from the image, if any.

        Returns:
            Dict containing 'misogynous' (bool), 'explanation' (str),
            'raw_response' (str), and 'latency_s' (float).
        """
        # Convert image to PIL
        import torch

        if isinstance(image, torch.Tensor):
            img_np = (image.detach().cpu().numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)
            pil_img = Image.fromarray(img_np)
        elif isinstance(image, np.ndarray):
            pil_img = Image.fromarray(image).convert("RGB")
        else:
            pil_img = image.convert("RGB")

        prompt_text = build_explainability_prompt(ocr_transcript)

        # Build model-specific prompt formats
        if self.model_type == "llava":
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt_text},
                    ],
                }
            ]
            prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
            inputs = self.processor(images=pil_img, text=prompt, return_tensors="pt").to(
                self.model.device
            )
        else:  # qwen
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": pil_img},
                        {"type": "text", "text": prompt_text},
                    ],
                }
            ]
            prompt = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.processor(
                text=[prompt], images=[pil_img], padding=True, return_tensors="pt"
            ).to(self.model.device)

        # Generate response
        t0 = time.perf_counter()
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=150,  # Allow enough tokens for the JSON block
                do_sample=False,
            )
        latency = time.perf_counter() - t0

        input_len = inputs["input_ids"].shape[1]
        raw_response = self.processor.decode(
            output_ids[0, input_len:], skip_special_tokens=True
        ).strip()

        # Parse the structured JSON output
        parsed = self._parse_json_response(raw_response)
        parsed["raw_response"] = raw_response
        parsed["latency_s"] = latency

        return parsed

    def _parse_json_response(self, text: str) -> dict[str, Any]:
        """Attempt to extract and parse JSON from the raw text response."""
        # Find JSON boundaries
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            json_str = match.group(0)
            try:
                data = json.loads(json_str)
                # Normalize keys and values
                misogynous = data.get("misogynous")
                if isinstance(misogynous, str):
                    misogynous = misogynous.lower() == "true"
                elif not isinstance(misogynous, bool):
                    misogynous = None

                explanation = data.get("explanation")
                return {
                    "misogynous": misogynous,
                    "explanation": str(explanation) if explanation else None,
                }
            except Exception:
                pass

        # Fallback heuristic parsing if JSON load fails
        cleaned = text.lower()
        misogynous = None
        if '"misogynous": true' in cleaned or '"misogynous":true' in cleaned:
            misogynous = True
        elif '"misogynous": false' in cleaned or '"misogynous":false' in cleaned:
            misogynous = False

        # Extract explanation string
        explanation_match = re.search(r'"explanation"\s*:\s*"([^"]+)"', text)
        explanation = explanation_match.group(1) if explanation_match else text

        return {
            "misogynous": misogynous,
            "explanation": explanation,
        }
