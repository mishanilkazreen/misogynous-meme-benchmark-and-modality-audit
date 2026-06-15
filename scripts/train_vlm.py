"""
Script to fine-tune generative VLMs (LLaVA, Qwen2-VL) using QLoRA and PEFT.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import time
from typing import Any

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader

from models.vlm.classifier import build_misogyny_prompt, build_subtype_prompt
from utils.dataset import DatasetManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).resolve().parents[1] / "results" / "models"


def image_to_numpy(image: Any) -> np.ndarray:
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    if isinstance(image, np.ndarray):
        if image.ndim == 3 and image.shape[0] == 3:
            image = image.transpose(1, 2, 0)
        if np.issubdtype(image.dtype, np.floating):
            image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
        elif image.dtype != np.uint8:
            image = image.astype(np.uint8)
        return image
    raise ValueError(f"Unsupported image type: {type(image)}")


class VLMCollate:
    """Collation helper that formats multimodal conversations and tokenizes them with masking."""

    def __init__(
        self, processor: Any, model_type: str, task: str, ocr_map: dict[str, str] | None = None
    ) -> None:
        self.processor = processor
        self.model_type = model_type
        self.task = task
        self.ocr_map = ocr_map

        if task == "singleclass":
            self.base_prompt_text = build_misogyny_prompt()
        else:
            self.base_prompt_text = build_subtype_prompt()

    def __call__(self, batch: list[dict]) -> dict[str, Any]:
        pils = []
        prompts = []
        responses = []

        # Prepare textual prompt and target response for each sample
        for sample in batch:
            arr = image_to_numpy(sample["image"])
            pil = Image.fromarray(arr)
            pils.append(pil)

            # Target response
            if self.task == "singleclass":
                resp = "yes" if sample["misogynous"] == 1 else "no"
            else:
                active = []
                for label in ["shaming", "stereotype", "objectification", "violence"]:
                    if sample[label] == 1:
                        active.append(label)
                resp = ", ".join(active) if active else "none"

            responses.append(resp)

            # Inject OCR text if available
            image_id = str(sample["image_id"])
            if self.ocr_map and image_id in self.ocr_map:
                ocr_text = self.ocr_map[image_id].strip()
                prompt_text = f'This meme contains the text: "{ocr_text}". {self.base_prompt_text}'
            else:
                prompt_text = self.base_prompt_text

            # Conversational format
            if "qwen2" in self.model_type:
                conversation = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": pil},
                            {"type": "text", "text": prompt_text},
                        ],
                    }
                ]
                # Format prompt for tokenizer
                prompt = self.processor.apply_chat_template(
                    conversation, tokenize=False, add_generation_prompt=True
                )
            else:
                # Default LLaVA formatting
                conversation = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": prompt_text},
                        ],
                    }
                ]
                prompt = self.processor.apply_chat_template(
                    conversation, add_generation_prompt=True
                )

            prompts.append(prompt)

        # Tokenize prompt + response end-to-end to compute outputs
        full_texts = [f"{p} {r}" for p, r in zip(prompts, responses, strict=True)]

        # Multimodal processor call
        inputs = self.processor(images=pils, text=full_texts, padding=True, return_tensors="pt")

        # Compute labels masking the user prompt tokens
        labels = inputs["input_ids"].clone()

        # Tokenize prompts alone to identify length of prompt tokens
        prompt_inputs = self.processor(images=pils, text=prompts, padding=True, return_tensors="pt")

        # Set all prompt tokens to -100 so the cross-entropy loss ignores them
        for i in range(len(batch)):
            prompt_len = (
                (prompt_inputs["input_ids"][i] != self.processor.tokenizer.pad_token_id)
                .sum()
                .item()
            )
            # Handle padding if left-padded
            labels[i, :prompt_len] = -100

        # Replace pad tokens with -100 in labels
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        inputs["labels"] = labels

        return inputs


def load_ocr_transcripts(split: str, ocr_engine: str, embeddings_dir: Path) -> dict[str, str]:
    """Load OCR-extracted texts from any pre-existing NPZ file for the split and engine."""
    pattern = f"{split}_*_{ocr_engine}.npz"
    files = list(embeddings_dir.glob(pattern))
    if not files:
        pattern = f"{split}_*_ocr_{ocr_engine}.npz"
        files = list(embeddings_dir.glob(pattern))

    if not files:
        logger.warning(
            f"No pre-extracted OCR NPZ file found for split '{split}' and engine '{ocr_engine}' in {embeddings_dir}. "
            "Please run scripts/extract_embeddings.py with --use-ocr --ocr-engine {ocr_engine} first."
        )
        return {}

    file_path = Path(files[0])
    logger.info("Loading OCR transcripts from %s...", file_path)
    data = np.load(file_path, allow_pickle=True)
    image_ids = data["image_ids"]
    raw_texts = data["raw_texts"]
    return {str(img_id): str(txt) for img_id, txt in zip(image_ids, raw_texts, strict=True)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune local VLMs on MAMI dataset")
    parser.add_argument(
        "--model-id", default="Qwen/Qwen2-VL-2B-Instruct", help="HuggingFace model ID"
    )
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument(
        "--task",
        default="singleclass",
        choices=["singleclass", "multiclass"],
        help="Target classification task",
    )
    parser.add_argument(
        "--quantize",
        default="4bit",
        choices=["none", "4bit", "8bit"],
        help="Quantization mode to reduce model footprint",
    )
    parser.add_argument("--device", default="cuda", help="Target device")
    parser.add_argument(
        "--limit", type=int, default=None, help="Limit sample count for smoke testing"
    )
    parser.add_argument(
        "--use-ocr",
        action="store_true",
        help="Use OCR-extracted text instead of dataset transcripts",
    )
    parser.add_argument(
        "--ocr-engine",
        default="easyocr",
        choices=["easyocr", "paddleocr"],
        help="OCR engine to load transcripts for",
    )
    args = parser.parse_args()

    # Pre-checks
    if not torch.cuda.is_available() and args.device == "cuda":
        logger.warning("CUDA not available, running on CPU (extremely slow for VLMs).")
        args.device = "cpu"
    device = torch.device(args.device)

    # 1. Load quant config
    from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig

    logger.info("Initializing processor for %s...", args.model_id)
    # Ensure Qwen models resize images cleanly within standard pixels to avoid memory spikes
    processor_kwargs = {}
    if "qwen2" in args.model_id.lower():
        processor_kwargs["max_pixels"] = 512 * 28 * 28

    processor = AutoProcessor.from_pretrained(args.model_id, **processor_kwargs)

    load_kwargs: dict[str, Any] = {
        "low_cpu_mem_usage": True,
        "device_map": args.device if args.quantize != "none" else None,
    }

    if args.quantize == "4bit":
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    elif args.quantize == "8bit":
        load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    else:
        load_kwargs["dtype"] = torch.float16

    logger.info("Loading model %s (%s)...", args.model_id, args.quantize)
    base_model = AutoModelForVision2Seq.from_pretrained(args.model_id, **load_kwargs)

    # 2. Setup PEFT/LoRA Adapters
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    # Prepare model for quantized training
    base_model = prepare_model_for_kbit_training(base_model)

    # Identify target modules dynamically or fall back to standard attention layers
    target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]

    # PEFT LoraConfig
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    logger.info("Injecting LoRA adapters...")
    model = get_peft_model(base_model, peft_config)
    model.print_trainable_parameters()

    # 3. Datasets and Loaders
    logger.info("Loading MAMI dataset split...")
    manager = DatasetManager()
    train_dataset = manager.load_dataset(split="train")

    if args.limit:
        train_dataset._records = train_dataset._records[: args.limit]
        logger.info("Capped training samples to %d", args.limit)

    ocr_map = None
    if args.use_ocr:
        ocr_map = load_ocr_transcripts("train", args.ocr_engine, MODELS_DIR.parent / "embeddings")

    collate_fn = VLMCollate(processor, args.model_id.lower(), args.task, ocr_map=ocr_map)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True,
    )

    # 4. Setup Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    # 5. Training Loop
    logger.info("Starting VLM fine-tuning loop...")
    model.train()

    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        t0 = time.perf_counter()

        for batch in train_loader:
            # Move inputs to device (except images, which are processed)
            model_inputs = {
                k: v.to(args.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()
            }

            optimizer.zero_grad()
            outputs = model(**model_inputs)
            loss = outputs.loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader) if train_loader else 0.0
        elapsed = time.perf_counter() - t0
        logger.info(
            "Epoch %d/%d completed | Avg Loss: %.4f | Time: %.2f seconds",
            epoch,
            args.epochs,
            avg_loss,
            elapsed,
        )

    # 6. Save Adapter Checkpoint
    model_name_clean = args.model_id.lower().split("/")[-1].replace("-", "_")
    output_dir = MODELS_DIR / f"lora_{model_name_clean}_{args.task}"
    output_dir.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(output_dir)
    logger.info("Successfully saved fine-tuned LoRA adapters to %s", output_dir)


if __name__ == "__main__":
    main()
