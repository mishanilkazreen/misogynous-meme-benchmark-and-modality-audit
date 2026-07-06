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
from torch.utils.data import DataLoader, WeightedRandomSampler

from models.vlm.classifier import (
    build_joint_prompt,
    build_joint_response,
    build_misogyny_prompt,
    build_subtype_prompt,
    build_subtype_response,
)
from utils.dataset import DatasetManager
from utils.seed import set_seed
from utils.task_names import TASK_CHOICES, canonical_task
from utils.text_source import load_text_source_transcripts, resolve_text_source

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
    """Collation helper that formats multimodal conversations and tokenizes them with masking.

    Forces the tokenizer's ``padding_side`` to ``"right"`` at construction
    time. Our label-masking assumes right-padding: we mask the first
    ``prompt_len`` positions as prompt tokens and let the tail (the response)
    remain as training targets. Left-padding (the Qwen2 tokenizer's inference
    default) would silently break this: the first ``prompt_len`` positions
    would be a mix of PAD and start-of-prompt tokens, and the rest of the
    prompt would leak through to the loss as a training target. Once during
    ``__call__`` we decode the surviving labels of the first batch to verify
    the masking is correct.
    """

    def __init__(
        self, processor: Any, model_type: str, task: str, ocr_map: dict[str, str] | None = None
    ) -> None:
        self.processor = processor
        # Force right-padding for training. Qwen2's tokenizer defaults to
        # left-padding for inference-time generation; that convention is
        # incompatible with our labels[i, :prompt_len] = -100 masking.
        # This overrides whatever the processor's tokenizer was doing.
        self.processor.tokenizer.padding_side = "right"

        self.model_type = model_type
        self.task = task
        self.ocr_map = ocr_map
        # One-shot sanity check on the first batch we collate, so that any
        # future refactor that resets padding_side fails loud in training.
        self._first_batch_verified = False

        if task == "singleclass":
            self.base_prompt_text = build_misogyny_prompt()
        elif task == "joint":
            self.base_prompt_text = build_joint_prompt()
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

            # Target response. Task A stays yes/no. Task B and the new
            # joint task both emit JSON so the fine-tuned model learns the
            # exact schema that ``models.vlm.classifier.extract_subtypes``
            # parses at inference (docs/CODE_REVIEW_ISSUES.md §6.1, §6.3).
            if self.task == "singleclass":
                resp = "yes" if sample["misogynous"] == 1 else "no"
            elif self.task == "joint":
                resp = build_joint_response(sample)
            else:
                resp = build_subtype_response(sample)

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

        # One-shot sanity check: decode the first batch's surviving labels and
        # verify they match the intended target response. Fails loud if a
        # future refactor flips padding_side back to "left".
        if not self._first_batch_verified:
            self._verify_label_masking(inputs["labels"], responses)
            self._first_batch_verified = True

        return inputs

    def _verify_label_masking(self, labels: Any, responses: list[str]) -> None:
        """Decode the surviving labels of every row and confirm they contain the response.

        We only require that each row's decoded target contain the expected
        response substring. This tolerates chat-template EOS tokens or other
        model-specific trailing tokens without brittle exact-match checks.
        Raises ``RuntimeError`` if a mismatch is detected, which surfaces the
        padding-side bug immediately at training start rather than after
        hours of wasted compute.
        """
        # Assert right-padding hasn't been reset since construction.
        assert self.processor.tokenizer.padding_side == "right", (
            "VLMCollate expects processor.tokenizer.padding_side='right'; "
            f"found {self.processor.tokenizer.padding_side!r}. Label masking "
            "will train the model on prompt tokens instead of responses. "
            "See docs/CODE_REVIEW_ISSUES.md \u00a71.1."
        )
        for i, expected_response in enumerate(responses):
            row = labels[i]
            kept = row[row != -100]
            decoded = self.processor.tokenizer.decode(kept, skip_special_tokens=True).strip()
            # For non-empty responses, the decoded target must include the
            # response. Empty responses (e.g. task-B "none") produce a short
            # decoded string ("none" or similar); skip the substring check
            # only when the expected string is empty.
            if expected_response and expected_response.lower() not in decoded.lower():
                raise RuntimeError(
                    "VLMCollate label masking is producing wrong training "
                    "targets. Row %d expected response %r inside decoded "
                    "target %r. This usually means padding_side is 'left' "
                    "somewhere in the tokenizer stack. See "
                    "docs/CODE_REVIEW_ISSUES.md \u00a71.1." % (i, expected_response, decoded)
                )


def load_ocr_transcripts(split: str, ocr_engine: str, embeddings_dir: Path) -> dict[str, str]:
    """Deprecated shim delegating to :func:`utils.text_source.load_text_source_transcripts`."""
    return load_text_source_transcripts(split, "ocr", ocr_engine, embeddings_dir)


def compute_vlm_sample_weights(records: list[dict[str, Any]]) -> list[float]:
    """Return per-sample weights for a ``WeightedRandomSampler`` on MAMI train.

    Task B sub-types are imbalanced (shaming ~14 %, stereotype ~31 %,
    objectification ~29 %, violence ~13 %). Uniform sampling means the
    QLoRA loss sees mostly negative-of-rare-class targets, and the trained
    model biases toward always-negative on the rare classes. Reweighting
    samples so those with rare-class positives are more likely to be
    drawn evens out the training signal (docs/CODE_REVIEW_ISSUES.md \u00a76.4).

    Weight scheme: base 1.0 per sample plus 3.0 for each rare-class
    positive (shaming, violence) and 1.0 for each common-class positive
    (stereotype, objectification). Empirically this triples the training
    frequency of shaming/violence-positive memes without starving the
    model of easy negatives.
    """
    weights: list[float] = []
    for row in records:
        rare = int(row.get("shaming", 0)) + int(row.get("violence", 0))
        common = int(row.get("stereotype", 0)) + int(row.get("objectification", 0))
        weights.append(1.0 + 3.0 * rare + 1.0 * common)
    return weights


@torch.no_grad()
def _validate_vlm(
    model: Any,
    processor: Any,
    val_dataset: Any,
    model_type: str,
    task: str,
    device: str,
    ocr_map: dict[str, str] | None = None,
    batch_size: int = 4,
    max_new_tokens: int = 100,
    limit: int | None = 200,
) -> float:
    """Run generation-based inference on the validation split and return the primary metric.

    Task A (singleclass): macro F1.
    Task B (multiclass) and joint: MAMI 2022 official mami_score_b.

    Temporarily switches the processor to left-padding for batched generation
    (Qwen2's inference default), then restores right-padding so training can
    resume. The ``VLMCollate._first_batch_verified`` flag is intentionally NOT
    reset here; the masking assertion only runs on the training path.

    See docs/CODE_REVIEW_ISSUES.md \u00a72.4.
    """
    from models.vlm.classifier import (
        MISOGYNY_LABELS,
        SUBTYPE_LABELS,
        build_joint_prompt,
        build_misogyny_prompt,
        build_subtype_prompt,
        extract_joint,
        extract_label,
        extract_subtypes,
    )
    from models.vlm.metrics_multilabel import compute_mami_score_b
    from sklearn.metrics import f1_score

    n_val = min(len(val_dataset), limit) if limit is not None else len(val_dataset)
    if n_val == 0:
        return 0.0

    was_training = model.training
    model.eval()

    # Restore left-padding for generation (Qwen2's inference default).
    # VLMCollate forced right-padding on the processor; undo that for the
    # duration of validation so batched generation is correct.
    prev_padding_side = processor.tokenizer.padding_side
    processor.tokenizer.padding_side = "left"

    if task == "singleclass":
        base_prompt = build_misogyny_prompt()
    elif task == "joint":
        base_prompt = build_joint_prompt()
    else:
        base_prompt = build_subtype_prompt()

    all_preds: list[int] = []
    all_gts: list[int] = []
    pred_dicts: list[dict[str, int]] = []
    gt_dicts: list[dict[str, int]] = []

    try:
        for batch_start in range(0, n_val, batch_size):
            batch = [val_dataset[i] for i in range(batch_start, min(batch_start + batch_size, n_val))]
            pils = []
            texts = []
            for sample in batch:
                arr = image_to_numpy(sample["image"])
                pil = Image.fromarray(arr)
                pils.append(pil)
                image_id = str(sample["image_id"])
                if ocr_map and image_id in ocr_map:
                    ocr_text = ocr_map[image_id].strip()
                    prompt_text = f'This meme contains the text: "{ocr_text}". {base_prompt}'
                else:
                    prompt_text = base_prompt
                if "qwen2" in model_type:
                    conversation = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": pil},
                                {"type": "text", "text": prompt_text},
                            ],
                        }
                    ]
                    texts.append(
                        processor.apply_chat_template(
                            conversation, tokenize=False, add_generation_prompt=True
                        )
                    )
                else:
                    conversation = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image"},
                                {"type": "text", "text": prompt_text},
                            ],
                        }
                    ]
                    texts.append(
                        processor.apply_chat_template(conversation, add_generation_prompt=True)
                    )
            inputs = processor(
                images=pils, text=texts, padding=True, return_tensors="pt"
            )
            inputs = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in inputs.items()
            }
            output_ids = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False
            )
            input_len = inputs["input_ids"].shape[1]
            responses = processor.batch_decode(
                output_ids[:, input_len:], skip_special_tokens=True
            )
            for sample, resp in zip(batch, responses):
                resp = resp.strip()
                if task == "singleclass":
                    matched = extract_label(resp, list(MISOGYNY_LABELS))
                    pred_int = 1 if matched == "yes" else 0
                    gt_int = int(sample["misogynous"])
                    all_preds.append(pred_int)
                    all_gts.append(gt_int)
                elif task == "joint":
                    joint_labels = ["misogynous"] + list(SUBTYPE_LABELS)
                    parsed = extract_joint(resp, joint_labels)
                    all_preds.append(parsed.get("misogynous", 0))
                    all_gts.append(int(sample["misogynous"]))
                    pred_dicts.append({lbl: parsed.get(lbl, 0) for lbl in SUBTYPE_LABELS})
                    gt_dicts.append({lbl: int(sample.get(lbl, 0)) for lbl in SUBTYPE_LABELS})
                else:
                    parsed = extract_subtypes(resp, list(SUBTYPE_LABELS))
                    pred_dicts.append(parsed)
                    gt_dicts.append({lbl: int(sample.get(lbl, 0)) for lbl in SUBTYPE_LABELS})
    finally:
        # Always restore padding side and training mode regardless of errors.
        processor.tokenizer.padding_side = prev_padding_side
        if was_training:
            model.train()

    if task == "singleclass":
        if not all_gts:
            return 0.0
        return float(f1_score(all_gts, all_preds, average="macro", zero_division=0))
    mami = compute_mami_score_b(pred_dicts, gt_dicts, list(SUBTYPE_LABELS))
    return float(mami["mami_score_b"])


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
        type=canonical_task,
        choices=TASK_CHOICES,
        help=(
            "Target classification task: 'binary'/'singleclass' = Task A binary "
            "misogyny (yes/no target), 'multilabel'/'multiclass' = Task B "
            "multi-label sub-types (4-key JSON target), 'joint' = both tasks "
            "in a single adapter (5-key JSON target). See "
            "docs/CODE_REVIEW_ISSUES.md \u00a74.1, \u00a76.3."
        ),
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
        "--text-source",
        default=None,
        choices=["provided", "ocr", "combined"],
        help=(
            "Where the text modality comes from. Default is 'provided' "
            "(MAMI's text-transcription column). Set 'ocr' or 'combined' to "
            "load pre-extracted NPZ transcripts from results/embeddings/."
        ),
    )
    parser.add_argument(
        "--use-ocr",
        action="store_true",
        help=(
            "Deprecated alias: equivalent to --text-source ocr. Kept for "
            "backward compatibility."
        ),
    )
    parser.add_argument(
        "--ocr-engine",
        default="easyocr",
        choices=["easyocr", "paddleocr"],
        help="OCR engine that produced the pre-extracted transcripts",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help=(
            "RNG seed for reproducible QLoRA training. Applied to Python's "
            "random module, NumPy, and PyTorch (CPU + CUDA). See "
            "docs/CODE_REVIEW_ISSUES.md \u00a73.1."
        ),
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=8,
        help=(
            "Number of micro-batches per optimizer step. Effective batch size "
            "is ``batch-size * gradient-accumulation-steps``. Default 8 lifts "
            "the effective batch from 2 to 16 without extra VRAM. See "
            "docs/CODE_REVIEW_ISSUES.md \u00a72.5."
        ),
    )
    parser.add_argument(
        "--sampler",
        default="uniform",
        choices=["uniform", "balanced"],
        help=(
            "Training sampler. 'uniform' shuffles the training set uniformly "
            "(default). 'balanced' uses a WeightedRandomSampler that "
            "up-weights samples with rare-class positives (shaming, violence). "
            "Recommended for multi-label and joint training. See "
            "docs/CODE_REVIEW_ISSUES.md \u00a76.4."
        ),
    )
    parser.add_argument(
        "--val-limit",
        type=int,
        default=200,
        help=(
            "Maximum number of validation samples used for best-val checkpoint "
            "selection after each training epoch. Default 200 keeps per-epoch "
            "validation under ~10 minutes even for 7B models. Set to 0 to "
            "disable per-epoch validation (last-epoch checkpoint is saved). "
            "See docs/CODE_REVIEW_ISSUES.md \u00a72.4."
        ),
    )
    args = parser.parse_args()

    # Seed every RNG before touching the dataset or the model.
    set_seed(args.seed)

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

    # LoRA target modules: ``all-linear`` covers self-attention *and* the MLP
    # projections (gate_proj, up_proj, down_proj), which contain the majority
    # of Qwen2-VL and LLaVA-1.5 parameters. The pre-fix config only touched
    # attention (q_proj, k_proj, v_proj, o_proj), halving effective LoRA
    # capacity. See docs/CODE_REVIEW_ISSUES.md \u00a72.6.
    #
    # ``task_type=None`` lets PEFT infer the task from the model class instead
    # of forcing ``CAUSAL_LM``, which is technically wrong for Vision2Seq
    # models and can trigger subtle mis-attachment on newer PEFT versions.
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules="all-linear",
        lora_dropout=0.05,
        bias="none",
        task_type=None,
    )

    logger.info("Injecting LoRA adapters...")
    model = get_peft_model(base_model, peft_config)
    model.print_trainable_parameters()

    # 3. Datasets and Loaders
    logger.info("Loading MAMI dataset split...")
    manager = DatasetManager()
    train_dataset = manager.load_dataset(split="train")

    if args.limit:
        train_dataset._records = train_dataset._records[: args.limit]  # pylint: disable=protected-access
        logger.info("Capped training samples to %d", args.limit)

    text_source = resolve_text_source(args.text_source, args.use_ocr)
    logger.info("Text source for training: %s", text_source)
    ocr_map: dict[str, str] | None = None
    if text_source != "provided":
        ocr_map = load_text_source_transcripts(
            "train", text_source, args.ocr_engine, MODELS_DIR.parent / "embeddings"
        )
        if not ocr_map:
            logger.warning(
                "Text-source NPZ not found; falling back to dataset transcripts."
            )
            ocr_map = None

    # Load validation dataset for best-val checkpoint selection
    # (docs/CODE_REVIEW_ISSUES.md \u00a72.4).
    val_dataset = manager.load_dataset(split="validation")
    val_ocr_map: dict[str, str] | None = None
    if text_source != "provided":
        val_ocr_map = load_text_source_transcripts(
            "validation", text_source, args.ocr_engine, MODELS_DIR.parent / "embeddings"
        ) or None
    logger.info(
        "Validation dataset loaded (%d samples; will validate on up to %s per epoch).",
        len(val_dataset),
        str(args.val_limit) if args.val_limit > 0 else "disabled",
    )

    collate_fn = VLMCollate(processor, args.model_id.lower(), args.task, ocr_map=ocr_map)

    loader_kwargs: dict[str, Any] = {
        "batch_size": args.batch_size,
        "collate_fn": collate_fn,
        "drop_last": True,
    }
    if args.sampler == "balanced":
        sample_weights = compute_vlm_sample_weights(train_dataset._records)  # pylint: disable=protected-access
        loader_kwargs["sampler"] = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_dataset),
            replacement=True,
        )
        logger.info(
            "Using class-balanced WeightedRandomSampler over %d training samples.",
            len(train_dataset),
        )
    else:
        loader_kwargs["shuffle"] = True

    train_loader = DataLoader(train_dataset, **loader_kwargs)

    # 4. Setup Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    # 5. Training Loop
    logger.info("Starting VLM fine-tuning loop...")
    model.train()

    best_val_metric = -1.0
    best_trainable_state: dict[str, torch.Tensor] | None = None

    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        t0 = time.perf_counter()

        # Gradient accumulation gives an effective batch size of
        # ``args.batch_size * args.gradient_accumulation_steps`` without the
        # VRAM cost of a larger per-step batch. QLoRA with batch size 2 has
        # extremely noisy gradients (docs/CODE_REVIEW_ISSUES.md \u00a72.5); an
        # accumulation of 8 raises the effective batch to 16 and stabilises
        # convergence.
        optimizer.zero_grad()
        n_accum = max(1, int(args.gradient_accumulation_steps))
        for step, batch in enumerate(train_loader):
            # Move inputs to device (except non-tensor entries).
            model_inputs = {
                k: v.to(args.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()
            }

            outputs = model(**model_inputs)
            loss = outputs.loss / n_accum
            loss.backward()

            # Step every ``n_accum`` micro-batches (or at the end of the loop
            # to avoid dropping the final residual).
            if ((step + 1) % n_accum == 0) or (step + 1 == len(train_loader)):
                torch.nn.utils.clip_grad_norm_(
                    (p for p in model.parameters() if p.requires_grad), 1.0
                )
                optimizer.step()
                optimizer.zero_grad()

            # Track the un-scaled loss so the logged epoch loss is comparable
            # across different accumulation settings.
            total_loss += loss.item() * n_accum

        avg_loss = total_loss / len(train_loader) if train_loader else 0.0
        elapsed = time.perf_counter() - t0
        logger.info(
            "Epoch %d/%d completed | Avg Loss: %.4f | Time: %.2f seconds",
            epoch,
            args.epochs,
            avg_loss,
            elapsed,
        )

        # Best-val checkpoint selection (docs/CODE_REVIEW_ISSUES.md \u00a72.4).
        # Skip if --val-limit 0 was passed (caller opts out of per-epoch validation).
        if args.val_limit != 0:
            logger.info(
                "Running validation on up to %d samples after epoch %d...",
                args.val_limit,
                epoch,
            )
            val_metric = _validate_vlm(
                model,
                processor,
                val_dataset,
                model_type=args.model_id.lower(),
                task=args.task,
                device=args.device,
                ocr_map=val_ocr_map,
                batch_size=args.batch_size,
                limit=args.val_limit if args.val_limit > 0 else None,
            )
            logger.info("Epoch %d val_metric=%.4f", epoch, val_metric)
            if val_metric > best_val_metric:
                best_val_metric = val_metric
                # Save only the trainable (LoRA) weights to CPU. The quantized
                # base-model weights are frozen and identical across epochs, so
                # we do not need to clone them.
                best_trainable_state = {
                    name: param.detach().cpu().clone()
                    for name, param in model.named_parameters()
                    if param.requires_grad
                }
                logger.info(
                    "New best val_metric=%.4f saved at epoch %d.", best_val_metric, epoch
                )

    # 6. Restore best-val LoRA weights before saving.
    # This ensures the persisted adapter corresponds to the checkpoint that
    # scored best on the validation split, not whatever the last epoch produced.
    if best_trainable_state is not None:
        logger.info(
            "Restoring best-val checkpoint (val_metric=%.4f) before saving.",
            best_val_metric,
        )
        for name, param in model.named_parameters():
            if name in best_trainable_state:
                param.data.copy_(best_trainable_state[name].to(param.device))
    elif args.val_limit != 0:
        logger.warning(
            "No best-val checkpoint recorded (val_limit=%d); saving last-epoch weights.",
            args.val_limit,
        )

    # 7. Save Adapter Checkpoint
    # Include the seed in the output path so that multi-seed runs do not
    # overwrite each other (docs/CODE_REVIEW_ISSUES.md \u00a73.1).
    model_name_clean = args.model_id.lower().split("/")[-1].replace("-", "_")
    output_dir = MODELS_DIR / f"lora_{model_name_clean}_{args.task}_seed{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(str(output_dir))
    logger.info("Successfully saved fine-tuned LoRA adapters to %s", output_dir)


if __name__ == "__main__":
    main()
