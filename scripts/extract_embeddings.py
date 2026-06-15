"""
Script to extract and save raw image and text embeddings for MAMI 2022 dataset splits.
Supports using either dataset ground-truth text or OCR-extracted text.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import time

import numpy as np
import open_clip  # type: ignore[import-untyped]
from PIL import Image
import torch
from tqdm import tqdm

from utils.dataset import DatasetManager
from utils.ocr import OCRPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Constants for sub-task labels in MAMI dataset
SUBTYPE_LABELS = ["shaming", "stereotype", "objectification", "violence"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract CLIP embeddings for MAMI 2022 dataset")
    parser.add_argument(
        "--model",
        default="ViT-L-14",
        help="CLIP model name (e.g. ViT-B-32, ViT-L-14)",
    )
    parser.add_argument(
        "--pretrained",
        default="openai",
        help="Pretrained weights name (e.g. openai, laion2b_s32b_b82k)",
    )
    parser.add_argument(
        "--split",
        default="validation",
        help="Comma-separated splits to process (e.g., train,validation,test)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap the number of samples processed per split (useful for testing)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device to run inference on (cuda or cpu)",
    )
    parser.add_argument(
        "--use-ocr",
        action="store_true",
        help="Use OCR pipeline to extract text from images instead of dataset transcripts",
    )
    parser.add_argument(
        "--ocr-engine",
        default="easyocr",
        choices=["easyocr", "paddleocr"],
        help="OCR engine to use if --use-ocr is enabled",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for embedding extraction",
    )
    parser.add_argument(
        "--output-dir",
        default="results/embeddings",
        help="Directory to save extracted embeddings",
    )
    args = parser.parse_args()

    # Determine device
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    logger.info("Using device: %s", device)

    # Output directory setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize CLIP model and transforms
    logger.info("Loading CLIP model %s (%s)...", args.model, args.pretrained)
    model, _, preprocess = open_clip.create_model_and_transforms(
        args.model, pretrained=args.pretrained
    )
    model = model.to(device)
    model.eval()
    tokenizer = open_clip.get_tokenizer(args.model)

    # Initialize OCR Pipeline if requested
    ocr_pipe = None
    if args.use_ocr:
        logger.info("Initializing OCR Pipeline (%s)...", args.ocr_engine)
        gpu_bool = device == "cuda"
        ocr_pipe = OCRPipeline(gpu=gpu_bool, engine=args.ocr_engine)

    # Initialize dataset manager
    manager = DatasetManager()
    splits = [s.strip() for s in args.split.split(",")]

    for split in splits:
        logger.info("Loading split: %s", split)
        try:
            dataset = manager.load_dataset(split=split)
        except Exception as e:
            logger.error("Failed to load split %s: %s", split, e)
            continue

        num_samples = len(dataset)
        if args.limit is not None:
            num_samples = min(num_samples, args.limit)

        logger.info("Processing %d samples for split %s", num_samples, split)

        # Arrays to collect results
        image_ids: list[str] = []
        labels: list[int] = []
        subtask_labels: list[list[int]] = []  # shaming, stereotype, objectification, violence
        raw_texts: list[str] = []

        # We will extract image/text features in batches
        img_features_list: list[torch.Tensor] = []
        txt_features_list: list[torch.Tensor] = []

        batch_imgs: list[torch.Tensor] = []
        batch_txts: list[str] = []

        start_time = time.perf_counter()

        for idx in tqdm(range(num_samples), desc=f"Extracting {split}"):
            sample = dataset[idx]

            image_ids.append(sample["image_id"])
            labels.append(sample["misogynous"])
            subtask_labels.append(
                [
                    sample["shaming"],
                    sample["stereotype"],
                    sample["objectification"],
                    sample["violence"],
                ]
            )

            # Get image and convert torch tensor [3, H, W] to PIL Image
            img_tensor = sample["image"]
            img_np = img_tensor.permute(1, 2, 0).numpy()
            img_np = (img_np * 255.0).astype(np.uint8)
            pil_img = Image.fromarray(img_np)

            # Get text (either from OCR or dataset transcription)
            if args.use_ocr and ocr_pipe is not None:
                text = ocr_pipe.extract_and_normalize(img_np)
            else:
                text = sample["text"]

            raw_texts.append(text)

            # Keep track of words to avoid context length overflow in CLIP tokenizer (max 77 tokens)
            text_cleaned = text.strip()
            words = text_cleaned.split()
            if len(words) > 60:
                text_cleaned = " ".join(words[:60])
            if not text_cleaned:
                text_cleaned = (
                    "empty text"  # fallback placeholder to avoid empty tokenization errors
                )

            # Add to batch buffers
            batch_imgs.append(preprocess(pil_img))
            batch_txts.append(text_cleaned)

            # If batch is full, execute encoding pass
            if len(batch_imgs) == args.batch_size or idx == num_samples - 1:
                # Process image batch
                img_batch_tensor = torch.stack(batch_imgs).to(device)
                with torch.no_grad():
                    img_feats = model.encode_image(img_batch_tensor)
                    # L2 Normalise features
                    img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
                    img_features_list.append(img_feats.cpu())

                # Process text batch
                txt_tokens = tokenizer(batch_txts).to(device)
                with torch.no_grad():
                    txt_feats = model.encode_text(txt_tokens)
                    # L2 Normalise features
                    txt_feats = txt_feats / txt_feats.norm(dim=-1, keepdim=True)
                    txt_features_list.append(txt_feats.cpu())

                batch_imgs = []
                batch_txts = []

        # Concatenate features across batches
        image_embeddings = torch.cat(img_features_list).numpy()
        text_embeddings = torch.cat(txt_features_list).numpy()

        elapsed = time.perf_counter() - start_time
        logger.info(
            "Extraction completed for split %s in %.2f seconds (Avg latency: %.4f s/sample)",
            split,
            elapsed,
            elapsed / num_samples,
        )

        # Build paths
        ocr_suffix = ""
        if args.use_ocr:
            ocr_suffix = f"_ocr_{args.ocr_engine}"
        out_filename = f"{split}_{args.model.lower().replace('-', '_')}{ocr_suffix}.npz"
        out_path = output_dir / out_filename

        # Save to disk
        np.savez_compressed(
            out_path,
            image_embeddings=image_embeddings,
            text_embeddings=text_embeddings,
            labels=np.array(labels, dtype=np.int32),
            subtask_labels=np.array(subtask_labels, dtype=np.int32),
            image_ids=np.array(image_ids),
            raw_texts=np.array(raw_texts),
        )
        logger.info("Saved embeddings to %s", out_path)


if __name__ == "__main__":
    main()
