"""EXP-04 / EXP-06: frozen-backbone features for the modality ablation.

Extracts, for the MAMI 2022 train, validation and test splits, three feature sets that are
consumed by ``scripts/revision/modality_ablation.py``:

``clip``
    CLIP ViT-L-14-quickgelu (OpenAI weights, ``open_clip``) image embeddings plus text
    embeddings of the PROVIDED MAMI transcription. The pipeline is a line-by-line copy of
    ``scripts/extract_embeddings.py --text-source provided``: images go through
    ``utils.dataset.DatasetManager`` and the same float-to-uint8 conversion, text is
    ``text.strip() or "empty text"`` truncated by the CLIP tokenizer at 77 tokens, and both
    embeddings are L2-normalised. Output name matches the main-benchmark configuration
    (``{split}_vit_l_14_quickgelu.npz``).
``masked``
    Text-masked images: EasyOCR (CRAFT detector) text boxes are rasterised into a mask, dilated,
    and filled with ``cv2.inpaint`` (Telea). The masked image is embedded with the same CLIP
    model. The text block is the provided-text CLIP embedding, so only the image block changes.
``siglip``
    SigLIP SO400M/14 at 384 px (``open_clip`` ``ViT-SO400M-14-SigLIP-384`` / ``webli``, the
    ``timm`` conversion of ``google/siglip-so400m-patch14-384``), image plus provided-text
    embeddings (64-token SigLIP context), L2-normalised.

Masking quality is audited on a seeded sample of test memes by running EasyOCR recognition on
the original and the masked image and counting recognised alphanumeric characters.
Five seeded-random test memes are saved as original|masked PNG pairs for figures.

Usage:
    $env:CUDA_VISIBLE_DEVICES = "0"
    uv run python -m scripts.revision.extract_features --features clip,masked,siglip
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import random
import sys
import time
from typing import Any

# utils.dataset imports `datasets`, which must be imported before torch (OpenMP segfault).
from utils.dataset import DatasetManager  # isort: split

import cv2
import easyocr
import numpy as np
import open_clip
from open_clip.pretrained import download_pretrained
from open_clip.transform import PreprocessCfg, image_transform_v2
from PIL import Image
from safetensors import safe_open
import torch
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("extract_features")

ROOT = Path(__file__).resolve().parents[2]
EMB_DIR = ROOT / "results" / "embeddings"
OUT_DIR = ROOT / "results" / "revision" / "exp04_exp06"
EXAMPLES_DIR = OUT_DIR / "examples"

SPLITS: tuple[str, ...] = ("train", "validation", "test")
CLIP_MODEL = "ViT-L-14-quickgelu"
CLIP_PRETRAINED = "openai"
SIGLIP_MODEL = "ViT-SO400M-14-SigLIP-384"
SIGLIP_PRETRAINED = "webli"
SIGLIP_HF_EQUIVALENT = "google/siglip-so400m-patch14-384"

# Text detector settings (EasyOCR CRAFT defaults except where noted).
DETECTOR = "easyocr-craft"
DET_MIN_SIZE = 10  # EasyOCR default 20; lowered so small caption text is also masked
DET_TEXT_THRESHOLD = 0.7
DET_LOW_TEXT = 0.4
DET_LINK_THRESHOLD = 0.4
DET_ADD_MARGIN = 0.1
MASK_DILATE_PX = 5  # grows boxes to cover the dark outline of meme (Impact) fonts
INPAINT_RADIUS = 5
AUDIT_CONF = 0.3
SEED = 20260708


@dataclass
class SplitBuffers:
    """Per-split accumulators for embeddings and metadata."""

    image_ids: list[str] = field(default_factory=list)
    labels: list[int] = field(default_factory=list)
    subtask_labels: list[list[int]] = field(default_factory=list)
    raw_texts: list[str] = field(default_factory=list)
    clip_img: list[np.ndarray] = field(default_factory=list)
    clip_txt: list[np.ndarray] = field(default_factory=list)
    masked_img: list[np.ndarray] = field(default_factory=list)
    siglip_img: list[np.ndarray] = field(default_factory=list)
    siglip_txt: list[np.ndarray] = field(default_factory=list)
    mask_fraction: list[float] = field(default_factory=list)
    n_boxes: list[int] = field(default_factory=list)


@dataclass
class Encoder:
    """An ``open_clip`` model with its preprocessing transform and tokenizer."""

    name: str
    pretrained: str
    model: Any
    preprocess: Any
    tokenizer: Any


def sha256_file(path: Path) -> str:
    """Return the hex SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_encoder(name: str, pretrained: str, device: str) -> Encoder:
    """Load a frozen ``open_clip`` model in eval mode on ``device``."""
    logger.info("Loading %s (%s)", name, pretrained)
    model, _, preprocess = open_clip.create_model_and_transforms(name, pretrained=pretrained)
    model = model.to(device)
    model.eval()
    return Encoder(name, pretrained, model, preprocess, open_clip.get_tokenizer(name))


def load_encoder_direct(name: str, pretrained: str, device: str) -> Encoder:
    """Load an ``open_clip`` model with its weights read straight onto ``device``.

    Equivalent to :func:`load_encoder` (same pretrained config, weights and eval transform), but
    the random-init model is moved to ``device`` before the safetensors file is read directly to
    ``device``. On the local Windows host the standard CPU-side load of the 3.5 GB SigLIP
    SO400M checkpoint terminates with an access violation (0xC0000005) under memory pressure.
    """
    logger.info("Loading %s (%s) directly to %s", name, pretrained, device)
    cfg = open_clip.get_pretrained_cfg(name, pretrained)
    model = open_clip.create_model(name, pretrained=None).to(device)
    weights = download_pretrained(cfg)
    target = model.state_dict()
    with safe_open(weights, framework="pt", device="cpu") as fh:
        keys = set(fh.keys())
        if keys != set(target):
            raise RuntimeError(
                f"checkpoint/model key mismatch: missing={sorted(set(target) - keys)[:5]} "
                f"unexpected={sorted(keys - set(target))[:5]}"
            )
        with torch.no_grad():
            for key in keys:  # one tensor at a time keeps peak host and device memory low
                target[key].copy_(fh.get_tensor(key))
    logger.info("Copied %d tensors from %s", len(keys), weights)
    model.eval()
    size = model.visual.image_size
    pp_cfg = PreprocessCfg(
        size=size if isinstance(size, int) else tuple(size),
        mean=tuple(cfg["mean"]),
        std=tuple(cfg["std"]),
        interpolation=cfg["interpolation"],
        resize_mode=cfg["resize_mode"],
    )
    preprocess = image_transform_v2(pp_cfg, is_train=False)
    return Encoder(name, pretrained, model, preprocess, open_clip.get_tokenizer(name))


def encode_images(enc: Encoder, images: list[Image.Image], device: str) -> np.ndarray:
    """Return L2-normalised image embeddings for a batch of PIL images."""
    batch = torch.stack([enc.preprocess(im) for im in images]).to(device)
    with torch.no_grad():
        feats = enc.model.encode_image(batch)
        feats = feats / feats.norm(dim=-1, keepdim=True)
    out: np.ndarray = feats.float().cpu().numpy()
    return out


def encode_texts(enc: Encoder, texts: list[str], device: str) -> np.ndarray:
    """Return L2-normalised text embeddings (tokenizer truncates to the model context)."""
    tokens = enc.tokenizer([t.strip() or "empty text" for t in texts]).to(device)
    with torch.no_grad():
        feats = enc.model.encode_text(tokens)
        feats = feats / feats.norm(dim=-1, keepdim=True)
    out: np.ndarray = feats.float().cpu().numpy()
    return out


def text_mask(reader: easyocr.Reader, img: np.ndarray) -> tuple[np.ndarray, int]:
    """Detect text regions with EasyOCR and return a dilated uint8 mask and the box count."""
    horizontal, free = reader.detect(
        img,
        min_size=DET_MIN_SIZE,
        text_threshold=DET_TEXT_THRESHOLD,
        low_text=DET_LOW_TEXT,
        link_threshold=DET_LINK_THRESHOLD,
        add_margin=DET_ADD_MARGIN,
    )
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    n = 0
    for x_min, x_max, y_min, y_max in horizontal[0]:
        x0, x1 = max(0, int(x_min)), min(w, int(x_max))
        y0, y1 = max(0, int(y_min)), min(h, int(y_max))
        if x1 > x0 and y1 > y0:
            mask[y0:y1, x0:x1] = 255
            n += 1
    for poly in free[0]:
        pts = np.array(poly, dtype=np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(mask, [pts], 255)
        n += 1
    if n and MASK_DILATE_PX > 0:
        k = 2 * MASK_DILATE_PX + 1
        mask = cv2.dilate(mask, np.ones((k, k), dtype=np.uint8))
    return mask, n


def mask_text(reader: easyocr.Reader, img: np.ndarray) -> tuple[np.ndarray, float, int]:
    """Return the inpainted image, masked-pixel fraction and number of detected boxes."""
    mask, n = text_mask(reader, img)
    if n == 0:
        return img.copy(), 0.0, 0
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    filled = cv2.inpaint(bgr, mask, INPAINT_RADIUS, cv2.INPAINT_TELEA)
    return cv2.cvtColor(filled, cv2.COLOR_BGR2RGB), float((mask > 0).mean()), n


def recognised_chars(reader: easyocr.Reader, img: np.ndarray) -> int:
    """Count alphanumeric characters EasyOCR recognises with confidence >= ``AUDIT_CONF``."""
    total = 0
    for _box, text, conf in reader.readtext(img):
        if float(conf) >= AUDIT_CONF:
            total += sum(ch.isalnum() for ch in str(text))
    return total


def save_example(img: np.ndarray, masked: np.ndarray, image_id: str) -> Path:
    """Save an original|masked side-by-side PNG (height 256 px) and return its path."""
    height = 256
    scale = height / img.shape[0]
    size = (max(1, round(img.shape[1] * scale)), height)
    left = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    right = cv2.resize(masked, size, interpolation=cv2.INTER_AREA)
    gap = np.full((height, 8, 3), 255, dtype=np.uint8)
    path = EXAMPLES_DIR / f"masked_{image_id}.png"
    Image.fromarray(np.concatenate([left, gap, right], axis=1)).save(path, optimize=True)
    return path


def flush(
    buf: SplitBuffers,
    pil_orig: list[Image.Image],
    pil_masked: list[Image.Image],
    texts: list[str],
    encoders: dict[str, Encoder],
    device: str,
) -> None:
    """Encode one batch with every active encoder and append the results to ``buf``."""
    if "clip" in encoders:
        buf.clip_img.append(encode_images(encoders["clip"], pil_orig, device))
        buf.clip_txt.append(encode_texts(encoders["clip"], texts, device))
        if pil_masked:
            buf.masked_img.append(encode_images(encoders["clip"], pil_masked, device))
    if "siglip" in encoders:
        buf.siglip_img.append(encode_images(encoders["siglip"], pil_orig, device))
        buf.siglip_txt.append(encode_texts(encoders["siglip"], texts, device))


def common_arrays(buf: SplitBuffers, split: str) -> dict[str, np.ndarray]:
    """Arrays shared by every output file (same keys as ``scripts/extract_embeddings.py``)."""
    return {
        "labels": np.array(buf.labels, dtype=np.int32),
        "subtask_labels": np.array(buf.subtask_labels, dtype=np.int32),
        "image_ids": np.array(buf.image_ids),
        "raw_texts": np.array(buf.raw_texts),
        "split": np.array(split),
        "text_source": np.array("provided"),
    }


def process_split(
    split: str,
    manager: DatasetManager,
    features: set[str],
    encoders: dict[str, Encoder],
    reader: easyocr.Reader | None,
    args: argparse.Namespace,
    log: dict[str, Any],
) -> None:
    """Extract all requested features for one split and write the ``.npz`` files."""
    dataset = manager.load_dataset(split=split)
    n = len(dataset) if args.limit is None else min(len(dataset), args.limit)
    rng = random.Random(SEED)
    example_idx = (
        set(rng.sample(range(n), 5)) if split == "test" and "masked" in features else set()
    )
    audit_idx = (
        set(rng.sample(range(n), min(args.audit_n, n)))
        if split == "test" and "masked" in features
        else set()
    )
    audit: list[dict[str, Any]] = []
    examples: list[str] = []
    buf = SplitBuffers()
    pil_orig: list[Image.Image] = []
    pil_masked: list[Image.Image] = []
    texts: list[str] = []
    t0 = time.perf_counter()
    for idx in tqdm(range(n), desc=split):
        sample = dataset[idx]
        buf.image_ids.append(sample["image_id"])
        buf.labels.append(sample["misogynous"])
        buf.subtask_labels.append(
            [sample[k] for k in ("shaming", "stereotype", "objectification", "violence")]
        )
        # Identical conversion to scripts/extract_embeddings.py.
        img_np = (sample["image"].permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
        text = str(sample.get("text", ""))
        buf.raw_texts.append(text)
        texts.append(text)
        pil_orig.append(Image.fromarray(img_np))
        if "masked" in features:
            assert reader is not None
            masked, frac, n_boxes = mask_text(reader, img_np)
            buf.mask_fraction.append(frac)
            buf.n_boxes.append(n_boxes)
            pil_masked.append(Image.fromarray(masked))
            if idx in example_idx:
                examples.append(str(save_example(img_np, masked, sample["image_id"])))
            if idx in audit_idx:
                audit.append(
                    {
                        "image_id": sample["image_id"],
                        "chars_original": recognised_chars(reader, img_np),
                        "chars_masked": recognised_chars(reader, masked),
                    }
                )
        if len(pil_orig) == args.batch_size or idx == n - 1:
            flush(buf, pil_orig, pil_masked, texts, encoders, args.device)
            pil_orig, pil_masked, texts = [], [], []
    elapsed = time.perf_counter() - t0
    logger.info("%s: %d samples in %.1f s", split, n, elapsed)

    EMB_DIR.mkdir(parents=True, exist_ok=True)
    base = common_arrays(buf, split)
    split_log: dict[str, Any] = {"n": n, "seconds": round(elapsed, 1), "files": {}}
    outputs: list[tuple[str, Path, dict[str, np.ndarray]]] = []
    if "clip" in features:
        clip_txt = np.concatenate(buf.clip_txt)
        outputs.append(
            (
                "clip",
                EMB_DIR / f"{split}_vit_l_14_quickgelu.npz",
                {
                    "image_embeddings": np.concatenate(buf.clip_img),
                    "text_embeddings": clip_txt,
                    "model": np.array(f"{CLIP_MODEL}/{CLIP_PRETRAINED}"),
                },
            )
        )
        if "masked" in features:
            outputs.append(
                (
                    "masked",
                    EMB_DIR / f"{split}_vit_l_14_quickgelu_textmasked_easyocr.npz",
                    {
                        "image_embeddings": np.concatenate(buf.masked_img),
                        "text_embeddings": clip_txt,
                        "mask_fraction": np.array(buf.mask_fraction, dtype=np.float32),
                        "n_boxes": np.array(buf.n_boxes, dtype=np.int32),
                        "model": np.array(f"{CLIP_MODEL}/{CLIP_PRETRAINED}"),
                        "masking": np.array(
                            f"{DETECTOR} min_size={DET_MIN_SIZE} dilate={MASK_DILATE_PX}px "
                            f"cv2.inpaint TELEA r={INPAINT_RADIUS}"
                        ),
                    },
                )
            )
    if "siglip" in features:
        outputs.append(
            (
                "siglip",
                EMB_DIR / f"{split}_vit_so400m_14_siglip_384.npz",
                {
                    "image_embeddings": np.concatenate(buf.siglip_img),
                    "text_embeddings": np.concatenate(buf.siglip_txt),
                    "model": np.array(f"{SIGLIP_MODEL}/{SIGLIP_PRETRAINED}"),
                },
            )
        )
    for key, path, arrays in outputs:
        payload: dict[str, Any] = {**arrays, **base}
        np.savez_compressed(path, **payload)
        split_log["files"][key] = {
            "path": str(path.relative_to(ROOT)).replace("\\", "/"),
            "sha256": sha256_file(path),
            "image_dim": int(arrays["image_embeddings"].shape[1]),
            "text_dim": int(arrays["text_embeddings"].shape[1]),
        }
        logger.info("Saved %s", path)
    if buf.mask_fraction:
        fracs = np.array(buf.mask_fraction)
        split_log["masking"] = {
            "mean_mask_fraction": float(fracs.mean()),
            "median_mask_fraction": float(np.median(fracs)),
            "share_no_text_detected": float(np.mean(np.array(buf.n_boxes) == 0)),
            "mean_boxes": float(np.mean(buf.n_boxes)),
        }
    if audit:
        orig = sum(a["chars_original"] for a in audit)
        left = sum(a["chars_masked"] for a in audit)
        split_log["mask_audit"] = {
            "n_images": len(audit),
            "recogniser": "easyocr en, conf >= 0.3, alphanumeric characters",
            "chars_original": orig,
            "chars_masked": left,
            "residual_char_ratio": left / orig if orig else None,
            "per_image": audit,
        }
    if examples:
        split_log["examples"] = [
            str(Path(p).relative_to(ROOT)).replace("\\", "/") for p in sorted(examples)
        ]
    log["splits"][split] = split_log


def main() -> int:
    """Parse arguments, run extraction for every split, and write the provenance log."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--features", default="clip,masked,siglip")
    parser.add_argument("--splits", default=",".join(SPLITS))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None, help="Debug: cap samples per split")
    parser.add_argument("--audit-n", type=int, default=200)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--log", default=str(OUT_DIR / "extraction_log.json"))
    args = parser.parse_args()

    features = {f.strip() for f in args.features.split(",") if f.strip()}
    unknown = features - {"clip", "masked", "siglip"}
    if unknown:
        parser.error(f"unknown features: {sorted(unknown)}")
    if "masked" in features:
        features.add("clip")  # the masked file reuses the provided-text CLIP text block
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.error("CUDA requested but unavailable (check CUDA_VISIBLE_DEVICES)")
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    encoders: dict[str, Encoder] = {}
    if "clip" in features:
        encoders["clip"] = load_encoder(CLIP_MODEL, CLIP_PRETRAINED, args.device)
    if "siglip" in features:
        encoders["siglip"] = load_encoder_direct(SIGLIP_MODEL, SIGLIP_PRETRAINED, args.device)
        logger.info("SigLIP preprocess: %s", encoders["siglip"].preprocess)
    reader = easyocr.Reader(["en"], gpu=args.device == "cuda") if "masked" in features else None

    log: dict[str, Any] = {
        "task": "EXP-04/EXP-06 feature extraction",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": " ".join(sys.argv),
        "features": sorted(features),
        "clip": {"model": CLIP_MODEL, "pretrained": CLIP_PRETRAINED, "context_tokens": 77},
        "siglip": {
            "model": SIGLIP_MODEL,
            "pretrained": SIGLIP_PRETRAINED,
            "hf_equivalent": SIGLIP_HF_EQUIVALENT,
            "context_tokens": 64,
        },
        "text_source": "provided",
        "l2_normalised": True,
        "masking": {
            "detector": DETECTOR,
            "easyocr_version": str(easyocr.__version__),
            "min_size": DET_MIN_SIZE,
            "text_threshold": DET_TEXT_THRESHOLD,
            "low_text": DET_LOW_TEXT,
            "link_threshold": DET_LINK_THRESHOLD,
            "add_margin": DET_ADD_MARGIN,
            "dilate_px": MASK_DILATE_PX,
            "fill": f"cv2.inpaint INPAINT_TELEA radius={INPAINT_RADIUS}",
            "example_and_audit_selection": f"random.Random({SEED}) over test indices",
        },
        "device": torch.cuda.get_device_name(0) if args.device == "cuda" else "cpu",
        "torch": torch.__version__,
        "open_clip": open_clip.__version__,
        "limit": args.limit,
        "splits": {},
    }
    manager = DatasetManager()
    t0 = time.perf_counter()
    for split in [s.strip() for s in args.splits.split(",")]:
        process_split(split, manager, features, encoders, reader, args, log)
    log["total_seconds"] = round(time.perf_counter() - t0, 1)
    Path(args.log).write_text(json.dumps(log, indent=2), encoding="utf-8")
    logger.info("Wrote %s", args.log)
    return 0


if __name__ == "__main__":
    sys.exit(main())
