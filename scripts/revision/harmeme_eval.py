"""DIS-01: HarMeme (Harm-C) second-dataset evaluation pipeline.

Implements the six pipeline steps of DIS-01:

1. ``download``: Download the official Harm-C JSONL split annotations (train: 3,013; validation:
   177; test: 354) from the MOMENTA repository (``LCS2-IIITD/MOMENTA``) and the official
   ``HarMeme_Images.zip`` archive, extracting all ``covid_memes_*.png`` images into
   ``data/harmeme/images/`` and verifying every split record and image.
2. ``extract``: Extract frozen CLIP ``ViT-L-14-quickgelu`` (``openai`` weights via ``open_clip``)
   L2-normalised image and text embeddings across the Harm-C train, validation, and test splits
   into ``results/embeddings/harmeme_{split}_vit_l_14_quickgelu.npz``.
3. ``classify``: Train ``SVM-RBF`` (with ``StandardScaler`` fitted on train only and
   ``class_weight="balanced"``) and ``XGBoost`` (default settings, no scaler) on the Harm-C
   training embeddings and evaluate on validation and test, writing provenance-verified JSON
   artefacts to ``results/{validation,test}/harmeme_<split>_clip_{svm_rbf,xgboost}.json``.
4. ``qwen``: Evaluate zero-shot ``Qwen/Qwen2-VL-7B-Instruct`` (4-bit NF4 on CUDA, greedy
   decoding) on the Harm-C validation and test splits, writing
   ``results/{validation,test}/harmeme_<split>_qwen2_vl_7b_instruct.json``.
5. ``gemini``: Evaluate zero-shot ``gemini-3.1-flash-lite`` (``BLOCK_NONE``, temperature 0.0) on
   the Harm-C validation and test splits, writing
   ``results/{validation,test}/harmeme_<split>_gemini_3_1_flash_lite.json``.
6. ``summarize``: Verify all ``results/test/harmeme_test_*.json`` files against the Harm-C test
   split ground truth, compute Accuracy, Macro-F1, and 2,000-resample bootstrap 95% confidence
   intervals, and write ``results/revision/dis01_harmeme/{metrics.json,table_rows.tex,summary.md}``.

Usage:
    uv run python -m scripts.revision.harmeme_eval download
    uv run python -m scripts.revision.harmeme_eval run-sciama --device cuda
    uv run python -m scripts.revision.harmeme_eval gemini
    uv run python -m scripts.revision.harmeme_eval summarize
    uv run python -m scripts.revision.harmeme_eval compare --tag _rerun

Prompt sensitivity and replication (see sciama/JOBS.md): ``gemini`` and ``qwen`` take
``--prompt-variant`` and ``--output-tag``. Tagged files are extra artefacts: ``summarize`` reads only
the untagged files, and ``gemini`` refuses to overwrite an existing result file.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import http.cookiejar
import io
import json
import logging
import os
from pathlib import Path
import re
import shutil
import sys
import time
import urllib.parse
import urllib.request
import zipfile

import numpy as np
from PIL import Image
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tqdm import tqdm
from xgboost import XGBClassifier

from models.vlm.classifier import extract_label, yesno_to_int
from scripts.official_mami_evaluation import compute_f1

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("harmeme_eval")

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HARMEME_DIR = ROOT / "data" / "harmeme"
EMB_DIR = ROOT / "results" / "embeddings"
RESULTS_DIR = ROOT / "results"
OUT_DIR = RESULTS_DIR / "revision" / "dis01_harmeme"

SPLITS: tuple[str, ...] = ("train", "validation", "test")
EXPECTED_SPLIT_SIZES: dict[str, int] = {
    "train": 3013,
    "validation": 177,
    "test": 354,
}
EXPECTED_POSITIVE_COUNTS: dict[str, int] = {
    "train": 1064,
    "validation": 61,
    "test": 124,
}

MOMENTA_ANNOTATIONS_BASE = (
    "https://raw.githubusercontent.com/LCS2-IIITD/MOMENTA/main/HarMeme_V1/Annotations/Harm-C"
)
REMOTE_SPLIT_FILES: dict[str, str] = {
    "train": "train.jsonl",
    "validation": "val.jsonl",
    "test": "test.jsonl",
}
GDRIVE_IMAGES_FILE_ID = "1aMvOHACrG5SgMl4tm9BP1VTiqv8zFn9b"

CLIP_MODEL = "ViT-L-14-quickgelu"
CLIP_PRETRAINED = "openai"
QWEN_MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct"
GEMINI_MODEL_ID = "gemini-3.1-flash-lite"

SEED = 20260708
N_BOOT = 2000
MAX_NEW_TOKENS = 20

HARMEME_SYSTEM_INSTRUCTION = (
    "You are a content moderation research assistant analysing internet memes for an academic "
    "harmful meme detection study. Your role is purely classificatory: answer with exactly one "
    "word - 'yes' or 'no'. Do not add explanations, warnings, or refusals."
)

HARMEME_BASE_PROMPT = (
    "Is this meme harmful? A harmful meme attacks, demeans, exploits, or promotes harm against "
    "an individual, organization, community, or society. Answer with exactly one word: yes or no."
)

# Wording taken from the HarMeme dataset paper (Pramanick et al., 2021, Sections 3 and 4.2), which
# names mocking and ridiculing as harm. HARMEME_BASE_PROMPT does not. Used only as a sensitivity check.
HARMEME_DEFINITION_PROMPT = (
    "Is this meme harmful? A meme is harmful if it is implicitly or explicitly intended to cause "
    "harm to an individual, an organization, a community, or society, depending on their personal, "
    "political, social, educational or industrial background. The harm can be obvious, such as "
    "abusing, offending, disrespecting, insulting, demeaning, or disregarding them, or subtle, such "
    "as mocking or ridiculing a person or an idea. Harm includes mental abuse, defamation, emotional "
    "disturbance, and damage to public image. Answer with exactly one word: yes or no."
)

HARMEME_PROMPTS: dict[str, str] = {
    "baseline": HARMEME_BASE_PROMPT,
    "definition": HARMEME_DEFINITION_PROMPT,
}

# Five backoff waits of 2-32 s total 62 s, so one per-minute API rate-limit window can pass.
GEMINI_MAX_ATTEMPTS = 6
# A re-run counts as reproducing the committed file if at most this share of memes changes answer.
REPRODUCE_MAX_CHANGED_SHARE = 0.02

# Refusal phrases exclude the word "harmful" because "harmful" is the target class name.
REFUSAL_PHRASES: tuple[str, ...] = (
    "i cannot",
    "i'm unable",
    "i am unable",
    "i'm not able",
    "inappropriate",
    "i can't",
    "as an ai",
)


@dataclass(frozen=True)
class HarmemeRecord:
    """Single Harm-C dataset entry with binary harmful label (1 = harmful, 0 = harmless)."""

    image_id: str
    image_filename: str
    text: str
    label: int
    raw_label: str


def harmeme_dir() -> Path:
    """Return the Harm-C data directory, honouring ``$HARMEME_DIR`` if set."""
    env_val = os.environ.get("HARMEME_DIR", "").strip()
    return Path(env_val) if env_val else DEFAULT_HARMEME_DIR


def map_harmeme_label(raw_labels: list[str] | str | int) -> tuple[int, str]:
    """Map a HarMeme label field to binary ``(0|1, raw_label_str)``.

    Following the standard Harm-C binary protocol (Pramanick et al., 2021; Cao et al., 2022,
    2023; Lin et al., 2023, 2024; Ji et al., 2024 Table 5), ``"not harmful"`` (or ``0``) maps
    to ``0`` (harmless) and ``"somewhat harmful"`` / ``"very harmful"`` (or ``1`` / ``2``) map
    to ``1`` (harmful).
    """
    if isinstance(raw_labels, int):
        return (0 if raw_labels == 0 else 1), str(raw_labels)
    first = raw_labels[0] if isinstance(raw_labels, list) and raw_labels else str(raw_labels)
    norm = str(first).strip().lower()
    if norm in ("not harmful", "harmless", "0"):
        return 0, norm
    if norm in ("somewhat harmful", "partially harmful", "very harmful", "harmful", "1", "2"):
        return 1, norm
    raise ValueError(f"Unrecognised HarMeme label: {raw_labels!r}")


def build_harmeme_prompt(ocr_text: str, prompt_variant: str = "baseline") -> str:
    """Construct the zero-shot binary harmful-meme classification prompt."""
    base_prompt = HARMEME_PROMPTS[prompt_variant]
    cleaned = ocr_text.strip()
    if cleaned:
        return f'This meme contains the text: "{cleaned}". {base_prompt}'
    return base_prompt


def is_refusal_response(text: str) -> bool:
    """Return True if the VLM response is an explicit safety refusal or empty."""
    stripped = text.strip().lower()
    if not stripped:
        return True
    return any(phrase in stripped for phrase in REFUSAL_PHRASES)


def parse_yesno_response(text: str) -> tuple[int, bool]:
    """Parse a VLM response into ``(binary_prediction, refusal_flag)``."""
    if is_refusal_response(text):
        return 0, True
    matched = extract_label(text.strip(), ["yes", "no"])
    if matched is None:
        return 0, True
    val = yesno_to_int(matched)
    return (1 if val == 1 else 0), False


def download_annotations(data_root: Path | None = None) -> dict[str, Path]:
    """Download and verify the official Harm-C JSONL split files from LCS2-IIITD/MOMENTA."""
    base_dir = data_root or harmeme_dir()
    ann_dir = base_dir / "annotations"
    ann_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for split, remote_name in REMOTE_SPLIT_FILES.items():
        target = ann_dir / f"{split}.jsonl"
        if not target.exists():
            url = f"{MOMENTA_ANNOTATIONS_BASE}/{remote_name}"
            logger.info("Downloading Harm-C %s annotations from %s", split, url)
            with urllib.request.urlopen(url, timeout=60) as resp:
                target.write_bytes(resp.read())
        records = load_split_records(split, data_root=base_dir)
        pos_count = sum(r.label for r in records)
        logger.info(
            "Verified Harm-C %s.jsonl: %d records (%d harmful, %d harmless)",
            split,
            len(records),
            pos_count,
            len(records) - pos_count,
        )
        paths[split] = target
    return paths


def load_split_records(
    split: str,
    data_root: Path | None = None,
    *,
    strict_counts: bool = True,
) -> list[HarmemeRecord]:
    """Load and validate all records for a Harm-C split (``train``, ``validation``, or ``test``)."""
    if split not in SPLITS:
        raise ValueError(f"Invalid split {split!r}; expected one of {SPLITS}")
    base_dir = data_root or harmeme_dir()
    jsonl_path = base_dir / "annotations" / f"{split}.jsonl"
    if not jsonl_path.exists():
        raise FileNotFoundError(
            f"Harm-C annotation file missing: {jsonl_path}. "
            "Run `uv run python -m scripts.revision.harmeme_eval download` first."
        )
    records: list[HarmemeRecord] = []
    seen_ids: set[str] = set()
    with jsonl_path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            obj = json.loads(stripped)
            raw_id = str(obj["id"]).strip()
            image_id = Path(raw_id).stem
            if image_id in seen_ids:
                raise ValueError(f"Duplicate id {image_id!r} in {jsonl_path}:{line_no}")
            seen_ids.add(image_id)
            img_name = str(obj.get("image") or f"{image_id}.png").strip()
            label_int, raw_label = map_harmeme_label(obj["labels"])
            text = str(obj.get("text", ""))
            records.append(
                HarmemeRecord(
                    image_id=image_id,
                    image_filename=img_name,
                    text=text,
                    label=label_int,
                    raw_label=raw_label,
                )
            )
    if strict_counts:
        expected_n = EXPECTED_SPLIT_SIZES[split]
        if len(records) != expected_n:
            raise ValueError(f"{jsonl_path}: expected {expected_n} records, found {len(records)}")
        pos_count = sum(r.label for r in records)
        expected_pos = EXPECTED_POSITIVE_COUNTS[split]
        if pos_count != expected_pos:
            raise ValueError(
                f"{jsonl_path}: expected {expected_pos} harmful records, found {pos_count}"
            )
    return records


def resolve_image_path(record: HarmemeRecord, data_root: Path | None = None) -> Path:
    """Resolve the image file path for a ``HarmemeRecord`` inside ``data/harmeme/images/``."""
    base_dir = data_root or harmeme_dir()
    img_dir = base_dir / "images"
    candidates = [
        img_dir / record.image_filename,
        img_dir / "memes_3013" / record.image_filename,
        img_dir / f"{record.image_id}.png",
        img_dir / "memes_3013" / f"{record.image_id}.png",
        img_dir / f"{record.image_id}.jpg",
    ]
    for cand in candidates:
        if cand.exists():
            return cand
    raise FileNotFoundError(
        f"Image {record.image_filename!r} not found in {img_dir}. "
        "Run `uv run python -m scripts.revision.harmeme_eval download` on the login node."
    )


def _download_gdrive_zip(file_id: str, dest_zip: Path) -> None:
    """Download a large file from Google Drive handling the virus-scan confirmation page."""
    dest_zip.parent.mkdir(parents=True, exist_ok=True)
    direct_url = (
        f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"
    )
    cookie_jar = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cookie_jar))
    logger.info("Downloading HarMeme_Images.zip from Google Drive (id=%s) -> %s", file_id, dest_zip)

    with opener.open(direct_url, timeout=120) as resp:
        content_type = resp.headers.get("Content-Type", "")
        if "text/html" in content_type:
            html = resp.read().decode("utf-8", errors="ignore")
            uuid_match = re.search(r'name="uuid"\s+value="([^"]+)"', html)
            params = {"id": file_id, "export": "download", "confirm": "t"}
            if uuid_match:
                params["uuid"] = uuid_match.group(1)
            confirm_url = "https://drive.usercontent.google.com/download?" + urllib.parse.urlencode(
                params
            )
            with opener.open(confirm_url, timeout=120) as resp2, dest_zip.open("wb") as out_f:
                shutil.copyfileobj(resp2, out_f, length=1024 * 1024)
        else:
            with dest_zip.open("wb") as out_f:
                shutil.copyfileobj(resp, out_f, length=1024 * 1024)


def download_and_extract_images(data_root: Path | None = None, zip_path: Path | None = None) -> int:
    """Download ``HarMeme_Images.zip`` (if needed), flatten ``covid_memes_*`` into ``images/``."""
    base_dir = data_root or harmeme_dir()
    img_dir = base_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    all_records = [r for sp in SPLITS for r in load_split_records(sp, data_root=base_dir)]
    missing_before = [r for r in all_records if not (img_dir / r.image_filename).exists()]
    if not missing_before:
        logger.info("All %d Harm-C images already present in %s", len(all_records), img_dir)
        return len(all_records)

    archive = zip_path or (base_dir / "HarMeme_Images.zip")
    if not archive.exists():
        _download_gdrive_zip(GDRIVE_IMAGES_FILE_ID, archive)

    logger.info("Extracting Harm-C images from %s into %s", archive, img_dir)
    needed_names = {r.image_filename for r in all_records} | {
        f"{r.image_id}.png" for r in all_records
    }
    extracted = 0
    with zipfile.ZipFile(archive, "r") as zf:
        for member in zf.infolist():
            if member.is_dir():
                continue
            basename = Path(member.filename).name
            if basename in needed_names or basename.startswith("covid_memes_"):
                target_path = img_dir / basename
                if not target_path.exists():
                    with zf.open(member) as src, target_path.open("wb") as dst:
                        shutil.copyfileobj(src, dst)
                extracted += 1

    # Verify every split image exists and opens cleanly
    for idx, rec in enumerate(all_records):
        p = resolve_image_path(rec, data_root=base_dir)
        if idx in (0, len(all_records) // 2, len(all_records) - 1):
            with Image.open(p) as im:
                im.convert("RGB")
    logger.info(
        "Verified all %d Harm-C images in %s (extracted %d files)",
        len(all_records),
        img_dir,
        extracted,
    )
    return len(all_records)


def extract_clip_embeddings(
    data_root: Path | None = None,
    emb_dir: Path | None = None,
    batch_size: int = 32,
    device: str | None = None,
) -> dict[str, Path]:
    """Extract L2-normalised frozen CLIP ViT-L-14-quickgelu embeddings for Harm-C splits."""
    import open_clip
    import torch

    base_dir = data_root or harmeme_dir()
    out_emb_dir = emb_dir or EMB_DIR
    out_emb_dir.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    logger.info("Loading OpenCLIP %s (%s) on %s", CLIP_MODEL, CLIP_PRETRAINED, device)
    model, _, preprocess = open_clip.create_model_and_transforms(
        CLIP_MODEL, pretrained=CLIP_PRETRAINED, device=device
    )
    model.eval()
    tokenizer = open_clip.get_tokenizer(CLIP_MODEL)

    saved_paths: dict[str, Path] = {}
    log_entries: dict[str, dict[str, object]] = {}

    for split in SPLITS:
        records = load_split_records(split, data_root=base_dir)
        img_embs: list[np.ndarray] = []
        txt_embs: list[np.ndarray] = []
        labels: list[int] = []
        image_ids: list[str] = []
        raw_texts: list[str] = []

        for start in tqdm(
            range(0, len(records), batch_size),
            desc=f"CLIP {split}",
            unit="batch",
        ):
            batch = records[start : start + batch_size]
            pil_images: list[torch.Tensor] = []
            batch_texts: list[str] = []
            for rec in batch:
                img_path = resolve_image_path(rec, data_root=base_dir)
                with Image.open(img_path) as pil_im:
                    rgb = pil_im.convert("RGB")
                    pil_images.append(preprocess(rgb))
                cleaned_text = rec.text.strip() or "empty text"
                batch_texts.append(cleaned_text)
                labels.append(rec.label)
                image_ids.append(rec.image_id)
                raw_texts.append(rec.text)

            image_Input = torch.stack(pil_images).to(device)
            text_tokens = tokenizer(batch_texts).to(device)

            with torch.no_grad():
                img_feat = model.encode_image(image_Input)
                txt_feat = model.encode_text(text_tokens)
                img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
                txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)

            img_embs.append(img_feat.detach().cpu().to(torch.float32).numpy())
            txt_embs.append(txt_feat.detach().cpu().to(torch.float32).numpy())

        img_arr = np.concatenate(img_embs, axis=0)
        txt_arr = np.concatenate(txt_embs, axis=0)
        lbl_arr = np.asarray(labels, dtype=np.int64)
        ids_arr = np.asarray(image_ids, dtype=object)
        txt_raw_arr = np.asarray(raw_texts, dtype=object)

        out_path = out_emb_dir / f"harmeme_{split}_vit_l_14_quickgelu.npz"
        np.savez_compressed(
            out_path,
            image_embeddings=img_arr,
            text_embeddings=txt_arr,
            labels=lbl_arr,
            image_ids=ids_arr,
            raw_texts=txt_raw_arr,
        )
        sha256 = hashlib.sha256(out_path.read_bytes()).hexdigest()
        saved_paths[split] = out_path
        log_entries[split] = {
            "path": out_path.relative_to(ROOT).as_posix()
            if out_path.is_relative_to(ROOT)
            else str(out_path),
            "sha256": sha256,
            "num_samples": len(records),
            "image_shape": list(img_arr.shape),
            "text_shape": list(txt_arr.shape),
        }
        logger.info("Saved %s (%s, sha256=%s)", out_path.name, img_arr.shape, sha256[:12])

    log_path = OUT_DIR / "extraction_log_clip.json"
    log_path.write_text(
        json.dumps(
            {
                "dataset": "HarMeme (Harm-C)",
                "model": CLIP_MODEL,
                "pretrained": CLIP_PRETRAINED,
                "extracted_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "splits": log_entries,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return saved_paths


def compute_binary_metrics(preds: list[int], gts: list[int]) -> dict[str, float]:
    """Compute Accuracy and official unweighted binary Macro-F1."""
    if len(preds) != len(gts) or not preds:
        raise ValueError("Predictions and ground truths must be non-empty and equal length")
    acc = sum(1 for p, g in zip(preds, gts, strict=True) if int(p) == int(g)) / len(gts)
    macro_f1 = float(compute_f1([int(p) for p in preds], [int(g) for g in gts]))
    return {"exact_match_accuracy": float(acc), "macro_f1": macro_f1}


def bootstrap_ci(
    preds: list[int],
    gts: list[int],
    n_boot: int = N_BOOT,
    seed: int = SEED,
) -> dict[str, list[float]]:
    """Compute 95% bootstrap confidence intervals for Accuracy and Macro-F1."""
    rng = np.random.default_rng(seed)
    pred_arr = np.asarray(preds, dtype=np.int64)
    gt_arr = np.asarray(gts, dtype=np.int64)
    n = len(gt_arr)
    acc_samples = np.empty(n_boot, dtype=np.float64)
    f1_samples = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        p_b = pred_arr[idx]
        g_b = gt_arr[idx]
        acc_samples[b] = float(np.mean(p_b == g_b))
        f1_samples[b] = float(compute_f1(p_b.tolist(), g_b.tolist()))
    return {
        "accuracy_ci95": [
            float(np.percentile(acc_samples, 2.5)),
            float(np.percentile(acc_samples, 97.5)),
        ],
        "macro_f1_ci95": [
            float(np.percentile(f1_samples, 2.5)),
            float(np.percentile(f1_samples, 97.5)),
        ],
    }


def build_benchmark_payload(
    *,
    model_name: str,
    split: str,
    image_ids: list[str],
    gts: list[int],
    preds: list[int],
    refusals: list[bool] | None = None,
    threshold_desc: str = "N/A",
    raw_responses: list[str] | None = None,
    run_metadata: dict[str, object] | None = None,
) -> list[dict[str, object]]:
    """Build a provenance-compliant benchmark JSON payload matching repo conventions.

    When ``raw_responses`` is given (VLM runs), each sample row also stores the model's raw
    reply under ``raw_response`` so the parsed prediction can be audited. ``run_metadata`` adds
    provenance fields (prompt, timing, model version) to the record.
    """
    metrics = compute_binary_metrics(preds, gts)
    ref_list = refusals if refusals is not None else [False] * len(preds)
    sample_rows: list[dict[str, object]] = [
        {
            "image_id": str(img_id),
            "ground_truth": int(gt),
            "prediction": int(pred),
            "correct": int(pred) == int(gt),
            "refusal": bool(ref),
        }
        for img_id, gt, pred, ref in zip(image_ids, gts, preds, ref_list, strict=True)
    ]
    if raw_responses is not None:
        for row, raw in zip(sample_rows, raw_responses, strict=True):
            row["raw_response"] = raw
    refusal_count = sum(1 for r in ref_list if r)
    record: dict[str, object] = {
        "model": model_name,
        "dataset": "harmeme_harm_c",
        "split": split,
        "task": "singleclass",
        "threshold": threshold_desc,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "mock": False,
        "error": None,
        "num_samples": len(sample_rows),
        "exact_match_accuracy": metrics["exact_match_accuracy"],
        "macro_f1": metrics["macro_f1"],
        "f1": metrics["macro_f1"],
        "refusal_count": refusal_count,
        "refusal_rate": refusal_count / len(sample_rows) if sample_rows else 0.0,
        **(run_metadata or {}),
        "sample_predictions": sample_rows,
    }
    return [record]


def run_classical_classifiers(
    emb_dir: Path | None = None,
    results_dir: Path | None = None,
) -> dict[str, dict[str, float]]:
    """Train SVM-RBF and XGBoost on Harm-C frozen CLIP embeddings and score val/test."""
    in_emb_dir = emb_dir or EMB_DIR
    out_res_dir = results_dir or RESULTS_DIR

    split_data: dict[str, dict[str, np.ndarray]] = {}
    for split in SPLITS:
        npz_path = in_emb_dir / f"harmeme_{split}_vit_l_14_quickgelu.npz"
        if not npz_path.exists():
            raise FileNotFoundError(
                f"Missing Harm-C CLIP embeddings: {npz_path}. "
                "Run `uv run python -m scripts.revision.harmeme_eval extract` first."
            )
        z = np.load(npz_path, allow_pickle=True)
        features = np.concatenate(
            [
                z["image_embeddings"].astype(np.float32),
                z["text_embeddings"].astype(np.float32),
            ],
            axis=1,
        )
        split_data[split] = {
            "X": features,
            "y": z["labels"].astype(np.int64),
            "ids": z["image_ids"],
        }

    classifiers: dict[str, tuple[str, Pipeline | XGBClassifier, str]] = {
        "svm_rbf": (
            "CLIP ViT-L-14 + SVM-RBF",
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "clf",
                        SVC(
                            kernel="rbf",
                            class_weight="balanced",
                            probability=False,
                            random_state=SEED,
                        ),
                    ),
                ]
            ),
            "margin 0",
        ),
        "xgboost": (
            "CLIP ViT-L-14 + XGBoost",
            XGBClassifier(
                n_estimators=100,
                random_state=SEED,
                eval_metric="logloss",
                n_jobs=1,
            ),
            "0.50",
        ),
    }

    summary_out: dict[str, dict[str, float]] = {}
    x_train = split_data["train"]["X"]
    y_train = split_data["train"]["y"]

    for slug, (display_name, clf, thr_desc) in classifiers.items():
        logger.info(
            "Fitting %s on Harm-C train split (%d samples, %d dims)",
            display_name,
            len(y_train),
            x_train.shape[1],
        )
        clf.fit(x_train, y_train)
        for eval_split in ("validation", "test"):
            x_eval = split_data[eval_split]["X"]
            y_eval = [int(v) for v in split_data[eval_split]["y"].tolist()]
            ids_eval = [str(i) for i in split_data[eval_split]["ids"].tolist()]
            preds_arr = clf.predict(x_eval)
            preds = [int(v) for v in preds_arr.tolist()]

            payload = build_benchmark_payload(
                model_name=display_name,
                split=eval_split,
                image_ids=ids_eval,
                gts=y_eval,
                preds=preds,
                threshold_desc=thr_desc,
            )
            split_dir = out_res_dir / eval_split
            split_dir.mkdir(parents=True, exist_ok=True)
            out_file = split_dir / f"harmeme_{eval_split}_clip_{slug}.json"
            out_file.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            metrics_eval = compute_binary_metrics(preds, y_eval)
            acc = metrics_eval["exact_match_accuracy"]
            f1 = metrics_eval["macro_f1"]
            summary_out[f"{slug}_{eval_split}"] = {"accuracy": acc, "macro_f1": f1}
            logger.info(
                "%s [%s]: Acc=%.2f%%, Macro-F1=%.2f%% -> %s",
                display_name,
                eval_split,
                acc * 100.0,
                f1 * 100.0,
                out_file.name,
            )

    return summary_out


def run_qwen_zeroshot(
    data_root: Path | None = None,
    results_dir: Path | None = None,
    model_id: str = QWEN_MODEL_ID,
    batch_size: int = 4,
    quantize: str = "4bit",
    device: str = "cuda",
    output_tag: str = "",
    prompt_variant: str = "baseline",
) -> dict[str, dict[str, float]]:
    """Evaluate zero-shot Qwen2-VL-7B-Instruct on Harm-C validation and test splits.

    ``output_tag`` is appended to the output file name (e.g. ``_raw_responses``) so a re-run
    never overwrites an earlier result file.
    """
    import torch

    from scripts.benchmark_qwen2vl import _load_model

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required for Qwen2-VL-7B zero-shot evaluation.")

    base_dir = data_root or harmeme_dir()
    out_res_dir = results_dir or RESULTS_DIR
    processor, model = _load_model(model_id, device, quantize=quantize, lora_path=None)

    summary_out: dict[str, dict[str, float]] = {}
    for eval_split in ("validation", "test"):
        records = load_split_records(eval_split, data_root=base_dir)
        preds: list[int] = []
        gts: list[int] = []
        ids: list[str] = []
        refusals: list[bool] = []
        raw_replies: list[str] = []

        for start in tqdm(
            range(0, len(records), batch_size),
            desc=f"Qwen2-VL {eval_split}",
            unit="batch",
        ):
            batch = records[start : start + batch_size]
            texts: list[str] = []
            pils: list[Image.Image] = []
            for rec in batch:
                img_path = resolve_image_path(rec, data_root=base_dir)
                with Image.open(img_path) as im:
                    pil_rgb = im.convert("RGB")
                prompt_text = build_harmeme_prompt(rec.text, prompt_variant)
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": pil_rgb},
                            {"type": "text", "text": prompt_text},
                        ],
                    }
                ]
                texts.append(
                    processor.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                )
                pils.append(pil_rgb)

            inputs = processor(
                text=texts,
                images=pils,
                padding=True,
                return_tensors="pt",
            ).to(model.device)

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False
                )
            input_len = inputs["input_ids"].shape[1]
            decoded = processor.batch_decode(output_ids[:, input_len:], skip_special_tokens=True)

            for rec, raw_reply in zip(batch, decoded, strict=True):
                pred_int, is_ref = parse_yesno_response(raw_reply)
                ids.append(rec.image_id)
                gts.append(rec.label)
                preds.append(pred_int)
                refusals.append(is_ref)
                raw_replies.append(raw_reply)

        payload = build_benchmark_payload(
            model_name=model_id,
            split=eval_split,
            image_ids=ids,
            gts=gts,
            preds=preds,
            refusals=refusals,
            raw_responses=raw_replies,
            run_metadata={
                "prompt_variant": prompt_variant,
                "prompt_template": HARMEME_PROMPTS[prompt_variant],
                "system_instruction": None,
            },
        )
        split_dir = out_res_dir / eval_split
        split_dir.mkdir(parents=True, exist_ok=True)
        out_file = split_dir / f"harmeme_{eval_split}_qwen2_vl_7b_instruct{output_tag}.json"
        out_file.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        metrics_eval = compute_binary_metrics(preds, gts)
        acc = metrics_eval["exact_match_accuracy"]
        f1 = metrics_eval["macro_f1"]
        summary_out[eval_split] = {"accuracy": acc, "macro_f1": f1}
        logger.info(
            "Qwen2-VL-7B [%s]: Acc=%.2f%%, Macro-F1=%.2f%% -> %s",
            eval_split,
            acc * 100.0,
            f1 * 100.0,
            out_file.name,
        )
    return summary_out


def run_gemini_zeroshot(
    data_root: Path | None = None,
    results_dir: Path | None = None,
    model_id: str = GEMINI_MODEL_ID,
    splits: tuple[str, ...] = ("validation", "test"),
    output_tag: str = "",
    prompt_variant: str = "baseline",
    *,
    strict_counts: bool = True,
) -> dict[str, dict[str, float]]:
    """Evaluate zero-shot Gemini (gemini-3.1-flash-lite) on Harm-C splits.

    Refuses to start if any output file already exists, so a re-run cannot replace an earlier
    result: pass ``output_tag`` (e.g. ``_rerun``) to write new file names.
    """
    existing = [
        path
        for path in (
            (results_dir or RESULTS_DIR)
            / eval_split
            / f"harmeme_{eval_split}_gemini_3_1_flash_lite{output_tag}.json"
            for eval_split in splits
        )
        if path.exists()
    ]
    if existing:
        names = ", ".join(path.name for path in existing)
        raise FileExistsError(f"Refusing to overwrite {names}. Use a new --output-tag.")

    from dotenv import load_dotenv
    from google import genai
    from google.genai import types as genai_types

    load_dotenv(ROOT / ".env")
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set in the environment or .env file.")

    client = genai.Client(api_key=api_key)
    safety_settings = [
        genai_types.SafetySetting(
            category=genai_types.HarmCategory.HARM_CATEGORY_HARASSMENT,
            threshold=genai_types.HarmBlockThreshold.BLOCK_NONE,
        ),
        genai_types.SafetySetting(
            category=genai_types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
            threshold=genai_types.HarmBlockThreshold.BLOCK_NONE,
        ),
        genai_types.SafetySetting(
            category=genai_types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
            threshold=genai_types.HarmBlockThreshold.BLOCK_NONE,
        ),
        genai_types.SafetySetting(
            category=genai_types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
            threshold=genai_types.HarmBlockThreshold.BLOCK_NONE,
        ),
    ]

    base_dir = data_root or harmeme_dir()
    out_res_dir = results_dir or RESULTS_DIR
    summary_out: dict[str, dict[str, float]] = {}

    for eval_split in splits:
        records = load_split_records(eval_split, data_root=base_dir, strict_counts=strict_counts)
        started = time.monotonic()
        retries = 0
        model_versions: set[str] = set()
        preds: list[int] = []
        gts: list[int] = []
        ids: list[str] = []
        refusals: list[bool] = []
        raw_replies: list[str] = []

        for rec in tqdm(records, desc=f"Gemini {eval_split}", unit="img"):
            img_path = resolve_image_path(rec, data_root=base_dir)
            with Image.open(img_path) as im:
                rgb = im.convert("RGB")
                buf = io.BytesIO()
                rgb.save(buf, format="PNG")
                image_bytes = buf.getvalue()

            prompt_text = build_harmeme_prompt(rec.text, prompt_variant)
            image_part = genai_types.Part.from_bytes(data=image_bytes, mime_type="image/png")
            text_part = genai_types.Part.from_text(text=prompt_text)
            user_content = genai_types.Content(role="user", parts=[image_part, text_part])
            config = genai_types.GenerateContentConfig(
                system_instruction=HARMEME_SYSTEM_INSTRUCTION,
                safety_settings=safety_settings,
                temperature=0.0,
                max_output_tokens=16,
            )

            raw_reply = ""
            for attempt in range(GEMINI_MAX_ATTEMPTS):
                try:
                    resp = client.models.generate_content(
                        model=model_id,
                        contents=user_content,
                        config=config,
                    )
                    raw_reply = (resp.text or "").strip()
                    if resp.model_version:
                        model_versions.add(str(resp.model_version))
                    break
                except Exception as exc:
                    if attempt == GEMINI_MAX_ATTEMPTS - 1:
                        raise RuntimeError(
                            f"Gemini API failed on {rec.image_id} after "
                            f"{GEMINI_MAX_ATTEMPTS} attempts: {exc!r}"
                        ) from exc
                    retries += 1
                    time.sleep(2.0 * (2**attempt))

            pred_int, is_ref = parse_yesno_response(raw_reply)
            ids.append(rec.image_id)
            gts.append(rec.label)
            preds.append(pred_int)
            refusals.append(is_ref)
            raw_replies.append(raw_reply)

        payload = build_benchmark_payload(
            model_name=model_id,
            split=eval_split,
            image_ids=ids,
            gts=gts,
            preds=preds,
            refusals=refusals,
            raw_responses=raw_replies,
            run_metadata={
                "prompt_variant": prompt_variant,
                "prompt_template": HARMEME_PROMPTS[prompt_variant],
                "system_instruction": HARMEME_SYSTEM_INSTRUCTION,
                "elapsed_seconds": round(time.monotonic() - started, 1),
                "retries": retries,
                "api_model_versions": sorted(model_versions),
            },
        )
        split_dir = out_res_dir / eval_split
        split_dir.mkdir(parents=True, exist_ok=True)
        out_file = split_dir / f"harmeme_{eval_split}_gemini_3_1_flash_lite{output_tag}.json"
        out_file.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        metrics_eval = compute_binary_metrics(preds, gts)
        acc = metrics_eval["exact_match_accuracy"]
        f1 = metrics_eval["macro_f1"]
        summary_out[eval_split] = {"accuracy": acc, "macro_f1": f1}
        logger.info(
            "Gemini [%s]: Acc=%.2f%%, Macro-F1=%.2f%% -> %s",
            eval_split,
            acc * 100.0,
            f1 * 100.0,
            out_file.name,
        )
    return summary_out


def verify_harmeme_output_file(
    path: Path,
    split: str,
    data_root: Path | None = None,
    *,
    strict_counts: bool = True,
) -> list[str]:
    """Verify a ``harmeme_<split>_*.json`` file against the official Harm-C ground truth."""
    errs: list[str] = []
    if not path.exists():
        return ["missing"]
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return [f"invalid JSON: {exc!r}"]

    rec = data[0] if isinstance(data, list) and data else {}
    if not rec:
        return ["empty or missing benchmark record"]
    if rec.get("mock") is True:
        errs.append("record marked as mock: true")
    if rec.get("error") is not None:
        errs.append(f"record contains error: {rec.get('error')!r}")
    if rec.get("split") != split:
        errs.append(f"split field {rec.get('split')!r} != {split!r}")
    if rec.get("dataset") != "harmeme_harm_c":
        errs.append(f"dataset field {rec.get('dataset')!r} != 'harmeme_harm_c'")
    if not rec.get("model"):
        errs.append("model field empty")

    records = load_split_records(split, data_root=data_root, strict_counts=strict_counts)
    truth_ids = [r.image_id for r in records]
    truth_map = {r.image_id: r.label for r in records}

    sp = rec.get("sample_predictions") or []
    got_ids = [str(p.get("image_id")) for p in sp]
    expected_len = EXPECTED_SPLIT_SIZES[split] if strict_counts else len(records)
    if (
        len(sp) != expected_len
        or set(got_ids) != set(truth_ids)
        or len(set(got_ids)) != len(got_ids)
    ):
        errs.append(
            f"{len(sp)} predictions; ids do not match the Harm-C {split} split ({expected_len})"
        )
        return errs

    bad_gt = sum(1 for p in sp if p.get("ground_truth") != truth_map.get(str(p.get("image_id"))))
    if bad_gt:
        errs.append(f"{bad_gt} ground-truth labels differ from Harm-C {split}.jsonl")

    preds = [p.get("prediction") for p in sp]
    invalid_preds = [p for p in preds if p not in (0, 1)]
    if invalid_preds:
        errs.append(f"{len(invalid_preds)} predictions not in {{0, 1}}")
    elif len(set(preds)) == 1:
        errs.append(
            f"degenerate predictions: all {len(preds)} samples predicted constant class {preds[0]}"
        )
    else:
        gts = [int(p["ground_truth"]) for p in sp]
        int_preds = [int(p) for p in preds]
        recomputed = compute_binary_metrics(int_preds, gts)
        rep_acc = rec.get("exact_match_accuracy")
        if rep_acc is None or abs(float(rep_acc) - recomputed["exact_match_accuracy"]) > 1e-5:
            errs.append(
                f"reported exact_match_accuracy {rep_acc} != recomputed {recomputed['exact_match_accuracy']:.6f}"
            )
        rep_f1 = rec.get("macro_f1", rec.get("f1"))
        if rep_f1 is None or abs(float(rep_f1) - recomputed["macro_f1"]) > 1e-3:
            errs.append(f"reported macro_f1 {rep_f1} != recomputed {recomputed['macro_f1']:.6f}")
    return errs


def _read_predictions(path: Path) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Return the benchmark record and its per-meme rows from a result file."""
    record = json.loads(path.read_text(encoding="utf-8"))[0]
    return record, list(record["sample_predictions"])


def _prediction_stats(rows: list[dict[str, object]]) -> dict[str, float]:
    """Accuracy, Macro-F1, harmful recall and share predicted harmful for per-meme rows."""
    preds = [int(str(row["prediction"])) for row in rows]
    gts = [int(str(row["ground_truth"])) for row in rows]
    metrics = compute_binary_metrics(preds, gts)
    harmful = sum(gts)
    caught = sum(1 for pred, gt in zip(preds, gts, strict=True) if pred == 1 and gt == 1)
    return {
        "accuracy": metrics["exact_match_accuracy"],
        "macro_f1": metrics["macro_f1"],
        "harmful_recall": caught / harmful if harmful else 0.0,
        "predicted_harmful_share": sum(preds) / len(preds),
    }


def compare_runs(
    tag: str,
    model_slug: str = "gemini_3_1_flash_lite",
    data_root: Path | None = None,
    results_dir: Path | None = None,
    out_dir: Path | None = None,
    *,
    strict_counts: bool = True,
) -> dict[str, object]:
    """Compare tagged result files with the committed untagged files for one model.

    Reports per-meme changes and metrics, then two pass/fail answers: ``reproduced`` (at most
    ``REPRODUCE_MAX_CHANGED_SHARE`` of memes changed answer in every compared split) and
    ``clears_validation_bar`` (validation Macro-F1 above the upper 95% bootstrap bound of the
    committed validation file). Writes ``compare_<model><tag>.json`` to the DIS-01 output folder.
    """
    if not tag:
        raise ValueError("compare needs a non-empty --tag; untagged files are the reference.")
    in_res_dir = results_dir or RESULTS_DIR
    target_out_dir = out_dir or OUT_DIR
    splits_out: dict[str, dict[str, object]] = {}
    for split in ("validation", "test"):
        tagged = in_res_dir / split / f"harmeme_{split}_{model_slug}{tag}.json"
        if not tagged.exists():
            continue
        reference = in_res_dir / split / f"harmeme_{split}_{model_slug}.json"
        _, ref_rows = _read_predictions(reference)
        record, new_rows = _read_predictions(tagged)
        ref_by_id = {str(row["image_id"]): row["prediction"] for row in ref_rows}
        changed = sum(
            1 for row in new_rows if ref_by_id.get(str(row["image_id"])) != row["prediction"]
        )
        splits_out[split] = {
            "verification_errors": verify_harmeme_output_file(
                tagged, split, data_root=data_root, strict_counts=strict_counts
            ),
            "memes": len(new_rows),
            "changed_answers": changed,
            "changed_share": changed / len(new_rows),
            "reference": _prediction_stats(ref_rows),
            "new": _prediction_stats(new_rows),
            "prompt_variant": record.get("prompt_variant"),
            "elapsed_seconds": record.get("elapsed_seconds"),
            "retries": record.get("retries"),
            "api_model_versions": record.get("api_model_versions"),
        }
    if not splits_out:
        raise FileNotFoundError(f"No results for {model_slug}{tag} in {in_res_dir}.")

    bar: float | None = None
    clears_bar: bool | None = None
    if "validation" in splits_out:
        reference = in_res_dir / "validation" / f"harmeme_validation_{model_slug}.json"
        _, ref_rows = _read_predictions(reference)
        ci95 = bootstrap_ci(
            [int(str(row["prediction"])) for row in ref_rows],
            [int(str(row["ground_truth"])) for row in ref_rows],
        )
        bar = ci95["macro_f1_ci95"][1]
        new_stats = splits_out["validation"]["new"]
        assert isinstance(new_stats, dict)
        clears_bar = float(new_stats["macro_f1"]) > bar

    result: dict[str, object] = {
        "model": model_slug,
        "tag": tag,
        "reproduced": all(
            float(str(entry["changed_share"])) <= REPRODUCE_MAX_CHANGED_SHARE
            for entry in splits_out.values()
        ),
        "validation_macro_f1_bar": bar,
        "clears_validation_bar": clears_bar,
        "splits": splits_out,
    }
    target_out_dir.mkdir(parents=True, exist_ok=True)
    out_file = target_out_dir / f"compare_{model_slug}{tag}.json"
    out_file.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for split, entry in splits_out.items():
        ref_stats, new_stats = entry["reference"], entry["new"]
        assert isinstance(ref_stats, dict) and isinstance(new_stats, dict)
        logger.info(
            "%s: %s of %s answers changed; Macro-F1 %.2f%% -> %.2f%%; harmful recall "
            "%.2f%% -> %.2f%%; elapsed %ss; errors %s",
            split,
            entry["changed_answers"],
            entry["memes"],
            ref_stats["macro_f1"] * 100.0,
            new_stats["macro_f1"] * 100.0,
            ref_stats["harmful_recall"] * 100.0,
            new_stats["harmful_recall"] * 100.0,
            entry["elapsed_seconds"],
            entry["verification_errors"] or "none",
        )
    logger.info("reproduced: %s", "yes" if result["reproduced"] else "NO")
    if bar is not None:
        logger.info(
            "clears validation bar (Macro-F1 above %.2f%%): %s",
            bar * 100.0,
            "yes" if clears_bar else "no",
        )
    logger.info("Wrote %s", out_file)
    return result


def summarize_results(
    data_root: Path | None = None,
    results_dir: Path | None = None,
    out_dir: Path | None = None,
    *,
    strict_counts: bool = True,
) -> dict[str, object]:
    """Verify all available Harm-C test outputs and write metrics.json, table_rows.tex, summary.md."""
    base_dir = data_root or harmeme_dir()
    in_res_dir = results_dir or RESULTS_DIR
    target_out_dir = out_dir or OUT_DIR
    target_out_dir.mkdir(parents=True, exist_ok=True)

    expected_models: list[tuple[str, str, str]] = [
        ("harmeme_test_clip_svm_rbf.json", "This work: SVM-RBF on CLIP", "Supervised, frozen CLIP"),
        ("harmeme_test_clip_xgboost.json", "This work: XGBoost on CLIP", "Supervised, frozen CLIP"),
        ("harmeme_test_qwen2_vl_7b_instruct.json", "This work: Qwen2-VL-7B", "Zero-shot VLM"),
        (
            "harmeme_test_gemini_3_1_flash_lite.json",
            "This work: Gemini 3.1 Flash Lite",
            "Zero-shot VLM",
        ),
    ]

    rows_summary: list[dict[str, object]] = []
    tex_lines: list[str] = ["\\midrule"]
    md_lines: list[str] = [
        "# DIS-01: HarMeme (Harm-C) Empirical Evaluation Summary",
        "",
        "Official Harm-C test split: 354 memes (124 harmful, 230 harmless).",
        "",
        "| Method | Setting | Test Acc (%) | Test Macro-F1 (%) | 95% CI Acc (%) | 95% CI Macro-F1 (%) | File |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]

    for filename, table_label, setting_label in expected_models:
        fpath = in_res_dir / "test" / filename
        if not fpath.exists():
            logger.warning("Result file not yet present: %s", fpath)
            continue
        errs = verify_harmeme_output_file(
            fpath, "test", data_root=base_dir, strict_counts=strict_counts
        )
        if errs:
            raise ValueError(f"Verification failed for {fpath}: {'; '.join(errs)}")
        rec = json.loads(fpath.read_text(encoding="utf-8"))[0]
        sp = rec["sample_predictions"]
        preds = [int(p["prediction"]) for p in sp]
        gts = [int(p["ground_truth"]) for p in sp]
        metrics = compute_binary_metrics(preds, gts)
        ci = bootstrap_ci(preds, gts)
        acc_pct = metrics["exact_match_accuracy"] * 100.0
        f1_pct = metrics["macro_f1"] * 100.0
        acc_ci_pct = [ci["accuracy_ci95"][0] * 100.0, ci["accuracy_ci95"][1] * 100.0]
        f1_ci_pct = [ci["macro_f1_ci95"][0] * 100.0, ci["macro_f1_ci95"][1] * 100.0]

        entry = {
            "file": f"results/test/{filename}",
            "sha256": hashlib.sha256(fpath.read_bytes()).hexdigest(),
            "model": rec["model"],
            "table_label": table_label,
            "setting": setting_label,
            "accuracy": metrics["exact_match_accuracy"],
            "macro_f1": metrics["macro_f1"],
            "accuracy_pct": round(acc_pct, 2),
            "macro_f1_pct": round(f1_pct, 2),
            "accuracy_ci95_pct": [round(acc_ci_pct[0], 2), round(acc_ci_pct[1], 2)],
            "macro_f1_ci95_pct": [round(f1_ci_pct[0], 2), round(f1_ci_pct[1], 2)],
            "refusal_rate": rec.get("refusal_rate", 0.0),
        }
        rows_summary.append(entry)
        tex_lines.append(
            f"\\revtab{{{table_label}}} & \\revtab{{{setting_label}}} & "
            f"\\revtab{{{acc_pct:.2f}}} & \\revtab{{{f1_pct:.2f}}} \\\\"
        )
        md_lines.append(
            f"| {table_label} | {setting_label} | {acc_pct:.2f} | {f1_pct:.2f} | "
            f"[{acc_ci_pct[0]:.2f}, {acc_ci_pct[1]:.2f}] | "
            f"[{f1_ci_pct[0]:.2f}, {f1_ci_pct[1]:.2f}] | `results/test/{filename}` |"
        )

    payload_out: dict[str, object] = {
        "task": "DIS-01",
        "dataset": "HarMeme (Harm-C)",
        "test_split_size": EXPECTED_SPLIT_SIZES["test"],
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "models": rows_summary,
    }
    (target_out_dir / "metrics.json").write_text(
        json.dumps(payload_out, indent=2) + "\n", encoding="utf-8"
    )
    (target_out_dir / "table_rows.tex").write_text("\n".join(tex_lines) + "\n", encoding="utf-8")
    (target_out_dir / "summary.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    logger.info(
        "Wrote DIS-01 summary (%d models) to %s",
        len(rows_summary),
        target_out_dir,
    )
    return payload_out


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point for the DIS-01 HarMeme evaluation pipeline."""
    parser = argparse.ArgumentParser(
        description="DIS-01: HarMeme (Harm-C) second-dataset evaluation pipeline."
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_dl = sub.add_parser("download", help="Download Harm-C annotations and images")
    p_dl.add_argument(
        "--annotations-only",
        action="store_true",
        help="Download and verify only the JSONL split annotations (skip 561 MB image zip)",
    )
    p_dl.add_argument(
        "--zip-path", type=Path, default=None, help="Existing HarMeme_Images.zip path"
    )

    p_ext = sub.add_parser("extract", help="Extract frozen CLIP ViT-L-14 embeddings on Harm-C")
    p_ext.add_argument("--batch-size", type=int, default=32)
    p_ext.add_argument("--device", default=None)

    sub.add_parser("classify", help="Fit SVM-RBF and XGBoost on Harm-C CLIP embeddings")

    p_qwen = sub.add_parser("qwen", help="Run zero-shot Qwen2-VL-7B-Instruct on Harm-C val/test")
    p_qwen.add_argument("--model-id", default=QWEN_MODEL_ID)
    p_qwen.add_argument("--batch-size", type=int, default=4)
    p_qwen.add_argument("--quantize", default="4bit", choices=("4bit", "none"))
    p_qwen.add_argument("--device", default="cuda")
    p_qwen.add_argument(
        "--output-tag",
        default="",
        help="Suffix for output file names (e.g. _raw_responses) so earlier results are kept",
    )
    p_qwen.add_argument("--prompt-variant", default="baseline", choices=tuple(HARMEME_PROMPTS))

    p_gem = sub.add_parser("gemini", help="Run zero-shot Gemini 3.1 Flash Lite on Harm-C val/test")
    p_gem.add_argument("--model-id", default=GEMINI_MODEL_ID)
    p_gem.add_argument(
        "--output-tag",
        default="",
        help="Suffix for output file names (e.g. _rerun); existing files are never overwritten",
    )
    p_gem.add_argument("--prompt-variant", default="baseline", choices=tuple(HARMEME_PROMPTS))
    p_gem.add_argument(
        "--splits",
        nargs="+",
        default=["validation", "test"],
        choices=("validation", "test"),
        help="Splits to evaluate",
    )

    sub.add_parser("summarize", help="Verify outputs and write metrics.json and table_rows.tex")

    p_cmp = sub.add_parser("compare", help="Compare tagged result files with the committed files")
    p_cmp.add_argument("--tag", required=True, help="Output tag of the new run (e.g. _rerun)")
    p_cmp.add_argument(
        "--model",
        default="gemini_3_1_flash_lite",
        choices=("gemini_3_1_flash_lite", "qwen2_vl_7b_instruct"),
    )

    p_sci = sub.add_parser(
        "run-sciama",
        help="Run extract + classify + qwen + summarize inside the SCIAMA S6 job",
    )
    p_sci.add_argument("--device", default="cuda")
    p_sci.add_argument("--batch-size", type=int, default=4)

    args = parser.parse_args(argv)

    if args.cmd == "download":
        download_annotations()
        if not args.annotations_only:
            download_and_extract_images(zip_path=args.zip_path)
    elif args.cmd == "extract":
        extract_clip_embeddings(batch_size=args.batch_size, device=args.device)
    elif args.cmd == "classify":
        run_classical_classifiers()
    elif args.cmd == "qwen":
        run_qwen_zeroshot(
            model_id=args.model_id,
            batch_size=args.batch_size,
            quantize=args.quantize,
            device=args.device,
            output_tag=args.output_tag,
            prompt_variant=args.prompt_variant,
        )
    elif args.cmd == "gemini":
        run_gemini_zeroshot(
            model_id=args.model_id,
            splits=tuple(args.splits),
            output_tag=args.output_tag,
            prompt_variant=args.prompt_variant,
        )
    elif args.cmd == "summarize":
        summarize_results()
    elif args.cmd == "compare":
        compare_runs(args.tag, model_slug=args.model)
    elif args.cmd == "run-sciama":
        extract_clip_embeddings(device=args.device)
        run_classical_classifiers()
        run_qwen_zeroshot(batch_size=args.batch_size, device=args.device)
        summarize_results()
    return 0


if __name__ == "__main__":
    sys.exit(main())
