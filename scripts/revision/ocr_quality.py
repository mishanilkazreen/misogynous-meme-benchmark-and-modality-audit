"""EXP-05: PaddleOCR transcript quality, OCR-error impact and CLIP 77-token truncation.

Scope:
    b. Report OCR CER/WER, and run the OCR ablation on the VLMs.
    c. Analyse the effect of the 77-token truncation.

The script is CPU-only and reads existing artefacts; it does not run any model.

Inputs
    * Gold transcripts: MAMI ``text`` column (the "Text Transcription" field) loaded through
      :class:`utils.dataset.DatasetManager` (kagglehub cache).
    * PaddleOCR transcripts: ``raw_texts`` in
      ``results/embeddings/{split}_vit_l_14_quickgelu_ocr_paddleocr.npz``, aligned by ``image_ids``.
    * Per-sample predictions in ``results/test/qwen2vl_test_*.json`` and the auto_benchmark
      tabular prediction CSVs used by ``scripts/recompute_paper_metrics.py``.

Outputs (``results/revision/exp05/``)
    * ``ocr_quality.json``: CER/WER per split, CER buckets, OCR-vs-provided scores per bucket.
    * ``truncation.json``: CLIP token-length statistics and truncated vs non-truncated accuracy.
    * ``summary.md``: short method/result summary generated from the two JSON files.

Usage::

    uv run --with jiwer==4.0.0 python -m scripts.revision.ocr_quality
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
import csv
from dataclasses import dataclass
import datetime as dt
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any
import unicodedata

import jiwer
import numpy as np
import open_clip
from scipy import stats

from scripts.official_mami_evaluation import compute_scoreA, compute_scoreB
from utils.dataset import DatasetManager

ROOT = Path(__file__).resolve().parents[2]
EMB_DIR = ROOT / "results" / "embeddings"
TEST_DIR = ROOT / "results" / "test"
AB_DIR = ROOT / "auto_benchmark" / "results" / "model_results"
OUT_DIR = ROOT / "results" / "revision" / "exp05"

SPLITS = ("train", "validation", "test")
SUBTYPES = ("shaming", "stereotype", "objectification", "violence")
CLIP_MODEL = "ViT-L-14"
CONTEXT_LENGTH = 77
BUCKETS: tuple[tuple[str, float, float], ...] = (
    ("0-0.1", -math.inf, 0.1),
    ("0.1-0.3", 0.1, 0.3),
    ("0.3-0.6", 0.3, 0.6),
    (">0.6", 0.6, math.inf),
)
LENGTH_BINS: tuple[tuple[str, int, int], ...] = (
    ("<=16", 0, 16),
    ("17-32", 17, 32),
    ("33-48", 33, 48),
    ("49-64", 49, 64),
    ("65-77", 65, 77),
    ("78-100", 78, 100),
    ("101-150", 101, 150),
    (">150", 151, 10**9),
)
N_BOOT = 10_000
SEED = 20260505

# Zero-shot Qwen2-VL runs: (label, provided-text file, PaddleOCR-text file).
QWEN_BINARY_PAIRS = (
    (
        "Qwen2-VL-2B-Instruct zero-shot",
        "qwen2vl_test_qwen2_vl_2b_instruct.json",
        "qwen2vl_test_qwen2_vl_2b_instruct_ocr_paddleocr.json",
    ),
    (
        "Qwen2-VL-7B-Instruct zero-shot",
        "qwen2vl_test_qwen2_vl_7b_instruct.json",
        "qwen2vl_test_qwen2_vl_7b_instruct_ocr_paddleocr.json",
    ),
)
QWEN_MULTI_PAIRS = (
    (
        "Qwen2-VL-2B-Instruct zero-shot",
        "qwen2vl_test_qwen2_vl_2b_instruct_multiclass.json",
        "qwen2vl_test_qwen2_vl_2b_instruct_ocr_paddleocr_multiclass.json",
    ),
    (
        "Qwen2-VL-7B-Instruct zero-shot",
        "qwen2vl_test_qwen2_vl_7b_instruct_multiclass.json",
        "qwen2vl_test_qwen2_vl_7b_instruct_ocr_paddleocr_multiclass.json",
    ),
)
QWEN_FT_SEED1 = "qwen2vl_test_qwen2_vl_7b_instruct_finetuned_seed1.json"
TABULAR_MODELS = ("Support Vector Machine - RBF SVC", "XGBoost")
TAB_CSV = {
    ("provided", "binary"): AB_DIR
    / "mami_tabular_model_provided_test"
    / "mami_tabular_model_provided_test_predictions.csv",
    ("provided", "multiclass"): AB_DIR
    / "mami_tabular_model_multiclass_provided_test"
    / "mami_tabular_model_multiclass_provided_test_predictions.csv",
    ("paddleocr", "binary"): AB_DIR
    / "mami_tabular_model_test"
    / "mami_tabular_model_test_predictions.csv",
    ("paddleocr", "multiclass"): AB_DIR
    / "mami_tabular_model_multiclass_test"
    / "mami_tabular_model_multiclass_test_predictions.csv",
}

# Binary tabular CSVs used for the truncation subgroup analysis (validation is supplementary).
TRUNC_TAB_CSV = {
    ("validation", "paddleocr"): AB_DIR
    / "mami_tabular_model"
    / "mami_tabular_model_predictions.csv",
    ("validation", "provided"): AB_DIR
    / "mami_tabular_model_provided"
    / "mami_tabular_model_provided_predictions.csv",
    ("test", "paddleocr"): TAB_CSV[("paddleocr", "binary")],
    ("test", "provided"): TAB_CSV[("provided", "binary")],
}
SHORT = {"Support Vector Machine - RBF SVC": "SVM-RBF", "XGBoost": "XGBoost"}

_WS_RE = re.compile(r"\s+")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


@dataclass
class SplitData:
    """Gold and PaddleOCR transcripts plus labels for one split, in dataset order."""

    split: str
    image_ids: list[str]
    gold: list[str]
    ocr: list[str]
    labels: list[int]
    subtypes: list[list[int]]


def sha256(path: Path) -> str:
    """Return the SHA-256 hex digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rel(path: Path) -> str:
    """Return a repository-relative POSIX path string."""
    return path.resolve().relative_to(ROOT).as_posix()


def load_split(manager: DatasetManager, split: str, inputs: dict[str, str]) -> SplitData:
    """Load gold transcripts from the dataset and align PaddleOCR transcripts by image id.

    Raises:
        ValueError: if ids, labels or subtask labels disagree between the NPZ and the TSV.
    """
    dataset = manager.load_dataset(split)
    tsv_path = Path(dataset._tsv_path)
    inputs[f"dataset:{tsv_path.name}"] = sha256(tsv_path)
    records = dataset._records
    ids = [Path(r["file_name"]).stem for r in records]
    gold = [r.get("text", "") or "" for r in records]
    labels = [int(r["label"]) for r in records]
    subtypes = [[int(r[s]) for s in SUBTYPES] for r in records]

    npz_path = EMB_DIR / f"{split}_vit_l_14_quickgelu_ocr_paddleocr.npz"
    inputs[rel(npz_path)] = sha256(npz_path)
    data = np.load(npz_path, allow_pickle=False)
    npz_ids = [str(x) for x in data["image_ids"]]
    if len(set(npz_ids)) != len(npz_ids) or set(npz_ids) != set(ids):
        raise ValueError(f"{split}: NPZ image_ids do not match the dataset split")
    pos = {img: i for i, img in enumerate(npz_ids)}
    order = [pos[i] for i in ids]
    ocr_all = [str(t) for t in data["raw_texts"]]
    ocr = [ocr_all[i] for i in order]
    npz_labels = data["labels"][order].astype(int).tolist()
    npz_sub = data["subtask_labels"][order].astype(int).tolist()
    if npz_labels != labels or npz_sub != subtypes:
        raise ValueError(f"{split}: NPZ labels disagree with the dataset TSV")
    return SplitData(split, ids, gold, ocr, labels, subtypes)


# ---------------------------------------------------------------------------
# OCR quality
# ---------------------------------------------------------------------------


def normalise(text: str) -> str:
    """NFKC, lowercase, remove Unicode punctuation (category P*), collapse whitespace."""
    text = unicodedata.normalize("NFKC", text).lower()
    text = "".join(" " if unicodedata.category(c).startswith("P") else c for c in text)
    return _WS_RE.sub(" ", text).strip()


def no_space(text: str) -> str:
    """Normalised text with all whitespace removed (word-segmentation-insensitive CER)."""
    return _WS_RE.sub("", normalise(text))


def raw(text: str) -> str:
    """Identity transform (jiwer default transforms still strip outer whitespace)."""
    return text


VARIANTS: dict[str, Callable[[str], str]] = {
    "raw": raw,
    "normalised": normalise,
    "normalised_no_space": no_space,
}


def edit_counts(ref: str, hyp: str, unit: str) -> tuple[int, int]:
    """Return (errors, reference length) for characters or words using jiwer.

    ``errors`` is substitutions + deletions + insertions. Empty references are handled by the
    caller and must not be passed here.
    """
    if unit == "char":
        c_out = jiwer.process_characters(ref, hyp)
        errs = c_out.substitutions + c_out.deletions + c_out.insertions
        return int(errs), int(c_out.substitutions + c_out.deletions + c_out.hits)
    w_out = jiwer.process_words(ref, hyp)
    errs = w_out.substitutions + w_out.deletions + w_out.insertions
    return int(errs), int(w_out.substitutions + w_out.deletions + w_out.hits)


def describe(values: Sequence[float]) -> dict[str, float]:
    """Return mean, median, quartiles, min and max of a sequence."""
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def bucket_of(cer: float) -> str:
    """Map a per-meme CER to its bucket label."""
    for name, lo, hi in BUCKETS:
        if lo <= cer < hi:
            return name
    raise ValueError(f"CER {cer} outside all buckets")


def ocr_quality_split(data: SplitData) -> tuple[dict[str, Any], dict[str, float | None]]:
    """Compute corpus and per-meme CER/WER for one split under each normalisation variant.

    Returns:
        The split report and the per-meme normalised CER keyed by image id (``None`` when the
        normalised gold transcript is empty).
    """
    report: dict[str, Any] = {
        "split": data.split,
        "n_memes": len(data.image_ids),
        "n_empty_gold_raw": sum(not g.strip() for g in data.gold),
        "n_empty_ocr_raw": sum(not o.strip() for o in data.ocr),
        "pct_empty_ocr_raw": 100.0 * sum(not o.strip() for o in data.ocr) / len(data.ocr),
        "variants": {},
    }
    per_meme_norm: dict[str, float | None] = {}
    for vname, fn in VARIANTS.items():
        refs = [fn(g) for g in data.gold]
        hyps = [fn(o) for o in data.ocr]
        keep = [i for i, r in enumerate(refs) if r.strip()]
        units = ("char",) if vname == "normalised_no_space" else ("char", "word")
        vrep: dict[str, Any] = {
            "n_scored": len(keep),
            "n_excluded_empty_gold": len(refs) - len(keep),
            "pct_empty_ocr": 100.0 * sum(not hyps[i].strip() for i in keep) / len(keep),
        }
        for unit in units:
            counts = [edit_counts(refs[i], hyps[i], unit) for i in keep]
            tot_err = sum(e for e, _ in counts)
            tot_ref = sum(n for _, n in counts)
            rates = [e / n for e, n in counts]
            corpus = tot_err / tot_ref
            # Cross-check against jiwer's own corpus-level implementation.
            kept_refs = [refs[i] for i in keep]
            kept_hyps = [hyps[i] for i in keep]
            check = (
                jiwer.cer(kept_refs, kept_hyps)
                if unit == "char"
                else jiwer.wer(kept_refs, kept_hyps)
            )
            if abs(float(check) - corpus) > 1e-9:
                raise ValueError(
                    f"{data.split}/{vname}/{unit}: corpus mismatch {check} vs {corpus}"
                )
            key = "cer" if unit == "char" else "wer"
            vrep[key] = {
                "corpus": corpus,
                "total_errors": tot_err,
                "total_reference_units": tot_ref,
                "per_meme": describe(rates),
            }
            if unit == "char":
                buckets = {name: 0 for name, _, _ in BUCKETS}
                for r in rates:
                    buckets[bucket_of(r)] += 1
                vrep["cer_buckets"] = {
                    k: {"n": v, "pct": 100.0 * v / len(rates)} for k, v in buckets.items()
                }
                if vname == "normalised":
                    rate_by_id = {data.image_ids[i]: r for i, r in zip(keep, rates, strict=True)}
                    per_meme_norm = {img: rate_by_id.get(img) for img in data.image_ids}
        report["variants"][vname] = vrep
    return report, per_meme_norm


# ---------------------------------------------------------------------------
# Prediction loading and scoring
# ---------------------------------------------------------------------------


def make_truth(data: SplitData) -> dict[str, list[bool]]:
    """Build the official-scorer truth dict: id -> [misogynous, shaming, stereotype, obj, violence]."""
    return {
        img: [bool(lbl), *(bool(x) for x in sub)]
        for img, lbl, sub in zip(data.image_ids, data.labels, data.subtypes, strict=True)
    }


def load_vlm(
    path: Path, truth: dict[str, list[bool]], inputs: dict[str, str], split: str = "test"
) -> dict[str, Any]:
    """Load a Qwen2-VL result file and validate ground truth against the dataset.

    Returns a dict with ``pred`` (id -> [misogynous, 4 subtypes] as bools), ``task``,
    ``stored`` summary metrics and ``n_parse_failures``.
    """
    inputs[rel(path)] = sha256(path)
    with path.open(encoding="utf-8") as fh:
        blob = json.load(fh)
    if not isinstance(blob, list) or len(blob) != 1:
        raise ValueError(f"{path.name}: expected a single-run list")
    run = blob[0]
    if run.get("split") != split or run.get("model") != "qwen2vl":
        raise ValueError(
            f"{path.name}: unexpected provenance {run.get('model')}/{run.get('split')}"
        )
    samples = run["sample_predictions"]
    if any(s.get("mock") for s in samples) or any("error" in s for s in samples):
        raise ValueError(f"{path.name}: contains mock or error records")
    ids = [str(s["image_id"]) for s in samples]
    if len(set(ids)) != len(ids) or set(ids) != set(truth):
        raise ValueError(f"{path.name}: image ids do not cover the {split} split exactly")
    pred: dict[str, list[bool]] = {}
    failures = 0
    task = run["task"]
    for s in samples:
        img = str(s["image_id"])
        if task == "singleclass":
            if bool(int(s["ground_truth"])) != truth[img][0]:
                raise ValueError(f"{path.name}: ground truth mismatch for {img}")
            pred[img] = [bool(int(s["prediction"])), False, False, False, False]
        else:
            gt = [bool(int(s["ground_truth"][k])) for k in SUBTYPES]
            if gt != truth[img][1:]:
                raise ValueError(f"{path.name}: subtype ground truth mismatch for {img}")
            p = s.get("prediction")
            if not isinstance(p, dict) or s.get("parse_failure"):
                failures += 1
            p_vec = [bool(int(p.get(k, 0))) if isinstance(p, dict) else False for k in SUBTYPES]
            pred[img] = [any(p_vec), *p_vec]
    stored = {k: run[k] for k in ("exact_match_accuracy", "mami_score_b") if k in run}
    return {"pred": pred, "task": task, "stored": stored, "n_parse_failures": failures}


def load_tabular(
    path: Path, task: str, test: SplitData, inputs: dict[str, str]
) -> dict[str, dict[str, list[bool]]] | None:
    """Load an auto_benchmark prediction CSV and align rows to the split's image ids.

    Rows follow the NPZ order (``auto_benchmark/src/autobenchmark/data.py`` concatenates NPZ
    arrays without shuffling); the ``Actual`` column is verified against the NPZ-aligned labels.
    Returns ``None`` if the CSV is not present locally.
    """
    if not path.exists():
        return None
    inputs[rel(path)] = sha256(path)
    npz_path = EMB_DIR / f"{test.split}_vit_l_14_quickgelu_ocr_paddleocr.npz"
    npz = np.load(npz_path, allow_pickle=False)
    npz_ids = [str(x) for x in npz["image_ids"]]
    idx = {img: i for i, img in enumerate(test.image_ids)}
    with path.open(encoding="utf-8", newline="") as fh:
        rows = [r for r in csv.reader(fh) if r]
    header, body = rows[0], rows[1:]
    if len(body) != len(npz_ids):
        raise ValueError(f"{path.name}: {len(body)} rows, expected {len(npz_ids)}")
    out: dict[str, dict[str, list[bool]]] = {m: {} for m in header[1:]}
    for img, row in zip(npz_ids, body, strict=True):
        i = idx[img]
        if task == "binary":
            if int(float(row[0])) != test.labels[i]:
                raise ValueError(f"{path.name}: Actual column misaligned at {img}")
            for m, cell in zip(header[1:], row[1:], strict=True):
                out[m][img] = [bool(int(float(cell))), False, False, False, False]
        else:
            gt = [int(float(x)) for x in row[0].split(",")]
            if gt != test.subtypes[i]:
                raise ValueError(f"{path.name}: Actual column misaligned at {img}")
            for m, cell in zip(header[1:], row[1:], strict=True):
                vec = [bool(int(float(x))) for x in cell.split(",")]
                out[m][img] = [any(vec), *vec]
    return out


def check_tabular(
    stored: dict[str, Any],
    truth: dict[str, list[bool]],
    pred: dict[str, list[bool]],
    task: str,
    label: str,
) -> None:
    """Verify recomputed full-test metrics against the committed tabular summary JSON."""
    got = score(truth, pred, sorted(truth), task)
    pairs = (
        [("accuracy", "accuracy"), ("macro_f1", "score_a")]
        if task == "binary"
        else [("exact_match_accuracy", "exact_match"), ("mami_score_b", "score_b")]
    )
    for s_key, g_key in pairs:
        if abs(float(stored[s_key]) - float(got[g_key])) > 1e-4:
            raise ValueError(f"{label}: {g_key} {got[g_key]} != stored {stored[s_key]}")


def score(
    truth: dict[str, list[bool]], pred: dict[str, list[bool]], ids: Sequence[str], task: str
) -> dict[str, Any]:
    """Official MAMI Score A (binary) or Score B (multiclass) plus accuracy on a subset."""
    sub_t = {i: truth[i] for i in ids}
    sub_p = {i: pred[i] for i in ids}
    if task == "binary":
        acc = float(np.mean([sub_t[i][0] == sub_p[i][0] for i in ids]))
        return {"n": len(ids), "accuracy": acc, "score_a": float(compute_scoreA(sub_t, sub_p))}
    n_pos = sum(sum(sub_t[i][1:]) for i in ids)
    exact = float(np.mean([sub_t[i][1:] == sub_p[i][1:] for i in ids]))
    score_b = float(compute_scoreB(sub_t, sub_p)) if n_pos > 0 else None
    return {"n": len(ids), "exact_match": exact, "score_b": score_b, "n_positive_labels": n_pos}


def correct_vector(
    truth: dict[str, list[bool]], pred: dict[str, list[bool]], ids: Sequence[str], task: str
) -> np.ndarray:
    """Per-sample correctness (binary label, or exact match of the four subtypes)."""
    if task == "binary":
        return np.array([truth[i][0] == pred[i][0] for i in ids], dtype=float)
    return np.array([truth[i][1:] == pred[i][1:] for i in ids], dtype=float)


def paired_diff(a: np.ndarray, b: np.ndarray, rng: np.random.Generator) -> dict[str, float]:
    """Paired accuracy difference b - a with bootstrap 95% CI and exact McNemar p-value."""
    diff = float(b.mean() - a.mean())
    n = len(a)
    idx = rng.integers(0, n, size=(N_BOOT, n))
    boots = b[idx].mean(axis=1) - a[idx].mean(axis=1)
    only_a = int(((a == 1) & (b == 0)).sum())
    only_b = int(((a == 0) & (b == 1)).sum())
    disc = only_a + only_b
    p = float(stats.binomtest(only_a, disc, 0.5).pvalue) if disc > 0 else 1.0
    return {
        "diff_ocr_minus_provided": diff,
        "ci95_low": float(np.percentile(boots, 2.5)),
        "ci95_high": float(np.percentile(boots, 97.5)),
        "provided_only_correct": only_a,
        "ocr_only_correct": only_b,
        "mcnemar_exact_p": p,
    }


def ocr_effect(
    label: str,
    task: str,
    truth: dict[str, list[bool]],
    pred_prov: dict[str, list[bool]] | None,
    pred_ocr: dict[str, list[bool]],
    buckets: dict[str, list[str]],
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Score provided vs OCR predictions overall and per normalised-CER bucket."""
    groups: dict[str, list[str]] = {"all": sorted(truth), **buckets}
    res: dict[str, Any] = {
        "model": label,
        "split": "test",
        "task": task,
        "provided_available": pred_prov is not None,
    }
    for gname, ids in groups.items():
        if not ids:
            res[gname] = {"n": 0}
            continue
        entry: dict[str, Any] = {"n": len(ids), "ocr": score(truth, pred_ocr, ids, task)}
        if pred_prov is not None:
            entry["provided"] = score(truth, pred_prov, ids, task)
            entry["paired"] = paired_diff(
                correct_vector(truth, pred_prov, ids, task),
                correct_vector(truth, pred_ocr, ids, task),
                rng,
            )
        res[gname] = entry
    return res


# ---------------------------------------------------------------------------
# Truncation
# ---------------------------------------------------------------------------


def token_lengths(tokenizer: Any, texts: Sequence[str]) -> list[int]:
    """CLIP BPE length incl. start/end tokens of the exact string that was embedded.

    ``scripts/extract_embeddings.py`` embeds ``text.strip() or "empty text"``.
    """
    return [len(tokenizer.encode(t.strip() or "empty text")) + 2 for t in texts]


def verify_tokenizer(tokenizer: Any, texts: Sequence[str], lengths: Sequence[int]) -> None:
    """Check that ``encode`` lengths match the truncated tensor produced by ``__call__``."""
    long_i = [i for i, n in enumerate(lengths) if n > CONTEXT_LENGTH][:20]
    short_i = [i for i, n in enumerate(lengths) if n <= CONTEXT_LENGTH][:20]
    for i in long_i + short_i:
        toks = tokenizer([texts[i].strip() or "empty text"])[0]
        n_nonzero = int((toks != 0).sum())
        if n_nonzero != min(lengths[i], CONTEXT_LENGTH):
            raise ValueError(f"tokenizer check failed at index {i}")
        if int(toks[n_nonzero - 1]) != tokenizer.eot_token_id:
            raise ValueError(f"missing end token at index {i}")


def length_report(lengths: Sequence[int]) -> dict[str, Any]:
    """Share of texts over the context length and the length distribution."""
    arr = np.asarray(lengths)
    n_over = int((arr > CONTEXT_LENGTH).sum())
    return {
        "n": len(arr),
        "n_over_77": n_over,
        "pct_over_77": 100.0 * n_over / len(arr),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": int(arr.max()),
        "histogram": {name: int(((arr >= lo) & (arr <= hi)).sum()) for name, lo, hi in LENGTH_BINS},
    }


def group_compare(correct: np.ndarray, flag: np.ndarray, rng: np.random.Generator) -> dict:
    """Accuracy of truncated vs non-truncated groups with z-test, Fisher test and bootstrap CI."""
    a, b = correct[flag], correct[~flag]
    out: dict[str, Any] = {
        "n_truncated": len(a),
        "n_not_truncated": len(b),
        "correct_truncated": int(a.sum()),
        "correct_not_truncated": int(b.sum()),
    }
    if len(a) == 0 or len(b) == 0:
        return out
    pa, pb = float(a.mean()), float(b.mean())
    pooled = (a.sum() + b.sum()) / (len(a) + len(b))
    se = math.sqrt(pooled * (1 - pooled) * (1 / len(a) + 1 / len(b)))
    z = (pa - pb) / se if se > 0 else 0.0
    table = [[int(a.sum()), len(a) - int(a.sum())], [int(b.sum()), len(b) - int(b.sum())]]
    boots = a[rng.integers(0, len(a), size=(N_BOOT, len(a)))].mean(axis=1) - b[
        rng.integers(0, len(b), size=(N_BOOT, len(b)))
    ].mean(axis=1)
    out.update(
        {
            "acc_truncated": pa,
            "acc_not_truncated": pb,
            "diff_truncated_minus_not": pa - pb,
            "two_proportion_z": z,
            "two_proportion_p": float(2 * stats.norm.sf(abs(z))),
            "fisher_exact_p": float(stats.fisher_exact(table).pvalue),
            "bootstrap_ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
        }
    )
    if min(len(a), len(b)) < 10:
        out["small_n_warning"] = (
            "fewer than 10 memes in a group: z-test and bootstrap CI are unreliable; use Fisher p"
        )
    return out


def truncation_subgroups(
    data: SplitData,
    truth: dict[str, list[bool]],
    lengths: dict[str, list[int]],
    tab_ocr: dict[str, dict[str, list[bool]]],
    tab_prov: dict[str, dict[str, list[bool]]] | None,
    qwen_ft: dict[str, list[bool]],
    qwen_file: str,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    """Compare accuracy on truncated vs non-truncated memes for one split.

    Each model is flagged by the length of the text it actually consumed: PaddleOCR tabular
    models by the OCR length, provided-text tabular models and Qwen2-VL (which does not use the
    CLIP tokenizer; a difficulty control) by the gold length. PaddleOCR tabular models are also
    reported against the gold-length flag.
    """
    ids = data.image_ids
    labels_arr = np.array(data.labels, dtype=bool)

    def entry(model: str, flag_text: str, pred: dict[str, list[bool]] | None) -> dict[str, Any]:
        flag = np.array(lengths[flag_text]) > CONTEXT_LENGTH
        row: dict[str, Any] = {
            "split": data.split,
            "model": model,
            "flag_text": flag_text,
            "pct_misogynous_truncated": float(labels_arr[flag].mean() * 100)
            if flag.any()
            else None,
            "pct_misogynous_not_truncated": float(labels_arr[~flag].mean() * 100),
        }
        if pred is None:
            row["comparison"] = {"unavailable": True, "n_truncated": int(flag.sum())}
        else:
            row["comparison"] = group_compare(correct_vector(truth, pred, ids, "binary"), flag, rng)
        return row

    rows = []
    for m in TABULAR_MODELS:
        rows.append(entry(f"Tabular {SHORT[m]} (PaddleOCR text)", "ocr", tab_ocr[m]))
        rows.append(entry(f"Tabular {SHORT[m]} (PaddleOCR text)", "gold", tab_ocr[m]))
        rows.append(
            entry(
                f"Tabular {SHORT[m]} (provided text)",
                "gold",
                tab_prov[m] if tab_prov is not None else None,
            )
        )
    qrow = entry("Qwen2-VL-7B QLoRA seed1 (provided text, no CLIP)", "gold", qwen_ft)
    qrow["file"] = qwen_file
    rows.append(qrow)
    return rows


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def fmt(x: float | None, nd: int = 3) -> str:
    """Format a float or return 'n/a'."""
    return "n/a" if x is None else f"{x:.{nd}f}"


def build_summary(ocr: dict[str, Any], trunc: dict[str, Any]) -> str:
    """Render summary.md (<= 50 lines) from the computed JSON reports."""
    lines = [
        "# EXP-05: OCR quality, OCR ablation and 77-token truncation",
        "",
        "## Method",
        "",
        f"- `{ocr['provenance']['command']}` (CPU, commit {ocr['provenance']['git_commit'][:7]}); numbers from the JSONs.",
        "- Gold = MAMI `text` column; OCR = PaddleOCR `raw_texts` (already NFKC + lowercased by `utils/ocr.py`).",
        "- jiwer 4.0.0; raw = as stored; norm = NFKC, lowercase, no punctuation, collapsed spaces; no-space = norm - spaces.",
        "- Buckets: per-meme norm CER. Scores: `scripts/official_mami_evaluation.py`. Tokens: open_clip ViT-L-14 + SOT/EOT.",
        "",
        "## OCR quality (corpus-level; mean/median are per meme, norm CER)",
        "",
        "| Split | raw CER | raw WER | norm CER | norm WER | no-space CER | mean / median | empty OCR % |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for s in SPLITS:
        rep = ocr["ocr_quality"][s]["variants"]
        r, n, ns = rep["raw"], rep["normalised"], rep["normalised_no_space"]
        lines.append(
            f"| {s} | {fmt(r['cer']['corpus'])} | {fmt(r['wer']['corpus'])} | "
            f"{fmt(n['cer']['corpus'])} | {fmt(n['wer']['corpus'])} | {fmt(ns['cer']['corpus'])} | "
            f"{fmt(n['cer']['per_meme']['mean'])} / {fmt(n['cer']['per_meme']['median'])} | "
            f"{fmt(ocr['ocr_quality'][s]['pct_empty_ocr_raw'], 1)} |"
        )
    sizes = ocr["test_bucket_sizes"]
    lines += [
        "",
        "## Test by norm-CER bucket (Qwen: OCR-provided acc/EM, Score provided->OCR; tabular: OCR acc/EM, Score)",
        "",
        "| Model | Task | all (n=1000) | "
        + " | ".join(f"{b[0]} (n={sizes[b[0]]})" for b in BUCKETS)
        + " |",
        "|---|---|---|" + "---|" * len(BUCKETS),
    ]
    for eff in ocr["ocr_effect"]:
        cells = []
        key = "score_a" if eff["task"] == "binary" else "score_b"
        acc_key = "accuracy" if eff["task"] == "binary" else "exact_match"
        for g in ("all", *(b[0] for b in BUCKETS)):
            e = eff[g]
            if "ocr" not in e:
                cells.append("n=0")
            elif "paired" in e:
                cells.append(
                    f"{e['paired']['diff_ocr_minus_provided']:+.3f} "
                    f"({key[-1].upper()} {fmt(e['provided'][key])}->{fmt(e['ocr'][key])})"
                )
            else:
                cells.append(f"{fmt(e['ocr'][acc_key])} ({key[-1].upper()} {fmt(e['ocr'][key])})")
        lines.append(f"| {eff['model']} | {eff['task']} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## 77-token truncation (subgroup flag = length of the text the model consumed)",
        "",
    ]
    for s in SPLITS:
        gr: dict[str, Any] = trunc["lengths"][s]["gold"]
        orp: dict[str, Any] = trunc["lengths"][s]["ocr"]
        lines.append(
            f"- {s}: gold {gr['n_over_77']} ({gr['pct_over_77']:.2f}%) > 77 tokens, median {gr['median']:.0f}, "
            f"max {gr['max']}; OCR {orp['n_over_77']} ({orp['pct_over_77']:.2f}%), median {orp['median']:.0f}."
        )
    for row in trunc["subgroups"]:
        c = row["comparison"]
        own_flag = ("PaddleOCR" in row["model"]) == (row["flag_text"] == "ocr")
        if "acc_truncated" not in c or not own_flag:
            continue
        lines.append(
            f"- {row['split']}, {row['model']}: acc truncated {c['correct_truncated']}/{c['n_truncated']} = "
            f"{c['acc_truncated']:.3f} vs {c['acc_not_truncated']:.3f}, Fisher p = {c['fisher_exact_p']:.3f}."
        )
    lines += ["", "## Caveats", ""]
    lines += [f"- {c}" for c in ocr["caveats"]]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def git_commit() -> str:
    """Return the current git HEAD commit hash."""
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()


def main(argv: Sequence[str] | None = None) -> int:
    """Run all EXP-05 analyses and write the JSON and Markdown outputs."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args(argv)
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    inputs: dict[str, str] = {}

    provenance = {
        "script": "scripts/revision/ocr_quality.py",
        "command": "uv run --with jiwer==4.0.0 python -m scripts.revision.ocr_quality",
        "git_commit": git_commit(),
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "jiwer_version": importlib.metadata.version("jiwer"),
        "open_clip_version": importlib.metadata.version("open_clip_torch"),
        "tokenizer": f"open_clip.get_tokenizer('{CLIP_MODEL}'), context_length {CONTEXT_LENGTH}",
        "bootstrap_resamples": N_BOOT,
        "seed": SEED,
        "inputs_sha256": inputs,
    }

    if os.environ.get("MAMI_SPLIT_DIR", "").strip():
        raise RuntimeError("MAMI_SPLIT_DIR is set; EXP-05 must use the official MAMI splits")
    manager = DatasetManager()
    data = {s: load_split(manager, s, inputs) for s in SPLITS}
    provenance["dataset_root"] = str(manager._dataset_path)

    # 1. OCR quality ------------------------------------------------------
    quality: dict[str, Any] = {}
    test_cer: dict[str, float | None] = {}
    for s in SPLITS:
        quality[s], per_meme = ocr_quality_split(data[s])
        if s == "test":
            test_cer = per_meme
        print(
            f"[{s}] CER raw/norm = {quality[s]['variants']['raw']['cer']['corpus']:.4f}/"
            f"{quality[s]['variants']['normalised']['cer']['corpus']:.4f}"
        )

    # 2. OCR effect per bucket ----------------------------------------------
    test = data["test"]
    truth = make_truth(test)
    buckets: dict[str, list[str]] = {b[0]: [] for b in BUCKETS}
    n_unbucketed = 0
    for img in test.image_ids:
        c = test_cer[img]
        if c is None:
            n_unbucketed += 1
        else:
            buckets[bucket_of(c)].append(img)

    effects: list[dict[str, Any]] = []
    parse_failures: dict[str, int] = {}
    for label, prov_f, ocr_f in QWEN_BINARY_PAIRS + QWEN_MULTI_PAIRS:
        prov = load_vlm(TEST_DIR / prov_f, truth, inputs)
        ocr_run = load_vlm(TEST_DIR / ocr_f, truth, inputs)
        task = "binary" if prov["task"] == "singleclass" else "multiclass"
        parse_failures[prov_f] = prov["n_parse_failures"]
        parse_failures[ocr_f] = ocr_run["n_parse_failures"]
        eff = ocr_effect(label, task, truth, prov["pred"], ocr_run["pred"], buckets, rng)
        eff["files"] = {"provided": f"results/test/{prov_f}", "ocr": f"results/test/{ocr_f}"}
        eff["stored_metrics"] = {"provided": prov["stored"], "ocr": ocr_run["stored"]}
        for side, run in (("provided", prov), ("ocr", ocr_run)):
            stored_key, mine_key = (
                ("exact_match_accuracy", "accuracy")
                if task == "binary"
                else ("mami_score_b", "score_b")
            )
            stored_v = run["stored"].get(stored_key)
            mine = eff["all"][side][mine_key]
            if stored_v is None or abs(stored_v - mine) > 1e-4:
                raise ValueError(f"{label} {side}: {mine_key} {mine} != stored {stored_v}")
        effects.append(eff)

    tab: dict[tuple[str, str], dict[str, dict[str, list[bool]]] | None] = {
        key: load_tabular(path, key[1], test, inputs) for key, path in TAB_CSV.items()
    }
    for task in ("binary", "multiclass"):
        ocr_tab = tab[("paddleocr", task)]
        prov_tab = tab[("provided", task)]
        if ocr_tab is None:
            raise FileNotFoundError(f"PaddleOCR tabular CSV missing for {task}")
        summary_name = (
            "tabular_paddleocr_binary.json"
            if task == "binary"
            else "tabular_paddleocr_multiclass_mami.json"
        )
        summary_path = TEST_DIR / summary_name
        inputs[rel(summary_path)] = sha256(summary_path)
        with summary_path.open(encoding="utf-8") as fh:
            stored_models = json.load(fh)["models"]
        for m in TABULAR_MODELS:
            check_tabular(stored_models[m], truth, ocr_tab[m], task, f"{summary_name}:{m}")
            eff = ocr_effect(
                f"Tabular {SHORT[m]} (CLIP ViT-L-14 concat)",
                task,
                truth,
                prov_tab[m] if prov_tab is not None else None,
                ocr_tab[m],
                buckets,
                rng,
            )
            eff["files"] = {
                "provided": rel(TAB_CSV[("provided", task)])
                if prov_tab is not None
                else f"MISSING: {TAB_CSV[('provided', task)].relative_to(ROOT).as_posix()}",
                "ocr": rel(TAB_CSV[("paddleocr", task)]),
            }
            effects.append(eff)

    caveats = [
        "Provided-text tabular CSVs/embeddings are absent locally (gitignored; SCIAMA unreachable): tabular is OCR-side only.",
        "Qwen2-VL also sees the image and can read the overlay: OCR-vs-provided gaps bound, not isolate, OCR-error effects.",
        "CER buckets are observational (layout and text length co-vary with CER); bucket gaps are not causal.",
        "Few test memes exceed 77 tokens, so those tests are underpowered; validation is supplementary.",
    ]
    if n_unbucketed:
        caveats.append(
            f"{n_unbucketed} test memes have an empty normalised gold text (unbucketed)."
        )
    ocr_report = {
        "task_id": "EXP-05",
        "split": "train/validation/test",
        "model": "PaddleOCR (OCR), Qwen2-VL zero-shot, tabular SVM-RBF/XGBoost",
        "provenance": provenance,
        "ocr_quality": quality,
        "cer_bucket_definition": "per-meme normalised CER; [0,0.1), [0.1,0.3), [0.3,0.6), >=0.6",
        "test_bucket_sizes": {k: len(v) for k, v in buckets.items()},
        "test_bucket_pct_misogynous": {
            k: 100.0 * float(np.mean([truth[i][0] for i in v])) if v else None
            for k, v in buckets.items()
        },
        "n_test_unbucketed_empty_gold": n_unbucketed,
        "vlm_parse_failures": parse_failures,
        "ocr_effect": effects,
        "caveats": caveats,
    }

    # 3. Truncation -------------------------------------------------------
    tokenizer = open_clip.get_tokenizer(CLIP_MODEL)
    lengths: dict[str, dict[str, Any]] = {}
    split_lengths: dict[str, dict[str, list[int]]] = {}
    for s in SPLITS:
        gl = token_lengths(tokenizer, data[s].gold)
        ol = token_lengths(tokenizer, data[s].ocr)
        verify_tokenizer(tokenizer, data[s].gold, gl)
        verify_tokenizer(tokenizer, data[s].ocr, ol)
        lengths[s] = {"split": s, "gold": length_report(gl), "ocr": length_report(ol)}
        split_lengths[s] = {"gold": gl, "ocr": ol}

    subgroups: list[dict[str, Any]] = []
    for s in ("validation", "test"):
        split_truth = truth if s == "test" else make_truth(data[s])
        tab_ocr = load_tabular(TRUNC_TAB_CSV[(s, "paddleocr")], "binary", data[s], inputs)
        tab_prov = load_tabular(TRUNC_TAB_CSV[(s, "provided")], "binary", data[s], inputs)
        if tab_ocr is None:
            raise FileNotFoundError(f"PaddleOCR tabular CSV missing for {s}")
        ft_path = ROOT / "results" / s / QWEN_FT_SEED1.replace("_test_", f"_{s}_")
        ft = load_vlm(ft_path, split_truth, inputs, split=s)
        subgroups += truncation_subgroups(
            data[s],
            split_truth,
            split_lengths[s],
            {m: tab_ocr[m] for m in TABULAR_MODELS},
            {m: tab_prov[m] for m in TABULAR_MODELS} if tab_prov is not None else None,
            ft["pred"],
            rel(ft_path),
            rng,
        )

    trunc_report = {
        "task_id": "EXP-05",
        "split": "train/validation/test",
        "model": f"open_clip {CLIP_MODEL} tokenizer; tabular SVM-RBF/XGBoost; Qwen2-VL-7B QLoRA seed1",
        "provenance": provenance,
        "definition": "length = len(tokenizer.encode(text.strip() or 'empty text')) + 2; truncated if > 77",
        "lengths": lengths,
        "subgroups": subgroups,
        "caveats": caveats,
    }

    for name, obj in (("ocr_quality.json", ocr_report), ("truncation.json", trunc_report)):
        with (out_dir / name).open("w", encoding="utf-8", newline="\n") as fh:
            json.dump(obj, fh, indent=2)
            fh.write("\n")
    summary = build_summary(ocr_report, trunc_report)
    if summary.count("\n") > 50:
        raise ValueError(f"summary.md has {summary.count(chr(10))} lines (> 50)")
    (out_dir / "summary.md").write_text(summary, encoding="utf-8", newline="\n")
    print(f"Wrote {rel(out_dir)}/ocr_quality.json, truncation.json, summary.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
