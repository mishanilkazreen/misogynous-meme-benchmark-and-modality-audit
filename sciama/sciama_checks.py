"""Prerequisite and output checks for the SCIAMA experiment jobs (see sciama/README.md).

Two subcommands, both exit non-zero on any failure:

``prereqs``  Run on the SCIAMA login node by ``sciama/submit.sh`` before ``sbatch`` and again at the
             start of every job (``--in-job``). A job is only queued when every input it needs exists
             and is valid: clean repo at ``origin/main``, dataset, model weights, adapters (with the
             right base model), OCR transcripts, code changes, free disk, and no existing output that
             would be overwritten.
``verify``   Run locally after syncing. Checks every expected output file against the real dataset
             (image ids, ground-truth labels, record counts, split, no mock/error records, reported
             metrics recomputed from per-sample predictions) and writes a report to
             ``results/sciama/verification/``.

Job keys: S1-seed, S1-zeroshot, S2, S3, S4, S5, S6, S6-qwen-raw, S6-qwen-def (parameters as in sciama/JOBS.md).
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
EMB = RESULTS / "embeddings"
VERIFY_DIR = RESULTS / "sciama" / "verification"
SPLITS = ("validation", "test")
SPLIT_SIZE = {"validation": 1000, "test": 1000}
SUBTYPES = ("shaming", "stereotype", "objectification", "violence")
DATASET_GLOB = "datasets/chukwuebukaanulunko/multimodal-misogyny-detection-mami-2022/versions/*"
LLAVA = "llava-hf/llava-1.5-7b-hf"
QWEN = {"2b": "Qwen/Qwen2-VL-2B-Instruct", "7b": "Qwen/Qwen2-VL-7B-Instruct"}
S3_MODELS = {  # rationale name -> (explainer family, base model id, singleclass adapter dir)
    "qwen7b": ("qwen", QWEN["7b"], "lora_qwen2_vl_7b_instruct_singleclass"),
    "qwen2b": ("qwen", QWEN["2b"], "lora_qwen2_vl_2b_instruct_singleclass"),
    "llava7b": ("llava", LLAVA, "lora_llava_1.5_7b_hf_singleclass"),
}
MIN_FREE_GB_TRAIN = 50
TASKS = ("singleclass", "multiclass")


# --------------------------------------------------------------------------- naming


def out_slug(model_id: str) -> str:
    """Model slug used in benchmark output file names."""
    return model_id.rsplit("/", maxsplit=1)[-1].lower().replace("-", "_").replace(".", "_")


def adapter_dir(model_id: str, task: str, seed: int | str, tag: str = "") -> Path:
    """Adapter directory written by scripts/train_vlm.py (dots are kept)."""
    clean = model_id.lower().split("/")[-1].replace("-", "_")
    suffix = f"_{tag}" if tag else ""
    return RESULTS / "models" / f"lora_{clean}_{task}_seed{seed}{suffix}"


def bench_file(
    prefix: str,
    split: str,
    model_id: str,
    task: str,
    ocr: bool = False,
    finetuned: bool = False,
    seed: int | str | None = None,
    tag: str = "",
) -> Path:
    """Benchmark output path, matching scripts/benchmark_{qwen2vl,llava}.py naming."""
    name = f"{prefix}_{split}_{out_slug(model_id)}"
    name += "_ocr_paddleocr" if ocr else ""
    name += "_multiclass" if task == "multiclass" else ""
    name += "_finetuned" if finetuned else ""
    name += f"_seed{seed}" if seed is not None else ""
    name += f"_{tag}" if tag else ""
    return RESULTS / split / f"{name}.json"


# --------------------------------------------------------------------------- job spec


@dataclass
class Spec:
    """Everything a job needs and produces."""

    weights: list[str] = field(default_factory=list)
    adapters: list[tuple[Path, str]] = field(default_factory=list)  # (dir, expected base model)
    inputs: list[Path] = field(default_factory=list)
    outputs: list[Path] = field(default_factory=list)
    rationale_n: int = 0
    trains: bool = False
    overwrite_intended: bool = False
    code_markers: list[tuple[str, str]] = field(default_factory=list)  # (file, required text)
    split_dir: Path | None = None


def ocr_npz(split: str) -> Path:
    """PaddleOCR transcript file for a split."""
    return EMB / f"{split}_vit_l_14_quickgelu_ocr_paddleocr.npz"


def build_spec(a: argparse.Namespace) -> Spec:
    """Return the Spec for the requested job and parameters."""
    s = Spec()
    if a.job == "S1-seed":
        s.weights, s.trains = [LLAVA], True
        s.outputs = [
            bench_file("llava", sp, LLAVA, a.task, finetuned=True, seed=a.seed) for sp in SPLITS
        ]
    elif a.job == "S1-zeroshot":
        s.weights, s.overwrite_intended = [LLAVA], True
        s.inputs = [ocr_npz(sp) for sp in SPLITS]
        s.outputs = [
            bench_file("llava", sp, LLAVA, t, ocr=o)
            for t in TASKS
            for o in (False, True)
            for sp in SPLITS
        ]
    elif a.job == "S2":
        mid = QWEN[a.model]
        s.weights = [mid]
        s.inputs = [ocr_npz(sp) for sp in SPLITS]
        s.adapters = [(adapter_dir(mid, t, sd), mid) for t in TASKS for sd in (1, 2, 3)]
        s.outputs = [
            bench_file("qwen2vl", sp, mid, t, ocr=True, finetuned=True, seed=sd)
            for t in TASKS
            for sd in (1, 2, 3)
            for sp in SPLITS
        ]
    elif a.job == "S3":
        s.weights = [m[1] for m in S3_MODELS.values()]
        s.adapters = [(RESULTS / "models" / m[2], m[1]) for m in S3_MODELS.values()]
        s.outputs = [RESULTS / "rationales" / f"{k}_test_n{a.n}.jsonl" for k in S3_MODELS]
        s.rationale_n = a.n
    elif a.job == "S4":
        mid = QWEN[a.model]
        s.weights, s.trains = [mid], True
        s.split_dir = ROOT / "data" / "splits" / "template_disjoint"
        s.inputs = [s.split_dir / f"{sp}.tsv" for sp in ("train", "validation", "test")]
        s.code_markers = [
            ("utils/dataset.py", "MAMI_SPLIT_DIR"),
            ("utils/dataset.py", "MAMI_SPLIT_TAG"),
            ("scripts/train_vlm.py", "MAMI_SPLIT_TAG"),
            ("scripts/benchmark_qwen2vl.py", "MAMI_SPLIT_TAG"),
        ]
        s.outputs = [
            bench_file("qwen2vl", sp, mid, a.task, finetuned=True, seed=a.seed, tag="tdisjoint")
            for sp in SPLITS
        ]
    elif a.job == "S5":
        s.weights, s.trains = [a.model_id], True
        s.code_markers = [
            ("scripts/benchmark_internvl.py", "--lora-path"),
            ("scripts/train_vlm.py", "internvl"),
        ]
        s.outputs = [
            bench_file(
                "internvl", sp, a.model_id, a.task, finetuned=ft, seed=a.seed if ft else None
            )
            for ft in (False, True)
            for sp in SPLITS
        ]
    elif a.job == "S6":
        s.weights = [QWEN["7b"]]
        s.code_markers = [
            ("scripts/revision/harmeme_eval.py", "run-sciama"),
        ]
        s.outputs = [
            RESULTS / sp / f"harmeme_{sp}_{slug}.json"
            for slug in ("clip_svm_rbf", "clip_xgboost", "qwen2_vl_7b_instruct")
            for sp in SPLITS
        ]
    elif a.job == "S6-qwen-raw":
        s.weights = [QWEN["7b"]]
        s.code_markers = [
            ("scripts/revision/harmeme_eval.py", "--output-tag"),
        ]
        s.outputs = [
            RESULTS / sp / f"harmeme_{sp}_qwen2_vl_7b_instruct_raw_responses.json" for sp in SPLITS
        ]
    elif a.job == "S6-qwen-def":
        s.weights = [QWEN["7b"]]
        s.code_markers = [
            ("scripts/revision/harmeme_eval.py", "--prompt-variant"),
        ]
        s.outputs = [
            RESULTS / sp / f"harmeme_{sp}_qwen2_vl_7b_instruct_definition.json" for sp in SPLITS
        ]
    else:
        raise SystemExit(f"unknown job {a.job}")
    return s


# --------------------------------------------------------------------------- prereqs


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=False
    ).stdout.strip()


def _find_dataset_root() -> Path | None:
    cache = Path(os.environ.get("KAGGLEHUB_CACHE", Path.home() / ".cache" / "kagglehub"))
    for cand in sorted(cache.glob(DATASET_GLOB), reverse=True):
        if all((cand / f"{sp}.tsv").exists() for sp in ("train", "validation", "test")):
            return cand
    return None


def check_dataset(errors: list[str]) -> None:
    """The MAMI dataset is cached and every split loads with the expected size and images."""
    root = _find_dataset_root()
    if root is None:
        errors.append("MAMI dataset not found in KAGGLEHUB_CACHE (README section 3)")
        return
    sys.path.insert(0, str(ROOT))
    try:
        from utils.dataset import MamiDataset

        for sp, size in (("train", None), *SPLIT_SIZE.items()):
            ds = MamiDataset(str(root), split=sp)
            if size is not None and len(ds) != size:
                errors.append(f"dataset {sp}: {len(ds)} records, expected {size}")
            for i in (0, len(ds) // 2, len(ds) - 1):
                ds[i]  # opens the image; raises if the file is missing or unreadable
    except Exception as exc:
        errors.append(f"dataset failed to load: {exc!r}")


def check_weights(model_ids: list[str], errors: list[str]) -> None:
    """Every model is fully present in the local Hugging Face cache (no download possible in jobs)."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        errors.append("huggingface_hub not importable; activate the conda env")
        return
    for mid in model_ids:
        try:
            path = Path(snapshot_download(mid, local_files_only=True))
            if not (path / "config.json").exists():
                errors.append(f"weights {mid}: snapshot has no config.json")
        except Exception as exc:
            errors.append(
                f"weights {mid} not in HF_HOME cache ({type(exc).__name__}); run predownload"
            )


def check_adapters(adapters: list[tuple[Path, str]], errors: list[str]) -> None:
    """Each adapter has config + weights and was trained on the expected base model."""
    for path, base in adapters:
        cfg = path / "adapter_config.json"
        if not cfg.exists():
            errors.append(f"adapter missing: {path.relative_to(ROOT)}")
            continue
        if not any((path / w).exists() for w in ("adapter_model.safetensors", "adapter_model.bin")):
            errors.append(f"adapter weights missing in {path.relative_to(ROOT)}")
        found = json.loads(cfg.read_text(encoding="utf-8")).get("base_model_name_or_path", "")
        if found != base:
            errors.append(f"adapter {path.name}: base model {found!r}, expected {base!r}")


def check_npz(path: Path, split: str, errors: list[str]) -> None:
    """OCR transcript file covers the whole split with non-empty text for most memes."""
    try:
        import numpy as np

        z = np.load(path, allow_pickle=True)
        ids, texts = z["image_ids"], z["raw_texts"]
        if len(ids) != SPLIT_SIZE[split] or len(set(ids.tolist())) != len(ids):
            errors.append(f"{path.name}: {len(ids)} ids (expected {SPLIT_SIZE[split]} unique)")
        empty = sum(1 for t in texts if not str(t).strip())
        if empty > 0.2 * len(texts):
            errors.append(f"{path.name}: {empty} empty OCR transcripts")
    except Exception as exc:
        errors.append(f"{path.name} unreadable: {exc!r}")


def check_template_split(split_dir: Path, errors: list[str]) -> None:
    """Template-disjoint split: required columns and no template shared between train and test."""
    templates: dict[str, set[str]] = {}
    for sp in ("train", "validation", "test"):
        with (split_dir / f"{sp}.tsv").open(encoding="utf-8") as f:
            rows = list(csv.DictReader(f, delimiter="\t"))
        if not rows or "template_id" not in rows[0]:
            errors.append(f"{sp}.tsv has no rows or no template_id column")
            return
        templates[sp] = {r["template_id"] for r in rows}
    for other in ("validation", "test"):
        shared = templates["train"] & templates[other]
        if shared:
            errors.append(
                f"template split not disjoint: {len(shared)} templates shared by train and {other}"
            )


def check_harmeme_dataset(errors: list[str]) -> None:
    """The HarMeme (Harm-C) splits and images exist and load with expected counts (3013/177/354)."""
    sys.path.insert(0, str(ROOT))
    try:
        from PIL import Image

        from scripts.revision.harmeme_eval import (
            SPLITS as HARMEME_SPLITS,
        )
        from scripts.revision.harmeme_eval import (
            load_split_records,
            resolve_image_path,
        )

        for sp in HARMEME_SPLITS:
            recs = load_split_records(sp)
            for idx in (0, len(recs) // 2, len(recs) - 1):
                img_path = resolve_image_path(recs[idx])
                with Image.open(img_path) as im:
                    im.convert("RGB")
    except Exception as exc:
        errors.append(f"Harm-C dataset failed to load: {exc!r}")


def prereqs(a: argparse.Namespace) -> list[str]:
    """Return every unmet prerequisite for the job."""
    spec, errors = build_spec(a), []
    if not a.in_job:
        # Code must be clean; allow current branch if SCIAMA_ALLOW_BRANCH=1, otherwise origin/main.
        if _git("status", "--porcelain", "--untracked-files=no", "--", ".", ":!results", ":!runs"):
            errors.append(
                "repo has modified tracked code files; do not submit from a dirty checkout"
            )
        if os.environ.get("SCIAMA_ALLOW_BRANCH", "").strip() != "1":
            _git("fetch", "origin", "main")
            if _git("rev-parse", "HEAD") != _git("rev-parse", "origin/main"):
                errors.append("HEAD is not origin/main; run git pull origin main")
    else:
        try:
            import torch

            if not torch.cuda.is_available():
                errors.append("no CUDA GPU visible in the job")
        except ImportError:
            errors.append("torch not importable in the job environment")
    for d in ("results/sciama/logs", "results/sciama/manifests"):
        if not (ROOT / d).is_dir():
            errors.append(
                f"missing directory {d} (mkdir -p it; Slurm needs the log dir before start)"
            )
    if a.job in ("S6", "S6-qwen-raw", "S6-qwen-def"):
        check_harmeme_dataset(errors)
    else:
        check_dataset(errors)
    check_weights(spec.weights, errors)
    check_adapters(spec.adapters, errors)
    for p in spec.inputs:
        if not p.exists():
            errors.append(f"required input missing: {p.relative_to(ROOT)}")
        elif p.suffix == ".npz":
            check_npz(p, p.name.split("_")[0], errors)
    for file, marker in spec.code_markers:
        fp = ROOT / file
        if not fp.exists() or marker.lower() not in fp.read_text(encoding="utf-8").lower():
            errors.append(f"code change missing: {file} must contain {marker!r} (JOBS.md)")
    if spec.split_dir is not None and all(p.exists() for p in spec.inputs):
        check_template_split(spec.split_dir, errors)
    if spec.trains:
        free_gb = shutil.disk_usage(ROOT).free / 1e9
        if free_gb < MIN_FREE_GB_TRAIN:
            errors.append(f"only {free_gb:.0f} GB free on Lustre, need {MIN_FREE_GB_TRAIN}")
        if a.job in ("S1-seed", "S4", "S5"):
            adir = adapter_dir(
                LLAVA if a.job == "S1-seed" else (QWEN[a.model] if a.job == "S4" else a.model_id),
                a.task,
                a.seed,
                "tdisjoint" if a.job == "S4" else "",
            )
            if adir.exists() and not a.in_job:
                errors.append(
                    f"adapter already exists, would be overwritten: {adir.relative_to(ROOT)}"
                )
    existing = [p for p in spec.outputs if p.exists()]
    overwrite_ok = spec.overwrite_intended and a.allow_overwrite
    if existing and not overwrite_ok and not a.in_job:
        errors += [
            f"output already exists, would be overwritten: {p.relative_to(ROOT)}" for p in existing
        ]
    return errors


# --------------------------------------------------------------------------- verify


def _truth(
    split: str, split_dir: Path | None = None
) -> tuple[list[str], dict[str, int], dict[str, dict[str, int]]]:
    """Ordered image ids and labels of a split.

    Random split: from the committed-source NPZ (dataset copy). Template-disjoint split (job S4,
    or ``split_dir`` / ``$MAMI_SPLIT_DIR`` given): from ``data/splits/template_disjoint/<split>.tsv``
    (or that folder), whose validation/test memes are drawn from all original MAMI splits.
    """
    import numpy as np

    if split_dir is None and os.environ.get("MAMI_SPLIT_DIR", "").strip():
        split_dir = Path(os.environ["MAMI_SPLIT_DIR"].strip())
    if split_dir is None:
        job = argparse.ArgumentParser(add_help=False)
        job.add_argument("--job")
        if job.parse_known_args(sys.argv[1:])[0].job == "S4":
            split_dir = ROOT / "data" / "splits" / "template_disjoint"
    if split_dir is not None:
        tsv = split_dir / f"{split}.tsv"
        if not tsv.exists():
            raise SystemExit(f"ground truth for {split} not found ({tsv})")
        with tsv.open(encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f, delimiter="\t"))
        ids = [Path(r["file_name"]).stem for r in rows]
        if len(set(ids)) != len(ids):
            raise SystemExit(f"{tsv}: duplicate image ids")
        lab = {i: int(r["label"]) for i, r in zip(ids, rows, strict=True)}
        sub = {i: {c: int(r[c]) for c in SUBTYPES} for i, r in zip(ids, rows, strict=True)}
        return ids, lab, sub

    candidates = [
        EMB,
        ROOT / ".sciama_backup" / "results" / "embeddings",
    ]
    backup_env = os.environ.get("SCIAMA_BACKUP_DIR", "").strip()
    if backup_env:
        candidates.append(Path(backup_env) / "results" / "embeddings")
    for base in candidates:
        if not base.exists():
            continue
        p = base / ocr_npz(split).name
        if p.exists():
            z = np.load(p, allow_pickle=True)
            ids = [str(i) for i in z["image_ids"].tolist()]
            lab = {i: int(v) for i, v in zip(ids, z["labels"].tolist(), strict=True)}
            sub = {
                i: {c: int(v) for c, v in zip(SUBTYPES, row, strict=True)}
                for i, row in zip(ids, z["subtask_labels"].tolist(), strict=True)
            }
            return ids, lab, sub
    raise SystemExit(f"ground truth for {split} not found (need {ocr_npz(split).name})")


def verify_bench(path: Path, split: str, task: str) -> list[str]:
    """A benchmark JSON contains real per-sample predictions for the whole split."""
    errs: list[str] = []
    data = json.loads(path.read_text(encoding="utf-8"))
    rec = data[0] if isinstance(data, list) and data else {}
    if not rec:
        return ["empty or missing benchmark record"]
    if rec.get("mock") is True:
        errs.append("record marked as mock: true (mock output not allowed)")
    if rec.get("error") is not None:
        errs.append(f"record contains error: {rec.get('error')!r}")
    if rec.get("split") != split:
        errs.append(f"split field {rec.get('split')!r} != {split}")
    if rec.get("task") and rec.get("task") != task:
        errs.append(f"task field {rec.get('task')!r} != {task}")
    if not rec.get("model"):
        errs.append("model field empty")
    sp = rec.get("sample_predictions") or []
    ids, lab, sub = _truth(split)
    got = [str(p.get("image_id")) for p in sp]
    if len(sp) != SPLIT_SIZE[split] or set(got) != set(ids) or len(set(got)) != len(got):
        errs.append(f"{len(sp)} predictions; ids do not match the {split} split exactly")
        return errs
    bad_gt = sum(
        1
        for p in sp
        if (
            p.get("ground_truth")
            != (lab if task == "singleclass" else sub).get(str(p.get("image_id")))
        )
    )
    if bad_gt:
        errs.append(f"{bad_gt} ground-truth labels differ from the dataset")

    # Validate prediction schema and reject degenerate outputs
    if task == "singleclass":
        preds = [p.get("prediction") for p in sp]
        invalid_preds = [p for p in preds if p not in (0, 1)]
        if invalid_preds:
            errs.append(f"{len(invalid_preds)} predictions not in {{0, 1}}")
        elif len(set(preds)) == 1:
            errs.append(
                f"degenerate singleclass predictions: all {len(preds)} samples predicted constant class {preds[0]}"
            )
        else:
            gts = [int(p["ground_truth"]) for p in sp]
            f1_per_class = []
            for lbl in (0, 1):
                tp = sum(1 for pr, gt in zip(preds, gts, strict=True) if pr == lbl and gt == lbl)
                fp = sum(1 for pr, gt in zip(preds, gts, strict=True) if pr == lbl and gt != lbl)
                fn = sum(1 for pr, gt in zip(preds, gts, strict=True) if pr != lbl and gt == lbl)
                p_c = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                r_c = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1_c = (2 * p_c * r_c) / (p_c + r_c) if (p_c + r_c) > 0 else 0.0
                f1_per_class.append(f1_c)
            recomputed_macro_f1 = sum(f1_per_class) / len(f1_per_class)
            rep_f1 = rec.get("macro_f1", rec.get("f1"))
            if rep_f1 is not None and abs(float(rep_f1) - recomputed_macro_f1) > 1e-3:
                errs.append(f"reported macro-F1 {rep_f1} != recomputed {recomputed_macro_f1:.6f}")
    else:
        bad_schema = 0
        preds_by_subtype: dict[str, list[int]] = {c: [] for c in SUBTYPES}
        for p in sp:
            pred = p.get("prediction")
            if not isinstance(pred, dict) or not all(
                c in pred and pred[c] in (0, 1) for c in SUBTYPES
            ):
                bad_schema += 1
            else:
                for c in SUBTYPES:
                    preds_by_subtype[c].append(int(pred[c]))
        if bad_schema:
            errs.append(
                f"{bad_schema} multiclass predictions do not match subtype schema (dict with 4 keys in {{0, 1}})"
            )
        elif all(len(set(preds_by_subtype[c])) == 1 for c in SUBTYPES):
            errs.append(
                "degenerate multiclass predictions: all 4 subtypes predicted constant values across all samples"
            )
        else:
            sys.path.insert(0, str(ROOT))
            from models.vlm.metrics_multilabel import compute_multilabel_metrics

            ml_preds = [p["prediction"] for p in sp]
            ml_gts = [p["ground_truth"] for p in sp]
            ml_metrics = compute_multilabel_metrics(ml_preds, ml_gts, list(SUBTYPES))
            rep_score_b = rec.get("mami_score_b")
            if (
                rep_score_b is not None
                and abs(float(rep_score_b) - float(ml_metrics["mami_score_b"])) > 1e-3
            ):
                errs.append(
                    f"reported mami_score_b {rep_score_b} != recomputed {ml_metrics['mami_score_b']:.6f}"
                )
            rep_macro_f1 = rec.get("macro_f1", rec.get("f1"))
            if (
                rep_macro_f1 is not None
                and abs(float(rep_macro_f1) - float(ml_metrics["macro_f1"])) > 1e-3
            ):
                errs.append(
                    f"reported macro_f1 {rep_macro_f1} != recomputed {ml_metrics['macro_f1']:.6f}"
                )

    correct = sum(1 for p in sp if p.get("prediction") == p.get("ground_truth")) / len(sp)
    reported = rec.get("exact_match_accuracy")
    if reported is None or abs(float(reported) - correct) > 1e-5:
        errs.append(f"reported exact_match_accuracy {reported} != recomputed {correct:.6f}")
    return errs


def verify_rationales(path: Path, n: int) -> list[str]:
    """A rationale JSONL has N real, error-free records for the first N test memes."""
    errs: list[str] = []
    rows = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    ids, lab, _ = _truth("test")
    if [str(r.get("image_id")) for r in rows] != ids[:n]:
        errs.append(f"{len(rows)} records; image ids are not the first {n} test memes in order")
    if len({r.get("model") for r in rows}) != 1:
        errs.append("more than one model value")
    for r in rows:
        if r.get("error") is not None or r.get("mock") is not False or r.get("split") != "test":
            errs.append(f"record {r.get('image_id')}: error/mock/split invalid")
            break
        if not str(r.get("explanation", "")).strip():
            errs.append(f"record {r.get('image_id')}: empty explanation")
            break
        if int(r.get("ground_truth", -1)) != lab.get(str(r.get("image_id")), -2):
            errs.append(f"record {r.get('image_id')}: ground truth differs from dataset")
            break
    return errs


def verify(a: argparse.Namespace) -> tuple[bool, dict[str, Any]]:
    """Verify all expected outputs and return (passed, report)."""
    spec = build_spec(a)
    files: list[dict[str, Any]] = []
    for p in spec.outputs:
        entry: dict[str, Any] = {"path": p.relative_to(ROOT).as_posix()}
        if not p.exists():
            entry["errors"] = ["missing"]
        else:
            entry["sha256"] = hashlib.sha256(p.read_bytes()).hexdigest()
            if p.suffix == ".jsonl":
                entry["errors"] = verify_rationales(p, spec.rationale_n)
            elif p.name.startswith("harmeme_"):
                sys.path.insert(0, str(ROOT))
                from scripts.revision.harmeme_eval import verify_harmeme_output_file

                entry["errors"] = verify_harmeme_output_file(p, p.parent.name)
            else:
                split = p.parent.name
                task = "multiclass" if "_multiclass" in p.name else "singleclass"
                entry["errors"] = verify_bench(p, split, task)
        files.append(entry)
    passed = all(not f["errors"] for f in files) and bool(files)
    relevant = {
        "S1-seed": ("task", "seed"),
        "S1-zeroshot": (),
        "S2": ("model",),
        "S3": ("n",),
        "S4": ("model", "task", "seed"),
        "S5": ("model_id", "task", "seed"),
        "S6": (),
        "S6-qwen-raw": (),
        "S6-qwen-def": (),
    }[a.job]
    params = {k: getattr(a, k) for k in relevant}
    report = {
        "job": a.job,
        "params": params,
        "passed": passed,
        "checked_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git("rev-parse", "HEAD"),
        "files": files,
    }
    return passed, report


# --------------------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    ap = argparse.ArgumentParser(description="SCIAMA job prerequisite and output checks.")
    ap.add_argument("cmd", choices=("prereqs", "verify"))
    ap.add_argument(
        "--job",
        required=True,
        choices=(
            "S1-seed",
            "S1-zeroshot",
            "S2",
            "S3",
            "S4",
            "S5",
            "S6",
            "S6-qwen-raw",
            "S6-qwen-def",
        ),
    )
    ap.add_argument("--task", choices=TASKS, default="singleclass")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--model", choices=tuple(QWEN), default="2b")
    ap.add_argument("--model-id", default="")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--in-job", action="store_true", help="running inside the Slurm job")
    ap.add_argument("--allow-overwrite", action="store_true", help="only honoured for S1-zeroshot")
    a = ap.parse_args(argv)
    if a.job == "S5" and not a.model_id:
        ap.error("S5 needs --model-id")

    if a.cmd == "prereqs":
        errors = prereqs(a)
        for e in errors:
            print(f"PREREQ FAIL: {e}", file=sys.stderr)
        print(
            "PREREQS OK"
            if not errors
            else f"PREREQS FAILED ({len(errors)}); job must NOT be queued"
        )
        return 1 if errors else 0

    passed, report = verify(a)
    VERIFY_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    tag = "_".join(
        str(v).replace("/", "_")
        for k, v in report["params"].items()
        if k != "job" and v not in ("", None)
    )
    out = VERIFY_DIR / f"{a.job}_{tag}_{stamp}.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for f in report["files"]:
        print(("OK   " if not f["errors"] else "FAIL ") + f["path"], "; ".join(f["errors"]))
    print(
        f"{'VERIFIED' if passed else 'VERIFICATION FAILED'}: report {out.relative_to(ROOT).as_posix()}"
    )
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
