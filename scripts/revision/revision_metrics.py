"""Revision metrics for EXP-01 and EXP-07 (official MAMI scorer, seeds, CIs, paired tests).

Run from the repository root:

    uv run python -m scripts.revision.revision_metrics

What it does:

1. Loads every per-split, per-seed prediction file behind the paper's Task A / Task B tables,
   rejects conflated or mislabelled files (the ``split`` field must equal the split, image ids
   must be a subset of that split's ids and ground truth must match the dataset labels), and
   scores each run by calling ``compute_scoreA`` / ``compute_scoreB`` / ``compute_f1`` from
   ``scripts/official_mami_evaluation.py``. Every official value is asserted equal to sklearn
   and to ``models/vlm/metrics_multilabel.py`` within 1e-6.
2. Aggregates fine-tuned models over all available seeds (mean, sample SD, file list).
3. Computes bootstrap 95% CIs (2000 resamples over memes, fixed RNG seed), paired bootstrap
   tests and McNemar tests on the test split, the all-negative Exact Match baseline, the
   decision thresholds actually used and the validation-to-test drop.
4. Compares every number in the paper's result tables with the recomputed value and greps
   ``submission/main.tex`` (read only) for each value that changes.

Outputs go to ``results/revision/exp01_exp07/``.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import textwrap
from types import ModuleType
from typing import Any

import numpy as np
from scipy.stats import binomtest, chi2
from sklearn.metrics import accuracy_score, f1_score
import yaml

from models.vlm.metrics_multilabel import compute_mami_score_b, compute_multilabel_metrics

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
AB_DIR = ROOT / "auto_benchmark"
OUT_DIR = RESULTS / "revision" / "exp01_exp07"
MAIN_TEX = ROOT / "submission" / "main.tex"
EMB_DIR = RESULTS / "embeddings"

LABELS = ["shaming", "stereotype", "objectification", "violence"]
SPLITS = ["test", "validation"]
TOL = 1e-6
N_BOOT = 2000
BOOT_SEED = 20220705
N_VERIFY_RESAMPLES = 20
SRCB_SCORE_B = 0.7310

# Classical estimators whose ``predict`` is a margin sign (decision_function = 0), not a
# probability cut-off. All other auto_benchmark estimators predict argmax of predict_proba,
# which for two classes is the 0.50 threshold.
MARGIN_MODELS = {
    "Support Vector Machine - RBF SVC",
    "Support Vector Machine - LinearSVC",
    "Ridge Classifier",
    "Perceptron",
    "PassiveAggressiveClassifier",
    "SGD Classifier",
}

ML_DISPLAY = {
    "KNN": "KNN",
    "Logistic Regression": "Logistic Regression",
    "SGD Classifier": "SGD Classifier",
    "Ridge Classifier": "Ridge Classifier",
    "Perceptron": "Perceptron",
    "PassiveAggressiveClassifier": "Passive-Aggressive",
    "Linear Discriminant Analysis": "Linear Discriminant Analysis",
    "Support Vector Machine - LinearSVC": "Linear SVC",
    "Support Vector Machine - RBF SVC": "SVM-RBF",
    "Neural Network - MLPClassifier": "MLP",
    "Extra Tree Classifier": "Extra Tree",
    "Bernoulli Naive Bayes": "Bernoulli NB",
    "Gaussian Naive Bayes": "Gaussian NB",
    "Decision Tree": "Decision Tree",
    "Random Forest": "Random Forest",
    "Hist Gradient Boosting Classifier": "HistGB",
    "Gradient Boosting Classifier": "Gradient Boosting",
    "XGBoost": "XGBoost",
    "LightGBM": "LightGBM",
    "Extra Trees Classifier": "Extra Trees",
    "AdaBoost Classifier": "AdaBoost",
    "Bagging Classifier": "Bagging Classifier",
}
# Extra aliases used in the paper's tables for the same estimators.
ML_ALIASES = {"LDA": "Linear Discriminant Analysis", "Ridge": "Ridge Classifier"}

# auto_benchmark model configs holding the PaddleOCR-text per-sample predictions.
TABULAR_PADDLE_CFG = {
    ("A", "validation"): "mami_tabular_model",
    ("A", "test"): "mami_tabular_model_test",
    ("B", "validation"): "mami_tabular_model_multiclass",
    ("B", "test"): "mami_tabular_model_multiclass_test",
}


# --------------------------------------------------------------------------------------------
# Input registry (sha256 of every file read)
# --------------------------------------------------------------------------------------------
class InputRegistry:
    """Record the sha256 of every input file read by this script."""

    def __init__(self) -> None:
        """Create an empty registry."""
        self.files: dict[str, str] = {}

    def register(self, path: Path) -> None:
        """Hash ``path`` and store it under its repository-relative name."""
        rel = path.resolve().relative_to(ROOT).as_posix()
        if rel not in self.files:
            self.files[rel] = hashlib.sha256(path.read_bytes()).hexdigest()

    def read_json(self, path: Path) -> Any:
        """Read a JSON file and register it."""
        self.register(path)
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)

    def read_text(self, path: Path) -> str:
        """Read a text file and register it."""
        self.register(path)
        return path.read_text(encoding="utf-8")


REG = InputRegistry()


def rel(path: Path) -> str:
    """Return ``path`` relative to the repository root in POSIX form."""
    return path.resolve().relative_to(ROOT).as_posix()


def load_official() -> ModuleType:
    """Import the verbatim official MAMI scorer from ``scripts/official_mami_evaluation.py``."""
    path = ROOT / "scripts" / "official_mami_evaluation.py"
    REG.register(path)
    spec = importlib.util.spec_from_file_location("official_mami_evaluation", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


OFFICIAL = load_official()


# --------------------------------------------------------------------------------------------
# Ground truth and runs
# --------------------------------------------------------------------------------------------
@dataclass
class Truth:
    """Dataset labels for one split, in the order of the pre-extracted embedding file."""

    split: str
    ids: list[str]
    task_a: np.ndarray
    task_b: np.ndarray
    index: dict[str, int]


def load_truth(split: str) -> Truth:
    """Load ids and labels of ``split`` from the committed CLIP embedding archive."""
    path = EMB_DIR / f"{split}_vit_l_14_quickgelu_ocr_paddleocr.npz"
    REG.register(path)
    data = np.load(path, allow_pickle=False)
    ids = [str(x) for x in data["image_ids"].tolist()]
    if len(set(ids)) != len(ids):
        raise ValueError(f"duplicate image ids in {path}")
    task_a = data["labels"].astype(int)
    task_b = data["subtask_labels"].astype(int)
    if task_b.shape != (len(ids), 4):
        raise ValueError(f"unexpected subtask shape in {path}")
    return Truth(split, ids, task_a, task_b, {k: i for i, k in enumerate(ids)})


@dataclass
class Run:
    """One prediction run (one model, one seed, one split, one task)."""

    path: str
    split: str
    task: str
    ids: list[str]
    gold: np.ndarray
    pred: np.ndarray
    complete: bool
    seed: int | None
    source_kind: str
    info: dict[str, Any] = field(default_factory=dict)


def seed_of(path: Path) -> int | None:
    """Return the seed encoded in a ``*_seed<k>.json`` file name, if any."""
    match = re.search(r"_seed(\d+)\.json$", path.name)
    return int(match.group(1)) if match else None


def load_json_run(path: Path, split: str, task: str, truth: Truth, allow_subset: bool) -> Run:
    """Load and validate a per-sample VLM / CLIP / Gemini / VisualBERT result file.

    Raises:
        ValueError: if the file is conflated, mislabelled, mock, contains errors or
            disagrees with the dataset labels.
    """
    data = REG.read_json(path)
    if not isinstance(data, list) or len(data) != 1:
        raise ValueError(f"{rel(path)}: expected exactly one result record")
    record = data[0]
    if record.get("split") != split:
        raise ValueError(f"{rel(path)}: split field {record.get('split')!r} != {split!r}")
    ok_tasks = {"singleclass", "a"} if task == "A" else {"multiclass"}
    if record.get("task") not in ok_tasks:
        raise ValueError(f"{rel(path)}: task field {record.get('task')!r} not in {ok_tasks}")
    if record.get("mock"):
        raise ValueError(f"{rel(path)}: mock record")
    samples = record["sample_predictions"]
    ids = [str(s["image_id"]) for s in samples]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{rel(path)}: duplicate image ids")
    unknown = set(ids) - set(truth.ids)
    if unknown:
        raise ValueError(f"{rel(path)}: {len(unknown)} ids not in the {split} split")
    complete = set(ids) == set(truth.ids)
    if not complete and not allow_subset:
        raise ValueError(f"{rel(path)}: covers {len(ids)}/{len(truth.ids)} {split} ids")
    by_id = {str(s["image_id"]): s for s in samples}
    ordered = [i for i in truth.ids if i in by_id]
    rows = [truth.index[i] for i in ordered]
    parse_fail = 0
    refusals = 0
    preds: list[Any] = []
    golds: list[Any] = []
    for image_id in ordered:
        sample = by_id[image_id]
        if sample.get("mock") or sample.get("error"):
            raise ValueError(f"{rel(path)}: mock/error sample {image_id}")
        parse_fail += int(bool(sample.get("parse_failure")))
        refusals += int(bool(sample.get("refusal")))
        if task == "A":
            preds.append(int(sample["prediction"]))
            golds.append(int(sample["ground_truth"]))
        else:
            preds.append([int(sample["prediction"][lbl]) for lbl in LABELS])
            golds.append([int(sample["ground_truth"][lbl]) for lbl in LABELS])
    pred = np.asarray(preds, dtype=int)
    gold = np.asarray(golds, dtype=int)
    expected = truth.task_a[rows] if task == "A" else truth.task_b[rows]
    if not np.array_equal(gold, expected):
        raise ValueError(f"{rel(path)}: ground truth disagrees with dataset labels")
    if not set(np.unique(pred).tolist()) <= {0, 1}:
        raise ValueError(f"{rel(path)}: non-binary predictions")
    info = {
        "n_samples": len(ordered),
        "split_size": len(truth.ids),
        "parse_failure_rate": parse_fail / len(ordered),
        "sample_refusal_rate": refusals / len(ordered),
        "file_refusal_rate": record.get("refusal_rate"),
        "degenerate_constant_prediction": bool(np.all(pred == pred[0])),
        "benchmark_date": record.get("benchmark_date"),
    }
    return Run(
        rel(path), split, task, ordered, gold, pred, complete, seed_of(path), "per_sample", info
    )


def tabular_csv_path(task: str, split: str) -> tuple[Path, Path]:
    """Return (predictions CSV, embedding NPZ) for the PaddleOCR tabular run, checked via configs."""
    cfg_name = TABULAR_PADDLE_CFG[(task, split)]
    model_cfg_path = AB_DIR / "config" / "model" / f"{cfg_name}.yaml"
    model_cfg = yaml.safe_load(REG.read_text(model_cfg_path))
    data_cfg_path = AB_DIR / "config" / "data" / f"{model_cfg['data_config_name']}.yaml"
    data_cfg = yaml.safe_load(REG.read_text(data_cfg_path))
    npz_name = Path(data_cfg["dataset"]["val_embeddings"]).name
    expected = f"{split}_vit_l_14_quickgelu_ocr_paddleocr.npz"
    if npz_name != expected:
        raise ValueError(f"{cfg_name}: evaluates {npz_name}, expected {expected}")
    csv_path = AB_DIR / "results" / "model_results" / cfg_name / f"{cfg_name}_predictions.csv"
    return csv_path, EMB_DIR / npz_name


def load_tabular_runs(task: str, split: str, truth: Truth) -> dict[str, Run]:
    """Load all PaddleOCR-text classical-classifier runs from the auto_benchmark CSV.

    The CSV has no image ids; rows follow the evaluation NPZ order. The ``Actual`` column is
    asserted equal to the NPZ labels in that order, which fixes the id alignment.
    """
    csv_path, _ = tabular_csv_path(task, split)
    REG.register(csv_path)
    with csv_path.open(encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    header, body = rows[0], [r for r in rows[1:] if r]
    if len(body) != len(truth.ids):
        raise ValueError(f"{rel(csv_path)}: {len(body)} rows, split has {len(truth.ids)}")

    def parse(cell: str) -> Any:
        if task == "A":
            return int(float(cell))
        return [int(float(x)) for x in cell.strip().strip('"').split(",")]

    gold = np.asarray([parse(r[0]) for r in body], dtype=int)
    expected = truth.task_a if task == "A" else truth.task_b
    if not np.array_equal(gold, expected):
        raise ValueError(f"{rel(csv_path)}: Actual column disagrees with NPZ labels")
    runs: dict[str, Run] = {}
    for j, name in enumerate(header[1:], start=1):
        pred = np.asarray([parse(r[j]) for r in body], dtype=int)
        info = {
            "n_samples": len(body),
            "split_size": len(truth.ids),
            "ids_source": "evaluation NPZ row order (CSV Actual column verified)",
            "degenerate_constant_prediction": bool(np.all(pred == pred[0])),
        }
        runs[name] = Run(
            rel(csv_path), split, task, list(truth.ids), gold, pred, True, None, "per_sample", info
        )
    return runs


# --------------------------------------------------------------------------------------------
# Scoring (official scorer + cross-checks)
# --------------------------------------------------------------------------------------------
def official_dicts(run: Run) -> tuple[dict[str, list[bool]], dict[str, list[bool]]]:
    """Build the (truth, submission) dicts expected by the official scorer."""
    truth: dict[str, list[bool]] = {}
    sub: dict[str, list[bool]] = {}
    for k, image_id in enumerate(run.ids):
        if run.task == "A":
            truth[image_id] = [bool(run.gold[k])]
            sub[image_id] = [bool(run.pred[k])]
        else:
            g = [bool(x) for x in run.gold[k]]
            p = [bool(x) for x in run.pred[k]]
            truth[image_id] = [any(g), *g]
            sub[image_id] = [any(p), *p]
    return truth, sub


def check(name: str, a: float, b: float, path: str) -> None:
    """Assert two metric values agree within ``TOL``."""
    if abs(a - b) > TOL:
        raise AssertionError(f"{path}: {name} mismatch {a:.8f} vs {b:.8f}")


def score_run(run: Run) -> dict[str, Any]:
    """Score one run with the official functions and assert agreement with sklearn / metrics."""
    truth, sub = official_dicts(run)
    if run.task == "A":
        acc = float(np.mean(run.gold == run.pred))
        score_a = float(OFFICIAL.compute_scoreA(truth, sub))
        sk_a = float(f1_score(run.gold, run.pred, average="macro", zero_division=0.0))
        check("score_a", score_a, sk_a, run.path)
        check("acc", acc, float(accuracy_score(run.gold, run.pred)), run.path)
        if score_a > acc + TOL:
            raise AssertionError(f"{run.path}: Score A {score_a} > accuracy {acc}")
        return {"acc": acc, "score_a": score_a}
    score_b = float(OFFICIAL.compute_scoreB(truth, sub))
    per_label_official = {
        lbl: float(
            OFFICIAL.compute_f1(
                [bool(x) for x in run.pred[:, i]], [bool(x) for x in run.gold[:, i]]
            )
        )
        for i, lbl in enumerate(LABELS)
    }
    pos_f1 = [
        float(f1_score(run.gold[:, i], run.pred[:, i], zero_division=0.0))
        for i in range(len(LABELS))
    ]
    subtype_f1 = float(np.mean(pos_f1))
    em = float(np.mean(np.all(run.gold == run.pred, axis=1)))
    preds_d = [dict(zip(LABELS, map(int, r), strict=True)) for r in run.pred]
    golds_d = [dict(zip(LABELS, map(int, r), strict=True)) for r in run.gold]
    ml = compute_multilabel_metrics(preds_d, golds_d, LABELS)
    check(
        "score_b",
        score_b,
        float(compute_mami_score_b(preds_d, golds_d, LABELS)["mami_score_b"]),
        run.path,
    )
    check("subtype_f1", subtype_f1, float(ml["macro_f1"]), run.path)
    check("em", em, float(ml["exact_match_accuracy"]), run.path)
    for i, lbl in enumerate(LABELS):
        sk = float(f1_score(run.gold[:, i], run.pred[:, i], average="macro", zero_division=0.0))
        check(f"binary_macro_f1[{lbl}]", per_label_official[lbl], sk, run.path)
    return {
        "score_b": score_b,
        "subtype_f1": subtype_f1,
        "em": em,
        "per_label_binary_macro_f1": per_label_official,
        "per_label_positive_f1": dict(zip(LABELS, pos_f1, strict=True)),
    }


def summary_metrics(path: Path, task: str, split: str, variant: str) -> dict[str, Any]:
    """Read aggregate metrics from a tabular summary JSON (no per-sample predictions exist)."""
    data = REG.read_json(path)
    if data.get("split") != split or data.get("variant") != variant:
        raise ValueError(f"{rel(path)}: split/variant fields do not match {split}/{variant}")
    out: dict[str, Any] = {}
    for name, m in data["models"].items():
        if task == "A":
            out[name] = {"acc": float(m["accuracy"]), "score_a": float(m["macro_f1"])}
            continue
        support = {lbl: int(m["per_class"][lbl]["support"]) for lbl in LABELS}
        plb = {lbl: float(m["per_label_binary_macro_f1"][lbl]) for lbl in LABELS}
        score_b = sum(plb[lbl] * support[lbl] for lbl in LABELS) / sum(support.values())
        if abs(score_b - float(m["mami_score_b"])) > 1e-5:
            raise ValueError(f"{rel(path)}: {name} Score B inconsistent with per-label values")
        subtype = float(np.mean([float(m["per_class"][lbl]["f1"]) for lbl in LABELS]))
        if abs(subtype - float(m["macro_f1"])) > 1e-5:
            raise ValueError(f"{rel(path)}: {name} macro_f1 inconsistent with per-class F1")
        # Use the stored values (6 d.p.); the recomputation above is a consistency check only,
        # because it is built from per-label values that are themselves rounded.
        out[name] = {
            "score_b": float(m["mami_score_b"]),
            "subtype_f1": float(m["macro_f1"]),
            "em": float(m["exact_match_accuracy"]),
            "per_label_binary_macro_f1": plb,
        }
    return {"source_csv_missing": data.get("source_csv"), "models": out}


# --------------------------------------------------------------------------------------------
# Bootstrap (vectorised form of the official formulas, verified against the official code)
# --------------------------------------------------------------------------------------------
def _safe_div(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    """Element-wise ``num / den`` with 0 where ``den == 0``."""
    return np.where(den > 0, num / np.maximum(den, 1), 0.0)


def macro_f1_last_axis(gold: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Binary macro-F1 (mean of class-1 and class-0 F1) along the last axis of bool arrays."""
    tp = np.sum(gold & pred, axis=-1)
    tn = np.sum(~gold & ~pred, axis=-1)
    fp = np.sum(~gold & pred, axis=-1)
    fn = np.sum(gold & ~pred, axis=-1)
    return (_safe_div(2 * tp, 2 * tp + fp + fn) + _safe_div(2 * tn, 2 * tn + fn + fp)) / 2


def boot_scores(run: Run, idx: np.ndarray) -> np.ndarray:
    """Primary score (Score A or Score B) of ``run`` on each bootstrap resample ``idx``."""
    g = run.gold.astype(bool)[idx]
    p = run.pred.astype(bool)[idx]
    if run.task == "A":
        return macro_f1_last_axis(g, p)
    per_label = macro_f1_last_axis(np.swapaxes(g, 1, 2), np.swapaxes(p, 1, 2))
    weights = g.sum(axis=1)
    return np.asarray((per_label * weights).sum(axis=1) / weights.sum(axis=1))


def verify_boot(run: Run, idx: np.ndarray, scores: np.ndarray) -> None:
    """Assert the vectorised bootstrap equals the official scorer on the first resamples."""
    for b in range(N_VERIFY_RESAMPLES):
        rows = idx[b]
        sub_run = Run(
            run.path,
            run.split,
            run.task,
            [str(k) for k in range(len(rows))],
            run.gold[rows],
            run.pred[rows],
            True,
            run.seed,
            run.source_kind,
        )
        truth, sub = official_dicts(sub_run)
        fn = OFFICIAL.compute_scoreA if run.task == "A" else OFFICIAL.compute_scoreB
        check(f"bootstrap[{b}]", float(fn(truth, sub)), float(scores[b]), run.path)


def ci(values: np.ndarray) -> list[float]:
    """Percentile 95% interval."""
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


# --------------------------------------------------------------------------------------------
# Model registry
# --------------------------------------------------------------------------------------------
@dataclass
class ModelSpec:
    """A paper table row backed by per-sample files."""

    key: str
    name: str
    origin: str
    thr: str
    trained: bool
    patterns: dict[str, str]
    allow_subset_val: bool = False
    in_paper: tuple[bool, bool] = (True, True)


VLM_SPECS = [
    ModelSpec(
        "gemini",
        "Gemini 3.1 Flash Lite",
        "API",
        "N/A",
        False,
        {"A": "gemini_{s}.json", "B": "gemini_{s}_multiclass.json"},
        True,
    ),
    ModelSpec(
        "qwen7b_ft",
        "Qwen2-VL-7B QLoRA",
        "VLM",
        "N/A",
        True,
        {
            "A": "qwen2vl_{s}_qwen2_vl_7b_instruct_finetuned_seed*.json",
            "B": "qwen2vl_{s}_qwen2_vl_7b_instruct_multiclass_finetuned_seed*.json",
        },
    ),
    ModelSpec(
        "qwen7b_zs",
        "Qwen2-VL-7B (Zero-Shot)",
        "VLM",
        "N/A",
        False,
        {
            "A": "qwen2vl_{s}_qwen2_vl_7b_instruct.json",
            "B": "qwen2vl_{s}_qwen2_vl_7b_instruct_multiclass.json",
        },
    ),
    ModelSpec(
        "qwen2b_ft",
        "Qwen2-VL-2B QLoRA",
        "VLM",
        "N/A",
        True,
        {
            "A": "qwen2vl_{s}_qwen2_vl_2b_instruct_finetuned_seed*.json",
            "B": "qwen2vl_{s}_qwen2_vl_2b_instruct_multiclass_finetuned_seed*.json",
        },
        in_paper=(False, True),
    ),
    ModelSpec(
        "qwen2b_zs",
        "Qwen2-VL-2B (Zero-Shot)",
        "VLM",
        "N/A",
        False,
        {
            "A": "qwen2vl_{s}_qwen2_vl_2b_instruct.json",
            "B": "qwen2vl_{s}_qwen2_vl_2b_instruct_multiclass.json",
        },
    ),
    ModelSpec(
        "llava_ft",
        "LLaVA-1.5-7B QLoRA",
        "VLM",
        "N/A",
        True,
        {
            "A": "llava_{s}_llava_1_5_7b_hf_finetuned.json",
            "B": "llava_{s}_llava_1_5_7b_hf_multiclass_finetuned.json",
        },
    ),
    ModelSpec(
        "llava_zs",
        "LLaVA-1.5-7B (Zero-Shot)",
        "VLM",
        "N/A",
        False,
        {"A": "llava_{s}_llava_1_5_7b_hf.json", "B": "llava_{s}_llava_1_5_7b_hf_multiclass.json"},
    ),
    ModelSpec(
        "clipL_ft",
        "CLIP ViT-L-14 (Fine-Tuned)",
        "CLIP",
        "0.50",
        True,
        {
            "A": "clip_{s}_finetuned_singleclass_vit_l_14_quickgelu_seed*.json",
            "B": "clip_{s}_finetuned_multiclass_vit_l_14_quickgelu_seed*.json",
        },
    ),
    ModelSpec(
        "clipB_ft",
        "CLIP ViT-B-32 (Fine-Tuned)",
        "CLIP",
        "0.50",
        True,
        {
            "A": "clip_{s}_finetuned_singleclass_vit_b_32_quickgelu_seed*.json",
            "B": "clip_{s}_finetuned_multiclass_vit_b_32_quickgelu_seed*.json",
        },
    ),
    ModelSpec(
        "clipL_zs",
        "CLIP ViT-L-14 (Zero-Shot)",
        "CLIP",
        "0.50",
        False,
        {
            "A": "clip_{s}_vit_l_14_quickgelu.json",
            "B": "clip_{s}_multiclass_vit_l_14_quickgelu.json",
        },
    ),
    ModelSpec(
        "clipB_zs",
        "CLIP ViT-B-32 (Zero-Shot)",
        "CLIP",
        "0.50",
        False,
        {
            "A": "clip_{s}_vit_b_32_quickgelu.json",
            "B": "clip_{s}_multiclass_vit_b_32_quickgelu.json",
        },
    ),
    ModelSpec(
        "visualbert",
        "VisualBERT (Zero-Shot)",
        "VLM",
        "N/A",
        False,
        {"A": "visualbert_{s}.json", "B": "visualbert_{s}_multiclass.json"},
        True,
    ),
]

THRESHOLD_SOURCES = {
    "gemini": "generated yes/no text and JSON booleans; no threshold",
    "qwen": "generated text parsed to labels; no threshold",
    "llava": "generated text parsed to labels; no threshold",
    "visualbert": "argmax of yes/no MLM logits",
    "clip_ft": "Task A argmax of 2 logits; Task B sigmoid >= 0.50 (scripts/train_clip.py)",
    "clip_zs": "Task A argmax softmax; Task B softmax prob >= 0.50 (models/vlm/clip_classifier.py)",
    "ml": "estimator.predict() in auto_benchmark/src/autobenchmark/models.py: argmax of "
    "predict_proba (= 0.50) or decision_function sign for margin models; the calibrated taus "
    "in results/threshold_calibration_results.csv are never applied to predictions",
}


def ml_key(variant: str, name: str) -> str:
    """Registry key of a classical classifier row."""
    return f"ml_{variant}::{name}"


def ml_thr(name: str) -> str:
    """Decision rule label used by ``estimator.predict`` for a classical classifier."""
    return "margin 0" if name in MARGIN_MODELS else "0.50"


def aggregate(per_seed: list[dict[str, Any]]) -> tuple[dict[str, float], dict[str, float | None]]:
    """Mean and sample SD (ddof=1) of every scalar metric across seeds."""
    keys = [k for k, v in per_seed[0].items() if isinstance(v, float)]
    mean = {k: float(np.mean([m[k] for m in per_seed])) for k in keys}
    sd: dict[str, float | None] = {
        k: float(np.std([m[k] for m in per_seed], ddof=1)) if len(per_seed) > 1 else None
        for k in keys
    }
    return mean, sd


def collect() -> tuple[dict[str, Any], dict[tuple[str, str, str], list[Run]]]:
    """Load, validate and score every model row; return (models dict, runs by key/task/split)."""
    truths = {s: load_truth(s) for s in SPLITS}
    models: dict[str, Any] = {}
    runs: dict[tuple[str, str, str], list[Run]] = {}

    for spec in VLM_SPECS:
        entry: dict[str, Any] = {
            "name": spec.name,
            "origin": spec.origin,
            "thr": spec.thr,
            "trained": spec.trained,
            "zero_shot": not spec.trained,
            "variant": None,
            "in_paper_task_a": spec.in_paper[0],
            "in_paper_task_b": spec.in_paper[1],
            "tasks": {},
        }
        for task, pattern in spec.patterns.items():
            entry["tasks"][task] = {}
            for split in SPLITS:
                files = sorted(
                    (RESULTS / split).glob(pattern.format(s=split)),
                    key=lambda p: (seed_of(p) or 0, p.name),
                )
                if not files:
                    conflated = sorted((RESULTS / split).glob("*test_validation*"))
                    entry["tasks"][task][split] = {
                        "available": False,
                        "reason": f"no file matching results/{split}/{pattern.format(s=split)}"
                        + (
                            "; only conflated test+validation files exist, which are rejected"
                            if conflated
                            else ""
                        ),
                    }
                    continue
                allow = spec.allow_subset_val and split == "validation"
                loaded = [load_json_run(f, split, task, truths[split], allow) for f in files]
                runs[(spec.key, task, split)] = loaded
                per_seed = [score_run(r) for r in loaded]
                mean, sd = aggregate(per_seed)
                entry["tasks"][task][split] = {
                    "available": True,
                    "source_kind": "per_sample",
                    "official_scorer": True,
                    "n_seeds": len(loaded),
                    "seeds": [r.seed for r in loaded],
                    "files": [r.path for r in loaded],
                    "complete_split": all(r.complete for r in loaded),
                    "run_info": [r.info for r in loaded],
                    "per_seed": per_seed,
                    "mean": mean,
                    "sd": sd,
                }
        models[spec.key] = entry

    # Classical classifiers: PaddleOCR text (per-sample CSV) and provided / combined (summary).
    paddle: dict[tuple[str, str], dict[str, Run]] = {
        (task, split): load_tabular_runs(task, split, truths[split])
        for task in ("A", "B")
        for split in SPLITS
    }
    summaries: dict[tuple[str, str, str], dict[str, Any]] = {}
    for variant in ("provided", "paddleocr", "combined"):
        for split in SPLITS:
            summaries[(variant, "A", split)] = summary_metrics(
                RESULTS / split / f"tabular_{variant}_binary.json", "A", split, variant
            )
            summaries[(variant, "B", split)] = summary_metrics(
                RESULTS / split / f"tabular_{variant}_multiclass_mami.json", "B", split, variant
            )
    for name, display in ML_DISPLAY.items():
        for variant in ("provided", "paddleocr", "combined"):
            entry = {
                "name": display,
                "estimator": name,
                "origin": "ML on emb",
                "thr": ml_thr(name),
                "trained": True,
                "zero_shot": False,
                "variant": variant,
                "in_paper_task_a": variant == "provided",
                "in_paper_task_b": variant == "provided",
                "tasks": {"A": {}, "B": {}},
            }
            for task in ("A", "B"):
                for split in SPLITS:
                    summ = summaries[(variant, task, split)]
                    if variant == "paddleocr":
                        run = paddle[(task, split)][name]
                        runs[(ml_key(variant, name), task, split)] = [run]
                        metrics = score_run(run)
                        for k in ("acc", "score_a", "score_b", "subtype_f1", "em"):
                            if k in metrics and abs(metrics[k] - summ["models"][name][k]) > 1e-5:
                                raise AssertionError(f"{run.path}: {name} {k} != summary JSON")
                        cell = {
                            "available": True,
                            "source_kind": "per_sample",
                            "official_scorer": True,
                            "files": [run.path],
                            "run_info": [run.info],
                            "complete_split": True,
                        }
                    else:
                        metrics = summ["models"][name]
                        fname = "binary" if task == "A" else "multiclass_mami"
                        cell = {
                            "available": True,
                            "source_kind": "summary_only",
                            "official_scorer": False,
                            "files": [f"results/{split}/tabular_{variant}_{fname}.json"],
                            "missing_per_sample_csv": summ["source_csv_missing"],
                            "complete_split": True,
                        }
                    cell.update(
                        {
                            "n_seeds": 1,
                            "seeds": [None],
                            "per_seed": [metrics],
                            "mean": {k: v for k, v in metrics.items() if isinstance(v, float)},
                            "sd": {k: None for k, v in metrics.items() if isinstance(v, float)},
                        }
                    )
                    entry["tasks"][task][split] = cell
            models[ml_key(variant, name)] = entry
    return models, runs


# --------------------------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------------------------
def mcnemar(x: Run, y: Run) -> dict[str, Any]:
    """Exact and continuity-corrected McNemar test on Task A correctness."""
    cx = x.pred == x.gold
    cy = y.pred == y.gold
    b = int(np.sum(cx & ~cy))
    c = int(np.sum(~cx & cy))
    n = b + c
    p_exact = float(binomtest(b, n, 0.5).pvalue) if n else 1.0
    stat = (abs(b - c) - 1) ** 2 / n if n else 0.0
    return {
        "x_only_correct": b,
        "y_only_correct": c,
        "p_exact": p_exact,
        "chi2_cc": float(stat),
        "p_chi2_cc": float(chi2.sf(stat, 1)) if n else 1.0,
        "acc_x": float(np.mean(cx)),
        "acc_y": float(np.mean(cy)),
    }


def paired_test(x: Run, y: Run, idx: np.ndarray) -> dict[str, Any]:
    """Paired bootstrap on the primary score difference (x - y), plus McNemar for Task A."""
    if x.ids != y.ids or x.task != y.task:
        raise ValueError(f"cannot pair {x.path} and {y.path}")
    sx, sy = score_run(x), score_run(y)
    metric = "score_a" if x.task == "A" else "score_b"
    observed = sx[metric] - sy[metric]
    deltas = boot_scores(x, idx) - boot_scores(y, idx)
    out: dict[str, Any] = {
        "metric": metric,
        "x_file": x.path,
        "y_file": y.path,
        "n": len(x.ids),
        "x": sx[metric],
        "y": sy[metric],
        "delta": observed,
        "delta_ci95": ci(deltas),
        "p_bootstrap_two_sided": float(np.mean(np.abs(deltas - observed) >= abs(observed))),
    }
    if x.task == "A":
        out["mcnemar"] = mcnemar(x, y)
    return out


def all_negative(truth: Truth) -> dict[str, Any]:
    """Metrics of a predictor that outputs 0 for every label, via the official scorer."""
    res: dict[str, Any] = {}
    run_a = Run(
        "all-negative",
        truth.split,
        "A",
        truth.ids,
        truth.task_a,
        np.zeros_like(truth.task_a),
        True,
        None,
        "synthetic",
    )
    run_b = Run(
        "all-negative",
        truth.split,
        "B",
        truth.ids,
        truth.task_b,
        np.zeros_like(truth.task_b),
        True,
        None,
        "synthetic",
    )
    res["task_a"] = score_run(run_a)
    res["task_b"] = score_run(run_b)
    res["frac_no_subtype_gold"] = float(np.mean(truth.task_b.sum(axis=1) == 0))
    return res


def compute_stats(
    models: dict[str, Any], runs: dict[tuple[str, str, str], list[Run]]
) -> dict[str, Any]:
    """Bootstrap CIs, paired tests, all-negative baseline, thresholds and drops."""
    truth_test = load_truth("test")
    rng = np.random.default_rng(BOOT_SEED)
    idx = rng.integers(0, len(truth_test.ids), size=(N_BOOT, len(truth_test.ids)))
    stats: dict[str, Any] = {
        "bootstrap": {
            "n_resamples": N_BOOT,
            "rng": "numpy.default_rng",
            "seed": BOOT_SEED,
            "unit": "meme",
            "interval": "percentile 2.5-97.5",
            "verified_against_official_resamples": N_VERIFY_RESAMPLES,
        },
        "ci95_test": {},
    }
    for (key, task, split), rs in sorted(runs.items()):
        if split != "test":
            continue
        first = rs[0]
        scores = boot_scores(first, idx)
        verify_boot(first, idx, scores)
        metric = "score_a" if task == "A" else "score_b"
        per_seed = [m[metric] for m in models[key]["tasks"][task]["test"]["per_seed"]]
        stats["ci95_test"].setdefault(key, {})[task] = {
            "metric": metric,
            "bootstrapped_file": first.path,
            "point": per_seed[0],
            "ci95": ci(scores),
            "n_seeds": len(rs),
            "seed_range": [min(per_seed), max(per_seed)],
        }
    for key, entry in models.items():
        if entry["variant"] in ("provided", "combined"):
            stats["ci95_test"].setdefault(key, {})["note"] = (
                "not computable: per-sample predictions for this text variant are not in the "
                "repository (only aggregate summary JSONs)"
            )

    svm = "Support Vector Machine - RBF SVC"
    pairs = [
        ("gemini_vs_qwen7b_qlora", "gemini", "qwen7b_ft", "A"),
        ("gemini_vs_qwen7b_qlora", "gemini", "qwen7b_ft", "B"),
        ("qwen7b_qlora_vs_svm_rbf_paddleocr", "qwen7b_ft", ml_key("paddleocr", svm), "B"),
        (
            "svm_rbf_vs_xgboost_paddleocr",
            ml_key("paddleocr", svm),
            ml_key("paddleocr", "XGBoost"),
            "A",
        ),
        ("qwen7b_qlora_vs_qwen7b_zero_shot", "qwen7b_ft", "qwen7b_zs", "A"),
        ("qwen7b_qlora_vs_qwen7b_zero_shot", "qwen7b_ft", "qwen7b_zs", "B"),
    ]
    tests: dict[str, Any] = {}
    for label, kx, ky, task in pairs:
        rx, ry = runs[(kx, task, "test")], runs[(ky, task, "test")]
        res = paired_test(rx[0], ry[0], idx)
        res["x_model"], res["y_model"] = models[kx]["name"], models[ky]["name"]
        res["x_variant"], res["y_variant"] = models[kx]["variant"], models[ky]["variant"]
        if len(rx) > 1 or len(ry) > 1:
            res["all_seed_pairs"] = [
                {k: v for k, v in paired_test(a, b, idx).items() if k != "mcnemar"}
                | ({"mcnemar_p_exact": mcnemar(a, b)["p_exact"]} if task == "A" else {})
                for a in rx
                for b in ry
            ]
        flags = [r.path for r in (rx[0], ry[0]) if r.info.get("degenerate_constant_prediction")]
        if flags:
            res["warning"] = f"constant-prediction (likely failed) run(s): {flags}"
        if "paddleocr" in label:
            res["note"] = (
                "SVM-RBF / XGBoost use PaddleOCR-text embeddings: the paper's provided-text "
                "rows have no per-sample predictions in the repository"
            )
        tests[f"{label}__task_{task}"] = res
    tests["svm_rbf_provided_vs_paddleocr__task_B"] = {
        "computed": False,
        "reason": "per-sample predictions for the provided-text run "
        "(auto_benchmark/results/model_results/mami_tabular_model_multiclass_provided_test/) "
        "are not in the repository; only aggregate metrics exist, so no paired test is possible",
    }
    stats["paired_tests_test"] = tests
    stats["all_negative"] = {s: all_negative(load_truth(s)) for s in SPLITS}
    stats["thresholds"] = threshold_audit(models)
    stats["validation_to_test_drop"] = drops(models)
    return stats


def threshold_audit(models: dict[str, Any]) -> dict[str, Any]:
    """Decide whether the tables used 0.50 / default predictions or the calibrated taus."""
    path = RESULTS / "threshold_calibration_results.csv"
    REG.register(path)
    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    norm = {re.sub(r"[^a-z0-9]", "", n.lower()): n for n in ML_DISPLAY}
    per_model = []
    default_hits = calibrated_hits = 0
    for row in rows:
        name = norm.get(re.sub(r"[^a-z0-9]", "", row["model"].lower()))
        if name is None:
            continue
        m = models[ml_key("provided", name)]["tasks"]["A"]["test"]["mean"]
        acc, sa = round(m["acc"], 4), round(m["score_a"], 4)
        is_default = acc == float(row["test_default_acc"]) and sa == float(
            row["test_default_macro_f1"]
        )
        is_cal = acc == float(row["test_calibrated_acc"]) and sa == float(
            row["test_calibrated_macro_f1"]
        )
        default_hits += int(is_default)
        calibrated_hits += int(is_cal and not is_default)
        per_model.append(
            {
                "model": name,
                "calibrated_tau": float(row["calibrated_tau"]),
                "matches_default": is_default,
                "matches_calibrated": is_cal,
            }
        )
    neither = [
        ML_DISPLAY[r["model"]]
        for r in per_model
        if not r["matches_default"] and not r["matches_calibrated"]
    ]
    return {
        "rules": THRESHOLD_SOURCES,
        "per_row_thr": {k: v["thr"] for k, v in models.items()},
        "calibration_csv": rel(path),
        "calibration_rows_matched": len(per_model),
        "provided_test_matches_default_tau": default_hits,
        "provided_test_matches_calibrated_only": calibrated_hits,
        "per_model": per_model,
        "conclusion": (
            f"Calibrated taus were never applied: {default_hits}/{len(per_model)} ML rows equal "
            "the tau=0.50 column of the calibration CSV, none equal only the calibrated column; "
            f"the rest ({', '.join(neither)}) match neither (SVC predict() uses the margin, "
            "not predict_proba >= 0.50; the tree ensembles differ by fit)."
            if calibrated_hits == 0
            else "Some rows match calibrated taus: see per_model."
        ),
    }


def drops(models: dict[str, Any]) -> dict[str, Any]:
    """Validation minus test Score A (points) per model, with range and mean per group."""
    rows = []
    for key, entry in models.items():
        cells = entry["tasks"].get("A", {})
        val, test = cells.get("validation", {}), cells.get("test", {})
        if not (val.get("available") and test.get("available")):
            continue
        d = 100 * (val["mean"]["score_a"] - test["mean"]["score_a"])
        rows.append(
            {
                "key": key,
                "name": entry["name"],
                "variant": entry["variant"],
                "trained": entry["trained"],
                "val_score_a": val["mean"]["score_a"],
                "test_score_a": test["mean"]["score_a"],
                "drop_points": d,
                "val_complete": val["complete_split"],
            }
        )

    def summarise(sel: list[dict[str, Any]]) -> dict[str, Any]:
        v = [r["drop_points"] for r in sel]
        return {
            "n": len(v),
            "min": min(v),
            "max": max(v),
            "mean": float(np.mean(v)),
            "min_model": min(sel, key=lambda r: r["drop_points"])["name"],
            "max_model": max(sel, key=lambda r: r["drop_points"])["name"],
            "val_score_a_range": [
                min(r["val_score_a"] for r in sel),
                max(r["val_score_a"] for r in sel),
            ],
            "test_score_a_range": [
                min(r["test_score_a"] for r in sel),
                max(r["test_score_a"] for r in sel),
            ],
        }

    trained_paper = [r for r in rows if r["trained"] and r["variant"] in (None, "provided")]
    return {
        "unit": "Score A points (x100), validation minus test; mean over seeds",
        "per_model": rows,
        "trained_paper_rows": summarise(trained_paper),
        "ml_provided": summarise([r for r in rows if r["variant"] == "provided"]),
        "ml_paddleocr": summarise([r for r in rows if r["variant"] == "paddleocr"]),
        "finetuned_vlm_clip": summarise([r for r in rows if r["trained"] and r["variant"] is None]),
        "zero_shot": [
            {
                "name": r["name"],
                "drop_points": r["drop_points"],
                "drops": r["drop_points"] > 0,
                "val_complete": r["val_complete"],
            }
            for r in rows
            if not r["trained"]
        ],
    }


# --------------------------------------------------------------------------------------------
# LaTeX tables
# --------------------------------------------------------------------------------------------
def fmt(cell: dict[str, Any], metric: str, pct: bool) -> str:
    """Format mean(+-SD) of ``metric`` in a table cell."""
    if not cell.get("available"):
        return "N/A"
    mean, sd = cell["mean"][metric], cell["sd"][metric]
    scale, dec, suf = (100, 2, r"\%") if pct else (1, 4, "")
    text = f"{mean * scale:.{dec}f}"
    if sd is not None:
        text += f"$\\pm${sd * scale:.{dec}f}"
    text += suf
    if not cell.get("complete_split", True):
        text += r"$^{\dagger}$"
    return text


def write_tables(models: dict[str, Any]) -> None:
    """Write table_task_a.tex and table_task_b.tex in the paper's column order."""
    specs = {
        "A": ("table_task_a.tex", "score_a", [("acc", True), ("score_a", False)]),
        "B": (
            "table_task_b.tex",
            "score_b",
            [("score_b", False), ("subtype_f1", False), ("em", True)],
        ),
    }
    header = [
        "% Auto-generated by scripts/revision/revision_metrics.py (do not edit by hand).",
        "% Columns: Model & Origin & Thr & test metrics & validation metrics.",
        "% Values are mean$\\pm$SD over seeds where (n=k) is shown; otherwise a single run.",
        "% $^{\\dagger}$ validation subset (Gemini: 400 memes, VisualBERT: 100 memes).",
        "% $^{s}$ aggregate summary JSON only (per-sample predictions not in the repository).",
        "% 'margin 0' = estimator.predict() uses the decision_function sign.",
    ]
    for task, (fname, sort_metric, cols) in specs.items():
        blocks: dict[str, list[tuple[float, str]]] = {"main": [], "paddle": []}
        for entry in models.values():
            if entry["variant"] == "combined":
                continue
            cells = entry["tasks"].get(task)
            if not cells or not cells["test"].get("available"):
                continue
            n = cells["test"]["n_seeds"]
            name = entry["name"] + (f" ($n{{=}}{n}$)" if n > 1 else "")
            if cells["test"].get("source_kind") == "summary_only":
                name += r"$^{s}$"
            parts = [name, entry["origin"], entry["thr"]]
            for split in ("test", "validation"):
                parts += [fmt(cells[split], m, pct) for m, pct in cols]
            line = " & ".join(parts) + r" \\"
            if task == "A" and not entry["in_paper_task_a"] and entry["variant"] is None:
                line += "  % not in the paper's Task A tables"
            block = "paddle" if entry["variant"] == "paddleocr" else "main"
            blocks[block].append((cells["test"]["mean"][sort_metric], line))
        lines = list(header)
        for _, line in sorted(blocks["main"], key=lambda t: -t[0]):
            lines.append(line)
        lines.append(r"\midrule")
        lines.append("% ML on emb, PaddleOCR text (per-sample predictions, official scorer)")
        for _, line in sorted(blocks["paddle"], key=lambda t: -t[0]):
            lines.append(line)
        (OUT_DIR / fname).write_text("\n".join(lines) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------------------------
# Paper comparison
# --------------------------------------------------------------------------------------------
VLM_NAMES = {
    "Gemini 3.1 Flash Lite": "gemini",
    "gemini-3.1-flash-lite": "gemini",
    "Qwen2-VL-7B (Zero-Shot)": "qwen7b_zs",
    "Qwen2-VL-7B QLoRA": "qwen7b_ft",
    "Qwen2-VL-2B (Zero-Shot)": "qwen2b_zs",
    "Qwen2-VL-2B QLoRA": "qwen2b_ft",
    "LLaVA-1.5-7B QLoRA": "llava_ft",
    "LLaVA-1.5-7B (Zero-Shot)": "llava_zs",
    "CLIP ViT-L-14 (Zero-Shot)": "clipL_zs",
    "CLIP ViT-B-32 (Zero-Shot)": "clipB_zs",
    "CLIP ViT-L-14 (Fine-Tuned)": "clipL_ft",
    "CLIP ViT-B-32 (Fine-Tuned)": "clipB_ft",
    "VisualBERT (Zero-Shot)": "visualbert",
    "Qwen2-VL-7B|Zero-Shot VLM": "qwen7b_zs",
    "Qwen2-VL-2B|Zero-Shot VLM": "qwen2b_zs",
    "CLIP ViT-B-32|Fine-Tuned CLIP": "clipB_ft",
    "CLIP ViT-L-14|Fine-Tuned CLIP": "clipL_ft",
}
TABLES = {
    "tab:task_a_results": ("A", ["test_acc", "test_score_a", "val_acc", "val_score_a"]),
    "tab:appendix_task_a": ("A", ["test_acc", "test_score_a", "val_acc", "val_score_a"]),
    "tab:task_b_results": (
        "B",
        ["test_score_b", "test_subtype_f1", "test_em", "val_score_b", "val_subtype_f1", "val_em"],
    ),
    "tab:appendix_task_b": (
        "B",
        ["test_score_b", "test_subtype_f1", "test_em", "val_score_b", "val_subtype_f1", "val_em"],
    ),
    "tab:appendix_task_b_part2": (
        "B",
        ["test_score_b", "test_subtype_f1", "test_em", "val_score_b", "val_subtype_f1", "val_em"],
    ),
}
PCT_METRICS = {"acc", "em"}


def clean_name(raw: str) -> str:
    """Strip citations and \\texttt from a table name cell."""
    raw = re.sub(r"~?\\cite\{[^}]*\}", "", raw)
    raw = re.sub(r"\\texttt\{([^}]*)\}", r"\1", raw)
    return raw.strip()


def table_rows(lines: list[str], label: str) -> list[tuple[int, list[str]]]:
    """Return (1-based line number, cells) for every data row of the table with ``label``."""
    start = next(i for i, ln in enumerate(lines) if f"\\label{{{label}}}" in ln)
    out: list[tuple[int, list[str]]] = []
    in_body = False
    for i in range(start, len(lines)):
        ln = lines[i].strip()
        if ln.startswith(r"\end{table}"):
            break
        if ln.startswith(r"\midrule"):
            in_body = True
            continue
        if ln.startswith(r"\bottomrule"):
            in_body = False
            continue
        if in_body and "&" in ln and ln.endswith("\\\\"):
            out.append((i + 1, [c.strip() for c in ln[:-2].split("&")]))
    return out


def resolve(cells: list[str]) -> str | None:
    """Map a paper row to a registry key (None for literature rows)."""
    name, second = clean_name(cells[0]), cells[1].strip()
    if second == "Literature":
        return None
    if second == "ML on emb":
        est = ML_ALIASES.get(name) or next((k for k, v in ML_DISPLAY.items() if v == name), None)
        if est is None:
            raise KeyError(f"unknown ML row {name!r}")
        return ml_key("provided", est)
    key = VLM_NAMES.get(f"{name}|{second}") or VLM_NAMES.get(name)
    if key is None:
        raise KeyError(f"unknown paper row {name!r} / {second!r}")
    return key


def value_of(models: dict[str, Any], key: str, task: str, col: str) -> tuple[str, float | None]:
    """Paper-formatted recomputed value (mean(+-SD)) and its rounded mean."""
    split_tag, metric = col.split("_", 1)
    split = "test" if split_tag == "test" else "validation"
    cell = models[key]["tasks"][task][split]
    if not cell.get("available"):
        return "N/A", None
    pct = metric in PCT_METRICS
    mean = cell["mean"][metric]
    return fmt(cell, metric, pct).replace(r"$\pm$", "±").replace(r"\%", "%").replace(
        r"$^{\dagger}$", " (subset)"
    ), round(mean * (100 if pct else 1), 2 if pct else 4)


def grep_lines(lines: list[str], token: str) -> list[int]:
    """1-based line numbers of ``main.tex`` containing the numeric ``token``."""
    pat = re.compile(r"(?<![\d.])" + re.escape(token) + r"(?![\d])")
    return [i + 1 for i, ln in enumerate(lines) if pat.search(ln)]


def paper_diffs(models: dict[str, Any], stats: dict[str, Any]) -> dict[str, Any]:
    """Every table/prose value in main.tex that differs from the recomputed value."""
    lines = REG.read_text(MAIN_TEX).splitlines()
    changes: list[dict[str, Any]] = []
    for label, (task, cols) in TABLES.items():
        for line_no, cells in table_rows(lines, label):
            key = resolve(cells)
            if key is None:
                continue
            for col, raw in zip(cols, cells[-len(cols) :], strict=True):
                old = raw.replace(r"\%", "").strip()
                new_text, new_val = value_of(models, key, task, col)
                if old == "N/A" and new_val is None:
                    continue
                if old != "N/A" and new_val is not None and float(old) == new_val:
                    continue
                changes.append(
                    {
                        "table": label,
                        "line": line_no,
                        "row": clean_name(cells[0]),
                        "key": key,
                        "column": col,
                        "old": old,
                        "new": new_text,
                        "main_tex_lines_with_old": grep_lines(lines, old) if old != "N/A" else [],
                    }
                )
    # Table A3 (text-modality ablation, test split).
    for line_no, cells in table_rows(lines, "tab:appendix_ocr_ablation"):
        est = ML_ALIASES.get(clean_name(cells[0])) or next(
            k for k, v in ML_DISPLAY.items() if v == clean_name(cells[0])
        )
        for (variant, metric), raw in zip(
            [(v, m) for v in ("provided", "paddleocr", "combined") for m in ("score_b", "em")],
            cells[1:],
            strict=True,
        ):
            old = raw.replace(r"\%", "").strip()
            new_text, new_val = value_of(models, ml_key(variant, est), "B", f"test_{metric}")
            if new_val is not None and float(old) == new_val:
                continue
            changes.append(
                {
                    "table": "tab:appendix_ocr_ablation",
                    "line": line_no,
                    "row": clean_name(cells[0]),
                    "key": ml_key(variant, est),
                    "column": f"test_{metric}",
                    "old": old,
                    "new": new_text,
                    "main_tex_lines_with_old": grep_lines(lines, old),
                }
            )
    d = stats["validation_to_test_drop"]
    ml = d["ml_provided"]
    tr = d["trained_paper_rows"]
    val_lo, val_hi = (100 * x for x in tr["val_score_a_range"])
    test_lo, test_hi = (100 * x for x in tr["test_score_a_range"])
    zs_small = [z for z in d["zero_shot"] if "Gemini" in z["name"] or "Qwen" in z["name"]]
    zs_other = [z for z in d["zero_shot"] if z not in zs_small]
    prose = [
        {
            "claim": "ML drop range 4.7%-17.7% (Sec. 5.2, Fig. 1 caption)",
            "old": "4.7% to 17.7%",
            "new": f"{ml['min']:.1f} to {ml['max']:.1f} Score A points "
            f"({ml['min_model']} to {ml['max_model']})",
            "changed": (round(ml["min"], 1), round(ml["max"], 1)) != (4.7, 17.7),
            "main_tex_lines_with_old": sorted(
                set(grep_lines(lines, "4.7") + grep_lines(lines, "17.7"))
            ),
        },
        {
            "claim": "ML mean drop 11.2%",
            "old": "11.2%",
            "new": f"{ml['mean']:.1f} points",
            "changed": round(ml["mean"], 1) != 11.2,
            "main_tex_lines_with_old": grep_lines(lines, "11.2"),
        },
        {
            "claim": "trained models validation 84%-89% F1",
            "old": "84%-89%",
            "new": f"{val_lo:.1f}-{val_hi:.1f} Score A ({tr['n']} trained rows)",
            "changed": (round(val_lo), round(val_hi)) != (84, 89),
            "main_tex_lines_with_old": grep_lines(lines, r"84\%--89"),
        },
        {
            "claim": "trained models test 68%-76% F1",
            "old": "68%-76%",
            "new": f"{test_lo:.1f}-{test_hi:.1f} Score A ({tr['n']} trained rows)",
            "changed": (round(test_lo), round(test_hi)) != (68, 76),
            "main_tex_lines_with_old": grep_lines(lines, r"68\%--76"),
        },
        {
            "claim": "zero-shot models (Gemini, zero-shot Qwen) suffer no drop",
            "old": "no drop",
            "new": "small: "
            + ", ".join(f"{z['name']} {z['drop_points']:+.1f}" for z in zs_small)
            + "; but "
            + ", ".join(f"{z['name']} {z['drop_points']:+.1f}" for z in zs_other),
            "changed": True,
            "main_tex_lines_with_old": [
                i + 1 for i, ln in enumerate(lines) if "no such drop" in ln
            ],
        },
    ]
    qwen = models["qwen7b_ft"]["tasks"]["B"]["test"]
    above = sum(m["score_b"] > SRCB_SCORE_B for m in qwen["per_seed"])
    prose.append(
        {
            "claim": "Qwen2-VL-7B QLoRA and SVM-RBF beat SRCB (0.7310) on Task B",
            "old": "0.7428 and 0.7321 vs. 0.7310",
            "new": f"Qwen2-VL-7B QLoRA {qwen['mean']['score_b']:.4f}±{qwen['sd']['score_b']:.4f} "
            f"(n={qwen['n_seeds']}, {above}/{qwen['n_seeds']} seeds above 0.7310); "
            "SVM-RBF 0.7321 unchanged",
            "changed": True,
            "main_tex_lines_with_old": grep_lines(lines, "0.7310"),
        }
    )
    return {"table_changes": changes, "prose_changes": prose}


# --------------------------------------------------------------------------------------------
# README
# --------------------------------------------------------------------------------------------
def write_readme(stats: dict[str, Any], diffs: dict[str, Any]) -> None:
    """Write the README with run instructions and the paper -> correct value list."""
    neg = stats["all_negative"]
    thr = stats["thresholds"]
    lines = [
        "# EXP-01 / EXP-07 revision metrics",
        "",
        "Run from the repository root: `uv run python -m scripts.revision.revision_metrics`.",
        "Scores call `compute_scoreA` / `compute_scoreB` / `compute_f1` from",
        "`scripts/official_mami_evaluation.py`, cross-checked with sklearn and",
        "`models/vlm/metrics_multilabel.py` (tolerance 1e-6).",
        "",
        "- `metrics.json`: every row, split, task and seed, with file lists and input sha256.",
        "- `table_task_a.tex`, `table_task_b.tex`: LaTeX rows (mean±SD over seeds where n>1).",
        "- `stats.json`: bootstrap CIs, paired tests, all-negative baseline, thresholds, drops.",
        "",
        "Caveats: provided-text ML rows come from summary JSONs only (their per-sample CSVs are",
        "absent), so they have no CI or paired test; PaddleOCR-text rows are scored per sample.",
        "Gemini Task B test predicts `shaming` only for all 1000 memes (failed run, re-run needed).",
        "LLaVA zero-shot validation exists only in conflated test+validation files (rejected).",
        *textwrap.wrap(
            f"All-negative Exact Match: test {100 * neg['test']['task_b']['em']:.1f}%, "
            f"validation {100 * neg['validation']['task_b']['em']:.1f}%. "
            f"Thresholds: {thr['conclusion']}",
            width=100,
        ),
        "",
        "## Paper value -> correct value (`submission/main.tex`)",
        "",
        "| Task | Row | Changes (old -> new) | main.tex lines |",
        "| --- | --- | --- | --- |",
    ]
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for ch in diffs["table_changes"]:
        task = "A3" if ch["table"] == "tab:appendix_ocr_ablation" else TABLES[ch["table"]][0]
        g = grouped.setdefault((task, ch["key"]), {"row": ch["row"], "cols": {}, "lines": set()})
        col = ch["column"] if task != "A3" else f"{ch['key'].split('::')[0][3:]} {ch['column']}"
        g["cols"][col] = f"{ch['old']} -> {ch['new']}"
        g["lines"].update(ch["main_tex_lines_with_old"])
    for (task, _), g in grouped.items():
        cols = "; ".join(f"{c} {v}" for c, v in g["cols"].items())
        lines.append(
            f"| {task} | {g['row']} | {cols} | {', '.join(map(str, sorted(g['lines'])))} |"
        )
    for p in diffs["prose_changes"]:
        if not p["changed"]:
            continue
        lines.append(
            f"| prose | {p['claim']} | {p['old']} -> {p['new']} | "
            f"{', '.join(map(str, p['main_tex_lines_with_old']))} |"
        )
    (OUT_DIR / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if len(lines) > 60:
        print(f"WARNING: README has {len(lines)} lines (> 60)")


def main() -> None:
    """Run the full EXP-01 / EXP-07 pipeline and write all outputs."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    models, runs = collect()
    stats = compute_stats(models, runs)
    diffs = paper_diffs(models, stats)
    write_tables(models)
    metrics = {
        "generator": "scripts/revision/revision_metrics.py",
        "scorer": "scripts/official_mami_evaluation.py (compute_scoreA, compute_scoreB, compute_f1)",
        "cross_checks": "sklearn.metrics and models/vlm/metrics_multilabel.py, tolerance 1e-6",
        "ground_truth": "results/embeddings/<split>_vit_l_14_quickgelu_ocr_paddleocr.npz; each "
        "file's ground_truth is asserted equal to it",
        "models": models,
        "paper_diffs": diffs,
        "inputs_sha256": dict(sorted(REG.files.items())),
    }
    (OUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    stats["inputs_sha256"] = dict(sorted(REG.files.items()))
    (OUT_DIR / "stats.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    write_readme(stats, diffs)
    for key in (
        "gemini",
        "qwen7b_ft",
        "qwen7b_zs",
        "qwen2b_ft",
        ml_key("provided", "XGBoost"),
        ml_key("provided", "Support Vector Machine - RBF SVC"),
    ):
        e = models[key]
        for task in ("A", "B"):
            c = e["tasks"].get(task, {}).get("test", {})
            if c.get("available"):
                print(
                    key,
                    task,
                    c["n_seeds"],
                    {k: round(v, 4) for k, v in c["mean"].items()},
                    {k: (round(v, 4) if v is not None else None) for k, v in c["sd"].items()},
                )
    print(f"Wrote {len(list(OUT_DIR.iterdir()))} files to {rel(OUT_DIR)}")


if __name__ == "__main__":
    main()
