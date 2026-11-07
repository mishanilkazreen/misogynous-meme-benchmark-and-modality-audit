"""EXP-04 / EXP-06: modality ablation, fusion variants and an alternative frozen backbone.

Trains the main benchmark's "ML on emb" classifiers on frozen embeddings with the SAME
preprocessing and settings as ``auto_benchmark`` (``autobenchmark.models.get_models_dict`` defaults, no
hyper-parameter search, ``class_weight='balanced'`` for SVM-RBF and Logistic Regression,
``StandardScaler`` pipeline for every non-tree model, L2-normalised embeddings, Task B as a
``MultiOutputClassifier`` over the four sub-types, decisions from ``predict`` at the default
threshold). Every model is fitted on the train split only; the validation split is used only to
select the best classifier per condition; the test split is scored once with the official
SemEval-2022 Task 5 scorer (``scripts/official_mami_evaluation.py``).

Conditions (feature files from ``scripts/revision/extract_features.py``):

* CLIP ViT-L-14 (provided text): image-only, text-only, concat (paper baseline), element-wise
  product, concat+product, late fusion (mean of image-only and text-only probabilities).
* CLIP ViT-L-14 on text-masked images: masked-image only, masked-image + text (concat).
* SigLIP SO400M/14-384: image-only, text-only, concat.

Permutation modality ablation: for every concat model the image block (then the text block) is
shuffled across memes at evaluation time, 10 repeats, on validation and test.

Usage:
    uv run python -m scripts.revision.modality_ablation --n-jobs 8
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import platform
import sys
import time
from typing import Any

import numpy as np
import sklearn
from sklearn.base import BaseEstimator, clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import xgboost
from xgboost import XGBClassifier

from scripts.official_mami_evaluation import compute_scoreA, compute_scoreB

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("modality_ablation")

ROOT = Path(__file__).resolve().parents[2]
EMB_DIR = ROOT / "results" / "embeddings"
OUT_DIR = ROOT / "results" / "revision" / "exp04_exp06"
PAPER_A = ROOT / "results" / "test" / "tabular_provided_binary.json"
PAPER_B = ROOT / "results" / "test" / "tabular_provided_multiclass_mami.json"

SPLITS: tuple[str, ...] = ("train", "validation", "test")
SUBTYPES: tuple[str, ...] = ("shaming", "stereotype", "objectification", "violence")
FEATURE_FILES: dict[str, str] = {
    "clip": "{split}_vit_l_14_quickgelu.npz",
    "clip_masked": "{split}_vit_l_14_quickgelu_textmasked_easyocr.npz",
    "siglip": "{split}_vit_so400m_14_siglip_384.npz",
}
CLASSIFIERS: tuple[str, ...] = ("SVM-RBF", "LR", "XGBoost", "MLP", "LDA")
PAPER_NAMES: dict[str, str] = {
    "SVM-RBF": "Support Vector Machine - RBF SVC",
    "LR": "Logistic Regression",
    "XGBoost": "XGBoost",
    "MLP": "Neural Network - MLPClassifier",
    "LDA": "Linear Discriminant Analysis",
}
TASKS: tuple[str, ...] = ("A", "B")
N_PERM = 10
N_BOOT = 1000
SEED = 20260708


@dataclass(frozen=True)
class Condition:
    """A feature construction: source file, blocks and how they are combined."""

    name: str
    source: str
    fusion: str  # image | text | concat | product | concat_product | late
    description: str


CONDITIONS: tuple[Condition, ...] = (
    Condition("clip_image", "clip", "image", "CLIP image block only"),
    Condition("clip_text", "clip", "text", "CLIP text block only (provided text)"),
    Condition("clip_concat", "clip", "concat", "CLIP [image; text] (paper baseline)"),
    Condition("clip_product", "clip", "product", "CLIP image * text (element-wise)"),
    Condition("clip_concat_product", "clip", "concat_product", "CLIP [image; text; image*text]"),
    Condition("clip_late", "clip", "late", "mean of image-only and text-only probabilities"),
    Condition("masked_image", "clip_masked", "image", "CLIP image of text-masked meme only"),
    Condition("masked_concat", "clip_masked", "concat", "CLIP [masked image; text]"),
    Condition("siglip_image", "siglip", "image", "SigLIP image block only"),
    Condition("siglip_text", "siglip", "text", "SigLIP text block only (provided text)"),
    Condition("siglip_concat", "siglip", "concat", "SigLIP [image; text]"),
)


# --------------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------------


@dataclass
class SplitData:
    """Embeddings and labels of one split from one feature file."""

    image: np.ndarray
    text: np.ndarray
    y_a: np.ndarray
    y_b: np.ndarray
    ids: np.ndarray


def sha256_file(path: Path) -> str:
    """Return the hex SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_sources() -> tuple[dict[str, dict[str, SplitData]], dict[str, dict[str, Any]]]:
    """Load every feature file; verify ids and labels agree across sources."""
    data: dict[str, dict[str, SplitData]] = {}
    meta: dict[str, dict[str, Any]] = {}
    for source, pattern in FEATURE_FILES.items():
        data[source] = {}
        meta[source] = {}
        for split in SPLITS:
            path = EMB_DIR / pattern.format(split=split)
            if not path.exists():
                raise FileNotFoundError(f"{path} missing; run scripts/revision/extract_features.py")
            npz = np.load(path)
            data[source][split] = SplitData(
                image=npz["image_embeddings"].astype(np.float32),
                text=npz["text_embeddings"].astype(np.float32),
                y_a=npz["labels"].astype(int),
                y_b=npz["subtask_labels"].astype(int),
                ids=npz["image_ids"],
            )
            meta[source][split] = {
                "path": str(path.relative_to(ROOT)).replace("\\", "/"),
                "sha256": sha256_file(path),
                "model": str(npz["model"]) if "model" in npz.files else None,
                "n": len(npz["labels"]),
                "image_dim": int(npz["image_embeddings"].shape[1]),
                "text_dim": int(npz["text_embeddings"].shape[1]),
            }
    for split in SPLITS:
        ref = data["clip"][split]
        for source in FEATURE_FILES:
            other = data[source][split]
            if not (np.array_equal(ref.ids, other.ids) and np.array_equal(ref.y_b, other.y_b)):
                raise ValueError(f"{source}/{split}: ids or labels differ from clip/{split}")
    return data, meta


def build_features(d: SplitData, fusion: str) -> tuple[np.ndarray, dict[str, slice]]:
    """Return the feature matrix for a fusion mode and the column slice of each block."""
    di, dt = d.image.shape[1], d.text.shape[1]
    if fusion == "image":
        return d.image, {"image": slice(0, di)}
    if fusion == "text":
        return d.text, {"text": slice(0, dt)}
    if fusion == "concat":
        return np.hstack([d.image, d.text]), {"image": slice(0, di), "text": slice(di, di + dt)}
    if fusion == "product":
        return d.image * d.text, {"product": slice(0, di)}
    if fusion == "concat_product":
        blocks = {
            "image": slice(0, di),
            "text": slice(di, di + dt),
            "product": slice(di + dt, 2 * di + dt),
        }
        return np.hstack([d.image, d.text, d.image * d.text]), blocks
    raise ValueError(f"fusion {fusion!r} has no direct feature matrix")


# --------------------------------------------------------------------------------------------
# Classifiers (settings copied from auto_benchmark/src/autobenchmark/models.py)
# --------------------------------------------------------------------------------------------


def make_classifier(name: str, probability: bool, xgb_threads: int) -> BaseEstimator:
    """Return an unfitted classifier with the main benchmark's ``get_models_dict`` settings.

    ``probability`` only toggles SVC's Platt scaling; ``SVC.predict`` uses the decision function
    and is unaffected, so it is enabled only where probabilities are consumed (late fusion).
    XGBoost runs on CPU (``tree_method='hist'``, the xgboost>=2 default).
    """
    if name == "SVM-RBF":
        est: BaseEstimator = SVC(class_weight="balanced", probability=probability, random_state=42)
    elif name == "LR":
        est = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
    elif name == "MLP":
        est = MLPClassifier(max_iter=500, random_state=42)
    elif name == "LDA":
        est = LinearDiscriminantAnalysis()
    elif name == "XGBoost":
        return XGBClassifier(
            verbosity=0, random_state=42, tree_method="hist", device="cpu", n_jobs=xgb_threads
        )
    else:
        raise ValueError(name)
    return Pipeline([("scaler", StandardScaler()), ("estimator", est)])


def fit_model(
    clf: str, task: str, x: np.ndarray, y_a: np.ndarray, y_b: np.ndarray, prob: bool, threads: int
) -> BaseEstimator:
    """Fit a Task A (binary) or Task B (``MultiOutputClassifier``) model on training data."""
    base = make_classifier(clf, prob, threads)
    if task == "A":
        return clone(base).fit(x, y_a)
    return MultiOutputClassifier(clone(base)).fit(x, y_b)


def predict_proba(model: BaseEstimator, task: str, x: np.ndarray) -> np.ndarray:
    """Positive-class probabilities: shape (n,) for Task A, (n, 4) for Task B."""
    if task == "A":
        out: np.ndarray = model.predict_proba(x)[:, 1]
        return out
    return np.column_stack([p[:, 1] for p in model.predict_proba(x)])


# --------------------------------------------------------------------------------------------
# Metrics (official scorer; vectorised equivalent for bootstrap/permutation, cross-checked)
# --------------------------------------------------------------------------------------------


def _official_truth(y_a: np.ndarray, y_b: np.ndarray) -> dict[str, list[bool]]:
    return {str(i): [bool(y_a[i])] + [bool(v) for v in y_b[i]] for i in range(len(y_a))}


def official_scores(
    task: str, y_a: np.ndarray, y_b: np.ndarray, pred: np.ndarray
) -> dict[str, Any]:
    """Score predictions with ``scripts/official_mami_evaluation.py`` plus diagnostics."""
    truth = _official_truth(y_a, y_b)
    if task == "A":
        sub = {str(i): [bool(pred[i]), False, False, False, False] for i in range(len(pred))}
        return {
            "acc": float(np.mean(pred == y_a)),
            "score_a": float(compute_scoreA(truth, sub)),
        }
    sub = {str(i): [bool(y_a[i])] + [bool(v) for v in pred[i]] for i in range(len(pred))}
    pos_f1 = []
    for j in range(len(SUBTYPES)):
        tp = int(np.sum((pred[:, j] == 1) & (y_b[:, j] == 1)))
        fp = int(np.sum((pred[:, j] == 1) & (y_b[:, j] == 0)))
        fn = int(np.sum((pred[:, j] == 0) & (y_b[:, j] == 1)))
        pos_f1.append(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0)
    return {
        "score_b": float(compute_scoreB(truth, sub)),
        "subtype_f1": float(np.mean(pos_f1)),
        "em": float(np.mean(np.all(pred == y_b, axis=1))),
    }


def _macro_f1_binary(gold: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Vectorised official ``compute_f1`` over the last axis (rows = resamples)."""
    tp = np.sum((gold == 1) & (pred == 1), axis=-1).astype(float)
    tn = np.sum((gold == 0) & (pred == 0), axis=-1).astype(float)
    fp = np.sum((gold == 0) & (pred == 1), axis=-1).astype(float)
    fn = np.sum((gold == 1) & (pred == 0), axis=-1).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        f_neg = np.where(tn > 0, 2 * tn / (2 * tn + fp + fn), 0.0)
        f_pos = np.where(tp > 0, 2 * tp / (2 * tp + fp + fn), 0.0)
    out: np.ndarray = (f_pos + f_neg) / 2
    return out


def fast_score(task: str, y_a: np.ndarray, y_b: np.ndarray, pred: np.ndarray) -> float:
    """Score A or Score B, numerically identical to the official scorer."""
    if task == "A":
        return float(_macro_f1_binary(y_a, pred))
    weights = y_b.sum(axis=0).astype(float)
    per = np.array([_macro_f1_binary(y_b[:, j], pred[:, j]) for j in range(y_b.shape[1])])
    return float(np.sum(per * weights) / weights.sum())


def bootstrap_ci(
    task: str, y_a: np.ndarray, y_b: np.ndarray, pred: np.ndarray, seed: int
) -> list[float]:
    """Percentile 95% bootstrap CI (1000 meme resamples) of Score A / Score B."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(y_a), size=(N_BOOT, len(y_a)))
    if task == "A":
        vals = _macro_f1_binary(y_a[idx], pred[idx])
    else:
        yb, pb = y_b[idx], pred[idx]
        w = yb.sum(axis=1).astype(float)
        per = np.stack([_macro_f1_binary(yb[..., j], pb[..., j]) for j in range(4)], axis=1)
        vals = (per * w).sum(axis=1) / w.sum(axis=1)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return [round(float(lo), 4), round(float(hi), 4)]


# --------------------------------------------------------------------------------------------
# Jobs
# --------------------------------------------------------------------------------------------


def run_job(
    cond: Condition,
    clf: str,
    task: str,
    data: dict[str, dict[str, SplitData]],
    threads: int,
) -> dict[str, Any]:
    """Fit one (condition, classifier, task) model and evaluate it on validation and test."""
    t0 = time.perf_counter()
    src = data[cond.source]
    tr = src["train"]
    record: dict[str, Any] = {"condition": cond.name, "classifier": clf, "task": task}
    preds: dict[str, np.ndarray] = {}
    perm: dict[str, Any] = {}
    if cond.fusion == "late":
        probs: dict[str, np.ndarray] = {}
        for split in ("validation", "test"):
            probs[split] = np.zeros_like(src[split].y_a if task == "A" else src[split].y_b, float)
        for block in ("image", "text"):
            x_tr, _ = build_features(tr, block)
            model = fit_model(clf, task, x_tr, tr.y_a, tr.y_b, True, threads)
            for split in ("validation", "test"):
                x, _ = build_features(src[split], block)
                probs[split] += predict_proba(model, task, x) / 2
        preds = {s: (p >= 0.5).astype(int) for s, p in probs.items()}
        record["threshold"] = 0.5
    else:
        x_tr, blocks = build_features(tr, cond.fusion)
        model = fit_model(clf, task, x_tr, tr.y_a, tr.y_b, False, threads)
        for split in ("validation", "test"):
            d = src[split]
            x, _ = build_features(d, cond.fusion)
            preds[split] = np.asarray(model.predict(x)).astype(int)
            if cond.fusion == "concat":
                perm[split] = permutation_drops(model, task, x, d, blocks, preds[split])
    for split, pred in preds.items():
        d = src[split]
        scores = official_scores(task, d.y_a, d.y_b, pred)
        key = "score_a" if task == "A" else "score_b"
        if abs(scores[key] - fast_score(task, d.y_a, d.y_b, pred)) > 1e-9:
            raise AssertionError(f"vectorised scorer disagrees with official on {record}")
        if split == "test":
            scores[f"{key}_ci95"] = bootstrap_ci(task, d.y_a, d.y_b, pred, SEED)
        record[split] = scores
    if perm:
        record["permutation"] = perm
    record["test_pred"] = preds["test"].tolist()
    record["seconds"] = round(time.perf_counter() - t0, 1)
    logger.info("%s / %s / Task %s done in %.0f s", cond.name, clf, task, record["seconds"])
    return record


def permutation_drops(
    model: BaseEstimator,
    task: str,
    x: np.ndarray,
    d: SplitData,
    blocks: dict[str, slice],
    base_pred: np.ndarray,
) -> dict[str, Any]:
    """Shuffle each block's rows across memes (``N_PERM`` repeats) and record metric drops."""
    base = fast_score(task, d.y_a, d.y_b, base_pred)
    base_acc = float(np.mean(base_pred == d.y_a)) if task == "A" else None
    out: dict[str, Any] = {"base": base}
    rng = np.random.default_rng(SEED)
    for block, sl in blocks.items():
        scores, accs = [], []
        for _ in range(N_PERM):
            xp = x.copy()
            xp[:, sl] = x[rng.permutation(len(x)), sl]
            pred = np.asarray(model.predict(xp)).astype(int)
            scores.append(fast_score(task, d.y_a, d.y_b, pred))
            if task == "A":
                accs.append(float(np.mean(pred == d.y_a)))
        drops = base - np.array(scores)
        entry: dict[str, Any] = {
            "mean_drop": float(drops.mean()),
            "std_drop": float(drops.std(ddof=1)),
            "mean_permuted_score": float(np.mean(scores)),
        }
        if base_acc is not None:
            entry["mean_acc_drop"] = float(base_acc - np.mean(accs))
        out[block] = entry
    return out


# --------------------------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------------------------


def paper_reference() -> dict[str, dict[str, dict[str, float]]]:
    """Paper's committed ML-on-emb test results (CLIP concat, provided text)."""
    a = json.loads(PAPER_A.read_text(encoding="utf-8"))["models"]
    b = json.loads(PAPER_B.read_text(encoding="utf-8"))["models"]
    ref: dict[str, dict[str, dict[str, float]]] = {}
    for short, long in PAPER_NAMES.items():
        ref[short] = {
            "A": {"acc": a[long]["accuracy"], "score_a": a[long]["macro_f1"]},
            "B": {
                "score_b": b[long]["mami_score_b"],
                "subtype_f1": b[long]["macro_f1"],
                "em": b[long]["exact_match_accuracy"],
            },
        }
    return ref


def select_on_validation(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Per condition and task, the classifier with the best validation Score A / Score B."""
    out: dict[str, dict[str, Any]] = {}
    for cond in CONDITIONS:
        out[cond.name] = {}
        for task in TASKS:
            key = "score_a" if task == "A" else "score_b"
            cands = [r for r in records if r["condition"] == cond.name and r["task"] == task]
            best = max(cands, key=lambda r: r["validation"][key])
            out[cond.name][task] = {
                "classifier": best["classifier"],
                "validation": best["validation"][key],
                "test": best["test"],
            }
    return out


def write_tex(records: list[dict[str, Any]], path: Path) -> None:
    """Write LaTeX table rows (booktabs body) for SVM-RBF and XGBoost, test split."""
    lookup = {(r["condition"], r["classifier"], r["task"]): r for r in records}
    lines = [
        "% Generated by scripts/revision/modality_ablation.py -- test split, official scorer.",
        "% Columns: Condition & Classifier & Acc & Score A & Score B & Subtype-F1 & EM \\\\",
    ]
    for cond in CONDITIONS:
        for clf in ("SVM-RBF", "XGBoost"):
            a = lookup[(cond.name, clf, "A")]["test"]
            b = lookup[(cond.name, clf, "B")]["test"]
            label = cond.description.replace("_", "\\_").replace("*", "$\\odot$")
            lines.append(
                f"{label} & {clf} & {100 * a['acc']:.2f}\\% & {a['score_a']:.4f} & "
                f"{b['score_b']:.4f} & {b['subtype_f1']:.4f} & {100 * b['em']:.2f}\\% \\\\"
            )
        lines.append("\\midrule")
    lines.append("% Permutation ablation (concat models, test, mean drop over 10 shuffles)")
    lines.append(
        "% Columns: Backbone & Classifier & Shuffled block & Delta Score A & Delta Score B"
    )
    for cname in ("clip_concat", "masked_concat", "siglip_concat"):
        for clf in ("SVM-RBF", "XGBoost"):
            pa = lookup[(cname, clf, "A")]["permutation"]["test"]
            pb = lookup[(cname, clf, "B")]["permutation"]["test"]
            cond_tex = cname.replace("_", "\\_")
            for block in ("image", "text"):
                lines.append(
                    f"{cond_tex} & {clf} & {block} & "
                    f"$-${pa[block]['mean_drop']:.4f} & $-${pb[block]['mean_drop']:.4f} \\\\"
                )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv(records: list[dict[str, Any]], path: Path) -> None:
    """Write a flat CSV of validation and test metrics for every model."""
    cols = ["acc", "score_a", "score_b", "subtype_f1", "em"]
    rows = ["condition,classifier,task,split," + ",".join(cols)]
    for r in records:
        for split in ("validation", "test"):
            vals = [f"{r[split][c]:.6f}" if c in r[split] else "" for c in cols]
            rows.append(f"{r['condition']},{r['classifier']},{r['task']},{split}," + ",".join(vals))
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


_WORKER_DATA: dict[str, dict[str, SplitData]] = {}


def _init_worker() -> None:
    """Load the feature files once per worker process."""
    _WORKER_DATA.update(load_sources()[0])


def _worker_job(cond: Condition, clf: str, task: str, threads: int) -> dict[str, Any]:
    """Run one job against the worker-local copy of the features."""
    return run_job(cond, clf, task, _WORKER_DATA, threads)


def job_priority(job: tuple[Condition, str, str]) -> int:
    """Sort key that schedules the longest jobs (SVM-RBF, then Task B) first."""
    return -2 * int(job[1] == "SVM-RBF") - int(job[2] == "B")


def main() -> int:
    """Run every job in parallel and write results.json, ablation_table.tex and a CSV."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--xgb-threads", type=int, default=4)
    parser.add_argument("--classifiers", default=",".join(CLASSIFIERS))
    parser.add_argument("--conditions", default=",".join(c.name for c in CONDITIONS))
    args = parser.parse_args()

    t0 = time.perf_counter()
    _, meta = load_sources()
    clfs = [c for c in args.classifiers.split(",") if c]
    conds = [c for c in CONDITIONS if c.name in set(args.conditions.split(","))]
    jobs: list[tuple[Condition, str, str]] = [(c, k, t) for c in conds for k in clfs for t in TASKS]
    jobs.sort(key=job_priority)
    logger.info("Running %d jobs with n_jobs=%d", len(jobs), args.n_jobs)
    with ProcessPoolExecutor(max_workers=args.n_jobs, initializer=_init_worker) as pool:
        futures = [pool.submit(_worker_job, c, k, t, args.xgb_threads) for c, k, t in jobs]
        records: list[dict[str, Any]] = [f.result() for f in futures]
    records.sort(key=lambda r: ([c.name for c in CONDITIONS].index(r["condition"]), r["task"]))

    ref = paper_reference()
    repro: dict[str, Any] = {}
    for r in records:
        if r["condition"] == "clip_concat" and r["classifier"] in ref:
            paper = ref[r["classifier"]][r["task"]]
            repro.setdefault(r["classifier"], {})[r["task"]] = {
                m: {"ours": r["test"][m], "paper": v, "diff": r["test"][m] - v}
                for m, v in paper.items()
            }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    test_preds = {
        f"{r['condition']}|{r['classifier']}|{r['task']}": r.pop("test_pred") for r in records
    }
    result = {
        "task": "EXP-04 / EXP-06 modality ablation",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": "uv run python -m scripts.revision.modality_ablation " + " ".join(sys.argv[1:]),
        "scorer": "scripts/official_mami_evaluation.py (compute_scoreA, compute_scoreB)",
        "protocol": {
            "fit": "train split only (n=9000)",
            "model_selection": "best classifier per condition by validation Score A / Score B",
            "evaluation": "test split (n=1000), threshold 0.5 / estimator predict()",
            "classifier_settings": "auto_benchmark get_models_dict defaults, hp_optimization none",
            "scaling": "StandardScaler pipeline for SVM-RBF, LR, MLP, LDA; none for XGBoost",
            "class_weight": "balanced for SVM-RBF and LR",
            "task_b": "MultiOutputClassifier over shaming, stereotype, objectification, violence",
            "late_fusion": "mean of image-only and text-only predict_proba, threshold 0.5",
            "permutation": f"{N_PERM} row shuffles per block, seed {SEED}",
            "bootstrap": f"{N_BOOT} meme resamples, percentile 95% CI, seed {SEED}",
            "xgboost": "CPU, tree_method=hist",
        },
        "versions": {
            "python": platform.python_version(),
            "sklearn": sklearn.__version__,
            "xgboost": xgboost.__version__,
            "numpy": np.__version__,
        },
        "embeddings": meta,
        "conditions": {
            c.name: {"source": c.source, "fusion": c.fusion, "desc": c.description} for c in conds
        },
        "paper_reference_files": [
            str(PAPER_A.relative_to(ROOT)).replace("\\", "/"),
            str(PAPER_B.relative_to(ROOT)).replace("\\", "/"),
        ],
        "reproduction_clip_concat_vs_paper": repro,
        "validation_selected": select_on_validation(records) if len(clfs) > 1 else None,
        "records": records,
        "total_seconds": round(time.perf_counter() - t0, 1),
    }
    (OUT_DIR / "results.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    (OUT_DIR / "test_predictions.json").write_text(json.dumps(test_preds), encoding="utf-8")
    write_csv(records, OUT_DIR / "metrics.csv")
    if set(CLASSIFIERS) <= set(clfs) and len(conds) == len(CONDITIONS):
        write_tex(records, OUT_DIR / "ablation_table.tex")
    logger.info("Done in %.0f s", result["total_seconds"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
