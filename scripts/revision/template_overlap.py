"""EXP-02: meme-template overlap across MAMI splits and a template-disjoint split.

Stages (all CPU, deterministic):

1. ``hash``: 64-bit perceptual hash (``imagehash.phash``) of every train/validation/test image,
   cached in ``results/revision/exp02/phash.csv``. Requires ``uv run --with imagehash``.
2. ``analyse``: "same template" = pHash Hamming distance <= ``h`` OR CLIP ViT-L-14 image
   cosine >= ``c`` (committed embeddings in ``results/embeddings/``). Thresholds are chosen
   without manual inspection (see :func:`choose_thresholds`) for three settings; templates are
   the connected components of the resulting graph. Writes cluster ids, ``overlap.json`` and
   ``seen_unseen.json`` (per-model test/validation Score A and accuracy on memes whose template
   is seen vs unseen in train, with bootstrap 95% CIs).
3. ``split``: template-disjoint train/validation/test TSVs (9000/1000/1000, label-balanced) in
   ``data/splits/template_disjoint/`` from the central threshold setting.

Usage::

    uv run --with imagehash python -m scripts.revision.template_overlap all
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
import csv
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np
from PIL import Image
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from utils.dataset import SPLIT_SOURCE_COLUMN, SPLIT_TSV_AND_IMAGES, DatasetManager

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "revision" / "exp02"
EMB = ROOT / "results" / "embeddings"
SPLIT_OUT = ROOT / "data" / "splits" / "template_disjoint"
SPLITS = ("train", "validation", "test")
SUBTYPES = ("shaming", "stereotype", "objectification", "violence")
TSV_COLUMNS = ("file_name", "label", *SUBTYPES, "text")
SEED = 20260517
N_BOOT = 2000
BLOCK = 1024
SPLIT_TARGET = {"train": 9000, "validation": 1000, "test": 1000}
MAX_EVAL_CLUSTER = 25  # templates larger than this always go to the template-disjoint train split

Row = dict[str, str]


# --------------------------------------------------------------------------- data


def dataset_root() -> Path:
    """Root of the Kaggle MAMI dataset (downloaded through ``DatasetManager`` if missing)."""
    manager = DatasetManager()
    return Path(manager.load_dataset(split="train").dataset_path)


def read_tsv(path: Path) -> list[Row]:
    """Read a MAMI TSV as a list of dicts (column order preserved by the reader)."""
    with path.open(encoding="utf-8", newline="") as fh:
        return [dict(r) for r in csv.DictReader(fh, delimiter="\t")]


def load_records(root: Path) -> list[Row]:
    """All 11,000 memes in fixed order (train, validation, test), tagged with their source split."""
    rows: list[Row] = []
    for sp in SPLITS:
        for r in read_tsv(root / SPLIT_TSV_AND_IMAGES[sp][0]):
            r[SPLIT_SOURCE_COLUMN] = sp
            rows.append(r)
    return rows


def stem(file_name: str) -> str:
    """Image id used in every result file (file name without extension)."""
    return Path(file_name).stem


def load_embeddings(rows: Sequence[Row]) -> np.ndarray:
    """L2-normalised CLIP ViT-L-14 image embeddings aligned with ``rows`` (ids are checked)."""
    parts = []
    for sp in SPLITS:
        z = np.load(EMB / f"{sp}_vit_l_14_quickgelu_ocr_paddleocr.npz", allow_pickle=True)
        want = [stem(r["file_name"]) for r in rows if r[SPLIT_SOURCE_COLUMN] == sp]
        if [str(i) for i in z["image_ids"].tolist()] != want:
            raise SystemExit(f"{sp} embedding ids do not match the dataset TSV order")
        parts.append(np.asarray(z["image_embeddings"], dtype=np.float32))
    emb = np.concatenate(parts)
    return emb / np.linalg.norm(emb, axis=1, keepdims=True)


# --------------------------------------------------------------------------- stage 1: hashing


def compute_phash(root: Path, rows: Sequence[Row], out: Path) -> None:
    """Write the 64-bit pHash (hex) of every image to ``out``."""
    import imagehash

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([SPLIT_SOURCE_COLUMN, "file_name", "phash"])
        for i, r in enumerate(rows):
            sp = r[SPLIT_SOURCE_COLUMN]
            path = root / "MAMI_2022_images" / SPLIT_TSV_AND_IMAGES[sp][1] / r["file_name"]
            with Image.open(path) as im:
                h = str(imagehash.phash(im.convert("RGB"), hash_size=8))
            w.writerow([sp, r["file_name"], h])
            if (i + 1) % 1000 == 0:
                print(f"hashed {i + 1}/{len(rows)}", flush=True)


def load_phash_bits(rows: Sequence[Row], path: Path) -> np.ndarray:
    """(N, 64) 0/1 float32 matrix of pHash bits aligned with ``rows``."""
    with path.open(encoding="utf-8", newline="") as fh:
        got = list(csv.DictReader(fh))
    if [(g[SPLIT_SOURCE_COLUMN], g["file_name"]) for g in got] != [
        (r[SPLIT_SOURCE_COLUMN], r["file_name"]) for r in rows
    ]:
        raise SystemExit(f"{path} does not match the dataset; rerun the hash stage")
    ints = np.array([int(g["phash"], 16) for g in got], dtype=np.uint64)
    shifts = np.arange(63, -1, -1, dtype=np.uint64)
    return ((ints[:, None] >> shifts[None, :]) & np.uint64(1)).astype(np.float32)


# --------------------------------------------------------------------------- similarity graph


def hamming_fn(bits: np.ndarray) -> Callable[[int, int], np.ndarray]:
    """Row-block function returning pHash Hamming distances to all images."""
    ones = bits.sum(axis=1)

    def fn(s: int, e: int) -> np.ndarray:
        dot = bits[s:e] @ bits.T
        return np.rint(ones[s:e, None] + ones[None, :] - 2.0 * dot)

    return fn


def cosine_fn(emb: np.ndarray) -> Callable[[int, int], np.ndarray]:
    """Row-block function returning CLIP cosine similarities to all images."""
    return lambda s, e: emb[s:e] @ emb.T


# Expected number of chance (different-template) matches per meme allowed by each setting.
CHANCE_MATCHES_PER_MEME = {"strict": 0.01, "central": 0.03, "loose": 0.1}
COS_GRID = 1000  # cosine histogram resolution (bins of 0.001)


def pair_histograms(bits: np.ndarray, emb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Exact histograms over all unordered pairs: Hamming 0..64 and cosine in 0.001 bins."""
    n = bits.shape[0]
    ham_h = np.zeros(65, dtype=np.int64)
    cos_h = np.zeros(2 * COS_GRID + 1, dtype=np.int64)
    ham, cos = hamming_fn(bits), cosine_fn(emb)
    for s in range(0, n, BLOCK):
        e = min(s + BLOCK, n)
        upper = np.arange(n)[None, :] > np.arange(s, e)[:, None]
        ham_h += np.bincount(ham(s, e)[upper].astype(np.int64), minlength=65)
        cbin = np.floor((cos(s, e)[upper] + 1.0) * COS_GRID).astype(np.int64)
        cos_h += np.bincount(np.clip(cbin, 0, 2 * COS_GRID), minlength=2 * COS_GRID + 1)
    return ham_h, cos_h


def tail_fit(x: np.ndarray, tail: np.ndarray, total: int) -> tuple[float, float, list[float]]:
    """Log-linear fit of the cumulative tail count over the chance-dominated region.

    The region is where the tail holds between 1e-4 and 1e-2 of all pairs: far enough into the
    tail to describe it, but still dominated by different-template pairs. Returns (intercept,
    slope, [region start, region end]).
    """
    reg = (tail >= 1e-4 * total) & (tail <= 1e-2 * total)
    slope, intercept = np.polyfit(x[reg], np.log(tail[reg]), 1)
    return float(intercept), float(slope), [float(x[reg].min()), float(x[reg].max())]


def choose_thresholds(ham_h: np.ndarray, cos_h: np.ndarray, n: int) -> dict[str, Any]:
    """Three reproducible threshold settings, no manual inspection.

    For each metric, the number of chance matches at a threshold is estimated by extrapolating a
    log-linear fit of the all-pairs tail (:func:`tail_fit`) from the chance-dominated region to
    the near-duplicate end. The excess of observed over extrapolated pairs are template matches.
    Each setting takes the loosest threshold whose estimated chance matches per meme
    (2 * pairs / n) do not exceed lambda = 0.01 (strict), 0.03 (central) or 0.1 (loose).
    The extrapolation overestimates chance matches for the (sub-exponential) pHash tail, which
    errs towards stricter thresholds.
    """
    total = int(ham_h.sum())
    d = np.arange(65, dtype=np.float64)
    ham_tail = np.cumsum(ham_h)  # pairs with distance <= d
    h_a, h_b, h_reg = tail_fit(d, ham_tail, total)
    c = np.arange(2 * COS_GRID + 1) / COS_GRID - 1.0
    cos_tail = np.cumsum(cos_h[::-1])[::-1]  # pairs with cosine >= c
    ok = cos_tail > 0
    c_a, c_b, c_reg = tail_fit(c[ok], cos_tail[ok], total)
    settings: dict[str, Any] = {}
    for name, lam in CHANCE_MATCHES_PER_MEME.items():
        null_h = 2 * np.exp(h_a + h_b * d) / n
        hmax = int(d[null_h <= lam].max()) if (null_h <= lam).any() else -1
        null_c = 2 * np.exp(c_a + c_b * c) / n
        cmin = round(float(c[(null_c <= lam) & (c <= 1.0)].min()), 3)
        ci = round((cmin + 1.0) * COS_GRID)
        settings[name] = {
            "chance_matches_per_meme_max": lam,
            "phash_max_hamming": hmax,
            "clip_min_cosine": cmin,
            "phash_pairs_observed": int(ham_tail[hmax]) if hmax >= 0 else 0,
            "phash_pairs_chance_est": round(float(np.exp(h_a + h_b * hmax)), 1),
            "clip_pairs_observed": int(cos_tail[ci]),
            "clip_pairs_chance_est": round(float(np.exp(c_a + c_b * cmin)), 1),
        }
    settings["_fit"] = {
        "all_pairs": total,
        "phash_log_tail_intercept_slope": [round(h_a, 4), round(h_b, 4)],
        "phash_fit_region_hamming": h_reg,
        "clip_log_tail_intercept_slope": [round(c_a, 4), round(c_b, 4)],
        "clip_fit_region_cosine": [round(v, 3) for v in c_reg],
        "phash_pairs_at_distance_0_to_20": ham_h[:21].tolist(),
    }
    return settings


def cluster(
    bits: np.ndarray, emb: np.ndarray, max_ham: int, min_cos: float
) -> tuple[np.ndarray, dict[str, int]]:
    """Connected components of the graph (hamming <= max_ham) OR (cosine >= min_cos)."""
    n = bits.shape[0]
    rows_i: list[np.ndarray] = []
    cols_i: list[np.ndarray] = []
    edge_counts = {"phash": 0, "clip": 0, "union": 0}
    ham, cos = hamming_fn(bits), cosine_fn(emb)
    for s in range(0, n, BLOCK):
        e = min(s + BLOCK, n)
        hit_h = ham(s, e) <= max_ham
        hit_c = cos(s, e) >= min_cos
        hit = hit_h | hit_c
        upper = np.arange(n)[None, :] > np.arange(s, e)[:, None]
        edge_counts["phash"] += int((hit_h & upper).sum())
        edge_counts["clip"] += int((hit_c & upper).sum())
        edge_counts["union"] += int((hit & upper).sum())
        r, c = np.nonzero(hit & upper)
        rows_i.append(r + s)
        cols_i.append(c)
    r_all, c_all = np.concatenate(rows_i), np.concatenate(cols_i)
    graph = coo_matrix((np.ones(len(r_all), dtype=np.int8), (r_all, c_all)), shape=(n, n))
    _, labels = connected_components(graph, directed=False)
    # Renumber deterministically by first occurrence in the fixed (train, val, test) order.
    first: dict[int, int] = {}
    out = np.array([first.setdefault(int(x), len(first)) for x in labels], dtype=np.int64)
    return out, edge_counts


# --------------------------------------------------------------------------- overlap


def overlap_stats(rows: Sequence[Row], cid: np.ndarray) -> dict[str, Any]:
    """Share of val/test memes whose template occurs in train, cluster sizes, label agreement."""
    src = np.array([r[SPLIT_SOURCE_COLUMN] for r in rows])
    lab = np.array([int(r["label"]) for r in rows])
    train_c = set(cid[src == "train"].tolist())
    val_c = set(cid[src == "validation"].tolist())
    sizes = np.bincount(cid)
    res: dict[str, Any] = {
        "n_clusters": len(sizes),
        "n_multi_member_clusters": int((sizes > 1).sum()),
        "memes_in_multi_member_clusters": int(sizes[sizes > 1].sum()),
        "largest_cluster_sizes": sorted(sizes.tolist(), reverse=True)[:10],
    }
    for sp in ("validation", "test"):
        m = src == sp
        seen = np.array([c in train_c for c in cid[m]])
        res[f"{sp}_seen_in_train_pct"] = round(100 * float(seen.mean()), 2)
        res[f"{sp}_seen_in_train_n"] = int(seen.sum())
    m = src == "test"
    seen_tv = np.array([c in train_c or c in val_c for c in cid[m]])
    res["test_seen_in_train_or_validation_pct"] = round(100 * float(seen_tv.mean()), 2)
    # Label agreement within multi-member clusters: fraction of same-label pairs vs. chance.
    same = total = 0
    for c in np.nonzero(sizes > 1)[0]:
        ls = lab[cid == c]
        k1 = int(ls.sum())
        k0 = len(ls) - k1
        same += k1 * (k1 - 1) // 2 + k0 * (k0 - 1) // 2
        total += len(ls) * (len(ls) - 1) // 2
    p = float(lab.mean())
    res["within_cluster_label_agreement"] = round(same / total, 4) if total else None
    res["chance_label_agreement"] = round(p * p + (1 - p) * (1 - p), 4)
    res["within_cluster_pairs"] = int(total)
    return res


def top_clusters(rows: Sequence[Row], cid: np.ndarray, k: int = 10) -> list[dict[str, Any]]:
    """Composition of the k largest clusters (split counts, misogynous share, example ids)."""
    sizes = np.bincount(cid)
    out = []
    for c in np.argsort(-sizes, kind="stable")[:k]:
        idx = np.nonzero(cid == c)[0]
        out.append(
            {
                "cluster_id": int(c),
                "size": len(idx),
                "by_split": {
                    sp: int(sum(rows[i][SPLIT_SOURCE_COLUMN] == sp for i in idx)) for sp in SPLITS
                },
                "misogynous_share": round(float(np.mean([int(rows[i]["label"]) for i in idx])), 3),
                "example_ids": [
                    f"{rows[i][SPLIT_SOURCE_COLUMN]}/{stem(rows[i]['file_name'])}" for i in idx[:5]
                ],
            }
        )
    return out


def label_distribution(rows: Sequence[Row]) -> dict[str, Any]:
    """Per-split label prevalences and chi-square tests (train vs test) for distribution shift."""
    from scipy.stats import chi2_contingency

    res: dict[str, Any] = {}
    for sp in SPLITS:
        rs = [r for r in rows if r[SPLIT_SOURCE_COLUMN] == sp]
        res[sp] = {"n": len(rs)} | {
            c: round(float(np.mean([int(r[c]) for r in rs])), 4) for c in ("label", *SUBTYPES)
        }
    tests = {}
    for c in ("label", *SUBTYPES):
        table = [
            [sum(int(r[c]) == v for r in rows if r[SPLIT_SOURCE_COLUMN] == sp) for v in (0, 1)]
            for sp in ("train", "test")
        ]
        chi2, pval, _, _ = chi2_contingency(table)
        tests[c] = {"chi2": round(float(chi2), 3), "p_value": float(f"{pval:.4g}")}
    res["train_vs_test_chi2"] = tests
    return res


def visual_shift(rows: Sequence[Row], emb: np.ndarray) -> dict[str, Any]:
    """Max CLIP cosine of each validation/test meme to any train meme (visual proximity)."""
    src = np.array([r[SPLIT_SOURCE_COLUMN] for r in rows])
    tr = emb[src == "train"]
    res = {}
    for sp in ("validation", "test"):
        best = (emb[src == sp] @ tr.T).max(axis=1)
        res[sp] = {
            "mean": round(float(best.mean()), 4),
            "median": round(float(np.median(best)), 4),
            "p10": round(float(np.quantile(best, 0.1)), 4),
            "p90": round(float(np.quantile(best, 0.9)), 4),
        }
    return res


# --------------------------------------------------------------------------- seen vs unseen


def official_score_a() -> Callable[[Sequence[int], Sequence[int]], float]:
    """Return a Score A function backed by ``compute_f1`` in scripts/official_mami_evaluation.py."""
    path = ROOT / "scripts" / "official_mami_evaluation.py"
    spec = importlib.util.spec_from_file_location("official_mami_evaluation", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    def score(gold: Sequence[int], pred: Sequence[int]) -> float:
        # Same path as compute_scoreA: booleans, index 0 of each record.
        return float(mod.compute_f1([bool(p) for p in pred], [bool(g) for g in gold]))

    return score


def model_sources(split: str) -> dict[str, list[Path]]:
    """Committed per-sample prediction files used for the seen/unseen analysis."""
    d = ROOT / "results" / split
    q7, q2 = "qwen2_vl_7b_instruct", "qwen2_vl_2b_instruct"
    return {
        "SVM-RBF (tabular, PaddleOCR)": [
            ROOT / "auto_benchmark" / "results" / "model_results" / tab_dir(split)
        ],
        "XGBoost (tabular, PaddleOCR)": [
            ROOT / "auto_benchmark" / "results" / "model_results" / tab_dir(split)
        ],
        "Qwen2-VL-7B QLoRA": [
            d / f"qwen2vl_{split}_{q7}_finetuned_seed{s}.json" for s in (1, 2, 3)
        ],
        "Qwen2-VL-2B QLoRA": [
            d / f"qwen2vl_{split}_{q2}_finetuned_seed{s}.json" for s in (1, 2, 3)
        ],
        "CLIP ViT-L-14 fine-tuned": [
            d / f"clip_{split}_finetuned_singleclass_vit_l_14_quickgelu_seed{s}.json"
            for s in (1, 2, 3)
        ],
        "CLIP ViT-B-32 fine-tuned": [
            d / f"clip_{split}_finetuned_singleclass_vit_b_32_quickgelu_seed{s}.json"
            for s in (1, 2, 3)
        ],
        "Gemini (zero-shot)": [d / f"gemini_{split}.json"],
        "Qwen2-VL-7B zero-shot": [d / f"qwen2vl_{split}_{q7}.json"],
        "Qwen2-VL-2B zero-shot": [d / f"qwen2vl_{split}_{q2}.json"],
    }


def tab_dir(split: str) -> str:
    """auto_benchmark result folder name of the PaddleOCR tabular run for a split."""
    name = "mami_tabular_model_test" if split == "test" else "mami_tabular_model"
    return f"{name}/{name}_predictions.csv"


TAB_COLUMN = {
    "SVM-RBF (tabular, PaddleOCR)": "Support Vector Machine - RBF SVC",
    "XGBoost (tabular, PaddleOCR)": "XGBoost",
}


def load_predictions(
    model: str, path: Path, split: str, truth: dict[str, int], order: list[str]
) -> tuple[dict[str, int], dict[str, Any]]:
    """Per-image binary predictions (invalid/refused -> wrong answer) and provenance info."""
    info: dict[str, Any] = {"file": path.relative_to(ROOT).as_posix()}
    if path.suffix == ".csv":
        # auto_benchmark writes predictions in the row order of the embedding NPZ it was given
        # (configs mami_pre_extracted[_test].yaml); the Actual column is checked against it.
        with path.open(encoding="utf-8", newline="") as fh:
            rs = list(csv.DictReader(fh))
        if [int(r["Actual"]) for r in rs] != [truth[i] for i in order]:
            raise SystemExit(f"{path}: Actual column does not match {split} NPZ label order")
        return {i: int(r[TAB_COLUMN[model]]) for i, r in zip(order, rs, strict=True)}, info
    rec = json.loads(path.read_text(encoding="utf-8"))[0]
    if rec.get("split") != split:
        raise SystemExit(f"{path}: split {rec.get('split')!r} != {split}")
    info["model_field"] = rec.get("model")
    preds: dict[str, int] = {}
    invalid = 0
    for p in rec["sample_predictions"]:
        iid = str(p["image_id"])
        if int(p["ground_truth"]) != truth[iid]:
            raise SystemExit(f"{path}: ground truth differs from dataset for {iid}")
        v = p.get("prediction")
        if isinstance(v, (bool, int)) and int(v) in (0, 1):
            preds[iid] = int(v)
        else:
            invalid += 1
            preds[iid] = 1 - truth[iid]
    if len(preds) != len(rec["sample_predictions"]) or not set(preds) <= set(order):
        raise SystemExit(f"{path}: duplicate ids or ids outside the {split} split")
    info["n_predictions"] = len(preds)
    info["invalid_predictions_scored_wrong"] = invalid
    return preds, info


def boot_ci(
    gold: np.ndarray, pred: np.ndarray, score: Callable[[Sequence[int], Sequence[int]], float]
) -> dict[str, Any]:
    """Point estimate and percentile bootstrap 95% CI of Score A and accuracy."""
    rng = np.random.default_rng(SEED)
    n = len(gold)
    idx = rng.integers(0, n, size=(N_BOOT, n))
    acc_b = (gold[idx] == pred[idx]).mean(axis=1)
    sa_b = np.array([score(gold[i].tolist(), pred[i].tolist()) for i in idx])
    return {
        "n": int(n),
        "misogynous_share": round(float(gold.mean()), 4),
        "score_a": round(score(gold.tolist(), pred.tolist()), 4),
        "score_a_ci95": [round(float(x), 4) for x in np.quantile(sa_b, [0.025, 0.975])],
        "accuracy": round(float((gold == pred).mean()), 4),
        "accuracy_ci95": [round(float(x), 4) for x in np.quantile(acc_b, [0.025, 0.975])],
    }


def diff_ci(
    gold: np.ndarray,
    pred: np.ndarray,
    seen: np.ndarray,
    score: Callable[[Sequence[int], Sequence[int]], float],
) -> dict[str, Any]:
    """Bootstrap 95% CI of (seen - unseen) Score A and accuracy, resampling each group."""
    rng = np.random.default_rng(SEED + 1)
    gs, ps, gu, pu = gold[seen], pred[seen], gold[~seen], pred[~seen]
    d_sa, d_acc = [], []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(gs), len(gs))
        j = rng.integers(0, len(gu), len(gu))
        d_sa.append(score(gs[i].tolist(), ps[i].tolist()) - score(gu[j].tolist(), pu[j].tolist()))
        d_acc.append(float((gs[i] == ps[i]).mean() - (gu[j] == pu[j]).mean()))
    return {
        "score_a_diff": round(score(gs.tolist(), ps.tolist()) - score(gu.tolist(), pu.tolist()), 4),
        "score_a_diff_ci95": [round(float(x), 4) for x in np.quantile(d_sa, [0.025, 0.975])],
        "accuracy_diff": round(float((gs == ps).mean() - (gu == pu).mean()), 4),
        "accuracy_diff_ci95": [round(float(x), 4) for x in np.quantile(d_acc, [0.025, 0.975])],
    }


def template_majority_baseline(rows: Sequence[Row], cid: np.ndarray, split: str) -> dict[str, Any]:
    """Accuracy of predicting a seen meme's label as the majority train label of its template."""
    src = np.array([r[SPLIT_SOURCE_COLUMN] for r in rows])
    lab = np.array([int(r["label"]) for r in rows])
    tr = src == "train"
    pos = np.bincount(cid[tr], weights=lab[tr], minlength=cid.max() + 1)
    cnt = np.bincount(cid[tr], minlength=cid.max() + 1)
    m = (src == split) & (cnt[cid] > 0)
    share = pos[cid[m]] / cnt[cid[m]]
    decided = share != 0.5
    pred = (share > 0.5).astype(int)
    return {
        "n_seen": int(m.sum()),
        "n_decided_no_tie": int(decided.sum()),
        "accuracy_on_decided": round(float((pred[decided] == lab[m][decided]).mean()), 4)
        if decided.any()
        else None,
    }


def seen_unseen(rows: Sequence[Row], cids: dict[str, np.ndarray], central: str) -> dict[str, Any]:
    """Seen/unseen-template metrics for every model, split and threshold setting."""
    score = official_score_a()
    src = np.array([r[SPLIT_SOURCE_COLUMN] for r in rows])
    ids = np.array([stem(r["file_name"]) for r in rows])
    out: dict[str, Any] = {}
    for split in ("validation", "test"):
        m = src == split
        order = ids[m].tolist()
        truth = {
            i: int(r["label"])
            for i, r in zip(
                order, [r for r in rows if r[SPLIT_SOURCE_COLUMN] == split], strict=True
            )
        }
        gold = np.array([truth[i] for i in order])
        seen_by = {}
        for name, cid in cids.items():
            train_c = set(cid[src == "train"].tolist())
            seen_by[name] = np.array([c in train_c for c in cid[m]])
        split_res: dict[str, Any] = {
            "counts": {
                k: {"seen": int(v.sum()), "unseen": int((~v).sum())} for k, v in seen_by.items()
            },
            "template_majority_baseline": {
                k: template_majority_baseline(rows, cids[k], split) for k in cids
            },
            "models": {},
        }
        for model, paths in model_sources(split).items():
            runs = []
            for path in paths:
                preds, info = load_predictions(model, path, split, truth, order)
                # Some files cover only part of a split (e.g. Gemini validation, 400 memes).
                cov = np.array([i in preds for i in order])
                g = gold[cov]
                pred = np.array([preds[i] for i in order if i in preds])
                run: dict[str, Any] = {"provenance": info, "all": boot_ci(g, pred, score)}
                for name, seen_all in seen_by.items():
                    seen = seen_all[cov]
                    res = {
                        "seen": boot_ci(g[seen], pred[seen], score),
                        "unseen": boot_ci(g[~seen], pred[~seen], score),
                    }
                    if name == central:
                        res["seen_minus_unseen"] = diff_ci(g, pred, seen, score)
                    run[name] = res
                runs.append(run)
            split_res["models"][model] = runs
        out[split] = split_res
    return out


# --------------------------------------------------------------------------- template-disjoint split


def make_split(rows: Sequence[Row], cid: np.ndarray) -> list[str]:
    """Assign every meme to train/validation/test with no template shared across splits.

    Clusters are visited in seeded random order; multi-member clusters (size <= MAX_EVAL_CLUSTER)
    are sent to a split drawn with probability proportional to its remaining capacity, provided
    the per-label capacity (targets are 50% misogynous) and file-name uniqueness (validation and
    test only, since ids key the prediction files) allow it, otherwise to train. Larger clusters always go to train. Singletons then fill validation and
    test exactly to their per-label targets; the rest go to train.
    """
    rng = random.Random(SEED)
    lab = [int(r["label"]) for r in rows]
    names = [r["file_name"] for r in rows]
    total_pos = sum(lab)
    cap: dict[str, list[int]] = {}
    for sp in ("validation", "test"):
        cap[sp] = [SPLIT_TARGET[sp] // 2, SPLIT_TARGET[sp] // 2]  # [neg, pos] remaining
    cap["train"] = [
        len(rows) - total_pos - cap["validation"][0] - cap["test"][0],
        total_pos - cap["validation"][1] - cap["test"][1],
    ]
    used: dict[str, set[str]] = {sp: set() for sp in SPLITS}
    members: dict[int, list[int]] = {}
    for i, c in enumerate(cid.tolist()):
        members.setdefault(c, []).append(i)
    order = sorted(members)
    rng.shuffle(order)
    assign = [""] * len(rows)

    def fits(sp: str, idx: list[int]) -> bool:
        n1 = sum(lab[i] for i in idx)
        if cap[sp][0] < len(idx) - n1 or cap[sp][1] < n1:
            return False
        if sp == "train":
            return True  # train predictions are never keyed by image id
        # Validation/test image ids must be unique (a few train and test file names coincide).
        return not any(names[i] in used[sp] for i in idx) and len({names[i] for i in idx}) == len(
            idx
        )

    def put(sp: str, idx: list[int]) -> None:
        for i in idx:
            assign[i] = sp
            cap[sp][lab[i]] -= 1
            used[sp].add(names[i])

    singles = []
    for c in order:
        idx = members[c]
        if len(idx) == 1:
            singles.append(idx)
            continue
        choices = ["train"]
        if len(idx) <= MAX_EVAL_CLUSTER:
            weights = {sp: sum(cap[sp]) for sp in SPLITS}
            first = rng.choices(list(weights), weights=list(weights.values()))[0]
            choices = [first] + [sp for sp in SPLITS if sp != first]
        for sp in choices:
            if fits(sp, idx):
                put(sp, idx)
                break
        else:
            raise SystemExit(f"cluster {c} (size {len(idx)}) cannot be placed")
    for idx in singles:
        for sp in ("validation", "test", "train"):
            if fits(sp, idx):
                put(sp, idx)
                break
        else:
            raise SystemExit(f"meme {names[idx[0]]} cannot be placed")
    if any(sum(v) for v in cap.values()):
        raise SystemExit(f"split targets not met exactly: remaining capacity {cap}")
    return assign


def write_split(rows: Sequence[Row], cid: np.ndarray, assign: Sequence[str]) -> dict[str, Any]:
    """Write the three TSVs and return their composition."""
    SPLIT_OUT.mkdir(parents=True, exist_ok=True)
    cols = [*TSV_COLUMNS, "template_id", SPLIT_SOURCE_COLUMN]
    stats: dict[str, Any] = {}
    for sp in SPLITS:
        sel = [i for i in range(len(rows)) if assign[i] == sp]
        with (SPLIT_OUT / f"{sp}.tsv").open("w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols, delimiter="\t", lineterminator="\n")
            w.writeheader()
            for i in sel:
                w.writerow(
                    {c: rows[i][c] for c in TSV_COLUMNS}
                    | {
                        "template_id": f"T{int(cid[i]):05d}",
                        SPLIT_SOURCE_COLUMN: rows[i][SPLIT_SOURCE_COLUMN],
                    }
                )
        back = read_tsv(SPLIT_OUT / f"{sp}.tsv")
        if [{c: b[c] for c in TSV_COLUMNS} for b in back] != [
            {c: rows[i][c] for c in TSV_COLUMNS} for i in sel
        ]:
            raise SystemExit(f"{sp}.tsv does not round-trip the dataset rows exactly")
        stats[sp] = {
            "n": len(sel),
            "misogynous": sum(int(rows[i]["label"]) for i in sel),
            "templates": len({int(cid[i]) for i in sel}),
            "from_source_split": {
                s: sum(rows[i][SPLIT_SOURCE_COLUMN] == s for i in sel) for s in SPLITS
            },
        } | {c: sum(int(rows[i][c]) for i in sel) for c in SUBTYPES}
    tmpl = {sp: {int(cid[i]) for i in range(len(rows)) if assign[i] == sp} for sp in SPLITS}
    stats["shared_templates"] = {
        "train_validation": len(tmpl["train"] & tmpl["validation"]),
        "train_test": len(tmpl["train"] & tmpl["test"]),
        "validation_test": len(tmpl["validation"] & tmpl["test"]),
    }
    return stats


# --------------------------------------------------------------------------- driver


def run_analysis(rows: list[Row]) -> None:
    """Stages 2 and 3: thresholds, clusters, overlap, seen/unseen metrics and the split."""
    bits = load_phash_bits(rows, OUT / "phash.csv")
    emb = load_embeddings(rows)
    n = len(rows)
    ham_h, cos_h = pair_histograms(bits, emb)
    fitted = choose_thresholds(ham_h, cos_h, n)
    fit_info = fitted.pop("_fit")
    settings: dict[str, Any] = fitted
    cids: dict[str, np.ndarray] = {}
    overlap: dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": "scripts/revision/template_overlap.py",
        "inputs": {
            "images": "Kaggle MAMI 2022 (training_images, test_images)",
            "phash": "results/revision/exp02/phash.csv (imagehash.phash, hash_size=8)",
            "clip": "results/embeddings/{split}_vit_l_14_quickgelu_ocr_paddleocr.npz",
        },
        "same_template_rule": "phash Hamming <= h OR CLIP image cosine >= c; connected components",
        "threshold_selection": {
            "rule": " ".join((choose_thresholds.__doc__ or "").split("\n\n")[1].split()),
        }
        | fit_info,
        "settings": {},
    }
    for name, st in settings.items():
        cid, edges = cluster(bits, emb, st["phash_max_hamming"], st["clip_min_cosine"])
        cids[name] = cid
        overlap["settings"][name] = st | {"edges": edges} | overlap_stats(rows, cid)
        overlap["settings"][name]["largest_clusters"] = top_clusters(rows, cid)
        print(
            name,
            {
                k: overlap["settings"][name][k]
                for k in (
                    "phash_max_hamming",
                    "clip_min_cosine",
                    "n_clusters",
                    "validation_seen_in_train_pct",
                    "test_seen_in_train_pct",
                )
            },
            flush=True,
        )
    overlap["label_distribution"] = label_distribution(rows)
    overlap["visual_proximity_to_train_max_cosine"] = visual_shift(rows, emb)
    with (OUT / "template_clusters.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([SPLIT_SOURCE_COLUMN, "image_id", *[f"cluster_{k}" for k in cids]])
        for i, r in enumerate(rows):
            w.writerow(
                [r[SPLIT_SOURCE_COLUMN], stem(r["file_name"]), *[int(c[i]) for c in cids.values()]]
            )
    assign = make_split(rows, cids["central"])
    overlap["template_disjoint_split"] = {
        "setting": "central",
        "seed": SEED,
        "max_eval_cluster": MAX_EVAL_CLUSTER,
    } | write_split(rows, cids["central"], assign)
    (OUT / "overlap.json").write_text(json.dumps(overlap, indent=2) + "\n", encoding="utf-8")
    su = seen_unseen(rows, cids, "central")
    su_out = {
        "generated_utc": overlap["generated_utc"],
        "script": "scripts/revision/template_overlap.py",
        "definition": "seen = the meme's template cluster contains at least one original train meme",
        "metric": "Score A = official MAMI macro-F1 (scripts/official_mami_evaluation.py)",
        "bootstrap": {"resamples": N_BOOT, "seed": SEED, "ci": "percentile 95%"},
        "invalid_prediction_policy": "refusals / unparseable outputs are scored as wrong",
        "thresholds": settings,
    } | su
    (OUT / "seen_unseen.json").write_text(json.dumps(su_out, indent=2) + "\n", encoding="utf-8")
    (OUT / "summary.md").write_text(render_summary(overlap, su_out), encoding="utf-8")


def _ci(x: list[float]) -> str:
    return f"[{x[0]:.3f}, {x[1]:.3f}]"


def render_summary(ov: dict[str, Any], su: dict[str, Any]) -> str:
    """Markdown summary (every number read from overlap/seen_unseen results)."""
    st = ov["settings"]
    lines = [
        "# EXP-02 template overlap (generated by scripts/revision/template_overlap.py)",
        "",
        "Same template: pHash Hamming <= h OR CLIP ViT-L-14 image cosine >= c, connected",
        "components over all 11,000 memes. Thresholds: loosest values whose estimated chance",
        "matches per meme (log-linear all-pairs tail extrapolation) stay <= lambda.",
        "",
        "| Setting | lambda | h | c | Clusters | Val seen in train | Test seen in train"
        " | Largest | Label agreement (chance) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for k, v in st.items():
        lines.append(
            f"| {k} | {v['chance_matches_per_meme_max']} | {v['phash_max_hamming']} "
            f"| {v['clip_min_cosine']} | {v['n_clusters']} "
            f"| {v['validation_seen_in_train_pct']}% ({v['validation_seen_in_train_n']}) "
            f"| {v['test_seen_in_train_pct']}% ({v['test_seen_in_train_n']}) "
            f"| {v['largest_cluster_sizes'][:3]} "
            f"| {v['within_cluster_label_agreement']} ({v['chance_label_agreement']}) |"
        )
    ld = ov["label_distribution"]
    vp = ov["visual_proximity_to_train_max_cosine"]
    lines += [
        "",
        "Label prevalence (misogynous) train/val/test: "
        f"{ld['train']['label']}/{ld['validation']['label']}/{ld['test']['label']}; "
        "train-vs-test chi-square p: "
        + ", ".join(f"{c} {v['p_value']}" for c, v in ld["train_vs_test_chi2"].items())
        + ".",
        "Sub-type prevalence train/val/test: "
        + "; ".join(f"{c} {ld['train'][c]}/{ld['validation'][c]}/{ld['test'][c]}" for c in SUBTYPES)
        + ".",
        f"Median max CLIP cosine to any train meme: validation {vp['validation']['median']}, "
        f"test {vp['test']['median']}.",
    ]
    for split in ("test", "validation"):
        res = su[split]
        cnt = res["counts"]["central"]
        base = res["template_majority_baseline"]["central"]
        lines += [
            "",
            f"## {split.capitalize()}: seen vs unseen templates (central; seen {cnt['seen']}, "
            f"unseen {cnt['unseen']})",
            "",
            f"Train-template majority-label baseline on seen memes: accuracy "
            f"{base['accuracy_on_decided']} (n={base['n_decided_no_tie']}).",
            "",
            "| Model (runs) | Score A seen | Score A unseen | Diff [95% CI] | Acc seen | Acc unseen |",
            "|---|---|---|---|---|---|",
        ]
        for model, runs in res["models"].items():
            c = [r["central"] for r in runs]
            sa_s = np.mean([x["seen"]["score_a"] for x in c])
            sa_u = np.mean([x["unseen"]["score_a"] for x in c])
            ac_s = np.mean([x["seen"]["accuracy"] for x in c])
            ac_u = np.mean([x["unseen"]["accuracy"] for x in c])
            diffs = "; ".join(
                f"{x['seen_minus_unseen']['score_a_diff']:+.3f} "
                f"{_ci(x['seen_minus_unseen']['score_a_diff_ci95'])}"
                for x in c
            )
            lines.append(
                f"| {model} ({len(runs)}) | {sa_s:.3f} | {sa_u:.3f} | {diffs} "
                f"| {ac_s:.3f} | {ac_u:.3f} |"
            )
    gap = []
    for model, runs in su["test"]["models"].items():
        v_runs = su["validation"]["models"][model]
        v_u = np.mean([r["central"]["unseen"]["score_a"] for r in v_runs])
        t_u = np.mean([r["central"]["unseen"]["score_a"] for r in runs])
        gap.append(f"{model} {v_u - t_u:+.3f}")
    lines += ["", "Validation-minus-test Score A on UNSEEN templates only: " + "; ".join(gap) + "."]
    sp = ov["template_disjoint_split"]
    lines += [
        "",
        "Template-disjoint split (central clusters, seed "
        f"{sp['seed']}): train/val/test = {sp['train']['n']}/{sp['validation']['n']}/"
        f"{sp['test']['n']}, misogynous {sp['train']['misogynous']}/"
        f"{sp['validation']['misogynous']}/{sp['test']['misogynous']}, shared templates "
        f"{sp['shared_templates']}. Multi-seed rows give per-seed differences; Score A and"
        " accuracy columns are seed means. Full CIs in seen_unseen.json.",
    ]
    tdis_rows = []
    for label, m, task_suffix, task_label in (
        ("Qwen2-VL-2B QLoRA", "2b", "", "Task A (singleclass)"),
        ("Qwen2-VL-7B QLoRA", "7b", "", "Task A (singleclass)"),
        ("Qwen2-VL-2B QLoRA", "2b", "_multiclass", "Task B (multiclass)"),
        ("Qwen2-VL-7B QLoRA", "7b", "_multiclass", "Task B (multiclass)"),
    ):
        vp_path = (
            ROOT
            / "results"
            / "validation"
            / f"qwen2vl_validation_qwen2_vl_{m}_instruct{task_suffix}_finetuned_seed1_tdisjoint.json"
        )
        tp_path = (
            ROOT
            / "results"
            / "test"
            / f"qwen2vl_test_qwen2_vl_{m}_instruct{task_suffix}_finetuned_seed1_tdisjoint.json"
        )
        if vp_path.is_file() and tp_path.is_file():
            vd = json.loads(vp_path.read_text(encoding="utf-8"))[0]
            td = json.loads(tp_path.read_text(encoding="utf-8"))[0]
            vs = vd.get("mami_score_b", vd["f1"])
            ts = td.get("mami_score_b", td["f1"])
            vm = vd.get("macro_f1", vd["f1"])
            tm = td.get("macro_f1", td["f1"])
            va = 100.0 * vd["exact_match_accuracy"]
            ta = 100.0 * td["exact_match_accuracy"]
            src = (
                "`results/{validation,test}/"
                f"qwen2vl_<split>_qwen2_vl_{m}_instruct{task_suffix}_finetuned_seed1_tdisjoint.json`"
            )
            tdis_rows.append(
                f"| {label} | {task_label} | {vs:.4f} | {ts:.4f} | {vm:.4f} | {tm:.4f} "
                f"| {va:.2f}% | {ta:.2f}% | {src} |"
            )
    if tdis_rows:
        lines += [
            "",
            "## Template-disjoint split QLoRA retraining (SCIAMA S4, seed 1)",
            "",
            "| Model | Task | Val Score A / B | Test Score A / B | Val Macro-F1 | Test Macro-F1"
            " | Val Acc / EM | Test Acc / EM | Source files |",
            "|---|---|---|---|---|---|---|---|---|",
            *tdis_rows,
        ]
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("stage", choices=("hash", "analyse", "all"))
    a = ap.parse_args(argv)
    root = dataset_root()
    rows = load_records(root)
    if a.stage in ("hash", "all"):
        compute_phash(root, rows, OUT / "phash.csv")
    if a.stage in ("analyse", "all"):
        run_analysis(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
