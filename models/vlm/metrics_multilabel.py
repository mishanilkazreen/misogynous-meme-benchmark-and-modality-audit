"""
Multi-label classification metrics for MAMI Sub-task B (Challenge 2).

Given per-image predicted dicts and ground-truth dicts over the four MAMI sub-type
labels (shaming, stereotype, objectification, violence), computes:

- Per-class precision, recall, f1, support for each of the four labels.
- macro_f1  = unweighted mean of per-class F1.
- micro_f1  = F1 computed over pooled TP/FP/FN across all classes.
- weighted_f1 = support-weighted mean of per-class F1.
- exact_match_accuracy = fraction of images where ALL four predicted labels match GT.

All returned floats are in [0, 1].
"""

from __future__ import annotations

from typing import Any


def compute_multilabel_metrics(
    predictions: list[dict[str, int]],
    ground_truths: list[dict[str, int]],
    labels: list[str],
) -> dict[str, Any]:
    """Compute multi-label classification metrics.

    Args:
        predictions: Per-image predicted label dicts, each mapping label → 0/1.
        ground_truths: Per-image ground-truth label dicts, same format.
        labels: Ordered list of label names (e.g. SUBTYPE_LABELS).

    Returns:
        Dict containing:
            per_class        - dict[label, {precision, recall, f1, support}]
            macro_f1         - unweighted mean of per-class F1
            micro_f1         - F1 over pooled TP/FP/FN
            weighted_f1      - support-weighted mean of per-class F1
            macro_precision  - unweighted mean of per-class precision
            macro_recall     - unweighted mean of per-class recall
            exact_match_accuracy - fraction of exact matches across all labels
    """
    n = len(predictions)
    if n == 0:
        empty_per_class = {
            lbl: {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0} for lbl in labels
        }
        return {
            "per_class": empty_per_class,
            "macro_f1": 0.0,
            "micro_f1": 0.0,
            "weighted_f1": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "exact_match_accuracy": 0.0,
        }

    # Per-class stats
    per_class: dict[str, dict[str, Any]] = {}
    total_support = 0
    pooled_tp = 0
    pooled_fp = 0
    pooled_fn = 0

    class_f1s: list[float] = []
    class_precs: list[float] = []
    class_recs: list[float] = []
    class_supports: list[int] = []

    for lbl in labels:
        tp = sum(
            1
            for pred, gt in zip(predictions, ground_truths, strict=True)
            if pred.get(lbl, 0) == 1 and gt.get(lbl, 0) == 1
        )
        fp = sum(
            1
            for pred, gt in zip(predictions, ground_truths, strict=True)
            if pred.get(lbl, 0) == 1 and gt.get(lbl, 0) == 0
        )
        fn = sum(
            1
            for pred, gt in zip(predictions, ground_truths, strict=True)
            if pred.get(lbl, 0) == 0 and gt.get(lbl, 0) == 1
        )
        support = tp + fn  # total ground-truth positives

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        per_class[lbl] = {
            "precision": round(prec, 6),
            "recall": round(rec, 6),
            "f1": round(f1, 6),
            "support": support,
        }

        pooled_tp += tp
        pooled_fp += fp
        pooled_fn += fn
        total_support += support

        class_f1s.append(f1)
        class_precs.append(prec)
        class_recs.append(rec)
        class_supports.append(support)

    # Macro averages (unweighted)
    macro_prec = sum(class_precs) / len(class_precs) if class_precs else 0.0
    macro_rec = sum(class_recs) / len(class_recs) if class_recs else 0.0
    macro_f1 = sum(class_f1s) / len(class_f1s) if class_f1s else 0.0

    # Micro F1 (pooled TP/FP/FN)
    micro_prec = pooled_tp / (pooled_tp + pooled_fp) if (pooled_tp + pooled_fp) > 0 else 0.0
    micro_rec = pooled_tp / (pooled_tp + pooled_fn) if (pooled_tp + pooled_fn) > 0 else 0.0
    micro_f1 = (
        2 * micro_prec * micro_rec / (micro_prec + micro_rec)
        if (micro_prec + micro_rec) > 0
        else 0.0
    )

    # Weighted F1 (support-weighted)
    if total_support > 0:
        weighted_f1 = (
            sum(f1 * sup for f1, sup in zip(class_f1s, class_supports, strict=True)) / total_support
        )
    else:
        weighted_f1 = 0.0

    # Exact match accuracy (all labels correct for an image)
    exact_matches = sum(
        1
        for pred, gt in zip(predictions, ground_truths, strict=True)
        if all(pred.get(lbl, 0) == gt.get(lbl, 0) for lbl in labels)
    )
    exact_match_accuracy = exact_matches / n

    return {
        "per_class": per_class,
        "macro_f1": round(macro_f1, 6),
        "micro_f1": round(micro_f1, 6),
        "weighted_f1": round(weighted_f1, 6),
        "macro_precision": round(macro_prec, 6),
        "macro_recall": round(macro_rec, 6),
        "exact_match_accuracy": round(exact_match_accuracy, 6),
    }
