"""
Multi-label classification metrics for MAMI Sub-task B (Challenge 2).

Given per-image predicted dicts and ground-truth dicts over the four MAMI sub-type
labels (shaming, stereotype, objectification, violence), computes:

- Per-class precision, recall, f1, support for each of the four labels.
- macro_f1  = unweighted mean of per-class positive-F1.
- micro_f1  = F1 computed over pooled TP/FP/FN across all classes.
- weighted_f1 = support-weighted mean of per-class positive-F1.
- exact_match_accuracy = fraction of images where ALL four predicted labels match GT.
- mami_score_b = official MAMI 2022 Sub-task B metric (see compute_mami_score_b).

``compute_multilabel_metrics`` returns all of the above (including ``mami_score_b``)
in a single dict, so downstream benchmark scripts do not need to call the two
functions separately. ``compute_mami_score_b`` remains a public entry point for
callers who only want the MAMI metric.

All returned floats are in [0, 1].
"""
# pylint: disable=too-many-locals

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
            macro_f1         - unweighted mean of per-class F1 (positive-class)
            micro_f1         - F1 over pooled TP/FP/FN
            weighted_f1      - support-weighted mean of per-class positive-F1
            macro_precision  - unweighted mean of per-class precision
            macro_recall     - unweighted mean of per-class recall
            exact_match_accuracy - fraction of exact matches across all labels
            mami_score_b     - MAMI 2022 official Sub-task B metric
            per_label_binary_macro_f1 - per-sub-type binary-macro F1 used by MAMI
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
            "mami_score_b": 0.0,
            "per_label_binary_macro_f1": dict.fromkeys(labels, 0.0),
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

    # MAMI 2022 official Sub-task B metric (uses binary-macro F1 per sub-type,
    # weighted by positive support). Reuses the same per-sub-type TP/FP/FN/TN.
    mami = compute_mami_score_b(predictions, ground_truths, labels)

    return {
        "per_class": per_class,
        "macro_f1": round(macro_f1, 6),
        "micro_f1": round(micro_f1, 6),
        "weighted_f1": round(weighted_f1, 6),
        "macro_precision": round(macro_prec, 6),
        "macro_recall": round(macro_rec, 6),
        "exact_match_accuracy": round(exact_match_accuracy, 6),
        "mami_score_b": mami["mami_score_b"],
        "per_label_binary_macro_f1": mami["per_label_binary_macro_f1"],
    }


def compute_mami_score_b(
    predictions: list[dict[str, int]],
    ground_truths: list[dict[str, int]],
    labels: list[str],
) -> dict[str, Any]:
    """Compute the MAMI 2022 official Sub-task B metric.

    Reproduces ``compute_scoreB`` from the MAMI shared-task evaluation script
    (``MIND-Lab/SemEval2022-Task-5.../Evaluation/evaluation.py``):

    * For each sub-type, compute the binary-macro F1 for that column, defined
      as ``(positive_F1 + negative_F1) / 2`` where each F1 treats the sub-type
      column as an independent binary problem.
    * Weight each sub-type's binary-macro F1 by its number of gold positives.
    * Return the positive-support weighted average.

    This differs from :func:`compute_multilabel_metrics`'s ``weighted_f1``:

    * The MAMI-official per-sub-type F1 is the average of the positive-class
      and negative-class F1, whereas ``weighted_f1`` uses only the positive
      class F1.
    * Both aggregate by positive support, but they aggregate different
      per-sub-type quantities.

    Args:
        predictions: Per-image predicted label dicts, mapping label to 0/1.
        ground_truths: Per-image ground-truth label dicts, same format.
        labels: Ordered list of sub-type label names (e.g. ``SUBTYPE_LABELS``).

    Returns:
        Dict with keys:
            ``mami_score_b`` (float in [0, 1]): the official Sub-task B score.
            ``per_label_binary_macro_f1`` (dict[str, float]): per-sub-type
                binary-macro F1 in [0, 1].
            ``per_label_support`` (dict[str, int]): per-sub-type count of gold
                positives.
    """
    empty_per_label = dict.fromkeys(labels, 0.0)
    empty_support = dict.fromkeys(labels, 0)

    # A length mismatch is a caller bug even if one side is empty; check first.
    if len(predictions) != len(ground_truths):
        raise ValueError(
            f"predictions and ground_truths length mismatch: "
            f"{len(predictions)} vs {len(ground_truths)}"
        )
    # Only after we know the lengths agree, a shared-empty case returns zeros.
    if not predictions:
        return {
            "mami_score_b": 0.0,
            "per_label_binary_macro_f1": empty_per_label,
            "per_label_support": empty_support,
        }

    per_label_f1: dict[str, float] = {}
    per_label_support: dict[str, int] = {}
    weighted_sum = 0.0
    total_weight = 0

    for lbl in labels:
        tp = fp = fn = tn = 0
        for pred, gt in zip(predictions, ground_truths, strict=True):
            p = pred.get(lbl, 0)
            g = gt.get(lbl, 0)
            if p == 1 and g == 1:
                tp += 1
            elif p == 1 and g == 0:
                fp += 1
            elif p == 0 and g == 1:
                fn += 1
            else:
                tn += 1

        # Positive-class F1
        pos_prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        pos_rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        pos_f1 = 2 * pos_prec * pos_rec / (pos_prec + pos_rec) if (pos_prec + pos_rec) > 0 else 0.0

        # Negative-class F1 (treat negative as the target class)
        neg_prec = tn / (tn + fn) if (tn + fn) > 0 else 0.0
        neg_rec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        neg_f1 = 2 * neg_prec * neg_rec / (neg_prec + neg_rec) if (neg_prec + neg_rec) > 0 else 0.0

        binary_macro_f1 = (pos_f1 + neg_f1) / 2
        support = tp + fn  # gold positives for this sub-type

        per_label_f1[lbl] = round(binary_macro_f1, 6)
        per_label_support[lbl] = support

        weighted_sum += binary_macro_f1 * support
        total_weight += support

    mami_score_b = weighted_sum / total_weight if total_weight > 0 else 0.0

    return {
        "mami_score_b": round(mami_score_b, 6),
        "per_label_binary_macro_f1": per_label_f1,
        "per_label_support": per_label_support,
    }
