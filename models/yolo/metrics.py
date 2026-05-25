from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DetectionPrediction:
    image_id: str
    bbox: tuple[float, float, float, float]
    confidence: float


@dataclass(frozen=True)
class GroundTruthBox:
    image_id: str
    bbox: tuple[float, float, float, float]


def intersection_over_union(
    box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]
) -> float:
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter_width = max(0.0, x2 - x1)
    inter_height = max(0.0, y2 - y1)
    inter_area = inter_width * inter_height

    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union_area = area_a + area_b - inter_area

    if union_area <= 0.0:
        return 0.0
    return inter_area / union_area


def _match_predictions_to_ground_truths(
    predictions: Iterable[DetectionPrediction],
    ground_truths: Iterable[GroundTruthBox],
    iou_threshold: float = 0.5,
) -> tuple[int, int, int]:
    ground_truths_by_image: dict[str, list[GroundTruthBox]] = {}
    for gt in ground_truths:
        ground_truths_by_image.setdefault(gt.image_id, []).append(gt)

    matched_gt_ids: set[tuple[str, int]] = set()
    tp = 0
    fp = 0

    sorted_predictions = sorted(predictions, key=lambda item: item.confidence, reverse=True)
    for prediction in sorted_predictions:
        matched = False
        image_gts = ground_truths_by_image.get(prediction.image_id, [])
        for index, gt in enumerate(image_gts):
            key = (gt.image_id, index)
            if key in matched_gt_ids:
                continue
            if intersection_over_union(prediction.bbox, gt.bbox) >= iou_threshold:
                matched = True
                matched_gt_ids.add(key)
                tp += 1
                break
        if not matched:
            fp += 1

    total_gts = sum(len(gts) for gts in ground_truths_by_image.values())
    fn = total_gts - tp
    return tp, fp, max(0, fn)


def _average_precision(
    predictions: Iterable[DetectionPrediction],
    ground_truths: Iterable[GroundTruthBox],
    iou_threshold: float = 0.5,
) -> float:
    ground_truths_by_image: dict[str, list[GroundTruthBox]] = {}
    for gt in ground_truths:
        ground_truths_by_image.setdefault(gt.image_id, []).append(gt)
    total_gts = sum(len(v) for v in ground_truths_by_image.values())
    if total_gts == 0:
        return 0.0

    sorted_predictions = sorted(predictions, key=lambda item: item.confidence, reverse=True)
    matched_gt_ids: set[tuple[str, int]] = set()
    tp = 0
    fp = 0
    precisions: list[float] = []
    recalls: list[float] = []

    for prediction in sorted_predictions:
        image_gts = ground_truths_by_image.get(prediction.image_id, [])
        matched = False
        for index, gt in enumerate(image_gts):
            key = (prediction.image_id, index)
            if key in matched_gt_ids:
                continue
            if intersection_over_union(prediction.bbox, gt.bbox) >= iou_threshold:
                matched = True
                matched_gt_ids.add(key)
                tp += 1
                break
        if not matched:
            fp += 1
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / total_gts
        precisions.append(precision)
        recalls.append(recall)

    if not precisions:
        return 0.0

    # COCO 101-point AP: precision envelope sampled at 101 recall thresholds.
    recalls_arr = np.asarray(recalls)
    precisions_arr = np.asarray(precisions)
    p_env = np.maximum.accumulate(precisions_arr[::-1])[::-1]
    r = np.linspace(0.0, 1.0, 101)
    indices = np.searchsorted(recalls_arr, r, side="left")
    p_interp = np.where(indices < len(p_env), p_env[np.clip(indices, 0, len(p_env) - 1)], 0.0)
    return float(np.mean(p_interp))


def compute_detection_metrics(
    predictions: Iterable[DetectionPrediction],
    ground_truths: Iterable[GroundTruthBox],
    iou_threshold: float = 0.5,
) -> dict[str, float]:
    tp, fp, fn = _match_predictions_to_ground_truths(predictions, ground_truths, iou_threshold)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    ap50 = _average_precision(predictions, ground_truths, iou_threshold=0.5)
    ap50_95 = 0.0
    thresholds = [0.5 + 0.05 * i for i in range(10)]
    if thresholds:
        ap50_95 = sum(
            _average_precision(predictions, ground_truths, iou_threshold=thr) for thr in thresholds
        ) / len(thresholds)
    return {
        "mAP50": ap50,
        "mAP50-95": ap50_95,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
