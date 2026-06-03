"""
Benchmark AWS Rekognition content moderation on HatefulIllusion.

Metric: any_detection_recall — did Rekognition fire at least one moderation label?
Stratified by visibility_score and subset so results join cleanly with the CLIP
and YOLO benchmarks in task 7.

Credentials are read from the boto3 default chain (env vars, ~/.aws/credentials,
IAM role). Never hardcode keys. Required policy: AmazonRekognitionReadOnlyAccess.

Usage:
    uv run python scripts/benchmark_aws_rekognition.py --subset digits
    uv run python scripts/benchmark_aws_rekognition.py --subset all
    uv run python scripts/benchmark_aws_rekognition.py --subset digits --min-confidence 70
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import time
from typing import Any

import boto3  # type: ignore[import-untyped]
from botocore.exceptions import ClientError  # type: ignore[import-untyped]
from dotenv import load_dotenv
import numpy as np
from PIL import Image

from utils.dataset import DatasetManager

load_dotenv()

SUBSET_NAMES = ["digits", "hate_slangs", "hate_symbols"]
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

# Rekognition default TPS limit is 50; enforce a floor of 1/50 s between calls.
_MIN_CALL_INTERVAL_S = 1.0 / 50.0
# Retry budget for throttling errors.
_MAX_RETRIES = 5
_RETRY_BACKOFF_BASE_S = 1.0


def image_to_jpeg_bytes(image: np.ndarray) -> bytes:
    """Convert a uint8 HWC NumPy array to JPEG bytes suitable for Rekognition."""
    pil = Image.fromarray(image)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def rekognition_detect(
    client: Any,
    image_bytes: bytes,
    min_confidence: float = 50.0,
) -> tuple[bool, list[dict[str, Any]]]:
    """Call detect_moderation_labels and return (fired, labels).

    Retries up to _MAX_RETRIES times on ProvisionedThroughputExceededException
    with exponential backoff.
    """
    for attempt in range(_MAX_RETRIES):
        try:
            response = client.detect_moderation_labels(
                Image={"Bytes": image_bytes},
                MinConfidence=min_confidence,
            )
            labels: list[dict[str, Any]] = response["ModerationLabels"]
            fired = len(labels) > 0
            return fired, labels
        except ClientError as exc:
            code = exc.response["Error"]["Code"]
            if code == "ProvisionedThroughputExceededException" and attempt < _MAX_RETRIES - 1:
                sleep_s = _RETRY_BACKOFF_BASE_S * (2**attempt)
                print(f"  Throttled — retrying in {sleep_s:.1f}s (attempt {attempt + 1})")
                time.sleep(sleep_s)
            else:
                raise
    return False, []  # unreachable; satisfies mypy


def collect_samples(subset: str, split: str = "train") -> list[dict[str, Any]]:
    subsets = SUBSET_NAMES if subset == "all" else [subset]
    manager = DatasetManager()
    samples: list[dict[str, Any]] = []
    for subset_name in subsets:
        dataset = manager.load_dataset(split=split, subset=subset_name)
        for index in range(len(dataset)):
            sample = dataset[index]
            sample["subset"] = subset_name
            sample["image_id"] = f"{subset_name}_{sample['image_id']}"
            samples.append(sample)
    return samples


def compute_any_detection_recall(fired_flags: list[bool]) -> float:
    if not fired_flags:
        return 0.0
    return sum(fired_flags) / len(fired_flags)


def build_visibility_metrics(
    fired_flags: list[bool],
    visibility_scores: list[int],
) -> dict[str, dict[str, float]]:
    scores_seen: dict[int, list[int]] = {}
    for i, v in enumerate(visibility_scores):
        scores_seen.setdefault(v, []).append(i)

    metrics_by_visibility: dict[str, dict[str, float]] = {}
    for v, indices in scores_seen.items():
        v_flags = [fired_flags[i] for i in indices]
        metrics_by_visibility[str(v)] = {
            "any_detection_recall": compute_any_detection_recall(v_flags),
            "num_images": len(v_flags),
            "num_detected": sum(v_flags),
        }
    return metrics_by_visibility


def run_benchmark(
    subset: str,
    min_confidence: float = 50.0,
    region: str = "us-east-1",
) -> dict[str, Any]:
    samples = collect_samples(subset)
    if not samples:
        raise ValueError(f"No samples found for subset '{subset}'")

    client = boto3.client("rekognition", region_name=region)
    print(f"Connected to Rekognition in {region}")

    fired_flags: list[bool] = []
    inference_times: list[float] = []
    sample_records: list[dict[str, Any]] = []

    print(f"Running inference on {len(samples)} images …")
    last_call_time: float = 0.0

    def _build_result(partial: bool = False) -> dict[str, Any]:
        vis_scores = [int(s["visibility_score"]) for s in samples[: len(fired_flags)]]
        any_detection_recall = compute_any_detection_recall(fired_flags)
        visibility_metrics = build_visibility_metrics(fired_flags, vis_scores)
        total_time = sum(inference_times)
        avg_time = total_time / len(inference_times) if inference_times else 0.0
        return {
            "benchmark_date": datetime.now(timezone.utc).isoformat(),
            "subset": subset,
            "partial": partial,
            "models": {
                "aws_rekognition": {
                    "region": region,
                    "min_confidence": min_confidence,
                    "num_images": len(fired_flags),
                    "computed_metrics": {
                        "any_detection_recall": any_detection_recall,
                        "num_detected": sum(fired_flags),
                    },
                    "average_inference_time_s": avg_time,
                    "total_inference_time_s": total_time,
                    "visibility_metrics": visibility_metrics,
                    "sample_predictions": sample_records,
                }
            },
        }

    try:
        for i, sample in enumerate(samples):
            # Convert image to bytes.
            image_np = sample["image"]
            if not isinstance(image_np, np.ndarray):
                image_np = np.array(image_np)
            if image_np.ndim == 3 and image_np.shape[0] == 3:
                image_np = image_np.transpose(1, 2, 0)
            if image_np.dtype != np.uint8:
                image_np = np.clip(image_np, 0, 255).astype(np.uint8)
            image_bytes = image_to_jpeg_bytes(image_np)

            # Enforce rate limit.
            elapsed_since_last = time.perf_counter() - last_call_time
            if elapsed_since_last < _MIN_CALL_INTERVAL_S:
                time.sleep(_MIN_CALL_INTERVAL_S - elapsed_since_last)

            t0 = time.perf_counter()
            last_call_time = t0
            fired, labels = rekognition_detect(client, image_bytes, min_confidence=min_confidence)
            elapsed = time.perf_counter() - t0

            fired_flags.append(fired)
            inference_times.append(elapsed)

            sample_records.append(
                {
                    "image_id": sample["image_id"],
                    "subset": sample["subset"],
                    "ground_truth": sample["message"],
                    "fired": fired,
                    "visibility_score": int(sample["visibility_score"]),
                    "inference_time_s": round(elapsed, 4),
                    "raw_labels": labels,
                }
            )

            if (i + 1) % 50 == 0 or (i + 1) == len(samples):
                recall_so_far = compute_any_detection_recall(fired_flags)
                print(f"  [{i + 1}/{len(samples)}] running recall={recall_so_far:.3f}")

    except Exception as exc:
        if sample_records:
            partial_path = RESULTS_DIR / f"aws_rekognition_benchmark_{subset}_partial.json"
            partial_path.parent.mkdir(parents=True, exist_ok=True)
            partial_path.write_text(
                json.dumps(_build_result(partial=True), indent=2), encoding="utf-8"
            )
            print(f"Saved partial results ({len(sample_records)} images) to {partial_path}")
        raise exc

    return _build_result(partial=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--subset",
        default="digits",
        choices=["digits", "hate_slangs", "hate_symbols", "all"],
        help="HatefulIllusion subset to evaluate on",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=50.0,
        help="Minimum confidence threshold for Rekognition labels (0-100)",
    )
    parser.add_argument(
        "--region",
        default="us-east-1",
        help="AWS region for Rekognition",
    )
    args = parser.parse_args()

    results = run_benchmark(
        subset=args.subset,
        min_confidence=args.min_confidence,
        region=args.region,
    )

    out_path = RESULTS_DIR / f"aws_rekognition_benchmark_{args.subset}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved benchmark results to {out_path}")


if __name__ == "__main__":
    main()
