"""Backfill provenance fields into legacy rationale generation files.

The rationale generation files produced on SCIAMA on 2026-08-05 predate the
provenance fields added to ``explain_with_vlm.py``. They record only
``image_id``, ``ground_truth``, ``predicted_misogynous``, ``explanation`` and
``latency_s``, so nothing inside the file states which model produced it or
which dataset split it came from. The filename was the only clue, which meant a
mislabelled file could not be detected by any automated check.

This script writes ``model`` and ``split`` into every record. The split is not
assumed: it is verified by matching each file's ``image_id`` sequence against
the actual dataset splits before anything is written. A file whose IDs do not
match the expected split is left untouched and reported as an error.

Idempotent. Running it twice is a no-op.

Usage:
    uv run python scripts/backfill_explanation_provenance.py
    uv run python scripts/backfill_explanation_provenance.py --check
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

RESULTS_DIR = PROJECT_ROOT / "results"

# Each legacy file, the model that produced it, and the split it is expected to
# come from. The split is verified against the dataset, never trusted blindly.
# Model ids are the BASE checkpoints, not fine-tuned adapters. Verified from the
# scripts that produced these files: scripts/submit_explainability.slurm and
# scripts/submit_explainability_all.slurm invoke explain_with_vlm.py with either
# no --model-id (falling back to the base default) or an explicit base id, and
# models/vlm/explainer.py has no PEFT or adapter loading path at all. Do not
# relabel these as QLoRA without first adding adapter support and re-running.
LEGACY_FILES: dict[str, dict[str, str]] = {
    "vlm_explanations.jsonl": {
        "model": "Qwen/Qwen2-VL-7B-Instruct",
        "split": "validation",
    },
    "llava_explanations.jsonl": {
        "model": "llava-hf/llava-1.5-7b-hf",
        "split": "validation",
    },
    "qwen2b_explanations.jsonl": {
        "model": "Qwen/Qwen2-VL-2B-Instruct",
        "split": "validation",
    },
}


def load_records(path: Path) -> list[dict]:
    """Read a JSONL file into a list of records."""
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def dataset_ids(split: str, count: int) -> list[str]:
    """Return the first ``count`` image ids of a dataset split, in order."""
    from utils.dataset import DatasetManager

    dataset = DatasetManager().load_dataset(split=split)
    return [str(dataset[i]["image_id"]) for i in range(min(count, len(dataset)))]


def verify_split(records: list[dict], expected_split: str) -> tuple[bool, str]:
    """Confirm the record ids really are the leading ids of ``expected_split``."""
    actual = [str(r["image_id"]) for r in records]
    expected = dataset_ids(expected_split, len(actual))
    if actual == expected:
        return True, f"ids match the first {len(actual)} of '{expected_split}'"
    overlap = len(set(actual) & set(expected))
    return False, (
        f"ids do NOT match '{expected_split}' (only {overlap}/{len(actual)} overlap). "
        f"file starts {actual[:4]}, split starts {expected[:4]}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify only; do not write. Exits non-zero if any file needs backfilling.",
    )
    args = parser.parse_args()

    needs_work = False
    failures: list[str] = []

    for filename, meta in LEGACY_FILES.items():
        path = RESULTS_DIR / filename
        if not path.exists():
            print(f"SKIP    {filename}: not present")
            continue

        records = load_records(path)
        already = all(r.get("model") and r.get("split") for r in records)

        ok, detail = verify_split(records, meta["split"])
        if not ok:
            print(f"ERROR   {filename}: {detail}")
            failures.append(filename)
            continue

        if already:
            print(f"OK      {filename}: provenance already present ({detail})")
            continue

        needs_work = True
        if args.check:
            print(f"PENDING {filename}: needs model/split ({detail})")
            continue

        for record in records:
            record.setdefault("model", meta["model"])
            record.setdefault("split", meta["split"])

        with path.open("w", encoding="utf-8") as fh:
            for record in records:
                fh.write(json.dumps(record) + "\n")

        print(
            f"WROTE   {filename}: model={meta['model']} split={meta['split']} "
            f"({len(records)} records; {detail})"
        )

    if failures:
        print(
            f"\nFAILED: {len(failures)} file(s) did not match their expected split. "
            "Nothing was written for those files.",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.check and needs_work:
        sys.exit(1)


if __name__ == "__main__":
    main()
