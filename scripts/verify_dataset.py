"""Verify the MAMI 2022 dataset is correctly downloaded and loadable.

Checks:
  1. Each split (train, validation, test) loads without error.
  2. Expected record counts match (train: 9000, validation: 1000, test: 1000).
  3. Required fields are present on every record.
  4. Labels are 0 or 1.
  5. Every image file is readable as RGB and non-zero.

Exits with code 0 on success, 1 on any failure.

Usage:
    uv run python scripts/verify_dataset.py
    uv run python scripts/verify_dataset.py --splits train
    uv run python scripts/verify_dataset.py --fast        # skips per-image pixel check
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from utils.dataset import MamiDataset, _kaggle_download  # noqa: E402

ALL_SPLITS = ["train", "validation", "test"]
EXPECTED_COUNTS = {"train": 9000, "validation": 1000, "test": 1000}
REQUIRED_FIELDS = {
    "image",
    "image_id",
    "text",
    "misogynous",
    "shaming",
    "stereotype",
    "objectification",
    "violence",
}


def check(condition: bool, msg: str, failures: list[str]) -> None:
    if not condition:
        failures.append(msg)
        print(f"  FAIL  {msg}")
    else:
        print(f"  ok    {msg}")


def verify_split(split: str, dataset_path: str, fast: bool) -> list[str]:
    failures: list[str] = []
    print(f"\n[{split}]")

    try:
        ds = MamiDataset(dataset_path=dataset_path, split=split)
    except Exception as exc:
        failures.append(f"Failed to load '{split}': {exc}")
        print(f"  FAIL  load: {exc}")
        return failures

    count = len(ds)
    expected = EXPECTED_COUNTS.get(split)
    check(
        expected is None or count == expected,
        f"record count: got {count}, expected {expected}",
        failures,
    )

    bad_fields: list[int] = []
    bad_labels: list[int] = []
    bad_images: list[int] = []

    for idx in range(count):
        if fast:
            # Only check metadata (no image load)
            row = ds._records[idx]
            missing = {"file_name", "label"} - set(row.keys())
            if missing:
                bad_fields.append(idx)
            lbl = int(row.get("label", -1))
            if lbl not in (0, 1):
                bad_labels.append(idx)
        else:
            try:
                sample = ds[idx]
                missing = REQUIRED_FIELDS - set(sample.keys())
                if missing:
                    bad_fields.append(idx)
                if sample.get("misogynous") not in (0, 1):
                    bad_labels.append(idx)
                img = sample["image"]
                if img.shape[0] != 3 or img.numel() == 0:
                    bad_images.append(idx)
            except Exception:
                bad_images.append(idx)

        if (idx + 1) % 500 == 0 or (idx + 1) == count:
            print(f"  checked {idx + 1}/{count}...", end="\r")

    print(f"  checked {count}/{count}        ")

    check(
        not bad_fields,
        f"all records have required fields ({bad_fields[:5] or 'none missing'})",
        failures,
    )
    check(not bad_labels, f"all labels in {{0, 1}} ({bad_labels[:5] or 'none bad'})", failures)
    if not fast:
        check(not bad_images, f"all images readable ({bad_images[:5] or 'none bad'})", failures)

    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--splits",
        nargs="+",
        default=ALL_SPLITS,
        choices=ALL_SPLITS,
        metavar="SPLIT",
        help="Splits to verify (default: all three)",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Skip per-image pixel read (only checks metadata)",
    )
    args = parser.parse_args()

    print("Downloading / locating MAMI 2022 dataset...")
    dataset_path = _kaggle_download()
    print(f"Dataset root: {dataset_path}")
    print(f"Splits : {', '.join(args.splits)}")
    print(f"Mode   : {'fast (metadata only)' if args.fast else 'full (metadata + images)'}\n")

    all_failures: list[str] = []
    for split in args.splits:
        all_failures.extend(verify_split(split, dataset_path, fast=args.fast))

    print()
    if all_failures:
        print(f"FAILED — {len(all_failures)} issue(s):")
        for f in all_failures:
            print(f"  * {f}")
        sys.exit(1)
    else:
        print("All checks passed.")


if __name__ == "__main__":
    main()
