"""Verify the HatefulIllusion dataset is correctly cached and loadable.

Checks:
  1. Each subset loads without error
  2. Expected record counts match (digits: 300, hate_slangs: 690, hate_symbols: 1170)
  3. Required fields present on every record (image, message, prompt, visibility)
  4. Visibility scores are integers in [0, 2]
  5. Every image file is readable as RGB and non-zero
  6. All three subsets are present when --subsets all is used

Exits with code 0 on success, 1 on any failure.

Usage:
    uv run python scripts/verify_dataset.py
    uv run python scripts/verify_dataset.py --subsets digits
    uv run python scripts/verify_dataset.py --fast        # skips per-image pixel check
"""

from __future__ import annotations

import argparse
import sys

from datasets import load_dataset
from huggingface_hub import hf_hub_download
from PIL import Image

REPO_ID = "yiting/HatefulIllusion_Dataset"
ALL_SUBSETS = ["digits", "hate_slangs", "hate_symbols"]
EXPECTED_COUNTS = {"digits": 300, "hate_slangs": 690, "hate_symbols": 1170}
REQUIRED_FIELDS = {"image", "message", "prompt", "visibility"}


def check(condition: bool, msg: str, failures: list[str]) -> None:
    if not condition:
        failures.append(msg)
        print(f"  FAIL  {msg}")
    else:
        print(f"  ok    {msg}")


def verify_subset(subset: str, fast: bool, cache_dir: str | None) -> list[str]:
    failures: list[str] = []
    print(f"\n[{subset}]")

    try:
        ds = load_dataset(REPO_ID, subset, cache_dir=cache_dir)
        split = ds["train"]
    except Exception as exc:
        failures.append(f"Failed to load '{subset}': {exc}")
        print(f"  FAIL  load: {exc}")
        return failures

    count = len(split)
    expected = EXPECTED_COUNTS.get(subset)
    check(
        expected is None or count == expected,
        f"record count: got {count}, expected {expected}",
        failures,
    )

    bad_fields: list[int] = []
    bad_visibility: list[int] = []
    bad_images: list[int] = []

    for idx, item in enumerate(split):
        missing = REQUIRED_FIELDS - set(item.keys())
        if missing:
            bad_fields.append(idx)

        vis = item.get("visibility")
        if not isinstance(vis, int) or not (0 <= vis <= 2):
            bad_visibility.append(idx)

        if not fast:
            try:
                image_path = item["image"]
                local = hf_hub_download(
                    repo_id=REPO_ID,
                    filename=f"{subset}/{image_path}",
                    repo_type="dataset",
                    cache_dir=cache_dir,
                )
                img = Image.open(local).convert("RGB")
                if img.width == 0 or img.height == 0:
                    bad_images.append(idx)
            except Exception:
                bad_images.append(idx)

        if (idx + 1) % 100 == 0 or (idx + 1) == count:
            print(f"  checked {idx + 1}/{count}...", end="\r")

    print(f"  checked {count}/{count}        ")

    check(
        not bad_fields,
        f"all records have required fields ({bad_fields[:5] or 'none missing'})",
        failures,
    )
    check(
        not bad_visibility,
        f"all visibility scores in [1,5] ({bad_visibility[:5] or 'none bad'})",
        failures,
    )
    if not fast:
        check(not bad_images, f"all images readable ({bad_images[:5] or 'none bad'})", failures)

    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--subsets",
        nargs="+",
        default=ALL_SUBSETS,
        choices=ALL_SUBSETS,
        metavar="SUBSET",
        help="Subsets to verify (default: all three)",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Skip per-image pixel read (only checks metadata)",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Hugging Face cache directory",
    )
    args = parser.parse_args()

    print(f"Verifying {REPO_ID}")
    print(f"Subsets : {', '.join(args.subsets)}")
    print(f"Mode    : {'fast (metadata only)' if args.fast else 'full (metadata + images)'}\n")

    all_failures: list[str] = []
    for subset in args.subsets:
        all_failures.extend(verify_subset(subset, fast=args.fast, cache_dir=args.cache_dir))

    print()
    if all_failures:
        print(f"FAILED — {len(all_failures)} issue(s):")
        for f in all_failures:
            print(f"  • {f}")
        sys.exit(1)
    else:
        print("All checks passed.")


if __name__ == "__main__":
    main()
