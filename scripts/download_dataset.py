"""Download the HatefulIllusion dataset from Hugging Face to the local cache.

Uses snapshot_download to pull all files in parallel rather than one-by-one.
Run this once before benchmarking so inference doesn't stall on per-image downloads.

Usage:
    uv run python scripts/download_dataset.py
    uv run python scripts/download_dataset.py --subsets digits
    uv run python scripts/download_dataset.py --subsets digits hate_slangs hate_symbols
    uv run python scripts/download_dataset.py --cache-dir D:/hf_cache
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import snapshot_download

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

REPO_ID = "yiting/HatefulIllusion_Dataset"
ALL_SUBSETS = ["digits", "hate_slangs", "hate_symbols"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--subsets",
        nargs="+",
        default=ALL_SUBSETS,
        choices=ALL_SUBSETS,
        metavar="SUBSET",
        help="Which subsets to download (default: all three)",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Hugging Face cache directory (default: ~/.cache/huggingface)",
    )
    args = parser.parse_args()

    print(f"Downloading {REPO_ID}")
    print(f"Subsets : {', '.join(args.subsets)}\n")

    # Build an allow-list of folder patterns so we only pull requested subsets.
    # Each subset lives under e.g. "digits/" in the repo.
    allow_patterns = ["*.md", "*.json", "*.parquet", "*.yaml"]
    for subset in args.subsets:
        allow_patterns.append(f"{subset}/*")

    token = os.environ.get("HF_TOKEN") or None
    if not token:
        print("Warning: HF_TOKEN not set — unauthenticated requests may be rate-limited.")

    print("Fetching files (parallel download)...")
    local_dir = snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        allow_patterns=allow_patterns,
        cache_dir=args.cache_dir,
        token=token,
    )
    print(f"Snapshot saved to: {local_dir}\n")

    # Load each subset through the datasets library so parquet metadata is
    # indexed and __getitem__ works correctly in subsequent scripts.
    total = 0
    for subset in args.subsets:
        print(f"Indexing '{subset}'...", end=" ", flush=True)
        ds = load_dataset(REPO_ID, subset, cache_dir=args.cache_dir, token=token)
        n = len(ds["train"])
        total += n
        print(f"{n} records")

    print(f"\nDone — {total} records across {len(args.subsets)} subset(s).")


if __name__ == "__main__":
    main()
