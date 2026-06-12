"""Download the MAMI 2022 dataset from Kaggle to the local cache.

Uses kagglehub to download the dataset.  Run this once before benchmarking.

Usage:
    uv run python scripts/download_dataset.py
"""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from utils.dataset import download_mami_dataset  # noqa: E402


def main() -> None:
    path = download_mami_dataset()
    print(f"Dataset ready at: {path}")


if __name__ == "__main__":
    main()
