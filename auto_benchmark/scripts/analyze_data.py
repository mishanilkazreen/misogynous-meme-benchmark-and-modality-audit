#!/usr/bin/env python3
"""
CLI script to profile datasets and output statistics before training models.
Usage:
  python scripts/analyze_data.py --init config/init_config.yaml --data config/data/data1.yaml
"""

import argparse
import os
from pathlib import Path

import yaml

from autobenchmark.data import load_data
from autobenchmark.data_analysis import run_data_profiling


def main():
    parser = argparse.ArgumentParser(description="Autobenchmark Data Profiling & Analysis CLI")
    parser.add_argument(
        "--init",
        type=str,
        default="config/init_config.yaml",
        help="Path to system initialization YAML configuration",
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to data preprocessing YAML configuration (e.g., config/data/data1.yaml)",
    )
    args = parser.parse_args()

    # Load configs
    if not os.path.exists(args.init):
        print(f"Error: Init config not found at: {args.init}")
        return
    if not os.path.exists(args.data):
        print(f"Error: Data config not found at: {args.data}")
        return

    with open(args.init, encoding="utf-8") as f:
        init_cfg = yaml.safe_load(f)

    with open(args.data, encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)

    data_config_name = Path(args.data).stem

    # Load dataset
    filepath = data_cfg.get("dataset", {}).get("filepath")
    if not filepath:
        print("Error: filepath not specified in data configuration.")
        return

    row_limit = data_cfg.get("dataset", {}).get("row_limit")
    df = load_data(filepath, init_cfg, nrows=row_limit)
    print(f"Data loaded successfully. Shape: {df.shape}")

    # Build output path
    base_dir = init_cfg.get("system", {}).get("base_dir", "C:/Github/auto_benchmark")
    results_dir = init_cfg.get("paths", {}).get("results_dir", "results")

    output_dir = os.path.join(base_dir, results_dir, "data_analysis", data_config_name)

    # Run profiling
    run_data_profiling(df, data_cfg, output_dir)
    print("\nData analysis complete!")


if __name__ == "__main__":
    main()
