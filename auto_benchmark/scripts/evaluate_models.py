#!/usr/bin/env python3
"""
CLI script to evaluate and rank model benchmarking results based on different metrics.
Usage:
  python scripts/evaluate_models.py --init config/init_config.yaml --model_config model1 --metric Accuracy
"""

import os
import argparse
import yaml

from autobenchmark.evaluation import load_and_evaluate_results


def main():
    parser = argparse.ArgumentParser(description="Autobenchmark Evaluation & Ranking CLI")
    parser.add_argument('--init', type=str, default='config/init_config.yaml',
                        help='Path to system initialization YAML configuration')
    parser.add_argument('--model_config', type=str, required=True,
                        help='Name of the model config whose results to evaluate (e.g., model1)')
    parser.add_argument('--metric', type=str, default='F1',
                        help='Metric to rank models by (options: Accuracy, F1, Precision, Recall, ROC_AUC)')
    args = parser.parse_args()
    
    # Load init config
    if not os.path.exists(args.init):
        print(f"Error: Init config not found at: {args.init}")
        return
        
    with open(args.init, 'r', encoding='utf-8') as f:
        init_cfg = yaml.safe_load(f)
        
    base_dir = init_cfg.get('system', {}).get('base_dir', 'C:/Github/auto_benchmark')
    results_dir = init_cfg.get('paths', {}).get('results_dir', 'results')
    
    # Locate evaluation file
    eval_csv_path = os.path.join(
        base_dir, results_dir, 'model_results', args.model_config, f"{args.model_config}_evaluation.csv"
    )
    
    if not os.path.exists(eval_csv_path):
        # Check relative paths
        eval_csv_path = os.path.join(
            results_dir, 'model_results', args.model_config, f"{args.model_config}_evaluation.csv"
        )
        
    if not os.path.exists(eval_csv_path):
        print(f"Error: Evaluation results CSV not found at: {eval_csv_path}. Run benchmarking script first.")
        return
        
    try:
        load_and_evaluate_results(eval_csv_path, args.metric)
    except Exception as e:
        print(f"Error executing ranking: {e}")


if __name__ == "__main__":
    main()
