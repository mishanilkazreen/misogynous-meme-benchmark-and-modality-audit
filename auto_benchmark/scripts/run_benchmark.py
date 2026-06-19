#!/usr/bin/env python3
"""
CLI script to execute model training, benchmarking, and uncertainty estimation.
Usage:
  python scripts/run_benchmark.py --init config/init_config.yaml --model config/model/model1.yaml
"""

import os
import argparse
from pathlib import Path
import yaml
import joblib
import re
import pandas as pd

from autobenchmark.data import load_data, prepare_data
from autobenchmark.models import train_benchmark_models
from autobenchmark.evaluation import save_and_rank_results, load_and_evaluate_results
from autobenchmark.uncertainty import calculate_prediction_uncertainty


def main():
    parser = argparse.ArgumentParser(description="Autobenchmark Model Training & Benchmarking CLI")
    parser.add_argument('--init', type=str, default='config/init_config.yaml',
                        help='Path to system initialization YAML configuration')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to model training YAML configuration (e.g., config/model/model1.yaml)')
    args = parser.parse_args()
    
    # Load configs
    if not os.path.exists(args.init):
        print(f"Error: Init config not found at: {args.init}")
        return
    if not os.path.exists(args.model):
        print(f"Error: Model config not found at: {args.model}")
        return
        
    with open(args.init, 'r', encoding='utf-8') as f:
        init_cfg = yaml.safe_load(f)
        
    with open(args.model, 'r', encoding='utf-8') as f:
        model_cfg = yaml.safe_load(f)
        
    model_config_name = Path(args.model).stem
    model_cfg['config_name'] = model_config_name
    
    # Locate data config
    data_config_name = model_cfg.get('data_config_name')
    if not data_config_name:
        print("Error: data_config_name not specified in model configuration.")
        return
        
    base_dir = init_cfg.get('system', {}).get('base_dir', 'C:/Github/auto_benchmark')
    data_cfg_path = os.path.join(base_dir, 'config', 'data', f"{data_config_name}.yaml")
    if not os.path.exists(data_cfg_path):
        # Fallback to relative check
        data_cfg_path = os.path.join('config', 'data', f"{data_config_name}.yaml")
        
    if not os.path.exists(data_cfg_path):
        print(f"Error: Data configuration file not found at: {data_cfg_path}")
        return
        
    with open(data_cfg_path, 'r', encoding='utf-8') as f:
        data_cfg = yaml.safe_load(f)
        
    # Ingest and preprocess data
    filepath = data_cfg.get('dataset', {}).get('filepath')
    if not filepath:
        print("Error: filepath not specified in data configuration.")
        return
        
    row_limit = data_cfg.get('dataset', {}).get('row_limit')
    df = load_data(filepath, init_cfg, nrows=row_limit)
    X_train, X_test, y_train, y_test, feat_labels, preprocessor = prepare_data(df, data_cfg, init_cfg)
    print(f"Data prepared: X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
    
    # Resolve output directory
    results_dir = init_cfg.get('paths', {}).get('results_dir', 'results')
    output_dir = os.path.join(base_dir, results_dir, 'model_results', model_config_name)
    
    # Fit models
    df_results, df_preds = train_benchmark_models(
        X_train, y_train, X_test, y_test, model_cfg, output_dir, feat_labels=feat_labels
    )
    
    # Save preprocessor for explainers/future pipelines
    preprocessor_path = os.path.join(output_dir, f"{model_config_name}_preprocessor.joblib")
    joblib.dump(preprocessor, preprocessor_path)
    print(f"  Saved preprocessor pipeline -> {preprocessor_path}")
    
    # Calculate uncertainty metrics for fitted models
    print("\n--- Calculating Prediction Uncertainty ---")
    models_subfolder = os.path.join(output_dir, 'models')
    uncertainty_subfolder = os.path.join(output_dir, 'uncertainty')
    os.makedirs(uncertainty_subfolder, exist_ok=True)
    
    overall_unc_records = []
    
    for idx, row in df_results.iterrows():
        model_name = row['Model']
        safe_name = re.sub(r'[^a-zA-Z0-9_-]', '_', model_name)
        model_file = os.path.join(models_subfolder, f"{model_config_name}_{safe_name}.joblib")
        if os.path.exists(model_file):
            try:
                model = joblib.load(model_file)
                df_unc = calculate_prediction_uncertainty(model, model_name, X_test, y_test, uncertainty_subfolder, model_cfg)
                if df_unc is not None:
                    mean_entropy = df_unc['Shannon_Entropy'].mean()
                    std_entropy = df_unc['Shannon_Entropy'].std()
                    mean_conf = df_unc['Confidence_Score'].mean()
                    mean_margin = df_unc['Confidence_Margin'].mean()
                    amb_rate_50 = (df_unc['Shannon_Entropy'] > 0.5).mean()
                    amb_rate_80 = (df_unc['Shannon_Entropy'] > 0.8).mean()
                    
                    overall_unc_records.append({
                        'Model': model_name,
                        'Mean_Shannon_Entropy': mean_entropy,
                        'StdDev_Shannon_Entropy': std_entropy,
                        'Mean_Confidence_Score': mean_conf,
                        'Mean_Confidence_Margin': mean_margin,
                        'Ambiguity_Rate_Entropy_Gt_0.5': amb_rate_50,
                        'Ambiguity_Rate_Entropy_Gt_0.8': amb_rate_80
                    })
            except Exception as e:
                print(f"  Could not compute uncertainty for {model_name}: {e}")
                
    if overall_unc_records:
        df_overall_unc = pd.DataFrame(overall_unc_records).sort_values(by='Mean_Shannon_Entropy', ascending=True)
        overall_unc_path = os.path.join(output_dir, 'overall_uncertainty_summary.csv')
        df_overall_unc.to_csv(overall_unc_path, index=False)
        print(f"\n  Saved overall uncertainty summary -> {overall_unc_path}")
                
    # Sort and rank results
    best_model_info = save_and_rank_results(df_results, model_cfg, output_dir)
    
    # Load and display model rankings
    eval_csv_path = os.path.join(output_dir, f"{model_config_name}_evaluation.csv")
    load_and_evaluate_results(eval_csv_path, model_cfg.get('optimize_metric', 'f1'))
    
    # Generate evaluation plots (bar charts, confidence curves)
    from autobenchmark.evaluation import generate_evaluation_plots
    generate_evaluation_plots(df_results, output_dir, model_cfg)
    
    print(f"\nBest Model: {best_model_info['Model']} ({best_model_info['Metric_Name']} = {best_model_info['Metric_Value']:.4f})")
    print("Benchmarking execution complete!")


if __name__ == "__main__":
    main()
