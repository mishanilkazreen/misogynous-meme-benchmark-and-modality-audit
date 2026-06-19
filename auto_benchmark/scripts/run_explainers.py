#!/usr/bin/env python3
"""
CLI script to run explainability pipelines on benchmarked models.
Usage:
  python scripts/run_explainers.py --init config/init_config.yaml --explainer config/explainer/explainer1.yaml
"""

import os
import argparse
from pathlib import Path
import yaml
import numpy as np
import pandas as pd
import joblib
import re

from autobenchmark.explain import (
    run_shap_explanations,
    run_lime_explanations,
    run_native_importance,
    run_permutation_importance,
    run_surrogate_model,
    run_text_explainers
)


def _safe_filename(name):
    return re.sub(r'[^a-zA-Z0-9_-]', '_', name)


def main():
    parser = argparse.ArgumentParser(description="Autobenchmark Model Explainability CLI")
    parser.add_argument('--init', type=str, default='config/init_config.yaml',
                        help='Path to system initialization YAML configuration')
    parser.add_argument('--explainer', type=str, required=True,
                        help='Path to explainer YAML configuration (e.g., config/explainer/explainer1.yaml)')
    args = parser.parse_args()
    
    # Load configs
    if not os.path.exists(args.init):
        print(f"Error: Init config not found at: {args.init}")
        return
    if not os.path.exists(args.explainer):
        print(f"Error: Explainer config not found at: {args.explainer}")
        return
        
    with open(args.init, 'r', encoding='utf-8') as f:
        init_cfg = yaml.safe_load(f)
        
    with open(args.explainer, 'r', encoding='utf-8') as f:
        explainer_cfg = yaml.safe_load(f)
        
    explainer_config_name = Path(args.explainer).stem
    
    model_config_name = explainer_cfg.get('model_config_name')
    if not model_config_name:
        print("Error: model_config_name not specified in explainer configuration.")
        return
        
    base_dir = init_cfg.get('system', {}).get('base_dir', 'C:/Github/auto_benchmark')
    results_dir = init_cfg.get('paths', {}).get('results_dir', 'results')
    
    # Locate trained models and datasets
    model_results_dir = os.path.join(base_dir, results_dir, 'model_results', model_config_name)
    data_path = os.path.join(model_results_dir, f"{model_config_name}_data.npz")
    eval_csv_path = os.path.join(model_results_dir, f"{model_config_name}_evaluation.csv")
    
    if not os.path.exists(data_path):
        print(f"Error: Preprocessed data file not found at: {data_path}. Run benchmark training first.")
        return
    if not os.path.exists(eval_csv_path):
        print(f"Error: Evaluation file not found at: {eval_csv_path}. Run benchmark training first.")
        return
        
    # Load prepared data
    data = np.load(data_path, allow_pickle=True)
    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']
    feat_labels = data['feat_labels'].tolist() if 'feat_labels' in data else None
    
    # Load evaluation df to find models
    df_eval = pd.read_csv(eval_csv_path)
    if df_eval.empty:
        print("Error: Model evaluation table is empty.")
        return
        
    output_dir = os.path.join(base_dir, results_dir, 'explanation_results', explainer_config_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Choose models to explain
    model_to_explain = explainer_cfg.get('model_to_explain', 'best')
    
    if model_to_explain == 'best':
        top_model_name = df_eval.iloc[0]['Model']
        print(f"Top performing model identified: {top_model_name}")
        models_to_explain = [top_model_name]
    elif isinstance(model_to_explain, list):
        models_to_explain = model_to_explain
    else:
        models_to_explain = [model_to_explain]
    
    # Check if the dataset is text type
    is_text = False
    data_config_name = None
    # Read model config
    model_cfg_path = os.path.join(base_dir, 'config', 'model', f"{model_config_name}.yaml")
    if os.path.exists(model_cfg_path):
        with open(model_cfg_path, 'r', encoding='utf-8') as f:
            model_cfg = yaml.safe_load(f)
            data_config_name = model_cfg.get('data_config_name')
            
    if data_config_name:
        data_cfg_path = os.path.join(base_dir, 'config', 'data', f"{data_config_name}.yaml")
        if os.path.exists(data_cfg_path):
            with open(data_cfg_path, 'r', encoding='utf-8') as f:
                data_cfg = yaml.safe_load(f)
                is_text = data_cfg.get('dataset', {}).get('data_type') == 'text'
                
    texts_train, texts_test = None, None
    if is_text:
        from autobenchmark.data import load_data, get_raw_text_splits
        filepath = data_cfg.get('dataset', {}).get('filepath')
        row_limit = data_cfg.get('dataset', {}).get('row_limit')
        df_raw = load_data(filepath, init_cfg, nrows=row_limit)
        texts_train, texts_test, _, _ = get_raw_text_splits(df_raw, data_cfg)
        
    # Load and run explainers for selected models
    for model_name in models_to_explain:
        safe_name = _safe_filename(model_name)
        models_subfolder = os.path.join(model_results_dir, 'models')
        model_file = os.path.join(models_subfolder, f"{model_config_name}_{safe_name}.joblib")
        
        if not os.path.exists(model_file):
            print(f"Warning: Model file not found at {model_file}. Skipping.")
            continue
            
        print(f"\n--- Explaining Model: {model_name} ---")
        
        # If it is a text dataset, run textual explainers (highlighting LIME and SHAP)
        if is_text:
            preprocessor_path = os.path.join(model_results_dir, f"{model_config_name}_preprocessor.joblib")
            preprocessor = joblib.load(preprocessor_path) if os.path.exists(preprocessor_path) else None
            run_text_explainers(model, model_name, texts_train, texts_test, y_test, preprocessor, output_dir, explainer_cfg, data_cfg=data_cfg)
            continue
        
        explainers_cfg = explainer_cfg.get('explainers', 'all')
        
        # 1. Native feature importance (for tree models only)
        if explainers_cfg == 'all' or 'native_importance' in explainers_cfg:
            run_native_importance(model, model_name, feat_labels, output_dir)
            
        # 2. Permutation importance
        if explainers_cfg == 'all' or 'permutation_importance' in explainers_cfg:
            perm_cfg = explainer_cfg.get('permutation_settings', {})
            run_permutation_importance(model, model_name, X_test, y_test, feat_labels, output_dir, perm_cfg)
            
        # 3. Decision Tree Surrogate Model
        if explainers_cfg == 'all' or 'surrogate' in explainers_cfg:
            surr_cfg = explainer_cfg.get('surrogate_settings', {})
            run_surrogate_model(model, model_name, X_train, X_test, feat_labels, output_dir, surr_cfg)
            
        # 4. SHAP
        if explainers_cfg == 'all' or 'shap' in explainers_cfg:
            shap_cfg = explainer_cfg.get('shap_settings', {})
            run_shap_explanations(model, model_name, X_train, X_test, feat_labels, output_dir, shap_cfg)
            
        # 5. LIME
        if explainers_cfg == 'all' or 'lime' in explainers_cfg:
            lime_cfg = explainer_cfg.get('lime_settings', {})
            run_lime_explanations(model, model_name, X_train, X_test, y_test, feat_labels, output_dir, lime_cfg)
            
    print(f"\nExplainability plots generated under: {output_dir}")


if __name__ == "__main__":
    main()
