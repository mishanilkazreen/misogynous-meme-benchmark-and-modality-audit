import argparse
import os
from pathlib import Path

import yaml


def run_profile(args):
    from autobenchmark.data import load_data
    from autobenchmark.data_analysis import run_data_profiling

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
    filepath = data_cfg.get("dataset", {}).get("filepath")
    if not filepath:
        print("Error: filepath not specified in data configuration.")
        return

    row_limit = data_cfg.get("dataset", {}).get("row_limit")
    df = load_data(filepath, init_cfg, nrows=row_limit)
    print(f"Data loaded successfully. Shape: {df.shape}")

    base_dir = init_cfg.get("system", {}).get("base_dir", "C:/Github/auto_benchmark")
    results_dir = init_cfg.get("paths", {}).get("results_dir", "results")
    output_dir = os.path.join(base_dir, results_dir, "data_analysis", data_config_name)

    run_data_profiling(df, data_cfg, output_dir)
    print("\nData analysis complete!")


def run_train(args):
    import re

    import joblib
    import pandas as pd

    from autobenchmark.data import load_data, prepare_data
    from autobenchmark.evaluation import load_and_evaluate_results, save_and_rank_results
    from autobenchmark.models import train_benchmark_models
    from autobenchmark.uncertainty import calculate_prediction_uncertainty

    if not os.path.exists(args.init):
        print(f"Error: Init config not found at: {args.init}")
        return
    if not os.path.exists(args.model):
        print(f"Error: Model config not found at: {args.model}")
        return

    with open(args.init, encoding="utf-8") as f:
        init_cfg = yaml.safe_load(f)
    with open(args.model, encoding="utf-8") as f:
        model_cfg = yaml.safe_load(f)

    model_config_name = Path(args.model).stem
    model_cfg["config_name"] = model_config_name

    data_config_name = model_cfg.get("data_config_name")
    if not data_config_name:
        print("Error: data_config_name not specified in model configuration.")
        return

    base_dir = init_cfg.get("system", {}).get("base_dir", "C:/Github/auto_benchmark")
    data_cfg_path = os.path.join(base_dir, "config", "data", f"{data_config_name}.yaml")
    if not os.path.exists(data_cfg_path):
        data_cfg_path = os.path.join("config", "data", f"{data_config_name}.yaml")
    if not os.path.exists(data_cfg_path):
        print(f"Error: Data configuration file not found at: {data_cfg_path}")
        return

    with open(data_cfg_path, encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)

    filepath = data_cfg.get("dataset", {}).get("filepath")
    if not filepath:
        print("Error: filepath not specified in data configuration.")
        return

    row_limit = data_cfg.get("dataset", {}).get("row_limit")
    df = load_data(filepath, init_cfg, nrows=row_limit)
    X_train, X_test, y_train, y_test, feat_labels, preprocessor = prepare_data(
        df, data_cfg, init_cfg
    )
    print(f"Data prepared: X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")

    results_dir = init_cfg.get("paths", {}).get("results_dir", "results")
    output_dir = os.path.join(base_dir, results_dir, "model_results", model_config_name)

    df_results, _df_preds = train_benchmark_models(
        X_train, y_train, X_test, y_test, model_cfg, output_dir, feat_labels=feat_labels
    )

    preprocessor_path = os.path.join(output_dir, f"{model_config_name}_preprocessor.joblib")
    joblib.dump(preprocessor, preprocessor_path)
    print(f"  Saved preprocessor pipeline -> {preprocessor_path}")

    print("\n--- Calculating Prediction Uncertainty ---")
    models_subfolder = os.path.join(output_dir, "models")
    uncertainty_subfolder = os.path.join(output_dir, "uncertainty")
    os.makedirs(uncertainty_subfolder, exist_ok=True)

    overall_unc_records = []

    for _idx, row in df_results.iterrows():
        model_name = row["Model"]
        safe_name = re.sub(r"[^a-zA-Z0-9_-]", "_", model_name)
        model_file = os.path.join(models_subfolder, f"{model_config_name}_{safe_name}.joblib")
        if os.path.exists(model_file):
            try:
                model = joblib.load(model_file)
                df_unc = calculate_prediction_uncertainty(
                    model, model_name, X_test, y_test, uncertainty_subfolder, model_cfg
                )
                if df_unc is not None:
                    mean_entropy = df_unc["Shannon_Entropy"].mean()
                    std_entropy = df_unc["Shannon_Entropy"].std()
                    mean_conf = df_unc["Confidence_Score"].mean()
                    mean_margin = df_unc["Confidence_Margin"].mean()
                    amb_rate_50 = (df_unc["Shannon_Entropy"] > 0.5).mean()
                    amb_rate_80 = (df_unc["Shannon_Entropy"] > 0.8).mean()

                    overall_unc_records.append(
                        {
                            "Model": model_name,
                            "Mean_Shannon_Entropy": mean_entropy,
                            "StdDev_Shannon_Entropy": std_entropy,
                            "Mean_Confidence_Score": mean_conf,
                            "Mean_Confidence_Margin": mean_margin,
                            "Ambiguity_Rate_Entropy_Gt_0.5": amb_rate_50,
                            "Ambiguity_Rate_Entropy_Gt_0.8": amb_rate_80,
                        }
                    )
            except Exception as e:
                print(f"  Could not compute uncertainty for {model_name}: {e}")

    if overall_unc_records:
        df_overall_unc = pd.DataFrame(overall_unc_records).sort_values(
            by="Mean_Shannon_Entropy", ascending=True
        )
        overall_unc_path = os.path.join(output_dir, "overall_uncertainty_summary.csv")
        df_overall_unc.to_csv(overall_unc_path, index=False)
        print(f"\n  Saved overall uncertainty summary -> {overall_unc_path}")

    best_model_info = save_and_rank_results(df_results, model_cfg, output_dir)

    eval_csv_path = os.path.join(output_dir, f"{model_config_name}_evaluation.csv")
    load_and_evaluate_results(eval_csv_path, model_cfg.get("optimize_metric", "f1"))

    from autobenchmark.evaluation import generate_evaluation_plots

    generate_evaluation_plots(df_results, output_dir, model_cfg)

    print(
        f"\nBest Model: {best_model_info['Model']} ({best_model_info['Metric_Name']} = {best_model_info['Metric_Value']:.4f})"
    )
    print("Benchmarking execution complete!")


def run_explain(args):
    import re

    import joblib
    import numpy as np
    import pandas as pd

    from autobenchmark.explain import (
        run_lime_explanations,
        run_native_importance,
        run_permutation_importance,
        run_shap_explanations,
        run_surrogate_model,
        run_text_explainers,
    )

    if not os.path.exists(args.init):
        print(f"Error: Init config not found at: {args.init}")
        return
    if not os.path.exists(args.explainer):
        print(f"Error: Explainer config not found at: {args.explainer}")
        return

    with open(args.init, encoding="utf-8") as f:
        init_cfg = yaml.safe_load(f)
    with open(args.explainer, encoding="utf-8") as f:
        explainer_cfg = yaml.safe_load(f)

    explainer_config_name = Path(args.explainer).stem
    model_config_name = explainer_cfg.get("model_config_name")
    if not model_config_name:
        print("Error: model_config_name not specified in explainer configuration.")
        return

    base_dir = init_cfg.get("system", {}).get("base_dir", "C:/Github/auto_benchmark")
    results_dir = init_cfg.get("paths", {}).get("results_dir", "results")

    model_results_dir = os.path.join(base_dir, results_dir, "model_results", model_config_name)
    data_path = os.path.join(model_results_dir, f"{model_config_name}_data.npz")
    eval_csv_path = os.path.join(model_results_dir, f"{model_config_name}_evaluation.csv")

    if not os.path.exists(data_path):
        print(
            f"Error: Preprocessed data file not found at: {data_path}. Run benchmark training first."
        )
        return
    if not os.path.exists(eval_csv_path):
        print(
            f"Error: Evaluation file not found at: {eval_csv_path}. Run benchmark training first."
        )
        return

    data = np.load(data_path, allow_pickle=True)
    X_train = data["X_train"]
    X_test = data["X_test"]
    data["y_train"]
    y_test = data["y_test"]
    feat_labels = data["feat_labels"].tolist() if "feat_labels" in data else None

    df_eval = pd.read_csv(eval_csv_path)
    if df_eval.empty:
        print("Error: Model evaluation table is empty.")
        return

    model_to_explain = explainer_cfg.get("model_to_explain", "best")
    if model_to_explain == "best":
        top_model_name = df_eval.iloc[0]["Model"]
        print(f"Top performing model identified: {top_model_name}")
        models_to_explain = [top_model_name]
    elif isinstance(model_to_explain, list):
        models_to_explain = model_to_explain
    else:
        models_to_explain = [model_to_explain]

    output_dir = os.path.join(base_dir, results_dir, "explanation_results", explainer_config_name)
    os.makedirs(output_dir, exist_ok=True)

    # Check if the dataset is text type
    is_text = False
    data_config_name = None
    # Read model config
    model_cfg_path = os.path.join(base_dir, "config", "model", f"{model_config_name}.yaml")
    if os.path.exists(model_cfg_path):
        with open(model_cfg_path, encoding="utf-8") as f:
            model_cfg = yaml.safe_load(f)
            data_config_name = model_cfg.get("data_config_name")

    if data_config_name:
        data_cfg_path = os.path.join(base_dir, "config", "data", f"{data_config_name}.yaml")
        if os.path.exists(data_cfg_path):
            with open(data_cfg_path, encoding="utf-8") as f:
                data_cfg = yaml.safe_load(f)
                is_text = data_cfg.get("dataset", {}).get("data_type") == "text"

    texts_train, texts_test = None, None
    if is_text:
        from autobenchmark.data import get_raw_text_splits, load_data

        filepath = data_cfg.get("dataset", {}).get("filepath")
        row_limit = data_cfg.get("dataset", {}).get("row_limit")
        df_raw = load_data(filepath, init_cfg, nrows=row_limit)
        texts_train, texts_test, _, _ = get_raw_text_splits(df_raw, data_cfg)

    for model_name in models_to_explain:
        safe_name = re.sub(r"[^a-zA-Z0-9_-]", "_", model_name)
        models_subfolder = os.path.join(model_results_dir, "models")
        model_file = os.path.join(models_subfolder, f"{model_config_name}_{safe_name}.joblib")

        if not os.path.exists(model_file):
            print(f"Warning: Model file not found at {model_file}. Skipping.")
            continue

        model = joblib.load(model_file)
        print(f"\n--- Explaining Model: {model_name} ---")

        # If it is a text dataset, run textual explainers (highlighting LIME and SHAP)
        if is_text:
            preprocessor_path = os.path.join(
                model_results_dir, f"{model_config_name}_preprocessor.joblib"
            )
            preprocessor = (
                joblib.load(preprocessor_path) if os.path.exists(preprocessor_path) else None
            )
            run_text_explainers(
                model,
                model_name,
                texts_train,
                texts_test,
                y_test,
                preprocessor,
                output_dir,
                explainer_cfg,
                data_cfg=data_cfg,
            )
            continue

        explainers_cfg = explainer_cfg.get("explainers", "all")

        if explainers_cfg == "all" or "native_importance" in explainers_cfg:
            run_native_importance(model, model_name, feat_labels, output_dir)

        if explainers_cfg == "all" or "permutation_importance" in explainers_cfg:
            perm_cfg = explainer_cfg.get("permutation_settings", {})
            run_permutation_importance(
                model, model_name, X_test, y_test, feat_labels, output_dir, perm_cfg
            )

        if explainers_cfg == "all" or "surrogate" in explainers_cfg:
            surr_cfg = explainer_cfg.get("surrogate_settings", {})
            run_surrogate_model(
                model, model_name, X_train, X_test, feat_labels, output_dir, surr_cfg
            )

        if explainers_cfg == "all" or "shap" in explainers_cfg:
            shap_cfg = explainer_cfg.get("shap_settings", {})
            run_shap_explanations(
                model, model_name, X_train, X_test, feat_labels, output_dir, shap_cfg
            )

        if explainers_cfg == "all" or "lime" in explainers_cfg:
            lime_cfg = explainer_cfg.get("lime_settings", {})
            run_lime_explanations(
                model, model_name, X_train, X_test, y_test, feat_labels, output_dir, lime_cfg
            )

    print(f"\nExplainability plots generated under: {output_dir}")


def run_detect(args):
    """Run YOLO object detection benchmark."""
    import os
    from pathlib import Path

    import yaml

    from autobenchmark.yolo_models import resolve_data_yaml, train_yolo_benchmark

    if not os.path.exists(args.init):
        print(f"Error: Init config not found at: {args.init}")
        return
    if not os.path.exists(args.model):
        print(f"Error: Model config not found at: {args.model}")
        return

    with open(args.init, encoding="utf-8") as f:
        init_cfg = yaml.safe_load(f)
    with open(args.model, encoding="utf-8") as f:
        model_cfg = yaml.safe_load(f)

    model_config_name = Path(args.model).stem
    model_cfg["config_name"] = model_config_name

    # Resolve the data config
    data_config_name = model_cfg.get("data_config_name")
    if not data_config_name:
        print("Error: data_config_name not specified in model configuration.")
        return

    base_dir = init_cfg.get("system", {}).get("base_dir", "C:/Github/auto_benchmark")
    data_cfg_path = os.path.join(base_dir, "config", "data", f"{data_config_name}.yaml")

    if not os.path.exists(data_cfg_path):
        print(f"Error: Data config not found at: {data_cfg_path}")
        return

    with open(data_cfg_path, encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)

    # Verify data_type is image
    data_type = data_cfg.get("dataset", {}).get("data_type", "")
    if data_type != "image":
        print(f"Error: data_type must be 'image' for the detect command, got '{data_type}'.")
        return

    # Resolve data.yaml path
    raw_yaml_path = data_cfg.get("dataset", {}).get("data_yaml_path", "")
    resolved_yaml = resolve_data_yaml(raw_yaml_path, init_cfg)
    data_cfg["dataset"]["data_yaml_path"] = resolved_yaml

    # Get models list
    models_list = model_cfg.get("models_to_run", [])
    if not models_list:
        print("Error: No models specified in models_to_run.")
        return

    # Output directory
    results_dir = init_cfg.get("paths", {}).get("results_dir", "results")
    output_dir = os.path.join(base_dir, results_dir, "model_results", model_config_name)

    print(f"Starting YOLO object detection benchmark: {model_config_name}")
    print(f"Data config: {data_config_name}")
    print(f"Models to benchmark: {len(models_list)}")

    df_results = train_yolo_benchmark(
        models_list=models_list,
        data_cfg=data_cfg,
        model_cfg=model_cfg,
        output_dir=output_dir,
        init_config=init_cfg,
    )

    if df_results is not None and not df_results.empty:
        print(f"\nYOLO benchmarking complete! Results saved to: {output_dir}")
    else:
        print("\nYOLO benchmarking completed with errors. Check output above.")


def run_classify(args):
    """Run Image Classification benchmarking."""
    from autobenchmark.image_classification import run_classification_benchmark

    if not os.path.exists(args.init):
        print(f"Error: Init config not found at: {args.init}")
        return
    if not os.path.exists(args.model):
        print(f"Error: Model config not found at: {args.model}")
        return

    with open(args.init, encoding="utf-8") as f:
        init_cfg = yaml.safe_load(f)
    with open(args.model, encoding="utf-8") as f:
        model_cfg = yaml.safe_load(f)

    model_config_name = Path(args.model).stem
    data_config_name = model_cfg.get("data_config_name")
    if not data_config_name:
        print("Error: data_config_name not specified in model configuration.")
        return

    base_dir = init_cfg.get("system", {}).get("base_dir", "C:/Github/auto_benchmark")
    data_cfg_path = os.path.join(base_dir, "config", "data", f"{data_config_name}.yaml")

    if not os.path.exists(data_cfg_path):
        print(f"Error: Data config not found at: {data_cfg_path}")
        return

    with open(data_cfg_path, encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)

    # Verify data_type is image_classification
    data_type = data_cfg.get("dataset", {}).get("data_type", "")
    if data_type != "image_classification":
        print(
            f"Error: data_type must be 'image_classification' for the classify command, got '{data_type}'."
        )
        return

    results_dir = init_cfg.get("paths", {}).get("results_dir", "results")
    output_dir = os.path.join(base_dir, results_dir, "model_results", model_config_name)

    print(f"Starting Image Classification benchmark: {model_config_name}")
    print(f"Data config: {data_config_name}")
    print(f"Mode: {data_cfg.get('mode', 'extract_features')}")

    try:
        df_results = run_classification_benchmark(
            data_cfg=data_cfg, model_cfg=model_cfg, output_dir=output_dir, init_config=init_cfg
        )
        if df_results is not None and not df_results.empty:
            print(f"\nImage Classification benchmarking complete! Results saved to: {output_dir}")
        else:
            print("\nImage Classification benchmarking completed but returned empty results.")
    except Exception as e:
        import traceback

        print(f"\nError running Image Classification benchmark: {e}")
        traceback.print_exc()


def run_vlmbench(args):
    """Run VLM benchmarking."""
    from autobenchmark.vlm_bench import run_vlm_benchmark

    if not os.path.exists(args.init):
        print(f"Error: Init config not found at: {args.init}")
        return
    if not os.path.exists(args.model):
        print(f"Error: Model config not found at: {args.model}")
        return

    with open(args.init, encoding="utf-8") as f:
        init_cfg = yaml.safe_load(f)
    with open(args.model, encoding="utf-8") as f:
        model_cfg = yaml.safe_load(f)

    model_config_name = Path(args.model).stem
    data_config_name = model_cfg.get("data_config_name")
    if not data_config_name:
        print("Error: data_config_name not specified in VLM configuration.")
        return

    base_dir = init_cfg.get("system", {}).get("base_dir", "C:/Github/auto_benchmark")
    data_cfg_path = os.path.join(base_dir, "config", "data", f"{data_config_name}.yaml")

    if not os.path.exists(data_cfg_path):
        print(f"Error: Data config not found at: {data_cfg_path}")
        return

    with open(data_cfg_path, encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)

    # Verify data_type is vlm
    data_type = data_cfg.get("dataset", {}).get("data_type", "")
    if data_type != "vlm":
        print(f"Error: data_type must be 'vlm' for the vlmbench command, got '{data_type}'.")
        return

    results_dir = init_cfg.get("paths", {}).get("results_dir", "results")
    output_dir = os.path.join(base_dir, results_dir, "model_results", model_config_name)

    print(f"Starting VLM benchmark: {model_config_name}")
    print(f"Data config: {data_config_name}")

    try:
        df_results = run_vlm_benchmark(
            data_cfg=data_cfg, model_cfg=model_cfg, output_dir=output_dir, init_config=init_cfg
        )
        if df_results is not None and not df_results.empty:
            print(f"\nVLM benchmarking complete! Results saved to: {output_dir}")
        else:
            print("\nVLM benchmarking completed but returned empty results.")
    except Exception as e:
        import traceback

        print(f"\nError running VLM benchmark: {e}")
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description="Unified Autobenchmark Command Line Interface")
    subparsers = parser.add_subparsers(dest="command", required=True, help="Autobenchmark commands")

    # 1. Profile command
    parser_profile = subparsers.add_parser("profile", help="Profile a dataset prior to training")
    parser_profile.add_argument(
        "--init",
        type=str,
        default="config/init_config.yaml",
        help="Path to system initialization YAML configuration",
    )
    parser_profile.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to data configuration (e.g. config/data/data1.yaml)",
    )

    # 2. Train command
    parser_train = subparsers.add_parser(
        "train", help="Train and benchmark ML models on tabular/text data"
    )
    parser_train.add_argument(
        "--init",
        type=str,
        default="config/init_config.yaml",
        help="Path to system initialization YAML configuration",
    )
    parser_train.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to model training configuration (e.g. config/model/model1.yaml)",
    )

    # 3. Explain command
    parser_explain = subparsers.add_parser(
        "explain", help="Generate explanations for a trained model"
    )
    parser_explain.add_argument(
        "--init",
        type=str,
        default="config/init_config.yaml",
        help="Path to system initialization YAML configuration",
    )
    parser_explain.add_argument(
        "--explainer",
        type=str,
        required=True,
        help="Path to explainer configuration (e.g. config/explainer/explainer1.yaml)",
    )

    # 4. Detect command (YOLO object detection benchmark)
    parser_detect = subparsers.add_parser(
        "detect", help="Benchmark YOLO object detection models on an image dataset"
    )
    parser_detect.add_argument(
        "--init",
        type=str,
        default="config/init_config.yaml",
        help="Path to system initialization YAML configuration",
    )
    parser_detect.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to YOLO model config (e.g. config/model/yolo_model1.yaml)",
    )

    # 5. Classify command (Image Classification benchmark)
    parser_classify = subparsers.add_parser(
        "classify", help="Benchmark Image Classification models on a directory of images"
    )
    parser_classify.add_argument(
        "--init",
        type=str,
        default="config/init_config.yaml",
        help="Path to system initialization YAML configuration",
    )
    parser_classify.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to image classification model config (e.g. config/model/image_class_model1.yaml)",
    )

    # 6. VLM benchmarking command
    parser_vlmbench = subparsers.add_parser(
        "vlmbench", help="Benchmark Vision-Language Models (VLMs) on an image dataset"
    )
    parser_vlmbench.add_argument(
        "--init",
        type=str,
        default="config/init_config.yaml",
        help="Path to system initialization YAML configuration",
    )
    parser_vlmbench.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to VLM model config (e.g. config/model/vlm_model1.yaml)",
    )

    args = parser.parse_args()

    if args.command == "profile":
        run_profile(args)
    elif args.command == "train":
        run_train(args)
    elif args.command == "explain":
        run_explain(args)
    elif args.command == "detect":
        run_detect(args)
    elif args.command == "classify":
        run_classify(args)
    elif args.command == "vlmbench":
        run_vlmbench(args)


if __name__ == "__main__":
    main()
