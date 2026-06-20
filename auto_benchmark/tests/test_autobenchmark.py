"""
Unit tests for the autobenchmark package.
"""

import os
import shutil
import unittest

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

# Import package components
from autobenchmark.data import prepare_data
from autobenchmark.data_analysis import run_data_profiling
from autobenchmark.evaluation import save_and_rank_results
from autobenchmark.models import train_single_model
from autobenchmark.uncertainty import calculate_prediction_uncertainty


class TestAutobenchmark(unittest.TestCase):
    def setUp(self):
        # Create a mock dataset for testing
        np.random.seed(42)
        n_samples = 100

        self.df = pd.DataFrame(
            {
                "feature_num_1": np.random.randn(n_samples),
                "feature_num_2": np.random.randn(n_samples) * 10,
                "feature_cat_1": np.random.choice(["GroupA", "GroupB", "GroupC"], size=n_samples),
                "feature_cat_2": np.random.choice(["Low", "Medium", "High"], size=n_samples),
                "target": np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3]),
            }
        )

        # Inject some missing values
        self.df.loc[5:10, "feature_num_1"] = np.nan
        self.df.loc[15:20, "feature_cat_1"] = np.nan

        self.data_config = {
            "dataset": {
                "target_column": "target",
                "classification_type": "binary",
                "features": "all",
                "exclude_columns": [],
            },
            "preprocessing": {
                "missing_value_handling": "mean",
                "scaling": "standard",
                "categorical_encoding": "onehot",
                "use_smote": False,
                "test_size": 0.2,
                "random_state": 42,
            },
        }

        self.model_config = {
            "config_name": "test_model",
            "optimize_metric": "f1",
            "hp_optimization": "none",
            "cv_settings": {"run_cv": True, "folds": 3},
        }

    def test_prepare_data(self):
        X_train, X_test, y_train, y_test, feat_labels, _preprocessor = prepare_data(
            self.df, self.data_config
        )

        # Verify splits shape
        self.assertEqual(len(X_train), 80)
        self.assertEqual(len(X_test), 20)
        self.assertEqual(len(y_train), 80)
        self.assertEqual(len(y_test), 20)

        # Verify no missing values remaining in preprocessed data
        self.assertFalse(np.isnan(X_train).any())
        self.assertFalse(np.isnan(X_test).any())

        # One-hot encoding of feature_cat_1 (3 categories) + feature_cat_2 (3 categories) + 2 numeric
        # Should result in at least 5 features
        self.assertGreater(len(feat_labels), 3)
        self.assertEqual(X_train.shape[1], len(feat_labels))

    def test_data_profiling(self):
        output_dir = "test_results_data_analysis"
        summary = run_data_profiling(self.df, self.data_config, output_dir)

        # Check that outputs are created
        self.assertTrue(os.path.exists(summary["profile_path"]))
        self.assertTrue(os.path.exists(summary["report_path"]))
        self.assertTrue(os.path.exists(os.path.join(output_dir, "target_distribution.csv")))
        self.assertTrue(os.path.exists(os.path.join(output_dir, "correlation_matrix.csv")))

        # Clean up files
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir, ignore_errors=True)

    def test_train_single_model_and_uncertainty(self):
        X_train, X_test, y_train, y_test, _feat_labels, _ = prepare_data(self.df, self.data_config)
        model = LogisticRegression(random_state=42)

        fitted_model, _preds, metrics = train_single_model(
            "Logistic Regression",
            model,
            X_train,
            y_train,
            X_test,
            y_test,
            self.model_config,
            is_multiclass=False,
        )

        # Verify metrics shape and content
        self.assertIn("Accuracy", metrics)
        self.assertIn("F1", metrics)
        self.assertIn("CV_Score_Mean", metrics)

        # Calculate uncertainty
        output_dir = "test_results_uncertainty"
        df_unc = calculate_prediction_uncertainty(
            fitted_model, "Logistic Regression", X_test, y_test, output_dir, self.model_config
        )

        self.assertIsNotNone(df_unc)
        self.assertEqual(len(df_unc), len(y_test))
        self.assertIn("Shannon_Entropy", df_unc.columns)
        self.assertIn("Confidence_Score", df_unc.columns)

        # Clean up files
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir, ignore_errors=True)

    def test_save_and_rank_results(self):
        df_results = pd.DataFrame(
            [
                {
                    "Model": "ModelA",
                    "Accuracy": 0.85,
                    "F1": 0.82,
                    "Training_Time": 0.1,
                    "Hyperparameters": "default",
                },
                {
                    "Model": "ModelB",
                    "Accuracy": 0.90,
                    "F1": 0.89,
                    "Training_Time": 0.2,
                    "Hyperparameters": "default",
                },
                {
                    "Model": "ModelC",
                    "Accuracy": 0.78,
                    "F1": 0.75,
                    "Training_Time": 0.05,
                    "Hyperparameters": "default",
                },
            ]
        )

        output_dir = "test_results_eval"
        best_info = save_and_rank_results(df_results, self.model_config, output_dir)

        # Best model should be ModelB because F1 is 0.89 (metric chosen is f1)
        self.assertEqual(best_info["Model"], "ModelB")
        self.assertEqual(best_info["Metric_Value"], 0.89)

        # Verify evaluation file saved
        eval_file = os.path.join(output_dir, "test_model_evaluation.csv")
        self.assertTrue(os.path.exists(eval_file))

        # Clean up files
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir, ignore_errors=True)

    def test_yolo_helpers(self):
        from autobenchmark.yolo_models import _compute_f1, _safe_filename, resolve_data_yaml

        # Test safe filename
        self.assertEqual(_safe_filename("YOLOv8 nano"), "YOLOv8_nano")
        self.assertEqual(_safe_filename("YOLO/v8:s"), "YOLO_v8_s")

        # Test F1 computation
        self.assertAlmostEqual(_compute_f1(0.8, 0.6), 0.6857142857142857)
        self.assertEqual(_compute_f1(0.0, 0.0), 0.0)

        # Test resolve data yaml
        init_cfg = {"system": {"base_dir": "/fake/base/dir"}}
        # If absolute path, returns as is
        self.assertEqual(resolve_data_yaml("/absolute/path.yaml", init_cfg), "/absolute/path.yaml")
        # If relative path and does not exist, defaults back to relative path
        self.assertEqual(resolve_data_yaml("relative.yaml", init_cfg), "relative.yaml")


if __name__ == "__main__":
    unittest.main()
