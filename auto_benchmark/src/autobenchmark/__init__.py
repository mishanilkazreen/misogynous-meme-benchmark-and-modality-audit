"""
Autobenchmark - Automated machine learning benchmarking and explainability package.
"""

__version__ = "0.1.0"

from .data import load_data, prepare_data
from .data_analysis import run_data_profiling
from .models import train_benchmark_models
from .evaluation import save_and_rank_results, load_and_evaluate_results
from .uncertainty import calculate_prediction_uncertainty
from .explain import (
    run_shap_explanations,
    run_lime_explanations,
    run_native_importance,
    run_permutation_importance,
    run_surrogate_model
)
