"""
Autobenchmark - Automated machine learning benchmarking and explainability package.
"""

__version__ = "0.1.0"

from .data import load_data as load_data
from .data import prepare_data as prepare_data
from .data_analysis import run_data_profiling as run_data_profiling
from .evaluation import load_and_evaluate_results as load_and_evaluate_results
from .evaluation import save_and_rank_results as save_and_rank_results
from .explain import (
    run_lime_explanations as run_lime_explanations,
)
from .explain import (
    run_native_importance as run_native_importance,
)
from .explain import (
    run_permutation_importance as run_permutation_importance,
)
from .explain import (
    run_shap_explanations as run_shap_explanations,
)
from .explain import (
    run_surrogate_model as run_surrogate_model,
)
from .models import train_benchmark_models as train_benchmark_models
from .uncertainty import calculate_prediction_uncertainty as calculate_prediction_uncertainty
