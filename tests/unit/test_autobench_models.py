"""Unit tests for the class-weight and scaler wrappers in autobenchmark.models.

Covers docs/CODE_REVIEW_ISSUES.md §7.3 (``class_weight='balanced'``) and
§7.4 (``StandardScaler`` pipeline for non-tree models). The tests never
fit a real model - they only inspect the constructor / pipeline
structure of what ``get_models_dict`` returns, which is fast enough to
run on any laptop.
"""

from __future__ import annotations

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier

from autobenchmark.models import get_models_dict


def _underlying_estimator(model):
    """Peel one level of Pipeline wrapping to reach the raw estimator."""
    if isinstance(model, Pipeline):
        return model.named_steps["estimator"]
    return model


def test_class_weight_balanced_is_applied_to_lr() -> None:
    """LogisticRegression is constructed with ``class_weight='balanced'``."""
    models = get_models_dict(num_classes=2)
    lr = _underlying_estimator(models["Logistic Regression"])
    assert isinstance(lr, LogisticRegression)
    assert lr.class_weight == "balanced"


def test_class_weight_balanced_subsample_on_random_forest() -> None:
    """Tree ensembles use ``balanced_subsample`` (per-bootstrap-sample balancing)."""
    models = get_models_dict(num_classes=2)
    rf = _underlying_estimator(models["Random Forest"])
    assert isinstance(rf, RandomForestClassifier)
    assert rf.class_weight == "balanced_subsample"


def test_class_weight_balanced_on_svm_variants() -> None:
    """Both SVM variants get ``class_weight='balanced'``."""
    models = get_models_dict(num_classes=2)
    linear_svc = _underlying_estimator(models["Support Vector Machine - LinearSVC"])
    rbf_svc = _underlying_estimator(models["Support Vector Machine - RBF SVC"])
    assert isinstance(linear_svc, LinearSVC)
    assert isinstance(rbf_svc, SVC)
    assert linear_svc.class_weight == "balanced"
    assert rbf_svc.class_weight == "balanced"


def test_class_weight_balanced_can_be_disabled() -> None:
    """The flag ``class_weight_balanced=False`` restores the pre-fix behaviour."""
    models = get_models_dict(num_classes=2, class_weight_balanced=False)
    lr = _underlying_estimator(models["Logistic Regression"])
    # ``class_weight=None`` is the sklearn default.
    assert lr.class_weight is None


def test_scaler_wraps_non_tree_models() -> None:
    """Non-tree models are wrapped in ``Pipeline(StandardScaler, model)``."""
    models = get_models_dict(num_classes=2)
    for name in [
        "Logistic Regression",
        "Support Vector Machine - LinearSVC",
        "Support Vector Machine - RBF SVC",
        "Neural Network - MLPClassifier",
        "KNN",
    ]:
        assert isinstance(models[name], Pipeline), f"{name} should be wrapped"
        steps = models[name].named_steps
        assert "scaler" in steps
        assert isinstance(steps["scaler"], StandardScaler)
        assert "estimator" in steps


def test_scaler_skipped_for_tree_models() -> None:
    """Tree-based models are scale-invariant and remain unwrapped."""
    models = get_models_dict(num_classes=2)
    for name in [
        "Decision Tree",
        "Random Forest",
        "Extra Trees Classifier",
        "XGBoost",
        "LightGBM",
        "Gradient Boosting Classifier",
    ]:
        assert not isinstance(
            models[name], Pipeline
        ), f"{name} is scale-invariant and should not be scaled"


def test_scale_features_can_be_disabled() -> None:
    """The flag ``scale_features=False`` returns bare estimators."""
    models = get_models_dict(num_classes=2, scale_features=False)
    lr = models["Logistic Regression"]
    assert not isinstance(lr, Pipeline)
    assert isinstance(lr, LogisticRegression)


def test_lda_gets_scaled_but_no_class_weight() -> None:
    """LDA has no ``class_weight`` kwarg but still benefits from scaling."""
    models = get_models_dict(num_classes=2)
    wrapped = models["Linear Discriminant Analysis"]
    assert isinstance(wrapped, Pipeline)
    inner = wrapped.named_steps["estimator"]
    assert isinstance(inner, LinearDiscriminantAnalysis)


def test_gaussian_nb_not_wrapped_or_weighted() -> None:
    """Gaussian NB has no class_weight kwarg and does not need scaling."""
    models = get_models_dict(num_classes=2)
    gnb = models["Gaussian Naive Bayes"]
    assert not isinstance(gnb, Pipeline)
    assert isinstance(gnb, GaussianNB)


def test_decision_tree_gets_class_weight_but_no_scaler() -> None:
    """Decision tree has ``class_weight`` but is scale-invariant."""
    models = get_models_dict(num_classes=2)
    dt = models["Decision Tree"]
    assert not isinstance(dt, Pipeline)
    assert isinstance(dt, DecisionTreeClassifier)
    assert dt.class_weight == "balanced"


def test_extra_trees_classifier_gets_balanced_subsample() -> None:
    """Extra Trees Classifier uses balanced_subsample (per-bootstrap balancing)."""
    models = get_models_dict(num_classes=2)
    etc = models["Extra Trees Classifier"]
    assert isinstance(etc, ExtraTreesClassifier)
    assert etc.class_weight == "balanced_subsample"


def test_sgd_classifier_gets_class_weight_and_scaler() -> None:
    """SGD Classifier gets both class_weight and StandardScaler wrapping."""
    models = get_models_dict(num_classes=2)
    wrapped = models["SGD Classifier"]
    assert isinstance(wrapped, Pipeline)
    inner = wrapped.named_steps["estimator"]
    assert isinstance(inner, SGDClassifier)
    assert inner.class_weight == "balanced"
