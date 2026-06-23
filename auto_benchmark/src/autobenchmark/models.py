"""
Model training and hyperparameter search module for the autobenchmark package.
Defines ML estimators, parameter grids, cross-validation, and fitting loops.
"""

import datetime
import os
import re
from timeit import default_timer as timer

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn import discriminant_analysis, linear_model, svm
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    cross_val_score,
)
from sklearn.naive_bayes import BernoulliNB, ComplementNB, GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from xgboost import XGBClassifier


def get_models_dict(num_classes=2):
    """
    Return a dictionary of model name -> model instance.
    Adjust parameters slightly if multiclass vs binary.
    """
    xgb_kwargs = {}
    try:
        import torch

        if torch.cuda.is_available():
            xgb_kwargs["device"] = "cuda"
            xgb_kwargs["tree_method"] = "hist"
    except Exception:
        pass

    if num_classes > 2:
        xgb_kwargs["objective"] = "multi:softprob"
        xgb_kwargs["num_class"] = num_classes

    models = {
        "KNN": KNeighborsClassifier(),
        "Logistic Regression": linear_model.LogisticRegression(max_iter=1000, random_state=42),
        "SGD Classifier": linear_model.SGDClassifier(random_state=42),
        "Ridge Classifier": linear_model.RidgeClassifier(random_state=42),
        "Perceptron": linear_model.Perceptron(random_state=42),
        "PassiveAggressiveClassifier": linear_model.PassiveAggressiveClassifier(random_state=42),
        "Linear Discriminant Analysis": discriminant_analysis.LinearDiscriminantAnalysis(),
        "Quadratic Discriminant Analysis": discriminant_analysis.QuadraticDiscriminantAnalysis(),
        "Support Vector Machine - LinearSVC": svm.LinearSVC(random_state=42),
        "Support Vector Machine - RBF SVC": svm.SVC(probability=True, random_state=42),
        "Neural Network - MLPClassifier": MLPClassifier(max_iter=500, random_state=42),
        "Extra Tree Classifier": ExtraTreeClassifier(random_state=42),
        "Bernoulli Naive Bayes": BernoulliNB(),
        "Gaussian Naive Bayes": GaussianNB(),
        "Multinomial Naive Bayes": MultinomialNB(),
        "Complement Naive Bayes": ComplementNB(),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "Hist Gradient Boosting Classifier": HistGradientBoostingClassifier(random_state=42),
        "Gradient Boosting Classifier": GradientBoostingClassifier(random_state=42),
        "XGBoost": XGBClassifier(verbosity=0, random_state=42, **xgb_kwargs),
        "LightGBM": lgb.LGBMClassifier(force_col_wise=True, n_jobs=1, verbose=-1, random_state=42),
        "Extra Trees Classifier": ExtraTreesClassifier(random_state=42),
        "AdaBoost Classifier": AdaBoostClassifier(random_state=42),
        "Bagging Classifier": BaggingClassifier(random_state=42),
    }
    return models


def get_param_grid(model_name):
    """
    Get hyperparameter grid for a given model.
    """
    param_grids = {
        "KNN": {
            "n_neighbors": [3, 5, 7, 9],
            "weights": ["uniform", "distance"],
            "metric": ["euclidean", "manhattan"],
        },
        "Logistic Regression": {"C": [0.01, 0.1, 1, 10, 100], "penalty": ["l2"]},
        "SGD Classifier": {
            "loss": ["hinge", "log_loss", "modified_huber"],
            "penalty": ["l2", "elasticnet"],
            "alpha": [0.0001, 0.001, 0.01],
        },
        "Random Forest": {
            "n_estimators": [50, 100, 200],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5, 10],
        },
        "XGBoost": {
            "n_estimators": [50, 100, 200],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.1, 0.3],
        },
        "LightGBM": {
            "n_estimators": [50, 100, 200],
            "max_depth": [3, 5, -1],
            "learning_rate": [0.01, 0.1, 0.3],
        },
        "Support Vector Machine - LinearSVC": {"C": [0.1, 1, 10], "max_iter": [1000, 2000]},
        "Support Vector Machine - RBF SVC": {
            "C": [0.1, 1, 10],
            "gamma": ["scale", "auto", 0.1, 0.01],
        },
        "Neural Network - MLPClassifier": {
            "hidden_layer_sizes": [(50,), (100,), (50, 50)],
            "alpha": [0.0001, 0.001, 0.01],
        },
        "Extra Tree Classifier": {"max_depth": [None, 5, 10, 20], "min_samples_split": [2, 5, 10]},
        "Decision Tree": {
            "max_depth": [None, 5, 10, 20],
            "min_samples_split": [2, 5, 10],
            "criterion": ["gini", "entropy"],
        },
        "Gradient Boosting Classifier": {
            "n_estimators": [50, 100],
            "learning_rate": [0.01, 0.1],
            "max_depth": [3, 5],
        },
        "AdaBoost Classifier": {"n_estimators": [50, 100], "learning_rate": [0.01, 0.1, 1.0]},
        "Extra Trees Classifier": {"n_estimators": [50, 100], "max_depth": [None, 10, 20]},
        "Hist Gradient Boosting Classifier": {
            "max_iter": [100, 200],
            "learning_rate": [0.01, 0.1],
            "max_depth": [3, 5],
        },
        "Perceptron": {"alpha": [0.0001, 0.001, 0.01], "penalty": ["l2", None]},
        "PassiveAggressiveClassifier": {"C": [0.1, 1, 10]},
        "Ridge Classifier": {"alpha": [0.1, 1, 10]},
        "Linear Discriminant Analysis": {"solver": ["svd", "lsqr"]},
        "Quadratic Discriminant Analysis": {"reg_param": [0.0, 0.1, 0.5]},
        "Bernoulli Naive Bayes": {"alpha": [0.1, 0.5, 1.0]},
        "Gaussian Naive Bayes": {"var_smoothing": [1e-9, 1e-8, 1e-7]},
        "Multinomial Naive Bayes": {"alpha": [0.1, 0.5, 1.0]},
        "Complement Naive Bayes": {"alpha": [0.1, 0.5, 1.0]},
        "Bagging Classifier": {"n_estimators": [5, 10, 20], "max_samples": [0.7, 1.0]},
    }
    return param_grids.get(model_name, {})


def get_sklearn_scoring(metric_name, is_multiclass=False):
    """
    Map user metric name to standard sklearn scorer name.
    """
    metric_map = {
        "accuracy": "accuracy",
        "f1": "f1_macro" if is_multiclass else "f1",
        "precision": "precision_macro" if is_multiclass else "precision",
        "recall": "recall_macro" if is_multiclass else "recall",
        "roc_auc": "roc_auc_ovr" if is_multiclass else "roc_auc",
    }
    return metric_map.get(metric_name.lower(), "accuracy")


def train_single_model(
    model_name, model, X_train, y_train, X_test, y_test, model_cfg, is_multiclass, inner_n_jobs=1
):
    """
    Train a single model, applying grid search or random search if specified.
    """
    hp_opt = model_cfg.get("hp_optimization", "none")
    optimize_metric = model_cfg.get("optimize_metric", "accuracy")
    cv_settings = model_cfg.get("cv_settings", {})
    cv_folds = cv_settings.get("folds", 5) if cv_settings.get("run_cv", True) else 5

    scoring_metric = get_sklearn_scoring(optimize_metric, is_multiclass)

    param_grid = get_param_grid(model_name)

    start = timer()
    if hp_opt == "grid" and param_grid:
        print(f"  Running GridSearchCV for {model_name}...")
        search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=cv_folds,
            scoring=scoring_metric,
            n_jobs=inner_n_jobs,
            verbose=0,
        )
        search.fit(X_train, y_train)
        best_model = search.best_estimator_
        hyperparams = str(search.best_params_)
    elif hp_opt == "random" and param_grid:
        print(f"  Running RandomizedSearchCV for {model_name}...")
        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_grid,
            n_iter=15,
            cv=cv_folds,
            scoring=scoring_metric,
            n_jobs=inner_n_jobs,
            verbose=0,
            random_state=42,
        )
        search.fit(X_train, y_train)
        best_model = search.best_estimator_
        hyperparams = str(search.best_params_)
    else:
        print(f"  Fitting {model_name} with default parameters...")
        best_model = model
        best_model.fit(X_train, y_train)
        hyperparams = "default"

    end = timer()
    train_time = end - start

    # Predict
    preds = best_model.predict(X_test)

    # Cross Validation score on training set
    cv_mean, cv_std = np.nan, np.nan
    if cv_settings.get("run_cv", True):
        try:
            cv_scores = cross_val_score(
                best_model,
                X_train,
                y_train,
                cv=cv_folds,
                scoring=scoring_metric,
                n_jobs=inner_n_jobs,
            )
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()
        except Exception as e:
            print(f"  Cross-validation failed for {model_name}: {e}")

    # Calculate performance metrics on test set
    acc = accuracy_score(y_test, preds)

    if is_multiclass:
        f1 = f1_score(y_test, preds, average="macro", zero_division=0)
        prec = precision_score(y_test, preds, average="macro", zero_division=0)
        rec = recall_score(y_test, preds, average="macro", zero_division=0)
        try:
            if hasattr(best_model, "predict_proba"):
                probs_list = best_model.predict_proba(X_test)
                if isinstance(probs_list, list):
                    # MultiOutputClassifier predict_proba returns a list of arrays of shape (n_samples, 2)
                    probs = np.column_stack([p[:, 1] for p in probs_list])
                    roc = roc_auc_score(y_test, probs, average="macro")
                else:
                    roc = roc_auc_score(y_test, probs_list, multi_class="ovr")
            elif hasattr(best_model, "decision_function"):
                dec = best_model.decision_function(X_test)
                roc = roc_auc_score(y_test, dec, multi_class="ovr")
            else:
                roc = np.nan
        except Exception:
            roc = np.nan

        # Confusion matrix for multiclass (skipped for multilabel)
        cm = confusion_matrix(y_test, preds) if y_test.ndim == 1 else None
        tp, fn, fp, tn = np.nan, np.nan, np.nan, np.nan
        fp_rate, fn_rate = np.nan, np.nan
    else:
        f1 = f1_score(y_test, preds, zero_division=0)
        prec = precision_score(y_test, preds, zero_division=0)
        rec = recall_score(y_test, preds, zero_division=0)
        try:
            if hasattr(best_model, "predict_proba"):
                probs = best_model.predict_proba(X_test)[:, 1]
                roc = roc_auc_score(y_test, probs)
            elif hasattr(best_model, "decision_function"):
                dec = best_model.decision_function(X_test)
                roc = roc_auc_score(y_test, dec)
            else:
                roc = np.nan
        except Exception:
            roc = np.nan

        cm = confusion_matrix(y_test, preds)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            fp_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            fn_rate = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        else:
            tp, fn, fp, tn = np.nan, np.nan, np.nan, np.nan
            fp_rate, fn_rate = np.nan, np.nan

    result = {
        "Model": model_name,
        "Training_Time": train_time,
        "Accuracy": acc,
        "CV_Score_Mean": cv_mean,
        "CV_Score_Std": cv_std,
        "F1": f1,
        "Precision": prec,
        "Recall": rec,
        "ROC_AUC": roc,
        "FP_Rate": fp_rate,
        "FN_Rate": fn_rate,
        "TP": tp,
        "FN": fn,
        "FP": fp,
        "TN": tn,
        "Hyperparameters": hyperparams,
    }

    return best_model, preds, result


def _train_task(model_name, model_obj, X_train, y_train, X_test, y_test, model_cfg, is_multiclass):
    if model_name in ["Multinomial Naive Bayes", "Complement Naive Bayes"]:
        min_val = X_train.min() if not hasattr(X_train, "toarray") else X_train.tocsr().min()
        if min_val < 0:
            print(
                f"  Skipping {model_name}: Features contain negative values (not supported by MultinomialNB/ComplementNB)."
            )
            return model_name, None, None, None
    try:
        from autobenchmark.models import train_single_model

        fitted_model, preds, metrics = train_single_model(
            model_name, model_obj, X_train, y_train, X_test, y_test, model_cfg, is_multiclass
        )
        return model_name, fitted_model, preds, metrics
    except Exception as e:
        print(f"  FAILED: {model_name}: {e}")
        return model_name, None, None, str(e)


def train_benchmark_models(
    X_train, y_train, X_test, y_test, model_cfg, output_dir, feat_labels=None
):
    """
    Train and evaluate all selected models based on configuration.
    Saves outputs in output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"Starting model benchmarking. Saving outputs to: {output_dir}")

    # Save the prepared datasets for explainers
    data_save_path = os.path.join(output_dir, f"{model_cfg['config_name']}_data.npz")
    np.savez(
        data_save_path,
        X_train=X_train,
        X_test=X_test,
        y_train=np.array(y_train),
        y_test=np.array(y_test),
        feat_labels=np.array(feat_labels if feat_labels is not None else [], dtype=object),
    )
    print(f"  Saved preprocessed data -> {data_save_path}")

    # Target Label Encoding
    from sklearn.preprocessing import LabelEncoder

    y_train_arr = np.array(y_train)
    y_test_arr = np.array(y_test)
    is_multilabel = y_train_arr.ndim > 1 and y_train_arr.shape[1] > 1

    if is_multilabel:

        class PassThroughEncoder:
            def fit_transform(self, y):
                return y

            def transform(self, y):
                return y

            def inverse_transform(self, y):
                return y

        le = PassThroughEncoder()
        y_train_encoded = y_train_arr
        y_test_encoded = y_test_arr
        num_classes = y_train_encoded.shape[1]
        is_multiclass = True
    else:
        le = LabelEncoder()
        y_train_encoded = le.fit_transform(y_train_arr)
        y_test_encoded = le.transform(y_test_arr)
        unique_classes = np.unique(y_train_encoded)
        num_classes = len(unique_classes)
        is_multiclass = num_classes > 2

    # Save label encoder for future decoding
    le_path = os.path.join(output_dir, f"{model_cfg['config_name']}_label_encoder.joblib")
    joblib.dump(le, le_path)
    print(f"  Saved label encoder -> {le_path}")

    # For multilabel, base models dictionary is initialized as binary classifiers
    # and then wrapped in MultiOutputClassifier
    base_models_dict = get_models_dict(num_classes=2 if is_multilabel else num_classes)

    models_dict = {}
    for name, model_obj in base_models_dict.items():
        if is_multilabel:
            from sklearn.multioutput import MultiOutputClassifier

            models_dict[name] = MultiOutputClassifier(model_obj)
        else:
            models_dict[name] = model_obj

    models_to_run = model_cfg.get("models_to_run", "all")
    if isinstance(models_to_run, list):
        selected_models = {k: v for k, v in models_dict.items() if k in models_to_run}
    else:
        selected_models = models_dict

    results_list = []
    predictions_df = pd.DataFrame()
    if is_multilabel:
        predictions_df["Actual"] = [",".join(map(str, row)) for row in y_test_encoded]
    else:
        predictions_df["Actual"] = np.array(y_test)

    # Run model training in parallel using n_jobs=-1
    print(f"  Training {len(selected_models)} models in parallel...")
    from joblib import Parallel, delayed

    parallel_results = Parallel(n_jobs=-1)(
        delayed(_train_task)(
            model_name,
            model_obj,
            X_train,
            y_train_encoded,
            X_test,
            y_test_encoded,
            model_cfg,
            is_multiclass,
        )
        for model_name, model_obj in selected_models.items()
    )

    successful_models = 0
    failed_models = []

    for model_name, fitted_model, preds, res in parallel_results:
        if fitted_model is not None:
            results_list.append(res)

            # Decode predictions
            if is_multilabel:
                preds_decoded = [",".join(map(str, row)) for row in preds]
            else:
                preds_decoded = le.inverse_transform(preds)
            predictions_df[model_name] = preds_decoded

            # Save serialized model if required
            if model_cfg.get("save_models", True):
                models_subfolder = os.path.join(output_dir, "models")
                os.makedirs(models_subfolder, exist_ok=True)
                safe_name = re.sub(r"[^a-zA-Z0-9_-]", "_", model_name)
                model_path = os.path.join(
                    models_subfolder, f"{model_cfg['config_name']}_{safe_name}.joblib"
                )
                joblib.dump(fitted_model, model_path)
                print(f"  Saved model -> {model_path}")

            successful_models += 1
        elif res is not None:
            failed_models.append((model_name, res))
        else:
            # Skipped due to negative values
            print(f"  Skipped: {model_name}")

    df_results = pd.DataFrame(results_list)

    # Save predictions df
    preds_path = os.path.join(output_dir, f"{model_cfg['config_name']}_predictions.csv")
    predictions_df.to_csv(preds_path, index=False)
    print(f"\n  Saved all model predictions -> {preds_path}")

    print(f"\nTabular benchmarking completed: {datetime.datetime.now()}")
    print(f"Successfully trained: {successful_models}/{len(selected_models)} models")
    if failed_models:
        print(f"Failed models ({len(failed_models)}):")
        for name, err in failed_models:
            print(f"  - {name}: {err}")

    return df_results, predictions_df
