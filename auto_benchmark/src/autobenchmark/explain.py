"""
Explainability and surrogate modeling module for the autobenchmark package.

Supports SHAP (Tree, Linear, Kernel), LIME, Permutation Importance,
Native Feature Importance, and interpretable Global Surrogates.
"""

# pylint: disable=too-many-arguments,too-many-positional-arguments
# pylint: disable=too-many-locals,too-many-statements,too-many-branches
# pylint: disable=import-outside-toplevel,broad-exception-caught
# pylint: disable=invalid-name

import os
import re

import matplotlib
import numpy as np
from sklearn.inspection import permutation_importance as sk_permutation_importance
from sklearn.tree import DecisionTreeClassifier, export_text

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _safe_filename(name: str) -> str:
    """Sanitize a string for use as a filename."""
    return re.sub(r"[^a-zA-Z0-9_-]", "_", name)


def _extract_binary_shap_values(shap_values):
    """Normalize SHAP values to 2D array representing class 1."""
    if isinstance(shap_values, list):
        return shap_values[1]
    if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        return shap_values[:, :, 1]
    return shap_values


def _get_predict_fn(model):
    """Return predict_proba if available, else predict."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba
    return model.predict


def _should_skip_kernel_shap(n_features: int, model_name: str) -> bool:
    """Check if Kernel SHAP should be skipped due to high dimensionality."""
    if n_features > 100:
        print(
            f"  Skipping SHAP: Model '{model_name}' is not tree/linear "
            f"and dataset has {n_features} features."
        )
        print("    Kernel SHAP is computationally intractable for high-dimensional feature spaces.")
        return True
    return False


def _plot_shap_multiclass(shap_values, x_test_sub, feat_labels, max_display, model_name, model_dir):
    """Generate SHAP plots for multiclass models."""
    import shap

    plt.figure()
    shap.summary_plot(
        shap_values,
        x_test_sub,
        feature_names=feat_labels,
        plot_type="bar",
        max_display=max_display,
        show=False,
    )
    plt.title(f"SHAP Feature Importance (Multiclass) - {model_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, "shap_bar.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # Beeswarm plot needs a 2D array, so we explain Class 0
    plt.figure()
    sv_class0 = shap_values[0] if isinstance(shap_values, list) else shap_values[:, :, 0]
    shap.summary_plot(
        sv_class0,
        x_test_sub,
        feature_names=feat_labels,
        max_display=max_display,
        show=False,
    )
    plt.title(f"SHAP Beeswarm (Class 0) - {model_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, "shap_beeswarm.png"), dpi=150, bbox_inches="tight")
    plt.close()


def _plot_shap_binary(shap_values, x_test_sub, feat_labels, max_display, model_name, model_dir):
    """Generate SHAP plots for binary classification models."""
    import shap

    sv = _extract_binary_shap_values(shap_values)

    plt.figure()
    shap.summary_plot(
        sv,
        x_test_sub,
        feature_names=feat_labels,
        plot_type="bar",
        max_display=max_display,
        show=False,
    )
    plt.title(f"SHAP Feature Importance - {model_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, "shap_bar.png"), dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure()
    shap.summary_plot(
        sv,
        x_test_sub,
        feature_names=feat_labels,
        max_display=max_display,
        show=False,
    )
    plt.title(f"SHAP Beeswarm - {model_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, "shap_beeswarm.png"), dpi=150, bbox_inches="tight")
    plt.close()


def _compute_shap_values_tree(model, x_test_sub, bg_samples):
    """Compute SHAP values for tree-based models with fallbacks."""
    import shap

    try:
        explainer = shap.TreeExplainer(model)
        return explainer.shap_values(x_test_sub)
    except Exception:
        pass

    # Fallback: generic Explainer with predict function
    try:
        predict_fn = _get_predict_fn(model)
        explainer = shap.Explainer(predict_fn, bg_samples)
        explanation = explainer(x_test_sub)
        return explanation.values  # pylint: disable=no-member
    except Exception:
        pass

    # Last resort: KernelExplainer (only if low dimensional)
    if x_test_sub.shape[1] > 100:
        raise ValueError(
            "Tree SHAP failed and features > 100 (Kernel SHAP skipped for performance)."
        )
    predict_fn = _get_predict_fn(model)
    explainer = shap.KernelExplainer(predict_fn, bg_samples)
    return explainer.shap_values(x_test_sub)


def _is_multiclass_shap(shap_values) -> bool:
    """Check whether SHAP values represent a multiclass problem."""
    if isinstance(shap_values, list) and len(shap_values) > 2:
        return True
    return (
        isinstance(shap_values, np.ndarray) and shap_values.ndim == 3 and shap_values.shape[2] > 2
    )


# 1. SHAP Explanations
def run_shap_explanations(model, model_name, x_train, x_test, feat_labels, output_dir, shap_cfg):
    """Generate global SHAP summary and bar plots."""
    import shap

    print(f"  Running SHAP for: {model_name}")
    model_dir = os.path.join(output_dir, _safe_filename(model_name))
    os.makedirs(model_dir, exist_ok=True)

    max_display = shap_cfg.get("max_display", 20)
    n_background = shap_cfg.get("n_background", 100)

    bg_samples = shap.sample(x_train, min(n_background, len(x_train)))

    is_tree = hasattr(model, "feature_importances_")
    is_linear = hasattr(model, "coef_")

    if not is_tree and not is_linear and _should_skip_kernel_shap(x_test.shape[1], model_name):
        return

    max_samples = 50 if (is_tree or is_linear) else 30
    x_test_sub = x_test[: min(max_samples, len(x_test))]

    try:
        if is_tree:
            shap_values = _compute_shap_values_tree(model, x_test_sub, bg_samples)
        elif is_linear:
            explainer = shap.LinearExplainer(model, bg_samples)
            shap_values = explainer.shap_values(x_test_sub)
        else:
            predict_fn = _get_predict_fn(model)
            explainer = shap.KernelExplainer(predict_fn, bg_samples)
            shap_values = explainer.shap_values(x_test_sub)

        if _is_multiclass_shap(shap_values):
            _plot_shap_multiclass(
                shap_values,
                x_test_sub,
                feat_labels,
                max_display,
                model_name,
                model_dir,
            )
        else:
            _plot_shap_binary(
                shap_values,
                x_test_sub,
                feat_labels,
                max_display,
                model_name,
                model_dir,
            )

        print(f"    Saved SHAP plots to: {model_dir}")
    except Exception as exc:
        print(f"    SHAP analysis failed for {model_name}: {exc}")


# 2. LIME Local Explanations
def run_lime_explanations(
    model, model_name, x_train, x_test, y_test, feat_labels, output_dir, lime_cfg
):
    """Generate local instance explanations using LIME."""
    from lime import lime_tabular

    print(f"  Running LIME for: {model_name}")
    model_dir = os.path.join(output_dir, _safe_filename(model_name))
    os.makedirs(model_dir, exist_ok=True)

    n_instances = lime_cfg.get("n_instances", 5)
    n_features = lime_cfg.get("n_features", 15)

    y_test_arr = np.array(y_test)
    classes = getattr(model, "classes_", np.unique(y_test_arr))
    class_names = [f"Class {c}" for c in classes]

    if hasattr(model, "predict_proba"):
        predict_fn = model.predict_proba
    else:

        def predict_fn(x):
            preds = model.predict(x)
            one_hot = np.zeros((len(x), len(classes)))
            for idx_class, val in enumerate(classes):
                one_hot[preds == val, idx_class] = 1.0
            return one_hot

    explainer = lime_tabular.LimeTabularExplainer(
        x_train,
        feature_names=feat_labels,
        class_names=class_names,
        mode="classification",
        discretize_continuous=True,
    )

    # Pick balanced instances from test set
    indices = []
    for c in np.unique(y_test_arr):
        cls_indices = np.where(y_test_arr == c)[0]
        n_per_class = max(1, n_instances // len(np.unique(y_test_arr)))
        indices.extend(cls_indices[:n_per_class].tolist())
    indices = indices[:n_instances]

    num_samples = lime_cfg.get("num_samples", 500)
    for i, idx in enumerate(indices):
        try:
            pred_probs = predict_fn(x_test[idx].reshape(1, -1))[0]
            pred_class_idx = np.argmax(pred_probs)

            exp = explainer.explain_instance(
                x_test[idx],
                predict_fn,
                num_features=n_features,
                labels=(pred_class_idx,),
                num_samples=num_samples,
            )
            fig = exp.as_pyplot_figure(label=pred_class_idx)
            true_label = y_test_arr[idx]
            pred_label = classes[pred_class_idx]
            fig.suptitle(
                f"LIME - {model_name} - Instance {i} (True={true_label}, Pred={pred_label})",
                fontsize=10,
            )
            fig.tight_layout()
            fig_path = os.path.join(model_dir, f"lime_instance_{i}.png")
            fig.savefig(fig_path, dpi=150)
            plt.close(fig)
        except Exception as exc:
            print(f"    LIME failed for instance {i}: {exc}")

    print(f"    Saved LIME plots to: {model_dir}")


# 3. Native Feature Importance
def run_native_importance(model, model_name, feat_labels, output_dir):
    """Plot native feature importances for tree-based models."""
    if not hasattr(model, "feature_importances_"):
        return

    print(f"  Running Native Feature Importance for: {model_name}")
    model_dir = os.path.join(output_dir, _safe_filename(model_name))
    os.makedirs(model_dir, exist_ok=True)

    importances = model.feature_importances_
    idx = np.argsort(importances)[-20:]  # Top 20 features

    plt.figure(figsize=(10, 6))
    plt.barh([feat_labels[i] for i in idx], importances[idx], color="steelblue")
    plt.xlabel("Importance")
    plt.title(f"Native Feature Importance - {model_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, "native_feature_importance.png"), dpi=150)
    plt.close()
    print(f"    Saved native feature importance plot to: {model_dir}")


# 4. Permutation Feature Importance
def run_permutation_importance(
    model, model_name, x_test, y_test, feat_labels, output_dir, perm_cfg
):
    """Generate model-agnostic permutation feature importances."""
    print(f"  Running Permutation Importance for: {model_name}")
    model_dir = os.path.join(output_dir, _safe_filename(model_name))
    os.makedirs(model_dir, exist_ok=True)

    n_repeats = perm_cfg.get("n_repeats", 10)
    top_n = perm_cfg.get("top_n", 25)

    if x_test.shape[1] > 100:
        print(f"  Skipping Permutation Importance: Dataset has {x_test.shape[1]} features.")
        print(
            "    Permutation importance scales O(N_features) "
            "and is too slow for high-dimensional spaces."
        )
        return

    try:
        result = sk_permutation_importance(
            model,
            x_test,
            y_test,
            n_repeats=n_repeats,
            random_state=42,
            scoring="accuracy",
            n_jobs=-1,
        )

        mean_imp = result.importances_mean
        std_imp = result.importances_std
        idx = np.argsort(mean_imp)[-top_n:]

        plt.figure(figsize=(10, 6))
        plt.barh(
            [feat_labels[i] for i in idx],
            mean_imp[idx],
            xerr=std_imp[idx],
            color="coral",
        )
        plt.xlabel("Mean Accuracy Decrease")
        plt.title(f"Permutation Feature Importance - {model_name}")
        plt.tight_layout()
        plt.savefig(os.path.join(model_dir, "permutation_importance.png"), dpi=150)
        plt.close()
        print(f"    Saved permutation importance plot to: {model_dir}")
    except Exception as exc:
        print(f"    Permutation importance failed for {model_name}: {exc}")


# 5. Global Surrogate Model
def run_surrogate_model(
    blackbox_model,
    model_name,
    x_train,
    x_test,
    feat_labels,
    output_dir,
    _surrogate_cfg=None,
):
    """Train an interpretable decision tree surrogate to approximate the blackbox."""
    print(f"  Running Surrogate Decision Tree for: {model_name}")
    model_dir = os.path.join(output_dir, _safe_filename(model_name))
    os.makedirs(model_dir, exist_ok=True)

    y_train_surr = blackbox_model.predict(x_train)
    y_test_surr = blackbox_model.predict(x_test)

    surrogate = DecisionTreeClassifier(max_depth=4, min_samples_split=10, random_state=42)
    surrogate.fit(x_train, y_train_surr)

    fidelity_train = surrogate.score(x_train, y_train_surr)
    fidelity_test = surrogate.score(x_test, y_test_surr)

    tree_rules = export_text(surrogate, feature_names=feat_labels)

    surr_report_path = os.path.join(model_dir, "surrogate_rules.txt")
    with open(surr_report_path, "w", encoding="utf-8") as fh:
        fh.write("=" * 80 + "\n")
        fh.write(f"GLOBAL DECISION TREE SURROGATE MODEL FOR: {model_name}\n")
        fh.write("=" * 80 + "\n\n")
        fh.write(f"Training Fidelity (align with blackbox): {fidelity_train:.4f}\n")
        fh.write(f"Testing Fidelity (align with blackbox):  {fidelity_test:.4f}\n\n")
        fh.write("Decision Rules:\n")
        fh.write("-" * 40 + "\n")
        fh.write(tree_rules)
        fh.write("=" * 80 + "\n")

    print(f"    Saved surrogate rules to: {surr_report_path}")
    return fidelity_test


# 6. Textual Explainers (LIME text + SHAP text)
def _build_text_predict_fn(model, preprocessor, classes, data_cfg):
    """Build a prediction function that handles tfidf/embeddings for text."""
    text_features_mode = "tfidf"
    if data_cfg:
        text_features_mode = data_cfg.get("preprocessing", {}).get("text_features", "both")

    sbert_model_cache = [None]

    def _get_sbert_model():
        if sbert_model_cache[0] is not None:
            return sbert_model_cache[0]

        from sentence_transformers import SentenceTransformer
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        emb_model_name = "all-MiniLM-L6-v2"
        if data_cfg:
            emb_model_name = data_cfg.get("preprocessing", {}).get(
                "embeddings_model", "all-MiniLM-L6-v2"
            )
        print(f"Loading SentenceTransformer: {emb_model_name} for text explanations...")
        try:
            sbert_model_cache[0] = SentenceTransformer(emb_model_name, device=device)
        except Exception as exc:
            print(f"Failed loading {emb_model_name} online: {exc}. Trying local_files_only=True...")
            sbert_model_cache[0] = SentenceTransformer(
                emb_model_name, device=device, local_files_only=True
            )
        return sbert_model_cache[0]

    def predict_fn(texts_list):
        if isinstance(texts_list, np.ndarray):
            texts_list = texts_list.tolist()

        features_tfidf = None
        features_emb = None

        if text_features_mode in ("tfidf", "both") and preprocessor is not None:
            features_tfidf = preprocessor.transform(texts_list).toarray()

        if text_features_mode in ("embeddings", "both"):
            sbert = _get_sbert_model()
            light_texts = [re.sub(r"\S*https?:\S*", "", t) for t in texts_list]
            features_emb = sbert.encode(light_texts, show_progress_bar=False)

        if features_tfidf is not None and features_emb is not None:
            features = np.hstack([features_tfidf, features_emb])
        elif features_tfidf is not None:
            features = features_tfidf
        else:
            features = features_emb

        if hasattr(model, "predict_proba"):
            return model.predict_proba(features)

        preds = model.predict(features)
        one_hot = np.zeros((len(texts_list), len(classes)))
        for idx_class, val in enumerate(classes):
            one_hot[preds == val, idx_class] = 1.0
        return one_hot

    return predict_fn


def _run_textual_lime(predict_fn, texts_test, y_test, class_names, explainer_cfg, model_dir):
    """Run textual LIME explanations."""
    from lime.lime_text import LimeTextExplainer

    print("    Running Textual LIME...")
    lime_explainer = LimeTextExplainer(class_names=class_names)

    y_test_arr = np.array(y_test)
    indices = []
    for c in np.unique(y_test_arr):
        cls_indices = np.where(y_test_arr == c)[0]
        if len(cls_indices) > 0:
            indices.append(cls_indices[0])
    indices = indices[:3]

    num_samples = explainer_cfg.get("lime_settings", {}).get("num_samples", 500)
    for i, idx in enumerate(indices):
        raw_text = texts_test[idx]
        if len(raw_text) > 1000:
            raw_text = raw_text[:1000] + "..."
        exp = lime_explainer.explain_instance(
            raw_text, predict_fn, num_features=10, num_samples=num_samples
        )
        lime_html_path = os.path.join(model_dir, f"lime_text_highlight_instance_{i}.html")
        exp.save_to_file(lime_html_path)
        print(f"      Saved LIME textual highlight -> {lime_html_path}")


def _run_textual_shap(predict_fn, texts_test, model_dir):
    """Run textual SHAP explanations."""
    import shap

    print("    Running Textual SHAP...")
    texts_sub = [texts_test[i] for i in range(min(3, len(texts_test)))]

    masker = shap.maskers.Text(tokenizer=r"\s+")
    explainer = shap.Explainer(predict_fn, masker)
    shap_values = explainer(texts_sub)

    shap_html_path = os.path.join(model_dir, "shap_text_highlight.html")
    html_string = shap.plots.text(shap_values, display=False)
    with open(shap_html_path, "w", encoding="utf-8") as fh:
        fh.write(html_string)
    print(f"      Saved SHAP textual highlight -> {shap_html_path}")


def run_text_explainers(
    model,
    model_name,
    _texts_train,
    texts_test,
    y_test,
    preprocessor,
    output_dir,
    explainer_cfg,
    data_cfg=None,
):
    """Generate textual local explanations (word-highlighting LIME and SHAP)."""
    print(f"  Running Textual Explainers for: {model_name}")
    model_dir = os.path.join(output_dir, _safe_filename(model_name))
    os.makedirs(model_dir, exist_ok=True)

    classes = getattr(model, "classes_", np.unique(y_test))
    class_names = [f"Class {c}" for c in classes]

    predict_fn = _build_text_predict_fn(model, preprocessor, classes, data_cfg)

    # Textual LIME
    try:
        _run_textual_lime(predict_fn, texts_test, y_test, class_names, explainer_cfg, model_dir)
    except Exception as exc:
        print(f"    Textual LIME failed: {exc}")

    # Textual SHAP
    try:
        _run_textual_shap(predict_fn, texts_test, model_dir)
    except Exception as exc:
        print(f"    Textual SHAP failed: {exc}")
