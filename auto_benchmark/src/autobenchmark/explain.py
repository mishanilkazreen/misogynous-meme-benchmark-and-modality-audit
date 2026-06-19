"""
Explainability and surrogate modeling module for the autobenchmark package.
Supports SHAP (Tree, Linear, Kernel), LIME, Permutation Importance,
Native Feature Importance, and interpretable Global Surrogates.
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance as sk_permutation_importance
from sklearn.tree import DecisionTreeClassifier, export_text


def _safe_filename(name):
    return re.sub(r'[^a-zA-Z0-9_-]', '_', name)


def _extract_binary_shap_values(shap_values):
    """Normalize SHAP values to 2D array representing class 1."""
    if isinstance(shap_values, list):
        return shap_values[1]
    if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        return shap_values[:, :, 1]
    return shap_values


# 1. SHAP Explanations
def run_shap_explanations(model, model_name, X_train, X_test, feat_labels, output_dir, shap_cfg):
    """
    Generate global SHAP summary and bar plots.
    """
    import shap
    print(f"  Running SHAP for: {model_name}")
    model_dir = os.path.join(output_dir, _safe_filename(model_name))
    os.makedirs(model_dir, exist_ok=True)
    
    max_display = shap_cfg.get('max_display', 20)
    n_background = shap_cfg.get('n_background', 100)
    
    # Subsample background data to speed up execution
    bg_samples = shap.sample(X_train, min(n_background, len(X_train)))
    
    # Subsample test set for SHAP to avoid running on too many samples
    # Kernel SHAP is especially slow, so we limit it to 30; tree/linear can handle 50.
    is_tree = hasattr(model, 'feature_importances_')
    is_linear = hasattr(model, 'coef_')
    
    # Check if features are too high for non-tree/non-linear SHAP explainers
    if not is_tree and not is_linear and X_test.shape[1] > 100:
        print(f"  Skipping SHAP: Model '{model_name}' is not tree/linear and dataset has {X_test.shape[1]} features.")
        print("    Kernel SHAP is computationally intractable for high-dimensional feature spaces.")
        return
        
    if is_tree or is_linear:
        X_test_sub = X_test[:min(50, len(X_test))]
    else:
        X_test_sub = X_test[:min(30, len(X_test))]
        
    try:
        if is_tree:
            try:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_test_sub)
            except Exception:
                try:
                    # Try using Explainer with callable predict function
                    predict_fn = model.predict_proba if hasattr(model, 'predict_proba') else model.predict
                    explainer = shap.Explainer(predict_fn, bg_samples)
                    shap_values = explainer(X_test_sub).values
                except Exception:
                    # Fallback to KernelExplainer only if low dimensional
                    if X_test.shape[1] > 100:
                        raise ValueError("Tree SHAP failed and features > 100 (Kernel SHAP skipped for performance).")
                    predict_fn = model.predict_proba if hasattr(model, 'predict_proba') else model.predict
                    explainer = shap.KernelExplainer(predict_fn, bg_samples)
                    shap_values = explainer.shap_values(X_test_sub)
        elif is_linear:
            explainer = shap.LinearExplainer(model, bg_samples)
            shap_values = explainer.shap_values(X_test_sub)
        else:
            predict_fn = model.predict_proba if hasattr(model, 'predict_proba') else model.predict
            explainer = shap.KernelExplainer(predict_fn, bg_samples)
            shap_values = explainer.shap_values(X_test_sub)
            
        # Check if shap_values is a list (multiclass) or 3D array
        is_multiclass = False
        if isinstance(shap_values, list) and len(shap_values) > 2:
            is_multiclass = True
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3 and shap_values.shape[2] > 2:
            is_multiclass = True
            
        if is_multiclass:
            # Bar plot natively supports multiclass inputs
            plt.figure()
            shap.summary_plot(shap_values, X_test_sub, feature_names=feat_labels, plot_type='bar', max_display=max_display, show=False)
            plt.title(f"SHAP Feature Importance (Multiclass) - {model_name}")
            plt.tight_layout()
            plt.savefig(os.path.join(model_dir, 'shap_bar.png'), dpi=150, bbox_inches='tight')
            plt.close()
            
            # Beeswarm plot needs a 2D array, so we explain Class 0
            plt.figure()
            if isinstance(shap_values, list):
                sv_class0 = shap_values[0]
            else:
                sv_class0 = shap_values[:, :, 0]
            shap.summary_plot(sv_class0, X_test_sub, feature_names=feat_labels, max_display=max_display, show=False)
            plt.title(f"SHAP Beeswarm (Class 0) - {model_name}")
            plt.tight_layout()
            plt.savefig(os.path.join(model_dir, 'shap_beeswarm.png'), dpi=150, bbox_inches='tight')
            plt.close()
        else:
            sv = _extract_binary_shap_values(shap_values)
            
            # Plot Bar Summary
            plt.figure()
            shap.summary_plot(sv, X_test_sub, feature_names=feat_labels, plot_type='bar', max_display=max_display, show=False)
            plt.title(f"SHAP Feature Importance - {model_name}")
            plt.tight_layout()
            plt.savefig(os.path.join(model_dir, 'shap_bar.png'), dpi=150, bbox_inches='tight')
            plt.close()
            
            # Plot Beeswarm Summary
            plt.figure()
            shap.summary_plot(sv, X_test_sub, feature_names=feat_labels, max_display=max_display, show=False)
            plt.title(f"SHAP Beeswarm - {model_name}")
            plt.tight_layout()
            plt.savefig(os.path.join(model_dir, 'shap_beeswarm.png'), dpi=150, bbox_inches='tight')
            plt.close()
        
        print(f"    Saved SHAP plots to: {model_dir}")
    except Exception as e:
        print(f"    SHAP analysis failed for {model_name}: {e}")


# 2. LIME Local Explanations
def run_lime_explanations(model, model_name, X_train, X_test, y_test, feat_labels, output_dir, lime_cfg):
    """
    Generate local instance explanations using LIME.
    """
    from lime import lime_tabular
    print(f"  Running LIME for: {model_name}")
    model_dir = os.path.join(output_dir, _safe_filename(model_name))
    os.makedirs(model_dir, exist_ok=True)
    
    n_instances = lime_cfg.get('n_instances', 5)
    n_features = lime_cfg.get('n_features', 15)
    
    y_test_arr = np.array(y_test)
    classes = getattr(model, 'classes_', np.unique(y_test_arr))
    class_names = [f"Class {c}" for c in classes]
    
    if hasattr(model, 'predict_proba'):
        predict_fn = model.predict_proba
    else:
        def predict_fn(x):
            preds = model.predict(x)
            one_hot = np.zeros((len(x), len(classes)))
            for idx_class, val in enumerate(classes):
                one_hot[preds == val, idx_class] = 1.0
            return one_hot
            
    explainer = lime_tabular.LimeTabularExplainer(
        X_train,
        feature_names=feat_labels,
        class_names=class_names,
        mode='classification',
        discretize_continuous=True
    )
    
    # Pick balanced instances from test set
    indices = []
    
    for c in np.unique(y_test_arr):
        cls_indices = np.where(y_test_arr == c)[0]
        indices.extend(cls_indices[:max(1, n_instances // len(np.unique(y_test_arr)))].tolist())
    indices = indices[:n_instances]
    
    num_samples = lime_cfg.get('num_samples', 500)
    for i, idx in enumerate(indices):
        try:
            pred_probs = predict_fn(X_test[idx].reshape(1, -1))[0]
            pred_class_idx = np.argmax(pred_probs)
            
            exp = explainer.explain_instance(
                X_test[idx], predict_fn, num_features=n_features, labels=(pred_class_idx,), num_samples=num_samples
            )
            fig = exp.as_pyplot_figure(label=pred_class_idx)
            fig.suptitle(f"LIME - {model_name} - Instance {i} (True={y_test_arr[idx]}, Pred={classes[pred_class_idx]})", fontsize=10)
            fig.tight_layout()
            fig_path = os.path.join(model_dir, f"lime_instance_{i}.png")
            fig.savefig(fig_path, dpi=150)
            plt.close(fig)
        except Exception as e:
            print(f"    LIME failed for instance {i}: {e}")
            
    print(f"    Saved LIME plots to: {model_dir}")


# 3. Native Feature Importance
def run_native_importance(model, model_name, feat_labels, output_dir):
    """
    Plot native feature importances for tree-based models.
    """
    if not hasattr(model, 'feature_importances_'):
        return
        
    print(f"  Running Native Feature Importance for: {model_name}")
    model_dir = os.path.join(output_dir, _safe_filename(model_name))
    os.makedirs(model_dir, exist_ok=True)
    
    importances = model.feature_importances_
    idx = np.argsort(importances)[-20:]  # Top 20 features
    
    plt.figure(figsize=(10, 6))
    plt.barh([feat_labels[i] for i in idx], importances[idx], color='steelblue')
    plt.xlabel('Importance')
    plt.title(f"Native Feature Importance - {model_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, 'native_feature_importance.png'), dpi=150)
    plt.close()
    print(f"    Saved native feature importance plot to: {model_dir}")


# 4. Permutation Feature Importance
def run_permutation_importance(model, model_name, X_test, y_test, feat_labels, output_dir, perm_cfg):
    """
    Generate model-agnostic permutation feature importances.
    """
    print(f"  Running Permutation Importance for: {model_name}")
    model_dir = os.path.join(output_dir, _safe_filename(model_name))
    os.makedirs(model_dir, exist_ok=True)
    
    n_repeats = perm_cfg.get('n_repeats', 10)
    top_n = perm_cfg.get('top_n', 25)
    
    # Check if features are too high for permutation importance
    if X_test.shape[1] > 100:
        print(f"  Skipping Permutation Importance: Dataset has {X_test.shape[1]} features.")
        print("    Permutation importance scales O(N_features) and is too slow for high-dimensional spaces.")
        return
        
    try:
        result = sk_permutation_importance(
            model, X_test, y_test, n_repeats=n_repeats, random_state=42, scoring='accuracy', n_jobs=-1
        )
        
        mean_imp = result.importances_mean
        std_imp = result.importances_std
        
        idx = np.argsort(mean_imp)[-top_n:]
        
        plt.figure(figsize=(10, 6))
        plt.barh([feat_labels[i] for i in idx], mean_imp[idx], xerr=std_imp[idx], color='coral')
        plt.xlabel('Mean Accuracy Decrease')
        plt.title(f"Permutation Feature Importance - {model_name}")
        plt.tight_layout()
        plt.savefig(os.path.join(model_dir, 'permutation_importance.png'), dpi=150)
        plt.close()
        print(f"    Saved permutation importance plot to: {model_dir}")
    except Exception as e:
        print(f"    Permutation importance failed for {model_name}: {e}")


# 5. Global Surrogate Model
def run_surrogate_model(blackbox_model, model_name, X_train, X_test, feat_labels, output_dir, surrogate_cfg):
    """
    Train an interpretable decision tree surrogate model to approximate the blackbox behavior.
    """
    print(f"  Running Surrogate Decision Tree for: {model_name}")
    model_dir = os.path.join(output_dir, _safe_filename(model_name))
    os.makedirs(model_dir, exist_ok=True)
    
    # Label targets using blackbox model predictions
    y_train_surr = blackbox_model.predict(X_train)
    y_test_surr = blackbox_model.predict(X_test)
    
    # Fit Decision Tree surrogate
    surrogate = DecisionTreeClassifier(max_depth=4, min_samples_split=10, random_state=42)
    surrogate.fit(X_train, y_train_surr)
    
    # Calculate fidelity (surrogate alignment with blackbox predictions)
    fidelity_train = surrogate.score(X_train, y_train_surr)
    fidelity_test = surrogate.score(X_test, y_test_surr)
    
    # Extract rules
    tree_rules = export_text(surrogate, feature_names=feat_labels)
    
    # Save results
    surr_report_path = os.path.join(model_dir, 'surrogate_rules.txt')
    with open(surr_report_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write(f"GLOBAL DECISION TREE SURROGATE MODEL FOR: {model_name}\n")
        f.write("="*80 + "\n\n")
        f.write(f"Training Fidelity (align with blackbox): {fidelity_train:.4f}\n")
        f.write(f"Testing Fidelity (align with blackbox):  {fidelity_test:.4f}\n\n")
        f.write("Decision Rules:\n")
        f.write("-"*40 + "\n")
        f.write(tree_rules)
        f.write("="*80 + "\n")
        
    print(f"    Saved surrogate rules to: {surr_report_path}")
    return fidelity_test


def run_text_explainers(model, model_name, texts_train, texts_test, y_test, preprocessor, output_dir, explainer_cfg, data_cfg=None):
    """
    Generate textual local explanations (word-highlighting LIME and SHAP).
    """
    import os
    import re
    import numpy as np
    import pandas as pd
    
    print(f"  Running Textual Explainers for: {model_name}")
    model_dir = os.path.join(output_dir, _safe_filename(model_name))
    os.makedirs(model_dir, exist_ok=True)
    
    # Define class labels
    classes = getattr(model, 'classes_', np.unique(y_test))
    class_names = [f"Class {c}" for c in classes]
    
    # 1. Define predict function
    text_features_mode = 'tfidf'
    if data_cfg:
        text_features_mode = data_cfg.get('preprocessing', {}).get('text_features', 'both')
        
    sbert_model = None
    
    def predict_fn(texts_list):
        if isinstance(texts_list, np.ndarray):
            texts_list = texts_list.tolist()
            
        X_tfidf = None
        X_emb = None
        
        if text_features_mode in ['tfidf', 'both'] and preprocessor is not None:
            X_tfidf = preprocessor.transform(texts_list).toarray()
            
        if text_features_mode in ['embeddings', 'both']:
            nonlocal sbert_model
            if sbert_model is None:
                from sentence_transformers import SentenceTransformer
                import torch
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                emb_model_name = 'all-MiniLM-L6-v2'
                if data_cfg:
                    emb_model_name = data_cfg.get('preprocessing', {}).get('embeddings_model', 'all-MiniLM-L6-v2')
                print(f"Loading SentenceTransformer: {emb_model_name} for text explanations...")
                try:
                    sbert_model = SentenceTransformer(emb_model_name, device=device)
                except Exception as e:
                    print(f"Failed loading {emb_model_name} online: {e}. Trying local_files_only=True...")
                    sbert_model = SentenceTransformer(emb_model_name, device=device, local_files_only=True)
            light_texts = [re.sub(r'\S*https?:\S*', '', t) for t in texts_list]
            X_emb = sbert_model.encode(light_texts, show_progress_bar=False)
            
        if X_tfidf is not None and X_emb is not None:
            X = np.hstack([X_tfidf, X_emb])
        elif X_tfidf is not None:
            X = X_tfidf
        else:
            X = X_emb
            
        if hasattr(model, 'predict_proba'):
            return model.predict_proba(X)
        else:
            preds = model.predict(X)
            one_hot = np.zeros((len(texts_list), len(classes)))
            for idx_class, val in enumerate(classes):
                one_hot[preds == val, idx_class] = 1.0
            return one_hot

    # 2. Textual LIME
    try:
        from lime.lime_text import LimeTextExplainer
        print("    Running Textual LIME...")
        lime_text_explainer = LimeTextExplainer(class_names=class_names)
        
        # Pick 3 test instances balanced by class
        y_test_arr = np.array(y_test)
        indices = []
        for c in np.unique(y_test_arr):
            cls_indices = np.where(y_test_arr == c)[0]
            if len(cls_indices) > 0:
                indices.append(cls_indices[0])  # 1 per class
        indices = indices[:3]
        
        num_samples = explainer_cfg.get('lime_settings', {}).get('num_samples', 500)
        for i, idx in enumerate(indices):
            raw_text = texts_test[idx]
            # Limit text length to 1000 characters for readability
            if len(raw_text) > 1000:
                raw_text = raw_text[:1000] + "..."
            exp = lime_text_explainer.explain_instance(raw_text, predict_fn, num_features=10, num_samples=num_samples)
            lime_html_path = os.path.join(model_dir, f"lime_text_highlight_instance_{i}.html")
            exp.save_to_file(lime_html_path)
            print(f"      Saved LIME textual highlight -> {lime_html_path}")
    except Exception as e:
        print(f"    Textual LIME failed: {e}")
        
    # 3. Textual SHAP
    try:
        import shap
        print("    Running Textual SHAP...")
        # Subsample test set for SHAP to avoid running on too many samples
        texts_sub = [texts_test[i] for i in range(min(3, len(texts_test)))]
        
        masker = shap.maskers.Text(tokenizer=r'\s+')
        explainer = shap.Explainer(predict_fn, masker)
        shap_values = explainer(texts_sub)
        
        # Save interactive HTML
        shap_html_path = os.path.join(model_dir, "shap_text_highlight.html")
        html_string = shap.plots.text(shap_values, display=False)
        with open(shap_html_path, "w", encoding="utf-8") as f:
            f.write(html_string)
        print(f"      Saved SHAP textual highlight -> {shap_html_path}")
    except Exception as e:
        print(f"    Textual SHAP failed: {e}")
