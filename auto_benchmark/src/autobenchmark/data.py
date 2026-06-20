"""
Data loading and preprocessing module for the autobenchmark package.
Supports CSV/Excel loading, train-test split, imputation, scaling,
categorical encoding, and class imbalance handling (SMOTE).
"""
# pylint: disable=invalid-name, import-outside-toplevel, too-many-locals, too-many-branches, too-many-statements
# pylint: disable=redefined-outer-name, reimported, too-many-nested-blocks, broad-exception-caught, line-too-long
# pylint: disable=import-error

import os
import re

from imblearn.over_sampling import SMOTE
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    RobustScaler,
    StandardScaler,
)


def resolve_path(filepath, init_config=None):
    """
    Resolve data path robustly across operating systems using init_config.
    """
    if os.path.isabs(filepath):
        return filepath

    if init_config:
        base_dir = init_config.get("system", {}).get("base_dir", "")
        if base_dir:
            full_path = os.path.join(base_dir, filepath)
            if os.path.exists(full_path):
                return full_path
            # Check inside data_dir
            data_dir = init_config.get("paths", {}).get("data_dir", "")
            if data_dir:
                full_path_in_data = os.path.join(base_dir, data_dir, os.path.basename(filepath))
                if os.path.exists(full_path_in_data):
                    return full_path_in_data

    # Backwards compatibility check
    if os.path.exists(filepath):
        return filepath

    # Try parent directory relative checks
    alt_path = os.path.join("data_files", os.path.basename(filepath))
    if os.path.exists(alt_path):
        return alt_path

    return filepath


def load_data(filepath, init_config=None, nrows=None):
    """
    Load data from CSV or Excel file.

    Args:
        filepath: Path to the file.
        init_config: Optional dict containing path resolution info.
        nrows: Optional number of rows to read.

    Returns:
        pd.DataFrame: Loaded dataset.
    """
    resolved_path = resolve_path(filepath, init_config)
    print(f"Loading data from: {resolved_path}")

    ext = os.path.splitext(resolved_path)[1].lower()
    if ext == ".csv":
        return pd.read_csv(resolved_path, nrows=nrows)
    elif ext in [".xlsx", ".xls"]:
        return pd.read_excel(resolved_path, nrows=nrows)
    else:
        raise ValueError(
            f"Unsupported file format: {ext}. Only CSV and Excel (.xlsx, .xls) are supported."
        )


# Cache stopwords at module-level to prevent redundant imports inside clean_text_string()
_CACHED_STOP_WORDS = None


def _get_stopwords():
    global _CACHED_STOP_WORDS
    if _CACHED_STOP_WORDS is not None:
        return _CACHED_STOP_WORDS

    try:
        import nltk
        from nltk.corpus import stopwords

        try:
            _CACHED_STOP_WORDS = set(stopwords.words("english"))
        except LookupError:
            nltk.download("stopwords", quiet=True)
            _CACHED_STOP_WORDS = set(stopwords.words("english"))
    except Exception:
        # Fallback list of standard English stopwords
        _CACHED_STOP_WORDS = {
            "i",
            "me",
            "my",
            "myself",
            "we",
            "our",
            "ours",
            "ourselves",
            "you",
            "you're",
            "you've",
            "you'll",
            "you'd",
            "your",
            "yours",
            "yourself",
            "yourselves",
            "he",
            "him",
            "his",
            "himself",
            "she",
            "she's",
            "her",
            "hers",
            "herself",
            "it",
            "it's",
            "its",
            "itself",
            "they",
            "them",
            "their",
            "theirs",
            "themselves",
            "what",
            "which",
            "who",
            "whom",
            "this",
            "that",
            "that'll",
            "these",
            "those",
            "am",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "having",
            "do",
            "does",
            "did",
            "doing",
            "a",
            "an",
            "the",
            "and",
            "but",
            "if",
            "or",
            "because",
            "as",
            "until",
            "while",
            "of",
            "at",
            "by",
            "for",
            "with",
            "about",
            "against",
            "between",
            "into",
            "through",
            "during",
            "before",
            "after",
            "above",
            "below",
            "to",
            "from",
            "up",
            "down",
            "in",
            "out",
            "on",
            "off",
            "over",
            "under",
            "again",
            "further",
            "then",
            "once",
            "here",
            "there",
            "when",
            "where",
            "why",
            "how",
            "all",
            "any",
            "both",
            "each",
            "few",
            "more",
            "most",
            "other",
            "some",
            "such",
            "no",
            "nor",
            "not",
            "only",
            "own",
            "same",
            "so",
            "than",
            "too",
            "very",
            "s",
            "t",
            "can",
            "will",
            "just",
            "don",
            "don't",
            "should",
            "should've",
            "now",
            "d",
            "ll",
            "m",
            "o",
            "re",
            "ve",
            "y",
            "ain",
            "aren",
            "aren't",
            "couldn",
            "couldn't",
            "didn",
            "didn't",
            "doesn",
            "doesn't",
            "hadn",
            "hadn't",
            "hasn",
            "hasn't",
            "haven",
            "haven't",
            "isn",
            "isn't",
            "ma",
            "mightn",
            "mightn't",
            "mustn",
            "mustn't",
            "needn",
            "needn't",
            "shan",
            "shan't",
            "shouldn",
            "shouldn't",
            "wasn",
            "wasn't",
            "weren",
            "weren't",
            "won",
            "won't",
            "wouldn",
            "wouldn't",
        }
    return _CACHED_STOP_WORDS


def clean_text_string(text):
    """
    Clean text for TF-IDF baseline classification.
    Removes URLs, non-alphabetic characters, lowercases, and removes English stopwords.
    """
    import re

    text = str(text).lower()
    # Remove URLs
    text = re.sub(r"\S*https?:\S*", "", text)
    # Remove numbers and punctuation
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    stop_words = _get_stopwords()

    words = text.split()
    cleaned = [w for w in words if w not in stop_words and len(w) > 1]
    return " ".join(cleaned)


def prepare_data(df, data_config, init_config=None):
    """
    Process dataset based on configuration (splitting, scaling, encoding, SMOTE).

    Args:
        df: Input raw pd.DataFrame.
        data_config: Preprocessing and dataset configuration dict.
        init_config: Optional dict containing path resolution info.

    Returns:
        tuple: (X_train_prep, X_test_prep, y_train, y_test, feat_labels, preprocessor)
    """
    dataset_cfg = data_config.get("dataset", {})
    prep_cfg = data_config.get("preprocessing", {})

    target_col = dataset_cfg.get("target_column")
    classification_type = dataset_cfg.get("classification_type", "binary")

    if target_col not in df.columns:
        raise ValueError(
            f"Target column '{target_col}' not found in dataset columns: {df.columns.tolist()}"
        )

    data_type = dataset_cfg.get("data_type", "tabular")

    if data_type in ["text", "image", "multimodal"]:
        y = df[target_col].values

        # Split indices first to avoid data leakage
        test_size = prep_cfg.get("test_size", 0.2)
        random_state = prep_cfg.get("random_state", 4711)
        stratify_y = y if classification_type in ["binary", "multiclass"] else None

        if "split" in df.columns:
            idx_train = np.where(df["split"] != "test")[0]
            idx_test = np.where(df["split"] == "test")[0]
            print(f"Using pre-defined split column: {len(idx_train)} train, {len(idx_test)} test")
        else:
            idx_train, idx_test = train_test_split(
                np.arange(len(y)),
                test_size=test_size,
                random_state=random_state,
                stratify=stratify_y,
            )

        y_train, y_test = y[idx_train], y[idx_test]

        X_train_prep = None
        X_test_prep = None
        feat_labels = []
        preprocessor = None

        # 1. Text feature extraction (if text column is specified and exists)
        text_col = dataset_cfg.get("text_column")
        if text_col and text_col in df.columns:
            texts = df[text_col].fillna("").astype(str).tolist()
            texts_train = [texts[i] for i in idx_train]
            texts_test = [texts[i] for i in idx_test]

            text_features_mode = prep_cfg.get("text_features", "both")  # tfidf, embeddings, both

            # 1a. TF-IDF feature extraction
            if text_features_mode in ["tfidf", "both"]:
                print("Fitting TF-IDF vectorizer...")
                # Preprocess texts for TF-IDF in parallel
                from joblib import Parallel, delayed
                from sklearn.feature_extraction.text import TfidfVectorizer

                cleaned_train = Parallel(n_jobs=-1)(
                    delayed(clean_text_string)(t) for t in texts_train
                )
                cleaned_test = Parallel(n_jobs=-1)(
                    delayed(clean_text_string)(t) for t in texts_test
                )

                tfidf_max_features = prep_cfg.get("tfidf_max_features", 1000)
                tfidf = TfidfVectorizer(max_features=tfidf_max_features)
                X_tfidf_train = tfidf.fit_transform(cleaned_train).toarray()
                X_tfidf_test = tfidf.transform(cleaned_test).toarray()

                tfidf_feat_names = [f"tfidf_{w}" for w in tfidf.get_feature_names_out()]

                X_train_prep = X_tfidf_train
                X_test_prep = X_tfidf_test
                feat_labels.extend(tfidf_feat_names)
                preprocessor = tfidf

            # 1b. Embedding feature extraction (SBERT)
            if text_features_mode in ["embeddings", "both"]:
                import hashlib

                corpus_str = "".join(texts)
                corpus_hash = hashlib.md5(corpus_str.encode("utf-8")).hexdigest()
                emb_model_name = prep_cfg.get("embeddings_model", "all-MiniLM-L6-v2")
                emb_model_safe = re.sub(r"[^a-zA-Z0-9_-]", "_", emb_model_name)

                base_dir = "C:/Github/auto_benchmark"
                if init_config:
                    base_dir = init_config.get("system", {}).get("base_dir", base_dir)

                cache_dir = os.path.join(base_dir, "results", "cache")
                os.makedirs(cache_dir, exist_ok=True)
                cache_file = os.path.join(
                    cache_dir, f"embeddings_{corpus_hash}_{emb_model_safe}.npz"
                )

                if os.path.exists(cache_file):
                    print(f"Loading cached sentence embeddings from: {cache_file}")
                    embeddings = np.load(cache_file)["embeddings"]
                else:
                    from sentence_transformers import SentenceTransformer
                    import torch

                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    print(f"Computing SBERT embeddings on device: {device}")
                    try:
                        model = SentenceTransformer(emb_model_name, device=device)
                    except Exception as e:
                        print(
                            f"Failed loading {emb_model_name} online: {e}. Trying local_files_only=True..."
                        )
                        model = SentenceTransformer(
                            emb_model_name, device=device, local_files_only=True
                        )
                    lightly_cleaned_texts = [re.sub(r"\S*https?:\S*", "", t) for t in texts]
                    embeddings = model.encode(lightly_cleaned_texts, show_progress_bar=True)
                    np.savez_compressed(cache_file, embeddings=embeddings)
                    print(f"Saved computed sentence embeddings -> {cache_file}")

                X_emb_train = embeddings[idx_train]
                X_emb_test = embeddings[idx_test]
                emb_feat_names = [f"emb_{i}" for i in range(embeddings.shape[1])]

                if X_train_prep is not None:
                    X_train_prep = np.hstack([X_train_prep, X_emb_train])
                    X_test_prep = np.hstack([X_test_prep, X_emb_test])
                else:
                    X_train_prep = X_emb_train
                    X_test_prep = X_emb_test

                feat_labels.extend(emb_feat_names)

        # 2. Image feature extraction (if image column is specified and exists)
        image_col = dataset_cfg.get("image_column")
        if image_col and image_col in df.columns:
            image_dir = dataset_cfg.get("image_directory", "data_files")
            images = df[image_col].fillna("").astype(str).tolist()

            base_dir = "C:/Github/auto_benchmark"
            if init_config:
                base_dir = init_config.get("system", {}).get("base_dir", base_dir)

            if not os.path.isabs(image_dir):
                image_dir = os.path.join(base_dir, image_dir)

            backbone = prep_cfg.get("image_backbone", "resnet50")
            image_size = prep_cfg.get("image_size", 224)

            # Compute hash of filenames list to cache embeddings
            import hashlib

            img_str = "".join(images)
            img_hash = hashlib.md5(img_str.encode("utf-8")).hexdigest()
            backbone_safe = backbone.replace("/", "_").replace(":", "_")

            cache_dir = os.path.join(base_dir, "results", "cache")
            os.makedirs(cache_dir, exist_ok=True)
            cache_file = os.path.join(
                cache_dir, f"image_embeddings_{img_hash}_{backbone_safe}_{image_size}.npz"
            )

            if os.path.exists(cache_file):
                print(f"Loading cached image embeddings from: {cache_file}")
                img_embeddings = np.load(cache_file)["embeddings"]
            else:
                print(f"Extracting image embeddings using {backbone}...")
                from PIL import Image
                import torch
                from tqdm import tqdm

                device = "cuda" if torch.cuda.is_available() else "cpu"

                from autobenchmark.image_classification import get_feature_extractor, get_transforms

                model, emb_dim = get_feature_extractor(backbone, pretrained=True)
                model = model.to(device)
                model.eval()

                transform = get_transforms(image_size, is_training=False)

                img_embeddings_list = []
                for img_name in tqdm(images, desc="Extracting image features"):
                    img_path = os.path.join(image_dir, img_name)
                    if not os.path.exists(img_path):
                        # Fallback for training_images, test_images, or images subfolders
                        for sub in ["training_images", "test_images", "images"]:
                            fallback = os.path.join(image_dir, sub, img_name)
                            if os.path.exists(fallback):
                                img_path = fallback
                                break

                    try:
                        with Image.open(img_path) as img:
                            img_rgb = img.convert("RGB")
                            tensor = transform(img_rgb).unsqueeze(0).to(device)
                            with torch.no_grad():
                                feat = model(tensor).cpu().numpy().flatten()
                            img_embeddings_list.append(feat)
                    except Exception as e:
                        print(f"Warning: Failed to load/extract {img_path}: {e}")
                        img_embeddings_list.append(np.zeros(emb_dim))

                img_embeddings = np.array(img_embeddings_list)
                np.savez_compressed(cache_file, embeddings=img_embeddings)
                print(f"Saved computed image embeddings -> {cache_file}")

            X_img_train = img_embeddings[idx_train]
            X_img_test = img_embeddings[idx_test]
            img_feat_names = [f"img_emb_{i}" for i in range(img_embeddings.shape[1])]

            if X_train_prep is not None:
                X_train_prep = np.hstack([X_train_prep, X_img_train])
                X_test_prep = np.hstack([X_test_prep, X_img_test])
            else:
                X_train_prep = X_img_train
                X_test_prep = X_img_test

            feat_labels.extend(img_feat_names)

        return X_train_prep, X_test_prep, y_train, y_test, feat_labels, preprocessor

    # Drop rows where target is missing
    df = df[df[target_col].notna()].copy()

    # Target and features selection
    y = df[target_col]

    features_cfg = dataset_cfg.get("features", "all")
    exclude_cols = dataset_cfg.get("exclude_columns", [])

    if isinstance(features_cfg, list):
        # Specific list of features requested
        feature_cols = [col for col in features_cfg if col in df.columns and col != target_col]
    else:
        # All columns except target and exclusions
        feature_cols = [col for col in df.columns if col != target_col and col not in exclude_cols]

    X = df[feature_cols].copy()

    # Perform train-test split first (to avoid data leak in fitting pipeline)
    test_size = prep_cfg.get("test_size", 0.2)
    random_state = prep_cfg.get("random_state", 4711)

    # Handle stratify for classification
    stratify_y = y if classification_type in ["binary", "multiclass"] else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify_y
    )

    # Identify numeric and categorical columns
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    # 1. Numeric pipeline
    num_impute_strategy = prep_cfg.get("missing_value_handling", "mean")
    num_scaling = prep_cfg.get("scaling", "standard")

    num_steps = []
    if num_impute_strategy == "mean":
        num_steps.append(("imputer", SimpleImputer(strategy="median")))
    elif num_impute_strategy == "zero":
        num_steps.append(("imputer", SimpleImputer(strategy="constant", fill_value=0.0)))
    elif num_impute_strategy == "ignore":
        # HistGradientBoosting supports NaNs natively, but standard imputer can be skipped or pass-through
        pass

    if num_scaling == "standard":
        num_steps.append(("scaler", StandardScaler()))
    elif num_scaling == "minmax":
        num_steps.append(("scaler", MinMaxScaler()))
    elif num_scaling == "robust":
        num_steps.append(("scaler", RobustScaler()))

    num_pipe = Pipeline(num_steps) if num_steps else "passthrough"

    # 2. Categorical pipeline
    cat_impute_strategy = prep_cfg.get("missing_value_handling", "mean")
    cat_encoding = prep_cfg.get("categorical_encoding", "onehot")

    cat_steps = []
    if cat_impute_strategy == "mean":
        cat_steps.append(("imputer", SimpleImputer(strategy="most_frequent")))
    elif cat_impute_strategy == "zero":
        cat_steps.append(("imputer", SimpleImputer(strategy="constant", fill_value="missing")))
    elif cat_impute_strategy == "ignore":
        pass

    if cat_encoding == "onehot":
        cat_steps.append(("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)))
    elif cat_encoding == "ordinal":
        cat_steps.append(
            ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
        )

    cat_pipe = Pipeline(cat_steps) if cat_steps else "passthrough"

    # Combined preprocessor
    transformers = []
    if numeric_cols:
        transformers.append(("num", num_pipe, numeric_cols))
    if categorical_cols:
        transformers.append(("cat", cat_pipe, categorical_cols))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

    # Fit and transform
    X_train_prep = preprocessor.fit_transform(X_train)
    X_test_prep = preprocessor.transform(X_test)

    # Extract feature labels after preprocessing
    try:
        raw_feat_labels = preprocessor.get_feature_names_out().tolist()
        feat_labels = [re.sub(r"^(num|cat|remainder)__", "", col) for col in raw_feat_labels]
    except Exception as e:
        print(f"Error retrieving feature names: {e}. Falling back to default list.")
        feat_labels = [f"feature_{i}" for i in range(X_train_prep.shape[1])]

    # Imbalance handling (SMOTE)
    use_smote = prep_cfg.get("use_smote", False)
    if use_smote and classification_type in ["binary", "multiclass"]:
        # Apply SMOTE only on training set
        # (Ensure there are no missing values remaining, SMOTE fails on NaN)
        if np.isnan(X_train_prep).any():
            print(
                "Warning: Missing values found in preprocessed data. Imputing temporarily with 0 for SMOTE."
            )
            X_train_prep = np.nan_to_num(X_train_prep)
            X_test_prep = np.nan_to_num(X_test_prep)

        # Count samples per class to adjust k_neighbors if necessary
        class_counts = pd.Series(y_train).value_counts()
        min_samples = class_counts.min()
        k_neighbors = min(5, min_samples - 1)

        if k_neighbors >= 1:
            sm = SMOTE(random_state=42, k_neighbors=k_neighbors)
            X_train_prep, y_train = sm.fit_resample(X_train_prep, y_train)
            print(f"Applied multiclass-capable SMOTE (k_neighbors={k_neighbors})")
        else:
            print("Warning: SMOTE skipped because one of the classes has only 1 sample.")

    return X_train_prep, X_test_prep, y_train, y_test, feat_labels, preprocessor


def get_raw_text_splits(df, data_config):
    """
    Extract raw text splits aligned with the train-test split indices.
    """
    dataset_cfg = data_config.get("dataset", {})
    prep_cfg = data_config.get("preprocessing", {})
    target_col = dataset_cfg.get("target_column")
    text_col = dataset_cfg.get("text_column")
    classification_type = dataset_cfg.get("classification_type", "binary")

    # Drop rows where target or text is missing
    df = df[df[target_col].notna() & df[text_col].notna()].copy()

    y = df[target_col].values
    texts = df[text_col].fillna("").astype(str).tolist()

    test_size = prep_cfg.get("test_size", 0.2)
    random_state = prep_cfg.get("random_state", 4711)
    stratify_y = y if classification_type in ["binary", "multiclass"] else None

    idx_train, idx_test = train_test_split(
        np.arange(len(y)), test_size=test_size, random_state=random_state, stratify=stratify_y
    )

    texts_train = [texts[i] for i in idx_train]
    texts_test = [texts[i] for i in idx_test]
    y_train = y[idx_train]
    y_test = y[idx_test]

    return texts_train, texts_test, y_train, y_test
