"""
Script to train and evaluate supervised classifiers (e.g., XGBoost, SVM, Logistic Regression)
on pre-extracted visual and textual embeddings.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import pickle
import sys
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import SVC
import xgboost as xgb

# Ensure project root is in sys.path so we can import models/metrics
sys.path.append(str(Path(__file__).resolve().parents[1]))

from models.vlm.metrics_multilabel import compute_multilabel_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

SUBTYPE_LABELS = ["shaming", "stereotype", "objectification", "violence"]


def load_embeddings(
    split: str,
    embeddings_dir: Path,
    model_name: str,
    use_ocr: bool = False,
    ocr_engine: str = "easyocr",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Load pre-extracted embeddings and labels for a split from disk."""
    ocr_suffix = ""
    if use_ocr:
        ocr_suffix = f"_ocr_{ocr_engine}"
    filename = f"{split}_{model_name.lower().replace('-', '_')}{ocr_suffix}.npz"
    file_path = embeddings_dir / filename

    if not file_path.exists():
        logger.warning("Embeddings file not found: %s", file_path)
        return None

    logger.info("Loading embeddings from %s...", file_path)
    data = np.load(file_path, allow_pickle=True)
    return (
        data["image_embeddings"],
        data["text_embeddings"],
        data["labels"],
        data["subtask_labels"],
        data["image_ids"],
        data["raw_texts"],
    )


def fuse_embeddings(img_emb: np.ndarray, txt_emb: np.ndarray, mode: str) -> np.ndarray:
    """Combine image and text embeddings based on selected fusion mode."""
    if mode == "concat":
        return np.concatenate([img_emb, txt_emb], axis=1)
    elif mode == "image":
        return img_emb
    elif mode == "text":
        return txt_emb
    elif mode == "mult":
        # Assumes they have matching dimensions (which CLIP encoders do)
        if img_emb.shape[1] != txt_emb.shape[1]:
            raise ValueError(
                f"Element-wise multiplication requires matching dimensions. "
                f"Image: {img_emb.shape[1]}, Text: {txt_emb.shape[1]}"
            )
        return img_emb * txt_emb
    else:
        raise ValueError(f"Unknown fusion mode: {mode}")


def get_base_classifier(name: str, hyperparams: dict) -> Any:
    """Instantiate classification estimator according to factory choice."""
    if name == "xgboost":
        # Set default XGBoost params if not provided
        params = {
            "n_estimators": 100,
            "max_depth": 5,
            "learning_rate": 0.1,
            "random_state": 42,
            "eval_metric": "logloss",
        }
        params.update(hyperparams)
        return xgb.XGBClassifier(**params)

    elif name == "logistic_regression":
        params = {"max_iter": 1000, "random_state": 42, "C": 1.0}
        params.update(hyperparams)
        return LogisticRegression(**params)

    elif name == "svm":
        params = {"kernel": "rbf", "C": 1.0, "probability": True, "random_state": 42}
        params.update(hyperparams)
        return SVC(**params)

    elif name == "random_forest":
        params = {"n_estimators": 100, "random_state": 42}
        params.update(hyperparams)
        return RandomForestClassifier(**params)

    else:
        raise ValueError(f"Unknown classifier choice: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a classifier on MAMI embeddings")
    parser.add_argument(
        "--embeddings-dir",
        default="results/embeddings",
        help="Directory where embeddings are saved",
    )
    parser.add_argument(
        "--model-name",
        default="ViT-L-14",
        help="CLIP model name used during extraction",
    )
    parser.add_argument(
        "--use-ocr",
        action="store_true",
        help="Whether to load OCR-extracted embeddings instead of default transcripts",
    )
    parser.add_argument(
        "--ocr-engine",
        default="easyocr",
        choices=["easyocr", "paddleocr"],
        help="OCR engine used during embedding extraction",
    )
    parser.add_argument(
        "--task",
        default="singleclass",
        choices=["singleclass", "multiclass"],
        help="Task type: singleclass (binary misogyny) or multiclass (Challenge 2 sub-types)",
    )
    parser.add_argument(
        "--classifier",
        default="xgboost",
        choices=["xgboost", "logistic_regression", "svm", "random_forest"],
        help="Classifier architecture to train",
    )
    parser.add_argument(
        "--fusion",
        default="concat",
        choices=["concat", "image", "text", "mult"],
        help="Fusion strategy for image and text representations",
    )
    parser.add_argument(
        "--hyperparams",
        default="{}",
        help="JSON string of classifier hyperparameter overrides",
    )
    parser.add_argument(
        "--output-dir",
        default="results/models",
        help="Directory to save the trained classifier model file",
    )
    args = parser.parse_args()

    embeddings_dir = Path(args.embeddings_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load Hyperparameters JSON
    try:
        hyperparams = json.loads(args.hyperparams)
    except json.JSONDecodeError as e:
        logger.error("Failed to parse hyperparameters JSON: %s", e)
        sys.exit(1)

    # 1. Load train/val embeddings
    train_data = load_embeddings(
        "train", embeddings_dir, args.model_name, args.use_ocr, args.ocr_engine
    )
    val_data = load_embeddings(
        "validation", embeddings_dir, args.model_name, args.use_ocr, args.ocr_engine
    )
    test_data = load_embeddings(
        "test", embeddings_dir, args.model_name, args.use_ocr, args.ocr_engine
    )

    if train_data is None:
        logger.critical("Train embeddings not found. Please extract them first.")
        sys.exit(1)
    if val_data is None:
        logger.critical("Validation embeddings not found. Please extract them first.")
        sys.exit(1)

    train_img, train_txt, train_y_bin, train_y_multi, _, _ = train_data
    val_img, val_txt, val_y_bin, val_y_multi, _, _ = val_data

    # 2. Fuse embeddings
    logger.info("Fusing embeddings using mode: %s...", args.fusion)
    train_X = fuse_embeddings(train_img, train_txt, args.fusion)
    val_X = fuse_embeddings(val_img, val_txt, args.fusion)

    # Determine targets based on task
    if args.task == "singleclass":
        train_y = train_y_bin
        val_y = val_y_bin
    else:  # multiclass (multilabel)
        train_y = train_y_multi
        val_y = val_y_multi

    logger.info("Train X shape: %s, targets shape: %s", train_X.shape, train_y.shape)
    logger.info("Validation X shape: %s, targets shape: %s", val_X.shape, val_y.shape)

    # 3. Instantiate model
    base_estimator = get_base_classifier(args.classifier, hyperparams)

    # For multiclass (multilabel), wrap model in sklearn MultiOutputClassifier
    if args.task == "multiclass":
        logger.info("Wrapping %s in MultiOutputClassifier for multilabel task...", args.classifier)
        model = MultiOutputClassifier(base_estimator)
    else:
        model = base_estimator

    # 4. Train model
    logger.info("Training %s classifier on train split...", args.classifier)
    model.fit(train_X, train_y)
    logger.info("Training completed successfully.")

    # 5. Evaluate on Validation split
    val_preds = model.predict(val_X)

    logger.info("--- Evaluation Results (Validation Split) ---")
    val_results = []
    if args.task == "singleclass":
        from sklearn.metrics import accuracy_score, classification_report, f1_score

        acc = accuracy_score(val_y, val_preds)
        f1_macro = f1_score(val_y, val_preds, average="macro")
        f1_bin = f1_score(val_y, val_preds, average="binary")

        print(f"Accuracy : {acc:.4f}")
        print(f"Macro F1 : {f1_macro:.4f}")
        print(f"Binary F1: {f1_bin:.4f}")
        print("\nClassification Report:")
        print(
            classification_report(val_y, val_preds, target_names=["not misogynous", "misogynous"])
        )
        val_results.append(
            {
                "model": args.classifier,
                "filter": "none",
                "split": "validation",
                "task": args.task,
                "exact_match_accuracy": float(acc),
                "f1": float(f1_macro),
                "macro_f1": float(f1_macro),
            }
        )
    else:
        # Format predicted matrix to list of dicts for compute_multilabel_metrics
        pred_dicts = []
        gt_dicts = []
        for i in range(len(val_preds)):
            pred_dicts.append({lbl: int(val_preds[i][j]) for j, lbl in enumerate(SUBTYPE_LABELS)})
            gt_dicts.append({lbl: int(val_y[i][j]) for j, lbl in enumerate(SUBTYPE_LABELS)})

        metrics = compute_multilabel_metrics(pred_dicts, gt_dicts, SUBTYPE_LABELS)
        print(f"Exact Match Accuracy: {metrics['exact_match_accuracy']:.4f}")
        print(f"Macro F1             : {metrics['macro_f1']:.4f}")
        print(f"Micro F1             : {metrics['micro_f1']:.4f}")
        print(f"Weighted F1          : {metrics['weighted_f1']:.4f}")
        print("\nPer-Class Metrics:")
        for label, scores in metrics["per_class"].items():
            print(
                f"  {label:<15}: Precision={scores['precision']:.4f}, "
                f"Recall={scores['recall']:.4f}, F1={scores['f1']:.4f} "
                f"(Support={scores['support']})"
            )
        val_results.append(
            {
                "model": args.classifier,
                "filter": "none",
                "split": "validation",
                "task": args.task,
                "exact_match_accuracy": float(metrics["exact_match_accuracy"]),
                "f1": float(metrics["macro_f1"]),
                "macro_f1": float(metrics["macro_f1"]),
            }
        )

    val_dir = RESULTS_DIR / "validation"
    val_dir.mkdir(parents=True, exist_ok=True)
    val_json_path = val_dir / f"{args.classifier}_validation_{args.classifier}_{args.task}.json"
    val_json_path.write_text(json.dumps(val_results, indent=2) + "\n", encoding="utf-8")
    logger.info("Saved validation metrics JSON to %s", val_json_path)

    # 6. Evaluate on Test split if present
    if test_data is not None:
        logger.info("Evaluating on Test split...")
        test_img, test_txt, test_y_bin, test_y_multi, _, _ = test_data
        test_X = fuse_embeddings(test_img, test_txt, args.fusion)
        test_y = test_y_bin if args.task == "singleclass" else test_y_multi
        test_preds = model.predict(test_X)

        test_results = []
        logger.info("--- Evaluation Results (Test Split) ---")
        if args.task == "singleclass":
            from sklearn.metrics import accuracy_score, f1_score

            acc = accuracy_score(test_y, test_preds)
            f1_macro = f1_score(test_y, test_preds, average="macro")
            print(f"Accuracy : {acc:.4f}")
            print(f"Macro F1 : {f1_macro:.4f}")
            test_results.append(
                {
                    "model": args.classifier,
                    "filter": "none",
                    "split": "test",
                    "task": args.task,
                    "exact_match_accuracy": float(acc),
                    "f1": float(f1_macro),
                    "macro_f1": float(f1_macro),
                }
            )
        else:
            pred_dicts = []
            gt_dicts = []
            for i in range(len(test_preds)):
                pred_dicts.append(
                    {lbl: int(test_preds[i][j]) for j, lbl in enumerate(SUBTYPE_LABELS)}
                )
                gt_dicts.append({lbl: int(test_y[i][j]) for j, lbl in enumerate(SUBTYPE_LABELS)})
            metrics = compute_multilabel_metrics(pred_dicts, gt_dicts, SUBTYPE_LABELS)
            print(f"Exact Match Accuracy: {metrics['exact_match_accuracy']:.4f}")
            print(f"Macro F1             : {metrics['macro_f1']:.4f}")
            test_results.append(
                {
                    "model": args.classifier,
                    "filter": "none",
                    "split": "test",
                    "task": args.task,
                    "exact_match_accuracy": float(metrics["exact_match_accuracy"]),
                    "f1": float(metrics["macro_f1"]),
                    "macro_f1": float(metrics["macro_f1"]),
                }
            )

        test_dir = RESULTS_DIR / "test"
        test_dir.mkdir(parents=True, exist_ok=True)
        test_json_path = test_dir / f"{args.classifier}_test_{args.classifier}_{args.task}.json"
        test_json_path.write_text(json.dumps(test_results, indent=2) + "\n", encoding="utf-8")
        logger.info("Saved test metrics JSON to %s", test_json_path)

    # 7. Save model
    ocr_suffix = ""
    if args.use_ocr:
        ocr_suffix = f"_ocr_{args.ocr_engine}"
    model_filename = f"{args.classifier}_{args.task}_{args.fusion}_{args.model_name.lower().replace('-', '_')}{ocr_suffix}.pkl"
    save_path = output_dir / model_filename
    with save_path.open("wb") as f:
        pickle.dump(model, f)
    logger.info("Saved trained classifier model to %s", save_path)


if __name__ == "__main__":
    main()
