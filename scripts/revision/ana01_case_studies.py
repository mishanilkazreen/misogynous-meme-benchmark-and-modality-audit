"""ANA-01: per-example SHAP case studies of misclassified test memes.

This script re-fits the XGBoost concatenated-fusion classifier on frozen CLIP
ViT-L-14-quickgelu embeddings with the exact settings used by
``scripts/revision/modality_ablation.py`` (``XGBClassifier(verbosity=0,
random_state=42, tree_method="hist", device="cpu")``, no scaler, concat columns
0:768 image / 768:1536 text), scores the test split with the official
SemEval-2022 Task 5 scorer, identifies the memes the model misclassifies from its
MEASURED predictions, computes per-example SHAP attributions with
``shap.TreeExplainer``, and selects a reproducible set of misclassified memes by a
deterministic measured rule (highest image-block absolute-SHAP share, fixed
tie-break and sample cap). It writes the committed artefacts
``results/revision/ana01/case_studies.json`` and ``summary.md`` with provenance
(model descriptor, split, embedding SHA-256) on every record, plus one per-example
SHAP figure per selected meme under ``results/figures/ana01_shap_<image_id>.png``.

Anti-expansion policy: this module reads ONLY committed or locally produced
artefacts (the CLIP ``.npz`` feature files written by
``scripts/revision/extract_features.py`` and the MAMI dataset accessed through
``utils.dataset.DatasetManager`` for verbatim transcripts). It never fabricates or
hand-picks examples; the selection is a documented deterministic function of the
measured SHAP values.

Usage:
    uv run python -m scripts.revision.ana01_case_studies
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import platform
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shap
import sklearn
import xgboost
from xgboost import XGBClassifier

from scripts.official_mami_evaluation import compute_scoreA
from utils.dataset import DatasetManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ana01_case_studies")

ROOT = Path(__file__).resolve().parents[2]
EMB_DIR = ROOT / "results" / "embeddings"
OUT_DIR = ROOT / "results" / "revision" / "ana01"
FIG_DIR = ROOT / "results" / "figures"
GLOBAL_SHAP_CSV = ROOT / "results" / "shap_modality_importance.csv"

SPLITS: tuple[str, ...] = ("train", "validation", "test")
IMAGE_DIM = 768
TEXT_DIM = 768
MAX_SAMPLES = 6
MODEL_DESCRIPTOR = (
    "XGBoost concat fusion on frozen CLIP ViT-L-14-quickgelu/openai embeddings "
    "(image cols 0:768, text cols 768:1536), re-fitted on the train split with "
    "auto_benchmark get_models_dict defaults (verbosity=0, random_state=42, "
    "tree_method='hist', device='cpu'), no scaler"
)
SELECTION_RULE = (
    "From the test split, keep memes the re-fitted model misclassifies on Task A "
    "(predicted label != true label, measured). Rank them by descending image-block "
    "absolute-SHAP share (sum of |SHAP| over image columns 0:768 divided by the sum "
    "over all 1536 columns). Break ties by ascending integer image_id. Keep the top "
    f"{MAX_SAMPLES}. No example is hand-picked; the set is a deterministic function "
    "of the measured SHAP values."
)


@dataclass
class SplitData:
    """Embeddings and labels of one split from one CLIP feature file."""

    image: np.ndarray
    text: np.ndarray
    y_a: np.ndarray
    ids: np.ndarray
    sha256: str
    path: str


def sha256_file(path: Path) -> str:
    """Return the hex SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_split(split: str) -> SplitData:
    """Load one CLIP ViT-L-14-quickgelu split, raising if the feature file is absent."""
    path = EMB_DIR / f"{split}_vit_l_14_quickgelu.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing; regenerate with "
            "'uv run python -m scripts.revision.extract_features --features clip'"
        )
    npz = np.load(path)
    image = npz["image_embeddings"].astype(np.float32)
    text = npz["text_embeddings"].astype(np.float32)
    if image.shape[1] != IMAGE_DIM or text.shape[1] != TEXT_DIM:
        raise ValueError(
            f"{path}: expected {IMAGE_DIM}+{TEXT_DIM} dims, got {image.shape[1]}+{text.shape[1]}"
        )
    return SplitData(
        image=image,
        text=text,
        y_a=npz["labels"].astype(int),
        ids=np.asarray([str(x) for x in npz["image_ids"]]),
        sha256=sha256_file(path),
        path=str(path.relative_to(ROOT)).replace("\\", "/"),
    )


def concat_features(data: SplitData) -> np.ndarray:
    """Return the concatenated [image; text] feature matrix (columns 0:768 image)."""
    return np.hstack([data.image, data.text])


def shap_to_class1(shap_values: Any, n_features: int) -> np.ndarray:
    """Normalise ``TreeExplainer`` output to a 2D positive-class array.

    ``shap`` returns a list (one array per class), a 3D array (n, features, classes)
    or a 2D array (n, features) depending on version; this collapses every form to
    the positive-class (index 1 / last) 2D array of shape (n_samples, n_features).
    """
    if isinstance(shap_values, list):
        arr = np.asarray(shap_values[1] if len(shap_values) > 1 else shap_values[0])
    else:
        arr = np.asarray(shap_values)
        if arr.ndim == 3:
            arr = arr[:, :, -1]
    if arr.ndim != 2 or arr.shape[1] != n_features:
        raise ValueError(f"unexpected SHAP shape {arr.shape}; expected (n, {n_features})")
    return arr


def tree_shap_values(model: XGBClassifier, x: np.ndarray) -> tuple[np.ndarray, str]:
    """Return per-example TreeSHAP values and the method used.

    Prefers ``shap.TreeExplainer``. When the installed ``shap`` cannot parse the
    XGBoost 3.x ``base_score`` config (shap issue #4184: ``base_score`` is stored
    as a bracketed vector string such as ``'[5E-1]'``), it falls back to the
    identical TreeSHAP computed natively by the fitted booster through
    ``Booster.predict(..., pred_contribs=True)``. Both paths run exact TreeSHAP on
    the same fitted model; the fallback returns the per-feature contributions with
    the trailing bias column dropped.
    """
    n_features = x.shape[1]
    try:
        explainer = shap.TreeExplainer(model)
        return shap_to_class1(explainer.shap_values(x), n_features), "shap.TreeExplainer"
    except ValueError:
        booster = model.get_booster()
        dmatrix = xgboost.DMatrix(x)
        contribs = np.asarray(booster.predict(dmatrix, pred_contribs=True))
        if contribs.ndim == 3:
            contribs = contribs[:, -1, :]
        values = contribs[:, :n_features]
        if values.shape != (x.shape[0], n_features):
            raise ValueError(
                f"unexpected pred_contribs shape {contribs.shape}; expected (n, {n_features}+1)"
            ) from None
        return values, "xgboost.Booster.pred_contribs"


def official_score_a(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Score A from ``scripts/official_mami_evaluation.py`` on binary Task A labels."""
    truth = {str(i): [bool(y_true[i]), False, False, False, False] for i in range(len(y_true))}
    submission = {str(i): [bool(y_pred[i]), False, False, False, False] for i in range(len(y_pred))}
    return float(compute_scoreA(truth, submission))


def load_transcripts() -> dict[str, str]:
    """Return a verbatim ``image_id -> text`` map for the test split (rule 6)."""
    dataset = DatasetManager().load_dataset(split="test")
    transcripts: dict[str, str] = {}
    for idx in range(len(dataset)):
        sample = dataset[idx]
        transcripts[str(sample["image_id"])] = str(sample.get("text", ""))
    return transcripts


def global_split_reference() -> dict[str, float]:
    """Read the committed global image/text SHAP percentages from the CSV."""
    reference: dict[str, float] = {}
    for line in GLOBAL_SHAP_CSV.read_text(encoding="utf-8").splitlines()[1:]:
        parts = line.split(",")
        if len(parts) == 3:
            reference[parts[0]] = float(parts[2])
    return reference


def save_shap_figure(
    image_id: str,
    image_share: float,
    text_share: float,
    true_label: int,
    pred_label: int,
) -> str:
    """Save a per-example image-vs-text absolute-SHAP share bar chart and return its filename."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(
        ["Image block (CLIP visual)", "Text block (CLIP text)"],
        [100.0 * image_share, 100.0 * text_share],
        color=["#4C72B0", "#DD8452"],
    )
    ax.set_ylabel("Per-example modality reliance (% of total |SHAP|)")
    ax.set_title(
        f"Meme {image_id}: true={true_label}, predicted={pred_label}",
    )
    ax.set_ylim(0, 100)
    for bar, share in zip(bars, [image_share, text_share], strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{100.0 * share:.1f}%",
            ha="center",
            va="bottom",
        )
    fig.tight_layout()
    filename = f"ana01_shap_{image_id}.png"
    fig.savefig(FIG_DIR / filename, dpi=150)
    plt.close(fig)
    return filename


def build_model() -> XGBClassifier:
    """Return the unfitted XGBoost concat classifier with the ablation settings."""
    return XGBClassifier(verbosity=0, random_state=42, tree_method="hist", device="cpu")


def main() -> int:
    """Re-fit, score, compute SHAP, select misclassified memes and write the artefacts."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--max-samples", type=int, default=MAX_SAMPLES)
    args = parser.parse_args()
    max_samples = args.max_samples

    splits = {name: load_split(name) for name in SPLITS}
    train, test = splits["train"], splits["test"]

    model = build_model()
    model.fit(concat_features(train), train.y_a)

    x_test = concat_features(test)
    y_pred = np.asarray(model.predict(x_test)).astype(int)
    score_a = official_score_a(test.y_a, y_pred)
    accuracy = float(np.mean(y_pred == test.y_a))

    misclassified = np.flatnonzero(y_pred != test.y_a)
    false_positives = int(np.sum((y_pred == 1) & (test.y_a == 0)))
    false_negatives = int(np.sum((y_pred == 0) & (test.y_a == 1)))
    logger.info(
        "Test Score A %.4f, accuracy %.4f, %d misclassified (%d FP, %d FN)",
        score_a,
        accuracy,
        len(misclassified),
        false_positives,
        false_negatives,
    )
    if len(misclassified) == 0:
        raise RuntimeError("no misclassified test memes; cannot build case studies")

    shap_values, shap_method = tree_shap_values(model, x_test)
    logger.info("Computed TreeSHAP via %s", shap_method)
    abs_shap = np.abs(shap_values)
    image_abs = abs_shap[:, :IMAGE_DIM].sum(axis=1)
    text_abs = abs_shap[:, IMAGE_DIM:].sum(axis=1)
    total_abs = image_abs + text_abs
    image_share = np.where(total_abs > 0, image_abs / total_abs, 0.0)

    def sort_key(index: int) -> tuple[float, int]:
        return (-float(image_share[index]), int(test.ids[index]))

    selected = sorted((int(i) for i in misclassified), key=sort_key)[:max_samples]

    transcripts = load_transcripts()
    reference = global_split_reference()

    records: list[dict[str, Any]] = []
    for rank, index in enumerate(selected, start=1):
        image_id = str(test.ids[index])
        img_share = float(image_share[index])
        txt_share = float(1.0 - img_share)
        figure = save_shap_figure(
            image_id, img_share, txt_share, int(test.y_a[index]), int(y_pred[index])
        )
        records.append(
            {
                "image_id": image_id,
                "split": "test",
                "model": MODEL_DESCRIPTOR,
                "embedding_file": test.path,
                "embedding_sha256": test.sha256,
                "true_label": int(test.y_a[index]),
                "predicted_label": int(y_pred[index]),
                "error_type": "FP" if y_pred[index] == 1 else "FN",
                "image_block_abs_shap": float(image_abs[index]),
                "text_block_abs_shap": float(text_abs[index]),
                "image_block_share": img_share,
                "text_block_share": txt_share,
                "selection_rank": rank,
                "transcript": transcripts.get(image_id, ""),
                "figure": figure,
            }
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "task": "ANA-01 qualitative case studies of misclassified memes",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": "uv run python -m scripts.revision.ana01_case_studies",
        "scorer": "scripts/official_mami_evaluation.py (compute_scoreA)",
        "model": MODEL_DESCRIPTOR,
        "shap_method": shap_method,
        "selection_rule": SELECTION_RULE,
        "max_samples": max_samples,
        "test_embedding_file": test.path,
        "test_embedding_sha256": test.sha256,
        "train_embedding_file": train.path,
        "train_embedding_sha256": train.sha256,
        "test_score_a": score_a,
        "test_accuracy": accuracy,
        "n_misclassified": len(misclassified),
        "n_false_positives": false_positives,
        "n_false_negatives": false_negatives,
        "global_modality_shap_percent": reference,
        "versions": {
            "python": platform.python_version(),
            "shap": shap.__version__,
            "xgboost": xgboost.__version__,
            "sklearn": sklearn.__version__,
            "numpy": np.__version__,
        },
        "records": records,
    }
    (OUT_DIR / "case_studies.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    summary_lines = [
        "# ANA-01 Qualitative Case Studies of Misclassified Memes",
        "",
        "Model:",
        "",
        "```text",
        MODEL_DESCRIPTOR,
        "```",
        "",
        f"Test Score A (official scorer): {score_a:.4f}; accuracy {accuracy:.4f}.",
        "",
        (
            f"Misclassified test memes: {len(misclassified)} "
            f"({false_positives} false positive, {false_negatives} false negative)."
        ),
        "",
        "## Selection rule",
        "",
        "```text",
        SELECTION_RULE,
        "```",
        "",
        "## Global modality SHAP reference",
        "",
        (
            f"Global XGBoost SHAP split (results/shap_modality_importance.csv): "
            f"image {reference.get('image', float('nan')):.2f}%, "
            f"text {reference.get('text', float('nan')):.2f}%."
        ),
        "",
        "## Selected memes",
        "",
        "| Rank | image_id | True | Predicted | Error | Image |SHAP| share | Figure |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for record in records:
        summary_lines.append(
            f"| {record['selection_rank']} | {record['image_id']} | "
            f"{record['true_label']} | {record['predicted_label']} | "
            f"{record['error_type']} | {100.0 * record['image_block_share']:.2f}% | "
            f"{record['figure']} |"
        )
    (OUT_DIR / "summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    logger.info(
        "Wrote %s, %s and %d figures",
        OUT_DIR / "case_studies.json",
        OUT_DIR / "summary.md",
        len(records),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
