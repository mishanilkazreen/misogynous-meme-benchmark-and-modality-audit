#!/usr/bin/env python3
"""CLIP Concept Activation Vectors (CAV) for the XGBoost Fusion classifier.

Implements Gist task ID 13 / GitHub issue #89, Part 2.

Goal: translate the abstract image-embedding dimensions that the XGBoost Fusion
model relies on into human-understandable concepts, using CLIP's shared
image/text latent space.

Method
------
1. Load the trained binary XGBoost model and read its native
   ``feature_importances_``. The fused feature vector is
   ``np.concatenate([img_emb, txt_emb], axis=1)`` (see
   ``scripts/train_classifier.py``), so columns ``0..767`` are the CLIP visual
   features. We take the top-K most important *image* dimensions.
2. Define a list of semantic concept text strings and encode them with the SAME
   CLIP model that produced the image embeddings (``ViT-L-14-quickgelu`` /
   ``openai``). The text and image towers share a normalised 768-dim space, so
   cosine similarity between them is meaningful.
3. For each important image dimension ``d`` the cosine similarity between the
   unit basis vector ``e_d`` and a normalised concept vector ``v_c`` reduces to
   ``v_c[d]``. That value says how strongly concept ``c`` loads onto the
   dimension the model cares about (e.g. "dim 112 aligns 0.31 with 'violence'").
4. We additionally report the dataset-level concept presence: the mean cosine
   similarity between every test image embedding and each concept vector.

Outputs (paths relative to the parent content-moderation repo):
    results/cav_concept_similarity.csv        (dimension x concept matrix)
    results/cav_global_concept_presence.csv   (mean image-vs-concept similarity)
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import pickle

import numpy as np

# Repo root = parent of auto_benchmark/. Outputs go to the parent repo's results/.
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = (
    REPO_ROOT
    / "results"
    / "models"
    / "xgboost_singleclass_concat_vit_l_14_quickgelu_ocr_paddleocr.pkl"
)
DEFAULT_EMB = REPO_ROOT / "results" / "embeddings" / "test_vit_l_14_quickgelu_ocr_paddleocr.npz"

# Semantic concepts spanning the four MAMI sub-types plus generic scene/visual
# cues. Kept human-readable; extend freely.
DEFAULT_CONCEPTS: list[str] = [
    "violence against women",
    "physical aggression",
    "body shaming",
    "fat shaming",
    "gender stereotype",
    "woman in the kitchen",
    "woman doing housework",
    "sexual objectification",
    "revealing clothing",
    "derogatory gesture",
    "insulting text",
    "smiling woman",
    "angry man",
    "cartoon drawing",
    "photograph of a person",
    "text overlay on image",
    "pornographic content",
    "neutral everyday scene",
]


def get_feature_importances(model: object, n_features: int) -> np.ndarray:
    """Return a length-``n_features`` importance vector from the model.

    Handles a bare XGBClassifier (``feature_importances_``) and a calibrated /
    pipeline wrapper that exposes the same attribute.
    """
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        raise SystemExit("Model has no feature_importances_; expected an XGBoost classifier.")
    importances = np.asarray(importances, dtype=float)
    if importances.shape[0] != n_features:
        raise SystemExit(
            f"feature_importances_ length {importances.shape[0]} != feature count {n_features}."
        )
    return importances


def encode_concepts(
    concepts: list[str], model_name: str, pretrained: str, device: str
) -> np.ndarray:
    """Encode concept strings into L2-normalised CLIP text embeddings."""
    import open_clip  # heavy import; only needed at run time
    import torch

    model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    model = model.to(device)
    model.eval()
    tokenizer = open_clip.get_tokenizer(model_name)

    tokens = tokenizer(concepts).to(device)
    with torch.no_grad():
        feats = model.encode_text(tokens)
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.cpu().numpy().astype(np.float64)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL))
    parser.add_argument("--embeddings", default=str(DEFAULT_EMB))
    parser.add_argument("--clip-model", default="ViT-L-14-quickgelu")
    parser.add_argument("--pretrained", default="openai")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--image-dim",
        type=int,
        default=768,
        help="Number of image-feature columns at the start of the fused vector",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Number of top XGBoost-important image dimensions to report",
    )
    args = parser.parse_args()

    model_path = Path(args.model_path)
    emb_path = Path(args.embeddings)
    if not model_path.exists():
        raise SystemExit(f"Model not found: {model_path}")
    if not emb_path.exists():
        raise SystemExit(f"Embeddings not found: {emb_path}")

    print(f"Loading model: {model_path}")
    with model_path.open("rb") as f:
        model = pickle.load(f)

    print(f"Loading embeddings: {emb_path}")
    data = np.load(emb_path, allow_pickle=True)
    img_emb = np.asarray(data["image_embeddings"], dtype=np.float64)
    txt_emb = np.asarray(data["text_embeddings"], dtype=np.float64)
    n_features = img_emb.shape[1] + txt_emb.shape[1]
    image_dim = args.image_dim
    if image_dim >= n_features:
        raise SystemExit(f"--image-dim {image_dim} >= total features {n_features}")

    importances = get_feature_importances(model, n_features)
    image_importances = importances[:image_dim]
    top_k = min(args.top_k, image_dim)
    top_dims = np.argsort(image_importances)[::-1][:top_k]
    print(f"Top {top_k} important image dimensions: {top_dims.tolist()}")

    print(f"Encoding {len(DEFAULT_CONCEPTS)} concepts with {args.clip_model} ({args.pretrained})")
    concept_vecs = encode_concepts(DEFAULT_CONCEPTS, args.clip_model, args.pretrained, args.device)
    if concept_vecs.shape[1] != image_dim:
        raise SystemExit(
            f"Concept embedding dim {concept_vecs.shape[1]} != image dim {image_dim}. "
            "The CLIP text tower must match the image-embedding model."
        )

    results_dir = REPO_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # 1) Dimension x concept matrix. cosine(e_d, v_c) = v_c[d] for unit basis e_d
    #    and L2-normalised concept vector v_c.
    matrix_path = results_dir / "cav_concept_similarity.csv"
    with matrix_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_dimension", "xgb_importance", *DEFAULT_CONCEPTS])
        for d in top_dims:
            sims = concept_vecs[:, d]
            writer.writerow([int(d), f"{image_importances[d]:.6f}", *[f"{s:.6f}" for s in sims]])
    print(f"Saved CAV dimension matrix: {matrix_path}")

    # 2) Dataset-level concept presence: mean cosine similarity between every
    #    L2-normalised image embedding and each concept vector.
    img_norm = img_emb / np.linalg.norm(img_emb, axis=1, keepdims=True)
    mean_sims = (img_norm @ concept_vecs.T).mean(axis=0)
    presence_path = results_dir / "cav_global_concept_presence.csv"
    order = np.argsort(mean_sims)[::-1]
    with presence_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["concept", "mean_cosine_similarity"])
        for i in order:
            writer.writerow([DEFAULT_CONCEPTS[i], f"{mean_sims[i]:.6f}"])
    print(f"Saved global concept presence: {presence_path}")

    print("\nTop concepts by mean image similarity:")
    for i in order[:5]:
        print(f"  {DEFAULT_CONCEPTS[i]:<28} {mean_sims[i]:.4f}")


if __name__ == "__main__":
    main()
