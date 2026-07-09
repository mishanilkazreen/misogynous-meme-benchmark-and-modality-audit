<!-- markdownlint-disable MD013 -->
<!-- cspell:ignore preprocessing -->
# Methodology Notes (for paper writing)

Working document capturing experimental design decisions and implementation
details that must be accurately described in the paper. Keep this updated as
the final methodology section is drafted.

Source code: <https://github.com/mishanilkazreen/content-moderation>

---

## 1. Input modalities by model family

| Model family | Image input | Text input | Notes |
| :--- | :--- | :--- | :--- |
| **VLMs (Qwen2-VL, Gemini)** | Raw meme image (as-is, no preprocessing) | None — the model reads overlay text from the image | See note below |
| **CLIP fine-tuned (frozen towers)** | CLIP-preprocessed (resize + centre crop + normalise) | CLIP text encoder applied to MAMI's provided transcript | Classification head trained on concatenated embeddings |
| **CLIP zero-shot** | CLIP-preprocessed | Phrase-ensemble text embeddings (see PROMPTS.md) | Cosine similarity, no training |
| **Tabular (XGBoost, SVM, etc.)** | Pre-extracted CLIP ViT-L-14 image embedding (768-d) | Pre-extracted CLIP text embedding of transcript (768-d) | Concatenated 1536-d vector |

### Critical distinction for VLMs (paper methodology must state this clearly)

Our headline VLM results use `--text-source provided`, which for VLMs means
**no text is injected into the prompt**. The model receives only:

1. The raw meme image (which visually contains the overlay text).
2. A classification prompt (see `paper/PROMPTS.md`).

The model must read and interpret any text on the meme using its own vision
encoder. This is the realistic content-moderation scenario — in production,
you receive an image and must classify it without a separately-maintained OCR
pipeline.

This contrasts with the tabular/CLIP pipeline, where "provided text" means the
MAMI dataset's manually-verified `Text Transcription` column is explicitly
encoded as a text feature. Both setups are valid but test different things:

- **VLM**: end-to-end meme understanding (vision + language reasoning).
- **Tabular/CLIP**: fusion of separately-extracted image and text features.

The `--text-source ocr` variant (where PaddleOCR-extracted text is prepended
to the VLM prompt as `This meme contains the text: "..."`) is reported as an
**ablation** only and is not the headline result.

---

## 2. Training recipe summary

| System | Backbone | Trainable params | Epochs | Effective batch | LR | Best-val selection | Seeds |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Qwen2-VL-7B QLoRA | Qwen2-VL-7B-Instruct (4-bit NF4) | All-linear LoRA r=16 | 3 | 32 (4×8 accum) | 2e-5 | Yes (per epoch) | 1–3 |
| Qwen2-VL-2B QLoRA | Qwen2-VL-2B-Instruct (4-bit NF4) | All-linear LoRA r=16 | 3 | 32 (4×8 accum) | 2e-5 | Yes (per epoch) | 1–3 |
| CLIP ViT-L-14 (frozen) | open_clip ViT-L-14-quickgelu | 2-layer MLP head | 10 | 32 | 1e-4 + cosine | Yes (per epoch) | 1–3 |
| CLIP ViT-B-32 (frozen) | open_clip ViT-B-32-quickgelu | 2-layer MLP head | 10 | 32 | 1e-4 + cosine | Yes (per epoch) | 1–3 |
| Tabular sweep | N/A (frozen CLIP embeddings) | Full model (XGBoost, SVM, etc.) | N/A | N/A | Defaults | N/A | 1 |

---

## 3. Metrics

- **Task A (binary):** macro-F1 = unweighted mean of positive-class F1 and
  negative-class F1 (matches sklearn `f1_score(average="macro")` and the
  SemEval MAMI leaderboard).
- **Task B (multi-label):** MAMI 2022 official `mami_score_b` =
  positive-support-weighted mean of per-sub-type binary-macro F1 (where
  binary-macro F1 = (pos_F1 + neg_F1) / 2 for that sub-type). Matches the
  "weighted-F1" on the SemEval leaderboard.
- **Selection:** best of 3 seeds, chosen on the validation split. The
  selected seed's test score is reported.

---

## 4. Hardware

All GPU work ran on the SCIAMA HPC cluster (University of Portsmouth):
single NVIDIA L40 (48 GB VRAM) per job, CUDA 12.7. Gemini inference used the
Google Cloud API (no local GPU).

---

## 5. Dataset

MAMI 2022 (SemEval Task 5): 10,000 memes split into train (9,000) /
validation (1,000) / test (1,000). Binary labels (misogynous yes/no) for
Task A; four independent binary sub-type labels for Task B (shaming,
stereotype, objectification, violence). The test split contains 500
misogynous + 500 non-misogynous memes (balanced).
