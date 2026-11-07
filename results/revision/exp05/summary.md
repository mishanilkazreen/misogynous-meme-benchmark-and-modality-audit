# EXP-05: OCR quality, OCR ablation and 77-token truncation

## Method

- `uv run --with jiwer==4.0.0 python -m scripts.revision.ocr_quality` (CPU, commit dd5196e); numbers from the JSONs.
- Gold = MAMI `text` column; OCR = PaddleOCR `raw_texts` (already NFKC + lowercased by `utils/ocr.py`).
- jiwer 4.0.0; raw = as stored; norm = NFKC, lowercase, no punctuation, collapsed spaces; no-space = norm - spaces.
- Buckets: per-meme norm CER. Scores: `scripts/official_mami_evaluation.py`. Tokens: open_clip ViT-L-14 + SOT/EOT.

## OCR quality (corpus-level; mean/median are per meme, norm CER)

| Split | raw CER | raw WER | norm CER | norm WER | no-space CER | mean / median | empty OCR % |
|---|---|---|---|---|---|---|---|
| train | 0.473 | 0.661 | 0.225 | 0.440 | 0.200 | 0.200 / 0.138 | 0.2 |
| validation | 0.463 | 0.653 | 0.214 | 0.422 | 0.192 | 0.189 / 0.129 | 0.2 |
| test | 0.621 | 0.801 | 0.220 | 0.597 | 0.169 | 0.219 / 0.171 | 0.3 |

## Test by norm-CER bucket (Qwen: OCR-provided acc/EM, Score provided->OCR; tabular: OCR acc/EM, Score)

| Model | Task | all (n=1000) | 0-0.1 (n=289) | 0.1-0.3 (n=436) | 0.3-0.6 (n=230) | >0.6 (n=45) |
|---|---|---|---|---|---|---|
| Qwen2-VL-2B-Instruct zero-shot | binary | -0.006 (A 0.725->0.718) | +0.003 (A 0.698->0.687) | -0.007 (A 0.719->0.713) | +0.009 (A 0.700->0.716) | -0.133 (A 0.910->0.778) |
| Qwen2-VL-7B-Instruct zero-shot | binary | -0.015 (A 0.766->0.750) | -0.017 (A 0.737->0.710) | -0.014 (A 0.759->0.745) | -0.004 (A 0.769->0.765) | -0.067 (A 0.867->0.800) |
| Qwen2-VL-2B-Instruct zero-shot | multiclass | -0.027 (B 0.443->0.518) | -0.024 (B 0.461->0.543) | -0.034 (B 0.435->0.510) | -0.017 (B 0.431->0.500) | -0.022 (B 0.408->0.482) |
| Qwen2-VL-7B-Instruct zero-shot | multiclass | -0.004 (B 0.609->0.594) | -0.010 (B 0.577->0.564) | +0.000 (B 0.612->0.595) | -0.004 (B 0.618->0.607) | +0.000 (B 0.707->0.667) |
| Tabular SVM-RBF (CLIP ViT-L-14 concat) | binary | 0.739 (A 0.729) | 0.689 (A 0.689) | 0.745 (A 0.725) | 0.765 (A 0.744) | 0.867 (A 0.863) |
| Tabular XGBoost (CLIP ViT-L-14 concat) | binary | 0.730 (A 0.719) | 0.696 (A 0.695) | 0.727 (A 0.706) | 0.765 (A 0.743) | 0.800 (A 0.793) |
| Tabular SVM-RBF (CLIP ViT-L-14 concat) | multiclass | 0.458 (B 0.733) | 0.488 (B 0.716) | 0.440 (B 0.738) | 0.448 (B 0.731) | 0.489 (B 0.740) |
| Tabular XGBoost (CLIP ViT-L-14 concat) | multiclass | 0.484 (B 0.686) | 0.564 (B 0.671) | 0.443 (B 0.678) | 0.457 (B 0.699) | 0.511 (B 0.698) |

## 77-token truncation (subgroup flag = length of the text the model consumed)

- train: gold 254 (2.82%) > 77 tokens, median 21, max 372; OCR 165 (1.83%), median 19.
- validation: gold 27 (2.70%) > 77 tokens, median 21, max 343; OCR 15 (1.50%), median 19.
- test: gold 2 (0.20%) > 77 tokens, median 18, max 100; OCR 2 (0.20%), median 16.
- validation, Tabular SVM-RBF (PaddleOCR text): acc truncated 14/15 = 0.933 vs 0.870, Fisher p = 0.708.
- validation, Tabular XGBoost (PaddleOCR text): acc truncated 14/15 = 0.933 vs 0.855, Fisher p = 0.710.
- validation, Qwen2-VL-7B QLoRA seed1 (provided text, no CLIP): acc truncated 23/27 = 0.852 vs 0.890, Fisher p = 0.530.
- test, Tabular SVM-RBF (PaddleOCR text): acc truncated 0/2 = 0.000 vs 0.740, Fisher p = 0.068.
- test, Tabular XGBoost (PaddleOCR text): acc truncated 1/2 = 0.500 vs 0.730, Fisher p = 0.467.
- test, Qwen2-VL-7B QLoRA seed1 (provided text, no CLIP): acc truncated 1/2 = 0.500 vs 0.788, Fisher p = 0.381.

## Caveats

- Provided-text tabular CSVs/embeddings are absent locally (gitignored; SCIAMA unreachable): tabular is OCR-side only.
- Qwen2-VL also sees the image and can read the overlay: OCR-vs-provided gaps bound, not isolate, OCR-error effects.
- CER buckets are observational (layout and text length co-vary with CER); bucket gaps are not causal.
- Few test memes exceed 77 tokens, so those tests are underpowered; validation is supplementary.
