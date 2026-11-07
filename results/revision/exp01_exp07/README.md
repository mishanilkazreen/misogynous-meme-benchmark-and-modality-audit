# EXP-01 / EXP-07 metrics

Run from the repository root: `uv run python -m scripts.revision.revision_metrics`.
Scores call `compute_scoreA` / `compute_scoreB` / `compute_f1` from
`scripts/official_mami_evaluation.py`, cross-checked with sklearn and
`models/vlm/metrics_multilabel.py` (tolerance 1e-6).

- `metrics.json`: every row, split, task and seed, with file lists and input sha256.
- `table_task_a.tex`, `table_task_b.tex`: LaTeX rows (mean±SD over seeds where n>1).
- `stats.json`: bootstrap CIs, paired tests, all-negative baseline, thresholds, drops.

Caveats: provided-text ML rows come from summary JSONs only (their per-sample CSVs are
absent), so they have no CI or paired test; PaddleOCR-text rows are scored per sample.
Gemini Task B was fixed and re-run 2026-10-01 (see below); the old 0.3651 all-`shaming` file was a parser bug.
LLaVA zero-shot validation exists only in conflated test+validation files (rejected).
All-negative Exact Match: test 50.0%, validation 52.3%. Thresholds: Calibrated taus were never
applied: 13/17 ML rows equal the tau=0.50 column of the calibration CSV, none equal only the
calibrated column; the rest (Bagging Classifier, Extra Trees, Random Forest, SVM-RBF) match neither
(SVC predict() uses the margin, not predict_proba >= 0.50; the tree ensembles differ by fit).

## Previously reported value -> corrected value

| Task | Row | Changes (old -> new) |
| --- | --- | --- |
| A | Qwen2-VL-7B QLoRA | test_acc 76.10 -> 77.60±1.00%; test_score_a 0.7599 -> 0.7694±0.0128; val_acc 79.00 -> 88.80±0.58%; val_score_a 0.7895 -> 0.8879±0.0058 |
| A | Qwen2-VL-2B | val_acc 73.00 -> 74.00%; val_score_a 0.7292 -> 0.7395 |
| A | CLIP ViT-B-32 | test_acc 63.20 -> 70.97±0.50%; test_score_a 0.6246 -> 0.6950±0.0062; val_acc 78.80 -> 84.67±0.59%; val_score_a 0.7876 -> 0.8467±0.0059 |
| A | CLIP ViT-L-14 | test_acc 62.10 -> 74.17±1.10%; test_score_a 0.6107 -> 0.7314±0.0121; val_acc 77.50 -> 86.73±0.06%; val_score_a 0.7747 -> 0.8673±0.0006 |
| A | CLIP ViT-L-14 (Zero-Shot) | test_acc 64.30 -> 63.20%; test_score_a 0.6280 -> 0.5826; val_acc 68.75 -> 72.80%; val_score_a 0.6135 -> 0.7171 |
| A | LLaVA-1.5-7B (Zero-Shot) | val_acc 61.80 -> N/A; val_score_a 0.5654 -> N/A |
| B | Qwen2-VL-7B QLoRA | test_score_b 0.7428 -> 0.7380±0.0124; test_subtype_f1 0.6014 -> 0.5937±0.0221; test_em 50.20 -> 51.78±1.28%; val_score_b 0.7686 -> 0.7657±0.0057; val_subtype_f1 0.5907 -> 0.5877±0.0107; val_em 65.40 -> 66.34±0.96% |
| B | Qwen2-VL-2B QLoRA | test_score_b 0.6893 -> 0.7043±0.0164; test_subtype_f1 0.5263 -> 0.5467±0.0321; test_em 45.90 -> 47.26±1.71%; val_score_b 0.7384 -> 0.7433±0.0089; val_subtype_f1 0.5353 -> 0.5513±0.0237; val_em 64.40 -> 63.94±0.71% |
| B | gemini-3.1-flash-lite | test_score_b 0.6730 -> 0.7536; test_subtype_f1 0.5015 -> 0.6390; test_em 42.80 -> 48.70%; val_score_b 0.7175 -> 0.7543; val_subtype_f1 0.4877 -> 0.5949; val_em 58.00 -> 55.10% (re-run 2026-10-01 after fixing the Task B parser bug in scripts/benchmark_gemini.py; the old 0.6730/0.3651 all-shaming values were a bug, not Gemini's answers) |
| B | CLIP ViT-L-14 (Fine-Tuned) | test_score_b 0.6165 -> 0.6177±0.0033; test_subtype_f1 0.5072 -> 0.5084±0.0019; test_em 17.00 -> 17.23±0.85%; val_score_b 0.6706 -> 0.6698±0.0052; val_subtype_f1 0.4908 -> 0.4888±0.0056; val_em 30.87 -> 30.40±0.80% |
| B | CLIP ViT-B-32 (Fine-Tuned) | test_score_b 0.5396 -> 0.5414±0.0063; test_subtype_f1 0.4677 -> 0.4689±0.0024; test_em 10.07 -> 10.57±0.32%; val_score_b 0.6206 -> 0.6203±0.0045; val_subtype_f1 0.4445 -> 0.4444±0.0034; val_em 24.23 -> 24.30±0.82% |
| B | CLIP ViT-L-14 (Zero-Shot) | test_score_b 0.5168 -> 0.5165; test_subtype_f1 0.3837 -> 0.3836; test_em 9.20 -> 9.10%; val_score_b 0.5204 -> 0.5207; val_subtype_f1 0.3215 -> 0.3217 |
| B | LLaVA-1.5-7B (Zero-Shot) | test_score_b 0.5049 -> 0.4139; test_subtype_f1 0.1396 -> 0.0000; test_em 50.80 -> 50.00%; val_score_b 0.4691 -> N/A; val_subtype_f1 0.1278 -> N/A; val_em 48.40 -> N/A |
| A3 | SVM-RBF | paddleocr test_score_b 0.7330 -> 0.7334; paddleocr test_em 45.50 -> 45.80% |
| A3 | SVM-RBF | combined test_score_b 0.7300 -> 0.7297; combined test_em 45.20 -> 44.80% |
| A3 | LDA | paddleocr test_score_b 0.7115 -> 0.7003; paddleocr test_em 44.40 -> 45.80% |
| A3 | LDA | combined test_score_b 0.7110 -> 0.7054; combined test_em 44.10 -> 43.90% |
| A3 | MLP | paddleocr test_score_b 0.7092 -> 0.7060; paddleocr test_em 45.30 -> 45.70% |
| A3 | MLP | combined test_score_b 0.7081 -> 0.6976; combined test_em 45.10 -> 44.70% |
| A3 | Ridge Classifier | paddleocr test_score_b 0.6834 -> 0.6740; paddleocr test_em 33.10 -> 32.60% |
| A3 | Ridge Classifier | combined test_score_b 0.6829 -> 0.6798; combined test_em 32.90 -> 31.70% |
| A3 | Logistic Regression | paddleocr test_score_b 0.6798 -> 0.6658; paddleocr test_em 34.20 -> 33.90% |
| A3 | Logistic Regression | combined test_score_b 0.6805 -> 0.6723; combined test_em 34.40 -> 33.50% |
| A3 | LightGBM | paddleocr test_score_b 0.6951 -> 0.6814; paddleocr test_em 50.40 -> 49.70% |
| A3 | LightGBM | combined test_score_b 0.6948 -> 0.6877; combined test_em 50.60 -> 51.30% |
| A3 | XGBoost | paddleocr test_score_b 0.6852 -> 0.6857; paddleocr test_em 49.70 -> 48.40% |
| A3 | XGBoost | combined test_score_b 0.6840 -> 0.6896; combined test_em 49.80 -> 50.30% |
| prose | trained models validation 84%-89% F1 | 84%-89% -> 63.9-88.8 Score A (27 trained rows) |
| prose | trained models test 68%-76% F1 | 68%-76% -> 59.2-76.9 Score A (27 trained rows) |
| prose | zero-shot models (Gemini, zero-shot Qwen) suffer no drop | no drop -> small: Gemini 3.1 Flash Lite +0.6, Qwen2-VL-7B (Zero-Shot) +0.1, Qwen2-VL-2B (Zero-Shot) +1.4; but CLIP ViT-L-14 (Zero-Shot) +13.5, CLIP ViT-B-32 (Zero-Shot) +14.9, VisualBERT (Zero-Shot) +10.7 |
| prose | Qwen2-VL-7B QLoRA and SVM-RBF beat SRCB (0.7310) on Task B | 0.7428 and 0.7321 vs. 0.7310 -> Qwen2-VL-7B QLoRA 0.7380±0.0124 (n=5, 4/5 seeds above 0.7310); SVM-RBF 0.7321 unchanged |
