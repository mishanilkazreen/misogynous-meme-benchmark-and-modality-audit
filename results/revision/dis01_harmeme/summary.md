# DIS-01: HarMeme (Harm-C) Empirical Evaluation Summary

Official Harm-C test split: 354 memes (124 harmful, 230 harmless).

| Method | Setting | Test Acc (%) | Test Macro-F1 (%) | 95% CI Acc (%) | 95% CI Macro-F1 (%) | File |
| --- | --- | --- | --- | --- | --- | --- |
| This work: SVM-RBF on CLIP | Supervised, frozen CLIP | 83.90 | 83.26 | [79.94, 87.57] | [78.97, 87.03] | `results/test/harmeme_test_clip_svm_rbf.json` |
| This work: XGBoost on CLIP | Supervised, frozen CLIP | 81.64 | 80.46 | [77.40, 85.59] | [75.87, 84.52] | `results/test/harmeme_test_clip_xgboost.json` |
| This work: Qwen2-VL-7B | Zero-shot VLM | 64.69 | 42.80 | [59.89, 69.50] | [39.34, 46.87] | `results/test/harmeme_test_qwen2_vl_7b_instruct.json` |
| This work: Gemini 3.1 Flash Lite | Zero-shot VLM | 65.82 | 46.42 | [61.02, 70.62] | [41.86, 51.12] | `results/test/harmeme_test_gemini_3_1_flash_lite.json` |
