# ANA-01 Qualitative Case Studies of Misclassified Memes

Model:

```text
XGBoost concat fusion on frozen CLIP ViT-L-14-quickgelu/openai embeddings (image cols 0:768, text cols 768:1536), re-fitted on the train split with auto_benchmark get_models_dict defaults (verbosity=0, random_state=42, tree_method='hist', device='cpu'), no scaler
```

Test Score A (official scorer): 0.7218; accuracy 0.7330.

Misclassified test memes: 267 (234 false positive, 33 false negative).

## Selection rule

```text
From the test split, keep memes the re-fitted model misclassifies on Task A (predicted label != true label, measured). Rank them by descending image-block absolute-SHAP share (sum of |SHAP| over image columns 0:768 divided by the sum over all 1536 columns). Break ties by ascending integer image_id. Keep the top 6. No example is hand-picked; the set is a deterministic function of the measured SHAP values.
```

## Global modality SHAP reference

Global XGBoost SHAP split (results/shap_modality_importance.csv): image 72.42%, text 27.58%.

## Selected memes

| Rank | image_id | True | Predicted | Error | Image |SHAP| share | Figure |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 15492 | 0 | 1 | FP | 72.83% | ana01_shap_15492.png |
| 2 | 15579 | 0 | 1 | FP | 71.55% | ana01_shap_15579.png |
| 3 | 15860 | 0 | 1 | FP | 71.41% | ana01_shap_15860.png |
| 4 | 15228 | 0 | 1 | FP | 71.38% | ana01_shap_15228.png |
| 5 | 15459 | 0 | 1 | FP | 71.14% | ana01_shap_15459.png |
| 6 | 15100 | 0 | 1 | FP | 70.48% | ana01_shap_15100.png |
