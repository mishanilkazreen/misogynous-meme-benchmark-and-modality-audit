# results/test/

Benchmark metrics computed on the **held-out test split** of the MAMI dataset.

These are the **final, reported numbers** used in the paper. Do **not** tune
models against these results — the test set must remain unseen until final
evaluation.

## File naming convention

```
<model>_test[_<variant>].json
```

| Segment | Meaning |
|---------|---------|
| `model` | `clip`, `llava`, `qwen2vl`, `visualbert`, `xgboost`, `gemini`, … |
| `test`  | always `test` for this directory |
| `variant` | optional: model size, LoRA checkpoint name, or task type (`multiclass`, `finetuned`, …) |

## Status

| Model | Single-class | Multi-class |
|-------|-------------|-------------|
| CLIP (zero-shot ViT-L/14) | ✅ `clip_test.json` | ✅ `clip_test_multiclass.json` |
| CLIP (fine-tuned ViT-L/14) | ✅ | ✅ |
| CLIP (fine-tuned ViT-B/32) | ✅ | — |
| LLaVA-1.5-7B (zero-shot) | ✅ | ✅ |
| LLaVA-1.5-7B (fine-tuned) | ✅ | ✅ |
| Qwen2-VL-2B (zero-shot) | ✅ | — |
| Qwen2-VL-2B (fine-tuned) | ✅ | ✅ |
| Qwen2-VL-7B (fine-tuned) | ✅ | ✅ |
| VisualBERT | ✅ | ✅ |
| XGBoost fusion | ✅ | ✅ |
| Gemini 3.1 Flash-Lite (zero-shot) | ✅ `gemini_test.json` | ✅ `gemini_test_multiclass.json` |

> [!NOTE]
> Gemini 3.1 Flash-Lite results must be generated on an internet-enabled device.
> Run: `uv run python scripts/benchmark_gemini.py --split test`
> and `uv run python scripts/benchmark_gemini.py --split test --task multiclass`
