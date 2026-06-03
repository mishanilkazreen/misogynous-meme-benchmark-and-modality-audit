# VLM & Cloud Moderation Benchmark

## Models

| Model | Type | Status | Doc |
|---|---|---|---|
| CLIP ViT-L/14 | Zero-shot classifier (local) | Done | [vlm_docs/clip.md](vlm_docs/clip.md) |
| YOLO-World | Text-prompted detector (local) | Not started | [vlm_docs/yolo_world.md](vlm_docs/yolo_world.md) |
| Qwen2-VL-72B 4-bit | Generative VLM (local, GPU) | Not started | [vlm_docs/qwen2vl.md](vlm_docs/qwen2vl.md) |
| AWS Rekognition | Cloud moderation API | Not started | [vlm_docs/aws_rekognition.md](vlm_docs/aws_rekognition.md) |
| Google SafeSearch | Cloud moderation API | Not started | [vlm_docs/google_safesearch.md](vlm_docs/google_safesearch.md) |
| GPT-4o-mini | Generative VLM (cloud) | Not started | [vlm_docs/gpt4omini.md](vlm_docs/gpt4omini.md) |
| Gemini 2.0 Flash | Generative VLM (cloud) | Not started | [vlm_docs/gemini.md](vlm_docs/gemini.md) |

## Cost summary

| Model | Cost (2,160 imgs) |
|---|---|
| CLIP | Free |
| YOLO-World | Free |
| Qwen2-VL-72B | Free (local GPU) |
| AWS Rekognition | ~$2.16 |
| Google SafeSearch | ~$3.24 |
| GPT-4o-mini | ~$0.28 |
| Gemini 2.0 Flash | ~$0.06 |
| **Total cloud** | **~$5.74** |

## Metric schema

- **Production APIs** (Rekognition, SafeSearch): `any_detection_recall`
- **Generative VLMs** (GPT-4o-mini, Gemini, Qwen2-VL): `exact_match_accuracy` + `refusal_rate`
- **CLIP**: `exact_match_accuracy` + `above_chance_rate`
- **YOLO-World**: `any_detection_recall`

All results files joinable on `subset` and `visibility_score` for task-7 comparison table.

## Paper narrative

> "Production content moderation APIs (AWS Rekognition, Google SafeSearch) detect
> hateful content in X% of HatefulIllusion images at visibility ≥ 3, falling to Y%
> at visibility 1–2. CLIP zero-shot classification achieves Z%, while generative VLMs
> (Qwen2-VL-72B, GPT-4o-mini, Gemini 2.0 Flash) reach A%, suggesting that image-level
> semantic reasoning is substantially more robust to obfuscation than detection-based
> or label-based moderation pipelines."
