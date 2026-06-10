# Comparison with Qu et al. (2025)

**Reference:** Qu et al., *HatefulIllusion: Benchmarking Vision-Language Models on Hateful
Steganographic Images*, arXiv 2507.22617.
**Their code:** <https://github.com/TrustAIRLab/HatefulIllusion>

---

## Context

Anna raised two questions after the task-4 VLM benchmark:

1. Can we report binary classification (hateful yes/no) to be directly comparable to
   Qu et al. Figure 10?
2. How did Qu et al. use Gemini successfully when we observed an 89 % refusal rate?

**Answer to (2):** There is no special research API. Their `vlms.py → inference()` shows
they call the standard Gemini API with all four harm-category safety filters set to
`BLOCK_NONE`, **and** their task is binary yes/no — the model never has to emit a slur.
Our high refusal rate came from asking Gemini to output the specific slur text with safety
filters at their default level.

---

## Methodology differences

| Aspect | Qu et al. | Our benchmark |
|---|---|---|
| Classification task | Binary (hateful / not hateful) | Identification (exact label) + binary |
| Gemini safety settings | `BLOCK_NONE` for all 4 categories | Default (causes ~89 % refusals in ID mode) |
| Models tested | CLIP, LLaVA-1.5, LLaVA-Next, Qwen2-VL, Gemini | Same set + GPT-4o-mini |
| Preprocessing | blur, histogram, blur+histogram, grid, … | Same filters |

---

## Binary mode results (to fill once GPU run completes)

Run command:

```bash
uv run python scripts/benchmark_vlm_classification.py --model all --subset all --binary
```

| Model | Subset | Filter | Binary accuracy | F1 | Qu et al. acc |
|---|---|---|---|---|---|
| clip | digits | none | — | — | — |
| llava | digits | none | — | — | — |
| llavanext | digits | none | — | — | — |
| qwen2vl | digits | none | — | — | — |
| gemini | digits | none | — | — | — |

---

## LLaVA-Next results (added issue #70)

- **Model:** `llava-hf/llava-v1.6-mistral-7b-hf`, 4-bit NF4 quantized
- **Binary mode accuracy:** [to fill once results are in]
- **Identification mode accuracy:** [to fill once results are in]
- **Notes on agreement/disagreement with Qu et al. Figure 10 filter trends:**
  [to fill once results are in]

---

## Filter trend analysis (to fill)

Cross-check against Qu et al. Figure 10:

- Does `blur_histogram` consistently help across models?
- Does `grid` reduce performance as in their results?
- Are per-subset trends consistent (digits vs hate_symbols vs hate_slangs)?

Generate the comparison figure:

```bash
uv run python scripts/plot_preprocessing_comparison.py --binary
```

---

## Honest caveats

- Binary mode inflates accuracy vs identification: with only two classes, random
  guessing achieves 50 %. Direct number comparison to Qu et al. should note this.
- Our Gemini identification-mode results are not comparable to theirs (different task
  difficulty + safety settings). Binary mode with `BLOCK_NONE` is the fair comparison.
- LLaVA-Next 4-bit quantization may reduce accuracy slightly vs fp16 baseline.
