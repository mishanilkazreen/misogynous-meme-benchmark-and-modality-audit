<!-- markdownlint-disable MD013 -->
<!-- cspell:ignore misogynistic objectifying sexualising -->
# Prompts Used in MAMI 2022 Experiments

All prompts are defined in `models/vlm/classifier.py`. This document records
the exact text sent to each model in our reported experiments.

---

## 1. Generative VLMs (Qwen2-VL, Gemini)

These models receive a chat message containing (1) the meme image and (2) a
text prompt. For our reported results (`--text-source provided`), no OCR
transcript is injected — the model sees only the image and the prompt below.
When running with `--text-source ocr` or `combined`, the prompt is prefixed
with: `This meme contains the text: "<extracted_text>".`

### Task A — Binary misogyny (yes/no)

```text
Is this meme misogynistic? A misogynistic meme demeans, objectifies, or
expresses hostility toward women. Answer with exactly one word: yes or no.
```

**Expected output:** `yes` or `no`

### Task B — Multi-label sub-types (JSON schema)

```text
This is a meme. Following the MAMI task, identify which TYPE(s) of misogyny
it expresses, if any. A meme can express more than one type at the same time.
Consider these four categories:
- shaming: insults or attacks a woman's body, appearance, or behaviour.
- stereotype: portrays women through oversimplified gender roles.
- objectification: reduces a woman to a sexual object.
- violence: depicts or threatens physical or sexual violence against women.
Reply with EXACTLY this JSON structure and nothing else:
{"shaming": <true|false>, "stereotype": <true|false>,
"objectification": <true|false>, "violence": <true|false>}
```

**Expected output (example):**
`{"shaming": false, "stereotype": true, "objectification": false, "violence": false}`

### Task B — Per-category binary prompts (alternative, `--task per_category`)

Four separate yes/no questions per image, one per sub-type:

| Sub-type | Prompt |
| :--- | :--- |
| shaming | Does this meme shame or insult a woman's body, appearance, or behaviour (for example body-shaming or slut-shaming)? Answer with exactly one word: yes or no. |
| stereotype | Does this meme reinforce oversimplified or traditional gender stereotypes about women? Answer with exactly one word: yes or no. |
| objectification | Does this meme reduce a woman to a sexual object or focus on her body for sexual purposes? Answer with exactly one word: yes or no. |
| violence | Does this meme depict, encourage, or threaten physical or sexual violence against women? Answer with exactly one word: yes or no. |

### Joint multi-task (Task A + Task B in one prompt, `--task joint`)

```text
This is a meme. Decide whether it is misogynistic and, if so, which TYPE(s)
of misogyny it expresses. A meme can express more than one type at the same
time. Consider these four sub-types:
- shaming: insults or attacks a woman's body, appearance, or behaviour.
- stereotype: portrays women through oversimplified gender roles.
- objectification: reduces a woman to a sexual object.
- violence: depicts or threatens physical or sexual violence against women.
Reply with EXACTLY this JSON structure and nothing else:
{"misogynous": <true|false>, "shaming": <true|false>,
"stereotype": <true|false>, "objectification": <true|false>,
"violence": <true|false>}
```

---

## 2. CLIP zero-shot (cosine similarity)

CLIP does not receive a text prompt at inference. Instead, per-class **text
embeddings** are precomputed from phrase banks and compared to the image
embedding via cosine similarity. The class whose text embedding is closest
to the image embedding is the prediction.

### Task A — Binary classification phrases

| Class | Prompt ensemble (averaged into one embedding per class) |
| :--- | :--- |
| Misogynistic | "a misogynistic meme", "a meme that demeans women", "a meme expressing hostility toward women", "a meme objectifying women", "a meme stereotyping women", "a sexist meme" |
| Not misogynistic | "a meme that does not target women", "a non-misogynistic meme", "a neutral meme", "a wholesome meme", "a meme with no gender content" |

### Task B — Per-sub-type phrase pairs (positive vs negative, per category)

Each sub-type is an independent binary decision: cosine similarity to the
positive phrase vs the negative phrase. Multiple prompts per side are
averaged into a single embedding before comparison.

| Sub-type | Positive phrases | Negative phrases |
| :--- | :--- | :--- |
| shaming | "a meme shaming or insulting a woman", "a meme body-shaming a woman", "a meme mocking a woman's appearance", "a meme slut-shaming a woman" | "a meme not shaming a woman", "a meme that does not insult a woman" |
| stereotype | "a meme reinforcing gender stereotypes about women", "a meme portraying women through traditional gender roles", "a meme depicting women as housewives", "a meme depicting women as bad drivers" | "a meme not reinforcing gender stereotypes", "a meme that does not stereotype women" |
| objectification | "a meme objectifying or sexualising a woman", "a meme reducing a woman to a sexual object", "a meme focusing on a woman's body for sexual purposes", "a meme sexualising a woman's appearance" | "a meme not objectifying women", "a meme that does not sexualise women" |
| violence | "a meme depicting or threatening violence against women", "a meme depicting physical aggression against a woman", "a meme depicting sexual violence against a woman", "a meme encouraging harm to a woman" | "a meme not depicting violence against women", "a meme that does not threaten women" |

---

## 3. Tabular models (XGBoost, SVM, etc.)

Tabular classifiers do not use text prompts. They operate on pre-extracted
numerical feature vectors (CLIP ViT-L-14 image embeddings concatenated with
CLIP text embeddings of the meme's transcript). The text is encoded by the
CLIP text encoder — no prompt is involved at classification time.

---

## 4. Which prompt was used for which reported result

| System in results table | Task | Prompt variant |
| :--- | :--- | :--- |
| Qwen2-VL (zero-shot + fine-tuned) | A | Section 1, Task A |
| Qwen2-VL (zero-shot + fine-tuned) | B | Section 1, Task B (JSON schema) |
| Gemini (gemini-3.1-flash-lite, zero-shot) | A | Section 1, Task A |
| Gemini (gemini-3.1-flash-lite, zero-shot) | B | Section 1, Task B (JSON schema) |
| CLIP ViT-L-14 / ViT-B-32 (zero-shot) | A | Section 2, Task A phrase ensemble |
| CLIP ViT-L-14 / ViT-B-32 (zero-shot) | B | Section 2, Task B phrase pairs |
| CLIP (fine-tuned, frozen towers) | A | N/A (trained classification head on embeddings) |
| CLIP (fine-tuned, frozen towers) | B | N/A (trained classification head on embeddings) |
| Tabular (XGBoost, SVM, etc.) | A + B | N/A (numerical features, no prompt) |

**Note on text-source for VLMs:** our headline VLM results use
`--text-source provided`, which means **no text is injected** into the prompt
(the model relies solely on the image and the classification question). The
`--text-source ocr` variant prepends the meme's extracted text, but those are
reported only as ablations.
