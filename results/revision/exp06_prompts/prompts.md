# EXP-06 prompt variants (exact text sent to the API)

Model `gemini-3.1-flash-lite`, temperature 0.0, all four safety categories `BLOCK_NONE`.
The image (PNG) is sent after the prompt text, as in `scripts/benchmark_gemini.py`.

## System instructions

`SYSTEM_A` (Task A; verbatim `_SYSTEM_INSTRUCTION` of `scripts/benchmark_gemini.py`):

```text
You are a content moderation research assistant analysing internet memes for an academic misogyny detection study. Your role is purely classificatory: answer with exactly one word — 'yes' or 'no'. Do not add explanations, warnings, or refusals.
```

`SYSTEM_B` (Task B; task-specific, replaces the one-word instruction that
contradicts the JSON reply format):

```text
You are a content moderation research assistant analysing internet memes for an academic misogyny detection study. Your role is purely classificatory: reply with only the JSON object requested in the prompt. Do not add explanations, warnings, or refusals.
```

## Prompts

### P0: Paper Task A prompt (baseline)

Task A, system instruction `SYSTEM_A`.

```text
Is this meme misogynistic? A misogynistic meme demeans, objectifies, or expresses hostility toward women. Answer with exactly one word: yes or no.
```

### P1: Paraphrased definition

Task A, system instruction `SYSTEM_A`.

```text
Does this meme express misogyny? We call a meme misogynistic when it degrades women, treats them as sexual objects, or conveys hatred or aggression towards them. Answer with exactly one word: yes or no.
```

### P2: Definition removed

Task A, system instruction `SYSTEM_A`.

```text
Is this meme misogynistic? Answer yes or no.
```

### P3: MAMI taxonomy (4 sub-types) then yes/no

Task A, system instruction `SYSTEM_A`.

```text
This is a meme. Following the MAMI task, a meme is misogynistic if it expresses at least one of these four types of misogyny:
- shaming: insults or attacks a woman's body, appearance, or behaviour.
- stereotype: portrays women through oversimplified gender roles.
- objectification: reduces a woman to a sexual object.
- violence: depicts or threatens physical or sexual violence against women.
Is this meme misogynistic? Answer with exactly one word: yes or no.
```

### B0: Paper Task B JSON prompt (baseline)

Task B, system instruction `SYSTEM_B`.

```text
This is a meme. Following the MAMI task, identify which TYPE(s) of misogyny it expresses, if any. A meme can express more than one type at the same time. Consider these four categories:
- shaming: insults or attacks a woman's body, appearance, or behaviour.
- stereotype: portrays women through oversimplified gender roles.
- objectification: reduces a woman to a sexual object.
- violence: depicts or threatens physical or sexual violence against women.
Reply with EXACTLY this JSON structure and nothing else:
{"shaming": <true|false>, "stereotype": <true|false>, "objectification": <true|false>, "violence": <true|false>}
```

### B1: Four per-sub-type yes/no questions in one JSON

Task B, system instruction `SYSTEM_B`.

```text
This is a meme. Answer the four questions below independently; each is a separate yes/no decision.
shaming: Does this meme shame or insult a woman's body, appearance, or behaviour (for example body-shaming or slut-shaming)?
stereotype: Does this meme reinforce oversimplified or traditional gender stereotypes about women?
objectification: Does this meme reduce a woman to a sexual object or focus on her body for sexual purposes?
violence: Does this meme depict, encourage, or threaten physical or sexual violence against women?
Reply with EXACTLY this JSON structure and nothing else:
{"shaming": "<yes|no>", "stereotype": "<yes|no>", "objectification": "<yes|no>", "violence": "<yes|no>"}
```
