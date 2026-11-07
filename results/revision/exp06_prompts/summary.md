# EXP-06 prompt sensitivity (Gemini, MAMI test, n=1000)

`gemini-3.1-flash-lite`, temperature 0, `BLOCK_NONE`; official MAMI scorer; 95% bootstrap CI
(1000 resamples). Primary scores use answered memes only; `all` scores the
full 1000 with failures as negative. Refusal = refusal + safety block.

| Var | Acc [CI] | Score A [CI] | Score A (all) | Refusal | Parse fail |
| --- | --- | --- | --- | --- | --- |
| P0 | 0.8720 [0.8500, 0.8930] | 0.8719 [0.8572, 0.8881] | 0.8719 | 0.0% | 0.0% |
| P1 | 0.8690 [0.8470, 0.8890] | 0.8688 [0.8534, 0.8850] | 0.8688 | 0.0% | 0.0% |
| P2 | 0.8680 [0.8470, 0.8880] | 0.8680 [0.8519, 0.8847] | 0.8680 | 0.0% | 0.0% |
| P3 | 0.8700 [0.8490, 0.8910] | 0.8692 [0.8535, 0.8850] | 0.8692 | 0.0% | 0.0% |

| Var | Score B [CI] | Subtype-F1 [CI] | EM [CI] | Score B (all) | Refusal | Parse fail |
| --- | --- | --- | --- | --- | --- | --- |
| B0 | 0.7469 [0.7350, 0.7580] | 0.6305 [0.6066, 0.6546] | 0.4740 [0.4410, 0.5030] | 0.7469 | 0.0% | 0.0% |
| B1 | 0.7682 [0.7571, 0.7800] | 0.6542 [0.6301, 0.6771] | 0.5040 [0.4720, 0.5350] | 0.7682 | 0.0% | 0.0% |

Spread (max - min, primary):

- A.primary.acc: 0.0040 (P0 0.8720 vs P2 0.8680)
- A.primary.score_a: 0.0039 (P0 0.8719 vs P2 0.8680)
- B.primary.score_b: 0.0213 (B1 0.7682 vs B0 0.7469)
- B.primary.em: 0.0300 (B1 0.5040 vs B0 0.4740)

Cohen's kappa (Task A): P0-P1 0.926, P0-P2 0.888, P0-P3 0.875, P1-P2 0.862, P1-P3 0.860, P2-P3 0.824
Cohen's kappa (Task B, mean over sub-types): 0.864
P0 re-run vs committed paper run (default temperature): agreement 0.972, kappa 0.944 (n=1000).
Estimated API cost (list price): $1.91.

Prompts: `prompts.md`. Full numbers: `metrics.json`.
