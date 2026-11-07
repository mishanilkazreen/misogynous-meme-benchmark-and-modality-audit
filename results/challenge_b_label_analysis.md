# Challenge B (MAMI Sub-task B) — label distribution and overlap

Exploratory analysis of the four MAMI 2022 sub-type labels (shaming,
stereotype, objectification, violence) on the evaluation set used in our
benchmarks (`test + validation`, 2,000 images). Train and the full
dataset show the same patterns.

Regenerate the underlying numbers with:

```bash
uv run python scripts/analyze_challenge_b_labels.py
```

This writes `results/challenge_b_label_stats.json` (machine-readable,
diagram-ready: class counts, co-occurrence matrix, active-count
histogram, conditional probabilities) for all three splits.

## What MAMI Sub-task B expects

MAMI (Fersini et al., SemEval-2022 Task 5) defines two tasks. Sub-task A
is binary misogyny detection. **Sub-task B is multi-label**: for a meme,
identify which of four overlapping *types* of misogyny it expresses. A
meme may express several types at once, and sub-type labels are only
positive for misogynous memes (all four are 0 for non-misogynous memes).

## Key findings

### The classes are not mutually exclusive

Among the 1,000 misogynous memes in the eval set, 58.1% carry two or
more sub-types (mean ≈ 1.72 active sub-types per misogynous meme):

| # sub-types active | Misogynous images | Share |
|---|---|---|
| 0 | 24 | 2.4% |
| 1 | 395 | 39.5% |
| 2 | 428 | 42.8% |
| 3 | 140 | 14.0% |
| 4 | 13 | 1.3% |

### Per-class distribution (eval set, n = 2,000)

| Sub-type | Positives | % of all | % of misogynous |
|---|---|---|---|
| stereotype | 626 | 31.3% | 62.6% |
| objectification | 576 | 28.8% | 57.6% |
| shaming | 269 | 13.5% | 26.9% |
| violence | 253 | 12.7% | 25.3% |

Misogyny is balanced 1,000 / 1,000. Sub-types are imbalanced (~2.5x
between the common and rare pairs). 51.1% of all eval images have zero
sub-types — this all-negative baseline is why a do-nothing model scores
~0.51 exact-match while detecting nothing.

### Class overlap

Co-occurrence (images where both are active):

| | shaming | stereotype | objectification | violence |
|---|---|---|---|---|
| shaming | 269 | 156 | 137 | 31 |
| stereotype | 156 | 626 | 330 | 144 |
| objectification | 137 | 330 | 576 | 128 |
| violence | 31 | 144 | 128 | 253 |

Conditional probability — P(column active given row active):

| given | shaming | stereotype | objectification | violence |
|---|---|---|---|---|
| shaming | — | 58.0% | 50.9% | 11.5% |
| stereotype | 24.9% | — | 52.7% | 23.0% |
| objectification | 23.8% | 57.3% | — | 22.2% |
| violence | 12.3% | 56.9% | 50.6% | — |

No pair is ever mutually exclusive. The strongest entanglement is
stereotype ↔ objectification (330 shared images). shaming and violence
rarely co-occur with each other (31 images) but both pull strongly into
stereotype/objectification.

### Annotation quirks

- 1 eval image (4 in the full dataset) is labelled non-misogynous yet
  has a sub-type active.
- 24 misogynous eval images (2.4%) have no sub-type assigned.

## Implications for evaluation

- Report **macro / micro / per-class F1**, not exact-match accuracy.
  Exact-match is dominated by the 51% all-negative baseline and rewards
  degenerate "predict nothing" behaviour.
- Expect stereotype/objectification confusion given their overlap;
  per-class precision/recall is the informative view.
- The task is genuinely multi-label, so the prompt must allow multiple
  selections (see below).

## Recommended Challenge B prompt

The current prompt lists the four category words without definitions.
Given how heavily the categories overlap, adding short MAMI-aligned
definitions and an explicit multi-label, fixed-vocabulary instruction
should reduce confusion and improve parse reliability. The category
keywords must stay exactly `shaming`, `stereotype`, `objectification`,
`violence` so `extract_subtypes()` word-boundary matching still works.

```text
This is a meme. Following the MAMI task, identify which TYPE(s) of
misogyny it expresses, if any. A meme can express more than one type at
the same time. Choose all that apply from these four categories:
- shaming: insults or attacks a woman's body, appearance, or behaviour
  (for example body-shaming or slut-shaming).
- stereotype: portrays women through oversimplified or traditional
  gender roles and expectations.
- objectification: reduces a woman to a sexual object or focuses on her
  body for sexual purposes.
- violence: depicts, encourages, or threatens physical or sexual
  violence against women.
Reply with a comma-separated list of every category that applies, using
only these exact words. If the meme is not misogynistic or none of the
categories apply, reply with exactly 'none'. Do not add any explanation.
```

Adopting this prompt requires re-running Challenge B for the affected
models (the binary Challenge A prompt is unchanged). CLIP is unaffected
because it scores each category with its own phrase pair
(`CLIP_SUBTYPE_LABELS`), but those phrases could be aligned to the same
definitions for consistency.

## Reproducibility

- Script: `scripts/analyze_challenge_b_labels.py`
- Data: `results/challenge_b_label_stats.json`
- Source: MAMI 2022 TSVs (`label`, `shaming`, `stereotype`,
  `objectification`, `violence` columns).
