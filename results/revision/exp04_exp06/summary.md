# EXP-04 and EXP-06: Modality Ablation and Alternative Frozen Backbone

All metrics are drawn from `results/revision/exp04_exp06/results.json`, produced by
`scripts/revision/modality_ablation.py` on the embeddings extracted by
`scripts/revision/extract_features.py`. Models are fitted on the train split
(n=9000), the best classifier per condition is selected on validation (n=1000),
and the test split (n=1000) is scored once with
`scripts/official_mami_evaluation.py`. Score A is binary Macro-F1 (Task A);
Score B is the weighted sub-type Macro-F1 (Task B). Reported numbers use the
validation-selected classifier per condition.

## Reproduction of the paper CLIP-concat baseline

The `reproduction_clip_concat_vs_paper` block compares the CLIP concat condition
against the committed paper files (`results/test/tabular_provided_binary.json`,
`results/test/tabular_provided_multiclass_mami.json`). SVM-RBF and LDA reproduce
to four decimals (e.g. SVM-RBF Score B 0.7321 vs 0.7321, Task A Score A 0.7204 vs
0.7204). The largest absolute difference across all five classifiers and both
tasks is 0.0104 (XGBoost Score A), consistent with CPU `tree_method=hist`
nondeterminism.

## EXP-04: image-only, text-only, fusion and permutation (CLIP ViT-L-14)

Validation-selected test metrics (SVM-RBF selected for every CLIP condition on
both tasks unless noted):

| Condition | Score A | Acc A | Score B | Sub-type F1 | EM |
|-----------|---------|-------|---------|-------------|-----|
| CLIP image-only | 0.7189 | 0.7320 | 0.7258 | 0.5972 | 0.4410 |
| CLIP text-only | 0.6753 | 0.6820 | 0.6700 | 0.4996 | 0.3860 |
| CLIP concat (paper baseline) | 0.7204 | 0.7320 | 0.7321 | 0.5958 | 0.4540 |
| CLIP product (image * text) | 0.6815 | 0.6870 | 0.6712 | 0.5011 | 0.4050 |
| CLIP concat + product | 0.7282 | 0.7390 | 0.7253 | 0.5815 | 0.4560 |
| CLIP late fusion | 0.7296 | 0.7400 | 0.6976 | 0.5271 | 0.4820 |

Observations:

- Image-only Score A (0.7189) is within 0.0015 of concat (0.7204); image-only
  Score B (0.7258) is 0.0063 below concat (0.7321). Text-only is lower on both
  tasks (Score A 0.6753, Score B 0.6700).
- Element-wise product is below concat on both tasks (Score A 0.6815, Score B
  0.6712).
- Concat + product raises Score A to 0.7282 and late fusion to 0.7296, but
  neither improves Score B over concat (0.7253 and 0.6976 vs 0.7321). Late fusion
  selects MLP on Task B.
- Test-split 95% bootstrap intervals overlap across the concat, image-only and
  fusion conditions (e.g. CLIP concat Score B [0.7166, 0.7481], image-only Score B
  [0.7071, 0.7410]), so the differences among them are within sampling noise.

Permutation modality ablation (concat conditions, SVM-RBF, test, mean drop over
10 row shuffles, standard deviation in parentheses):

| Condition | Task | Base | Image-block drop | Text-block drop |
|-----------|------|------|------------------|-----------------|
| CLIP concat | A | 0.7204 | 0.1945 (0.0112) | 0.0473 (0.0097) |
| CLIP concat | B | 0.7321 | 0.1641 (0.0049) | 0.0504 (0.0062) |
| masked concat | A | 0.7272 | 0.1387 (0.0117) | 0.1258 (0.0192) |
| masked concat | B | 0.7324 | 0.1230 (0.0045) | 0.1079 (0.0068) |
| SigLIP concat | A | 0.6941 | 0.1986 (0.0140) | 0.0171 (0.0099) |
| SigLIP concat | B | 0.7327 | 0.1929 (0.0095) | 0.0267 (0.0063) |

For the standard CLIP concat model, shuffling the image block reduces Score A by
0.1945 and Score B by 0.1641, while shuffling the text block reduces them by
0.0473 and 0.0504. The image block carries more of the decision on the
un-masked memes.

## EXP-04: text-masked image inputs (R3.1)

The text-masked condition removes recognisable embedded text from the image
(EasyOCR-CRAFT detection plus Telea inpainting; the committed
`extraction_log_clip.json` records 0 residual recognised characters on the
200-image seeded audit). The text block is unchanged (provided-text CLIP
embedding).

| Condition | Score A | Acc A | Score B | Sub-type F1 | EM |
|-----------|---------|-------|---------|-------------|-----|
| CLIP masked image-only | 0.6621 | 0.6740 | 0.6738 | 0.5392 | 0.3600 |
| CLIP masked concat | 0.7272 | 0.7360 | 0.7324 | 0.6061 | 0.4670 |

Masking the embedded text lowers image-only Score A from 0.7189 to 0.6621 and
Score B from 0.7258 to 0.6738. Re-adding the provided text (masked concat)
restores Score A to 0.7272 and Score B to 0.7324, matching the un-masked concat
baseline (0.7204 / 0.7321). The permutation ablation on masked concat shows image
and text block drops of comparable size (Task A 0.1387 image vs 0.1258 text; Task
B 0.1230 image vs 0.1079 text), whereas on the un-masked CLIP concat the image
drop is about four times the text drop (Task A 0.1945 vs 0.0473). This indicates
that part of the image-block importance measured on un-masked memes comes from
embedded text that the frozen encoder reads, so a visual-block attribution is not
by itself evidence of reliance on non-textual visual content.

## EXP-06: alternative frozen backbone (SigLIP SO400M/14-384)

SigLIP image and text embeddings (dim 1152) were extracted with the same
provided-text pipeline (`extraction_log_siglip.json`; model
`ViT-SO400M-14-SigLIP-384` / `webli`, 64-token context, L2-normalised).

| Condition | Score A | Acc A | Score B | Sub-type F1 | EM |
|-----------|---------|-------|---------|-------------|-----|
| SigLIP image-only | 0.6962 | 0.7140 | 0.7405 | 0.6067 | 0.4770 |
| SigLIP text-only | 0.6440 | 0.6530 | 0.6550 | 0.4787 | 0.3640 |
| SigLIP concat | 0.6941 | 0.7100 | 0.7327 | 0.5953 | 0.4670 |

CLIP-vs-SigLIP on the matched concat condition: Score B is 0.7321 (CLIP) vs
0.7327 (SigLIP), a 0.0006 difference with overlapping 95% intervals ([0.7166,
0.7481] vs [0.7167, 0.7502]). Score A is 0.7204 (CLIP) vs 0.6941 (SigLIP). On
Task B the SigLIP image-only condition reaches the highest Score B of the study
(0.7405, interval [0.7239, 0.7566]); on Task A CLIP is higher. Both backbones
place more weight on the image block in the permutation ablation (SigLIP concat
Task B image drop 0.1929 vs text drop 0.0267). The ranking of frozen backbones
depends on the task: the two are within sampling noise on Task B Score B, and
CLIP leads on Task A Score A.
