"""Regression test: Task A macro-F1 must be the mean of per-class F1.

Guards against the bug where macro-F1 was computed as the F1 of the averaged
precision and recall (2*P*R/(P+R)), which overstates F1 whenever macro-precision
and macro-recall diverge and is not the SemEval MAMI Task A metric.
"""

from __future__ import annotations

from sklearn.metrics import f1_score

from scripts.benchmark_vlm_classification import compute_classification_metrics

LABELS = ["yes", "no"]


def _harmonic(p, r):
    return 2 * p * r / (p + r) if (p + r) else 0.0


def test_macro_f1_equals_mean_of_per_class_f1_not_harmonic_of_pr():
    # Confusion (positive=yes): TP=8, FP=4, FN=2, TN=6 -> divergent macro P/R.
    preds = ["yes"] * 12 + ["no"] * 8
    gts = ["yes"] * 8 + ["no"] * 4 + ["yes"] * 2 + ["no"] * 6

    m = compute_classification_metrics(preds, gts, LABELS)

    # Reference: sklearn macro-F1 (mean of per-class F1).
    ref = f1_score(gts, preds, labels=LABELS, average="macro", zero_division=0)
    assert abs(m["f1"] - ref) < 1e-6, f"got {m['f1']}, sklearn macro-F1 {ref}"

    # The buggy formula (harmonic mean of macro-P and macro-R) is higher here;
    # the correct value must NOT equal it.
    buggy = _harmonic(m["precision"], m["recall"])
    assert abs(m["f1"] - buggy) > 1e-4, "macro-F1 must differ from F1-of-averaged-PR"
    assert m["f1"] < buggy


def test_macro_f1_matches_sklearn_on_balanced_case():
    # When P and R coincide the two formulas agree; still must match sklearn.
    preds = ["yes", "yes", "no", "no", "yes", "no"]
    gts = ["yes", "no", "no", "yes", "yes", "no"]
    m = compute_classification_metrics(preds, gts, LABELS)
    ref = f1_score(gts, preds, labels=LABELS, average="macro", zero_division=0)
    assert abs(m["f1"] - ref) < 1e-6
