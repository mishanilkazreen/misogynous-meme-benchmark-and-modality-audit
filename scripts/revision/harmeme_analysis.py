"""DIS-01: descriptive statistics for the reported Harm-C results.

Runs no model. Every number comes from the committed ``results/{validation,test}/harmeme_*.json``
prediction files and the Harm-C annotations:

- Accuracy, Macro-F1, bootstrap 95% CIs, harmful recall and predicted-harmful share per run.
- An all-negative baseline and exact McNemar tests on per-meme correctness.
- How well the rerun of the original Gemini prompt reproduces the committed replies.
- The best single-word rule, selected on the training split only, as an exploratory shortcut check.
- Harm-target composition per split and keyword counts for named topics.

Usage:
    uv run python -m scripts.revision.harmeme_analysis
"""

from __future__ import annotations

from collections import Counter
import json
from math import comb
from pathlib import Path
import re

from scripts.revision.harmeme_eval import (
    OUT_DIR,
    RESULTS_DIR,
    _prediction_stats,
    _read_predictions,
    bootstrap_ci,
    compute_binary_metrics,
    harmeme_dir,
)

Rows = list[dict[str, object]]

RUNS: dict[str, str] = {
    "svm_rbf": "clip_svm_rbf",
    "xgboost": "clip_xgboost",
    "gemini_original": "gemini_3_1_flash_lite",
    "gemini_rerun": "gemini_3_1_flash_lite_rerun",
    "gemini_definition": "gemini_3_1_flash_lite_definition",
    "qwen_original": "qwen2_vl_7b_instruct",
    "qwen_definition": "qwen2_vl_7b_instruct_definition",
}
PAIRED_TESTS: tuple[tuple[str, str], ...] = (
    ("gemini_original", "all_negative"),
    ("gemini_definition", "all_negative"),
    ("qwen_original", "all_negative"),
    ("qwen_definition", "all_negative"),
    ("gemini_definition", "gemini_original"),
    ("qwen_definition", "qwen_original"),
    ("gemini_original", "qwen_original"),
    ("svm_rbf", "gemini_original"),
    ("svm_rbf", "gemini_definition"),
)
TOPICS: dict[str, str] = {
    "trump": r"\btrump\b|donald",
    "vaccine": r"vaccin|\bvax",
    "political_term": (
        r"trump|donald|biden|obama|hillary|clinton|pelosi|democrat|republican|liberal|president"
        r"|sanders|bernie|election"
    ),
}
MIN_WORD_COUNT = 30


def exact_mcnemar_p(only_first_right: int, only_second_right: int) -> float:
    """Two-sided exact McNemar p-value from the two discordant counts."""
    total = only_first_right + only_second_right
    if total == 0:
        return 1.0
    tail = sum(comb(total, k) for k in range(min(only_first_right, only_second_right) + 1))
    return min(1.0, 2.0 * tail / 2**total)


def _load_annotations(split: str, data_root: Path) -> list[dict[str, object]]:
    path = data_root / "annotations" / f"{split}.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _is_harmful(record: dict[str, object]) -> int:
    labels = record["labels"]
    assert isinstance(labels, list)
    return int(labels[0] != "not harmful")


def _words(text: object) -> set[str]:
    return set(re.findall(r"[a-z']+", str(text).lower()))


def _macro_f1(preds: list[int], gts: list[int]) -> float:
    return compute_binary_metrics(preds, gts)["macro_f1"]


def select_keyword(train: list[dict[str, object]], min_count: int = MIN_WORD_COUNT) -> str:
    """Return the word whose presence alone best predicts ``harmful`` on the training split."""
    gts = [_is_harmful(rec) for rec in train]
    tokens = [_words(rec["text"]) for rec in train]
    counts = Counter(word for words in tokens for word in words)
    candidates = sorted(word for word, count in counts.items() if count >= min_count)
    return max(candidates, key=lambda w: _macro_f1([int(w in words) for words in tokens], gts))


def _stats_with_ci(rows: Rows) -> dict[str, object]:
    preds = [int(str(row["prediction"])) for row in rows]
    gts = [int(str(row["ground_truth"])) for row in rows]
    return {**_prediction_stats(rows), **bootstrap_ci(preds, gts), "memes": len(rows)}


def _correct_by_id(rows: Rows) -> dict[str, bool]:
    return {str(row["image_id"]): row["prediction"] == row["ground_truth"] for row in rows}


def _paired_test(first: Rows, second: Rows) -> dict[str, object]:
    right_a, right_b = _correct_by_id(first), _correct_by_id(second)
    only_a = sum(1 for key, value in right_a.items() if value and not right_b[key])
    only_b = sum(1 for key, value in right_a.items() if right_b[key] and not value)
    return {
        "only_first_right": only_a,
        "only_second_right": only_b,
        "exact_mcnemar_p": exact_mcnemar_p(only_a, only_b),
    }


def _all_negative(rows: Rows) -> Rows:
    return [{**row, "prediction": 0} for row in rows]


def _subset_stats(rows: Rows, keep: set[str]) -> dict[str, float]:
    return _prediction_stats([row for row in rows if str(row["image_id"]) in keep])


def analyse(
    data_root: Path | None = None, results_dir: Path | None = None, out_dir: Path | None = None
) -> dict[str, object]:
    """Compute every Harm-C statistic and write ``analysis.json`` to the DIS-01 output folder."""
    base_dir = data_root or harmeme_dir()
    in_dir = results_dir or RESULTS_DIR
    target = out_dir or OUT_DIR
    annotations = {
        split: _load_annotations(split, base_dir) for split in ("train", "validation", "test")
    }
    keyword = select_keyword(annotations["train"])
    result: dict[str, object] = {"keyword_selected_on_train": keyword, "splits": {}}
    splits_out = result["splits"]
    assert isinstance(splits_out, dict)

    for split, records in annotations.items():
        harmful = [rec for rec in records if _is_harmful(rec)]
        harmless = [rec for rec in records if not _is_harmful(rec)]
        targets = Counter(str(rec["labels"][1]) for rec in harmful)  # type: ignore[index]
        entry: dict[str, object] = {
            "memes": len(records),
            "harmful": len(harmful),
            "harmful_targets": dict(targets.most_common()),
            "topic_mentions": {
                name: {
                    "harmful": sum(bool(re.search(p, str(r["text"]), re.I)) for r in harmful),
                    "harmless": sum(bool(re.search(p, str(r["text"]), re.I)) for r in harmless),
                }
                for name, p in TOPICS.items()
            },
        }
        gts = [_is_harmful(rec) for rec in records]
        word_preds = [int(keyword in _words(rec["text"])) for rec in records]
        entry["keyword_rule"] = compute_binary_metrics(word_preds, gts)
        splits_out[split] = entry
        if split == "train":
            continue

        rows = {
            name: _read_predictions(in_dir / split / f"harmeme_{split}_{slug}.json")[1]
            for name, slug in RUNS.items()
        }
        rows["all_negative"] = _all_negative(rows["svm_rbf"])
        entry["runs"] = {name: _stats_with_ci(value) for name, value in rows.items()}
        entry["paired_tests"] = {
            f"{first}_vs_{second}": _paired_test(rows[first], rows[second])
            for first, second in PAIRED_TESTS
        }
        original = {str(r["image_id"]): r["raw_response"] for r in rows["gemini_original"]}
        entry["gemini_rerun_identical_replies"] = sum(
            original[str(r["image_id"])] == r["raw_response"] for r in rows["gemini_rerun"]
        )
        entry["gemini_rerun_elapsed_seconds"] = _read_predictions(
            in_dir / split / f"harmeme_{split}_{RUNS['gemini_rerun']}.json"
        )[0].get("elapsed_seconds")

        with_word = {str(rec["id"]) for rec in records if keyword in _words(rec["text"])}
        without_word = {str(rec["id"]) for rec in records} - with_word
        entry["keyword_strata"] = {
            name: {
                "with_word": _subset_stats(rows[name], with_word),
                "without_word": _subset_stats(rows[name], without_word),
            }
            for name in ("svm_rbf", "xgboost", "gemini_original", "gemini_definition")
        }
        entry["keyword_strata_sizes"] = {
            "with_word": len(with_word),
            "without_word": len(without_word),
            "with_word_harmful": sum(
                _is_harmful(rec) for rec in records if str(rec["id"]) in with_word
            ),
            "without_word_harmful": sum(
                _is_harmful(rec) for rec in records if str(rec["id"]) in without_word
            ),
        }

    target.mkdir(parents=True, exist_ok=True)
    (target / "analysis.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    analyse()
