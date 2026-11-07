"""Unit tests for the HarMeme (Harm-C) second-dataset evaluation pipeline (DIS-01)."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from sciama import sciama_checks
from scripts.revision.harmeme_analysis import exact_mcnemar_p, select_keyword
from scripts.revision.harmeme_eval import (
    HARMEME_BASE_PROMPT,
    HARMEME_SYSTEM_INSTRUCTION,
    build_benchmark_payload,
    build_harmeme_prompt,
    compare_runs,
    compute_binary_metrics,
    load_split_records,
    map_harmeme_label,
    parse_yesno_response,
    resolve_image_path,
    run_classical_classifiers,
    run_gemini_zeroshot,
    summarize_results,
    verify_harmeme_output_file,
)


def _write_synthetic_annotations(data_root: Path, counts: dict[str, int]) -> None:
    """Create synthetic Harm-C JSONL split files with balanced harmful/harmless records."""
    ann_dir = data_root / "annotations"
    ann_dir.mkdir(parents=True, exist_ok=True)
    for split, n in counts.items():
        out_path = ann_dir / f"{split}.jsonl"
        lines: list[str] = []
        for idx in range(n):
            is_harmful = idx % 2 == 1
            raw_label = (
                ("very harmful" if idx % 4 == 1 else "somewhat harmful")
                if is_harmful
                else "not harmful"
            )
            record = {
                "id": f"covid_memes_{split}_{idx:04d}",
                "image": f"covid_memes_{split}_{idx:04d}.png",
                "labels": [raw_label],
                "text": f"meme transcript {split} {idx}",
            }
            lines.append(json.dumps(record))
        out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_synthetic_embeddings(emb_dir: Path, data_root: Path, splits: tuple[str, ...]) -> None:
    """Create separable synthetic 768-D CLIP embeddings for each split."""
    emb_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(12345)
    for split in splits:
        records = load_split_records(split, data_root=data_root, strict_counts=False)
        n = len(records)
        labels = np.asarray([r.label for r in records], dtype=np.int64)
        ids = np.asarray([r.image_id for r in records], dtype=str)

        direction = np.ones((1, 768), dtype=np.float32)
        y_sign = (labels * 2 - 1).reshape(-1, 1).astype(np.float32)

        img_emb = rng.normal(0.0, 0.5, size=(n, 768)).astype(np.float32) + 1.0 * y_sign * direction
        txt_emb = rng.normal(0.0, 0.5, size=(n, 768)).astype(np.float32) + 1.5 * y_sign * direction

        img_emb /= np.linalg.norm(img_emb, axis=1, keepdims=True)
        txt_emb /= np.linalg.norm(txt_emb, axis=1, keepdims=True)

        np.savez_compressed(
            emb_dir / f"harmeme_{split}_vit_l_14_quickgelu.npz",
            image_embeddings=img_emb,
            text_embeddings=txt_emb,
            labels=labels,
            image_ids=ids,
        )


def test_map_harmeme_label_collapses_three_way_intensity() -> None:
    """Verify binary label mapping matches Pramanick et al. (2021) and CapAlign (2024)."""
    assert map_harmeme_label(["not harmful"]) == (0, "not harmful")
    assert map_harmeme_label(["somewhat harmful"]) == (1, "somewhat harmful")
    assert map_harmeme_label(["very harmful"]) == (1, "very harmful")
    assert map_harmeme_label("not harmful") == (0, "not harmful")
    assert map_harmeme_label("somewhat harmful") == (1, "somewhat harmful")
    assert map_harmeme_label(0) == (0, "0")
    assert map_harmeme_label(2) == (1, "2")

    with pytest.raises(ValueError, match="Unrecognised HarMeme label"):
        map_harmeme_label(["unknown_category"])


def test_parse_yesno_response_does_not_treat_harmful_as_refusal() -> None:
    """Verify 'harmful' in VLM responses is parsed as a valid label rather than a refusal."""
    assert parse_yesno_response("Yes") == (1, False)
    assert parse_yesno_response("No") == (0, False)
    assert parse_yesno_response("Yes, this meme is harmful.") == (1, False)
    assert parse_yesno_response("No, the meme is not harmful.") == (0, False)
    assert parse_yesno_response("I cannot fulfill this request as an AI.") == (0, True)
    assert parse_yesno_response("") == (0, True)


def test_build_harmeme_prompt_includes_ocr_transcript() -> None:
    """Verify prompt formatting with and without OCR text."""
    p_with_text = build_harmeme_prompt("stay home save lives")
    assert 'This meme contains the text: "stay home save lives".' in p_with_text
    assert "Answer with exactly one word: yes or no." in p_with_text

    p_empty = build_harmeme_prompt("   ")
    assert "This meme contains the text:" not in p_empty
    assert "Answer with exactly one word: yes or no." in p_empty


def test_load_split_records_and_resolve_image_path(tmp_path: Path) -> None:
    """Verify JSONL parsing, strict count check, and image path resolution."""
    _write_synthetic_annotations(tmp_path, {"train": 10, "validation": 6, "test": 8})
    records = load_split_records("test", data_root=tmp_path, strict_counts=False)
    assert len(records) == 8
    assert records[0].image_id == "covid_memes_test_0000"
    assert records[0].image_filename == "covid_memes_test_0000.png"
    assert records[0].label == 0
    assert records[1].label == 1

    with pytest.raises(ValueError, match="expected 354 records"):
        load_split_records("test", data_root=tmp_path, strict_counts=True)

    nested_dir = tmp_path / "images" / "memes_3013"
    nested_dir.mkdir(parents=True)
    img_file = nested_dir / "covid_memes_test_0000.png"
    img_file.write_bytes(b"fake_png")

    resolved = resolve_image_path(records[0], data_root=tmp_path)
    assert resolved == img_file


_ISOLATED_ENV = "HARMEME_TEST_ISOLATED"


def test_classical_classifiers_and_verification_pipeline(tmp_path: Path) -> None:
    """Run classical classifiers on synthetic embeddings and verify all generated artefacts."""
    if sys.platform == "darwin" and "torch" in sys.modules and not os.environ.get(_ISOLATED_ENV):
        # On macOS, torch and xgboost each load their own OpenMP runtime; fitting XGBoost after
        # torch was imported by an earlier test module segfaults. Re-run in a fresh interpreter.
        node = f"{Path(__file__).resolve()}::test_classical_classifiers_and_verification_pipeline"
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", node, "-q", "-p", "no:cacheprovider"],
            env={**os.environ, _ISOLATED_ENV: "1"},
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 0, proc.stdout + proc.stderr
        return

    data_root = tmp_path / "data" / "harmeme"
    emb_dir = tmp_path / "results" / "embeddings"
    results_dir = tmp_path / "results"
    out_dir = results_dir / "revision" / "dis01_harmeme"

    counts = {"train": 40, "validation": 16, "test": 20}
    _write_synthetic_annotations(data_root, counts)
    _write_synthetic_embeddings(emb_dir, data_root, ("train", "validation", "test"))

    summary = run_classical_classifiers(
        emb_dir=emb_dir,
        results_dir=results_dir,
    )

    assert "svm_rbf_test" in summary
    assert "xgboost_test" in summary
    assert summary["svm_rbf_test"]["accuracy"] > 0.7

    # Verify generated per-sample JSON file
    sample_file = results_dir / "test" / "harmeme_test_clip_svm_rbf.json"
    assert sample_file.is_file()
    errs = verify_harmeme_output_file(
        sample_file,
        split="test",
        data_root=data_root,
        strict_counts=False,
    )
    assert errs == []

    # Corrupt accuracy in a copy and confirm verification fails
    corrupted = json.loads(sample_file.read_text(encoding="utf-8"))
    corrupted[0]["exact_match_accuracy"] = 0.0
    bad_file = results_dir / "test" / "harmeme_test_corrupted.json"
    bad_file.write_text(json.dumps(corrupted), encoding="utf-8")
    errs_bad = verify_harmeme_output_file(
        bad_file,
        split="test",
        data_root=data_root,
        strict_counts=False,
    )
    assert any("exact_match_accuracy" in e for e in errs_bad)

    # Test summarize_results with synthetic VLM zero-shot outputs
    test_records = load_split_records("test", data_root=data_root, strict_counts=False)
    ids = [r.image_id for r in test_records]
    gts = [r.label for r in test_records]
    for filename, model_name in [
        ("harmeme_test_qwen2_vl_7b_instruct.json", "Qwen/Qwen2-VL-7B-Instruct"),
        ("harmeme_test_gemini_3_1_flash_lite.json", "gemini-3.1-flash-lite"),
    ]:
        payload = build_benchmark_payload(
            model_name=model_name,
            split="test",
            image_ids=ids,
            gts=gts,
            preds=gts,
            refusals=[False] * len(gts),
        )
        out_p = results_dir / "test" / filename
        out_p.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    m = compute_binary_metrics(gts, gts)
    assert m["exact_match_accuracy"] == pytest.approx(1.0)

    merged = summarize_results(
        data_root=data_root,
        results_dir=results_dir,
        out_dir=out_dir,
        strict_counts=False,
    )
    assert (out_dir / "metrics.json").is_file()
    assert (out_dir / "table_rows.tex").is_file()
    assert (out_dir / "summary.md").is_file()
    tex_content = (out_dir / "table_rows.tex").read_text(encoding="utf-8")
    assert "This work: SVM-RBF on CLIP" in tex_content
    assert "This work: XGBoost on CLIP" in tex_content
    assert "This work: Qwen2-VL-7B" in tex_content
    assert "This work: Gemini 3.1 Flash Lite" in tex_content
    models_list = merged["models"]
    assert isinstance(models_list, list)
    assert len(models_list) == 4


def test_sciama_checks_s6_spec() -> None:
    """Verify sciama_checks builds a complete specification for job S6."""
    ns = argparse.Namespace(job="S6")
    spec = sciama_checks.build_spec(ns)
    assert "Qwen/Qwen2-VL-7B-Instruct" in spec.weights
    assert len(spec.outputs) == 6


def test_sciama_checks_s6_qwen_def_spec() -> None:
    """The prompt-sensitivity job writes tagged files, so S6 outputs are never touched."""
    spec = sciama_checks.build_spec(argparse.Namespace(job="S6-qwen-def"))
    assert "Qwen/Qwen2-VL-7B-Instruct" in spec.weights
    assert sorted(p.name for p in spec.outputs) == [
        "harmeme_test_qwen2_vl_7b_instruct_definition.json",
        "harmeme_validation_qwen2_vl_7b_instruct_definition.json",
    ]


def test_baseline_prompt_is_unchanged_and_definition_names_mockery() -> None:
    """The committed runs' wording stays byte-identical; only the new variant adds mockery."""
    assert HARMEME_BASE_PROMPT == (
        "Is this meme harmful? A harmful meme attacks, demeans, exploits, or promotes harm against "
        "an individual, organization, community, or society. Answer with exactly one word: yes or no."
    )
    assert build_harmeme_prompt("hi") == f'This meme contains the text: "hi". {HARMEME_BASE_PROMPT}'
    definition = build_harmeme_prompt("hi", "definition")
    assert "mocking or ridiculing a person or an idea" in definition
    assert "mocking" not in build_harmeme_prompt("hi")
    assert definition.endswith("Answer with exactly one word: yes or no.")


class _FakeGeminiModels:
    """Answers from a fixed function and records every prompt and system instruction."""

    def __init__(self, reply_for: object, calls: list[tuple[str, str]]) -> None:
        self.reply_for = reply_for
        self.calls = calls

    def generate_content(self, *, model: str, contents: object, config: object) -> SimpleNamespace:
        """Return one fake reply; ``model`` is unused."""
        del model
        prompt = contents.parts[1].text  # type: ignore[attr-defined]
        self.calls.append((prompt, config.system_instruction))  # type: ignore[attr-defined]
        assert callable(self.reply_for)
        return SimpleNamespace(text=self.reply_for(prompt), model_version="fake-model-001")


def _prepare_gemini_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reply_for: object
) -> tuple[Path, Path, list[tuple[str, str]]]:
    """Create synthetic Harm-C data and replace the Gemini client with a fake."""
    from google import genai

    data_root = tmp_path / "harmeme"
    results_dir = tmp_path / "results"
    _write_synthetic_annotations(data_root, {"validation": 6, "test": 8})
    (data_root / "images").mkdir(parents=True)
    for split in ("validation", "test"):
        for rec in load_split_records(split, data_root=data_root, strict_counts=False):
            Image.new("RGB", (4, 4)).save(data_root / "images" / rec.image_filename)
    calls: list[tuple[str, str]] = []
    fake_client = SimpleNamespace(models=_FakeGeminiModels(reply_for, calls))
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(genai, "Client", lambda **_: fake_client)
    return data_root, results_dir, calls


def test_gemini_tagged_run_records_provenance_and_refuses_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A tagged run keeps earlier files, stores its prompt and timing, and cannot be repeated."""

    def reply_for(prompt: str) -> str:
        return "yes" if prompt.split('"')[1].endswith(("1", "3", "5", "7")) else "no"

    data_root, results_dir, calls = _prepare_gemini_run(tmp_path, monkeypatch, reply_for)
    out = results_dir / "validation" / "harmeme_validation_gemini_3_1_flash_lite_definition.json"
    run_kwargs = {
        "data_root": data_root,
        "results_dir": results_dir,
        "splits": ("validation",),
        "output_tag": "_definition",
        "prompt_variant": "definition",
        "strict_counts": False,
    }

    run_gemini_zeroshot(**run_kwargs)  # type: ignore[arg-type]

    assert len(calls) == 6
    assert all("mocking or ridiculing" in prompt for prompt, _ in calls)
    assert all(system == HARMEME_SYSTEM_INSTRUCTION for _, system in calls)
    assert not (results_dir / "test").exists()
    record = json.loads(out.read_text(encoding="utf-8"))[0]
    assert record["prompt_variant"] == "definition"
    assert record["api_model_versions"] == ["fake-model-001"]
    assert record["retries"] == 0
    assert record["elapsed_seconds"] >= 0
    before = out.read_bytes()

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        run_gemini_zeroshot(**run_kwargs)  # type: ignore[arg-type]

    assert len(calls) == 6
    assert out.read_bytes() == before


def test_compare_runs_reports_reproduction_and_validation_bar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Identical answers reproduce; answers tracking the labels clear the bar over all-"no"."""

    def always_no(_: str) -> str:
        return "no"

    data_root, results_dir, _ = _prepare_gemini_run(tmp_path, monkeypatch, always_no)
    out_dir = tmp_path / "out"
    base = {"data_root": data_root, "results_dir": results_dir, "strict_counts": False}
    run_gemini_zeroshot(**base)  # type: ignore[arg-type]
    run_gemini_zeroshot(**base, output_tag="_rerun")  # type: ignore[arg-type]

    same = compare_runs("_rerun", out_dir=out_dir, **base)  # type: ignore[arg-type]

    assert same["reproduced"] is True
    assert same["clears_validation_bar"] is False
    assert (out_dir / "compare_gemini_3_1_flash_lite_rerun.json").exists()

    def tracks_label(prompt: str) -> str:
        return "yes" if int(prompt.split(chr(34))[1].rsplit(maxsplit=1)[-1]) % 2 == 1 else "no"

    fake = _FakeGeminiModels(tracks_label, [])
    from google import genai

    monkeypatch.setattr(genai, "Client", lambda **_: SimpleNamespace(models=fake))
    run_gemini_zeroshot(  # type: ignore[arg-type]
        **base, splits=("validation",), output_tag="_definition", prompt_variant="definition"
    )

    changed = compare_runs("_definition", out_dir=out_dir, **base)  # type: ignore[arg-type]

    assert changed["reproduced"] is False
    assert changed["clears_validation_bar"] is True
    assert list(changed["splits"]) == ["validation"]  # type: ignore[call-overload]


def test_compare_runs_requires_a_tag(tmp_path: Path) -> None:
    """Untagged files are the reference, so comparing them to themselves is rejected."""
    with pytest.raises(ValueError, match="non-empty"):
        compare_runs("", results_dir=tmp_path)


def test_exact_mcnemar_p_matches_known_values() -> None:
    """The exact test is two-sided and handles the no-discordance case."""
    assert exact_mcnemar_p(0, 0) == 1.0
    assert exact_mcnemar_p(5, 5) == 1.0
    assert exact_mcnemar_p(10, 0) == pytest.approx(2 / 2**10)


def test_select_keyword_uses_only_the_given_split() -> None:
    """The rule is the word that best separates the classes in the records it is given."""
    train = [
        {"labels": ["very harmful", "individual"], "text": "alpha shared"},
        {"labels": ["somewhat harmful", "society"], "text": "alpha shared"},
        {"labels": ["not harmful"], "text": "beta shared"},
        {"labels": ["not harmful"], "text": "beta shared"},
    ]
    assert select_keyword(train, min_count=2) == "alpha"
