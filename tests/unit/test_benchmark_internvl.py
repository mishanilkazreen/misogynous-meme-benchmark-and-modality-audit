"""Unit tests for the InternVL benchmark/train glue (EXP-03, SCIAMA job S5).

These exercise the InternVL code paths with NO GPU and NO model download, mirroring
the module-loading/mocking approach in ``test_split_override.py`` and the
parse-status cases in ``test_benchmark_refusal.py``. Importing the benchmark
module pulls in torch (via ``utils.dataset``), so the whole module is gated on
torch being importable, exactly as the sibling tests are.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import ModuleType

import pytest

pytest.importorskip("torch")
pytest.importorskip("tqdm")

ROOT = Path(__file__).resolve().parents[2]
SUBTYPE_LABELS = ["shaming", "stereotype", "objectification", "violence"]


def _load(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _bench() -> ModuleType:
    return _load(ROOT / "scripts" / "benchmark_internvl.py", "_bench_internvl_under_test")


def test_transformers_guard_defined_without_import_error() -> None:
    """The module imports and defines the availability flag even if transformers is absent."""
    bench = _bench()
    assert hasattr(bench, "_TRANSFORMERS_AVAILABLE")
    assert isinstance(bench._TRANSFORMERS_AVAILABLE, bool)
    assert bench.MODEL_ID == "OpenGVLab/InternVL3-1B-hf"


def test_output_file_name_prefix_and_seed_order(monkeypatch: pytest.MonkeyPatch) -> None:
    """Output names carry the internvl_ prefix and place _<tag> after _seedN."""
    bench = _bench()
    slug = "internvl3_1b_hf"
    assert (
        bench.output_file_name("test", slug, "_finetuned_seed1")
        == f"internvl_test_{slug}_finetuned_seed1.json"
    )
    monkeypatch.setenv("MAMI_SPLIT_TAG", "tdisjoint")
    name = bench.output_file_name("test", slug, "_multiclass_finetuned_seed1")
    assert name == f"internvl_test_{slug}_multiclass_finetuned_seed1_tdisjoint.json"


def test_output_name_matches_sciama_checks_bench_file() -> None:
    """The emitted name equals sciama_checks.bench_file for the same InternVL id/params."""
    bench = _bench()
    checks = _load(ROOT / "sciama" / "sciama_checks.py", "_sciama_checks_for_internvl")
    model_id = "OpenGVLab/InternVL3-1B-hf"
    model_slug = model_id.rsplit("/", maxsplit=1)[-1].lower().replace("-", "_").replace(".", "_")

    zero_shot = bench.output_file_name("validation", model_slug, "")
    expected_zero = checks.bench_file(
        "internvl", "validation", model_id, "singleclass", finetuned=False, seed=None
    )
    assert zero_shot == expected_zero.name

    finetuned = bench.output_file_name("test", model_slug, "_finetuned_seed1")
    expected_ft = checks.bench_file(
        "internvl", "test", model_id, "singleclass", finetuned=True, seed=1
    )
    assert finetuned == expected_ft.name

    multiclass = bench.output_file_name("test", model_slug, "_multiclass_finetuned_seed1")
    expected_mc = checks.bench_file(
        "internvl", "test", model_id, "multiclass", finetuned=True, seed=1
    )
    assert multiclass == expected_mc.name


def test_slug_derivation_matches_out_slug() -> None:
    """The script's model_slug expression equals sciama_checks.out_slug for InternVL ids."""
    checks = _load(ROOT / "sciama" / "sciama_checks.py", "_sciama_checks_out_slug")
    for model_id in (
        "OpenGVLab/InternVL3-1B-hf",
        "OpenGVLab/InternVL3_5-8B-hf",
        "OpenGVLab/InternVL3-14B-hf",
    ):
        local = model_id.split("/")[-1].lower().replace("-", "_").replace(".", "_")
        assert local == checks.out_slug(model_id)


def test_subtype_parse_status_regression() -> None:
    """subtype_parse_status behaves identically to the Qwen regression cases."""
    bench = _bench()
    all_false = (
        '{"shaming": false, "stereotype": false, "objectification": false, "violence": false}'
    )
    assert bench.subtype_parse_status(all_false, SUBTYPE_LABELS) == "json"
    assert bench.subtype_parse_status("none", SUBTYPE_LABELS) == "none"
    assert bench.subtype_parse_status("", SUBTYPE_LABELS) == "empty"
    assert bench.subtype_parse_status("   ", SUBTYPE_LABELS) == "empty"
    assert bench.subtype_parse_status("This is stereotype and shaming", SUBTYPE_LABELS) == "labels"
    assert bench.subtype_parse_status("I think this is funny", SUBTYPE_LABELS) == "unparseable"


def test_cli_flags_present_in_source() -> None:
    """The benchmark CLI exposes the SCIAMA-required flags (parser lives inside main())."""
    src = (ROOT / "scripts" / "benchmark_internvl.py").read_text(encoding="utf-8")
    for flag in ("--lora-path", "--seed", "--task", "--quantize", "--text-source", "--split"):
        assert flag in src, f"missing CLI flag {flag}"


def test_benchmark_uses_internvl_model_class() -> None:
    """The benchmark loads InternVL via the native transformers class, not Qwen's."""
    src = (ROOT / "scripts" / "benchmark_internvl.py").read_text(encoding="utf-8")
    assert "InternVLForConditionalGeneration" in src
    assert "Qwen2VLForConditionalGeneration" not in src
    assert '"model": "internvl"' in src


def test_train_vlm_has_internvl_branch() -> None:
    """train_vlm.py selects an InternVL conversation-formatting branch in both code paths."""
    src = (ROOT / "scripts" / "train_vlm.py").read_text(encoding="utf-8")
    assert 'elif "internvl" in self.model_type:' in src  # VLMCollate.__call__
    assert 'elif "internvl" in model_type:' in src  # _validate_vlm
    assert "internvl" in src  # S5 code marker
