"""Unit tests for the MAMI_SPLIT_DIR / MAMI_SPLIT_TAG split override (EXP-02, sciama/JOBS.md S4)."""

from __future__ import annotations

import csv
import importlib.util
from pathlib import Path
import sys
from types import ModuleType

from PIL import Image
import pytest
import torch

from utils.dataset import (
    SPLIT_SOURCE_COLUMN,
    DatasetManager,
    MamiDataset,
    split_dir_override,
    split_tag_suffix,
)

ROOT = Path(__file__).resolve().parents[2]
COLS = ["file_name", "label", "shaming", "stereotype", "objectification", "violence", "text"]


def _write_tsv(path: Path, rows: list[dict[str, str]], extra: tuple[str, ...] = ()) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=[*COLS, *extra], delimiter="\t")
        w.writeheader()
        w.writerows(rows)


def _row(name: str, label: int, **extra: str) -> dict[str, str]:
    base = dict.fromkeys(COLS[2:6], "0")
    return {"file_name": name, "label": str(label), "text": f"text {name}", **base, **extra}


@pytest.fixture()
def kaggle_root(tmp_path: Path) -> Path:
    """Fake Kaggle root: original TSVs plus distinguishable images in both folders."""
    root = tmp_path / "kaggle"
    for sub, colour in (("training_images", (255, 0, 0)), ("test_images", (0, 0, 255))):
        d = root / "MAMI_2022_images" / sub
        d.mkdir(parents=True)
        for name in ("1.jpg", "2.jpg", "3.jpg"):
            Image.new("RGB", (8, 8), colour).save(d / name)
    _write_tsv(root / "train.tsv", [_row("1.jpg", 1), _row("2.jpg", 0)])
    _write_tsv(root / "validation.tsv", [_row("3.jpg", 1)])
    _write_tsv(root / "test.tsv", [_row("1.jpg", 0), _row("2.jpg", 1), _row("3.jpg", 0)])
    return root


@pytest.fixture()
def split_dir(tmp_path: Path) -> Path:
    """Alternative split whose test set mixes original train and test memes."""
    d = tmp_path / "tdisjoint"
    d.mkdir()
    extra = ("template_id", SPLIT_SOURCE_COLUMN)
    _write_tsv(d / "train.tsv", [_row("2.jpg", 0, template_id="T1", source_split="train")], extra)
    _write_tsv(
        d / "validation.tsv", [_row("3.jpg", 1, template_id="T2", source_split="validation")], extra
    )
    _write_tsv(
        d / "test.tsv",
        [
            _row("1.jpg", 1, template_id="T3", source_split="train"),
            _row("2.jpg", 1, template_id="T4", source_split="test"),
        ],
        extra,
    )
    return d


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MAMI_SPLIT_DIR", raising=False)
    monkeypatch.delenv("MAMI_SPLIT_TAG", raising=False)


def _red(img: torch.Tensor) -> bool:
    """True when a (3, H, W) image tensor is red (training_images fixture colour)."""
    return bool(img[0, 0, 0] > 0.9 and img[2, 0, 0] < 0.1)


def test_default_behaviour_unchanged(kaggle_root: Path) -> None:
    """Without the env vars the Kaggle TSVs and the split's own image folder are used."""
    assert split_dir_override() is None
    assert split_tag_suffix() == ""
    ds = MamiDataset(str(kaggle_root), split="test")
    assert ds.split_dir is None
    assert len(ds) == 3
    assert not _red(ds[0]["image"])  # test_images are blue


def test_env_split_dir_reads_override_tsv(
    kaggle_root: Path, split_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """MAMI_SPLIT_DIR replaces the TSVs; images follow each row's source_split."""
    monkeypatch.setenv("MAMI_SPLIT_DIR", str(split_dir))
    ds = MamiDataset(str(kaggle_root), split="test")
    assert len(ds) == 2
    first, second = ds[0], ds[1]
    assert first["image_id"] == "1" and first["misogynous"] == 1
    assert _red(first["image"])  # source_split=train -> training_images (red)
    assert not _red(second["image"])  # source_split=test -> test_images (blue)
    assert ds.image_path(ds._records[0]).parent.name == "training_images"


def test_manager_uses_override_and_does_not_mix_cache(
    kaggle_root: Path, split_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """DatasetManager honours MAMI_SPLIT_DIR and caches random and overridden splits apart."""
    mgr = DatasetManager(dataset_path=str(kaggle_root))
    assert len(mgr.load_dataset("test")) == 3
    monkeypatch.setenv("MAMI_SPLIT_DIR", str(split_dir))
    assert len(mgr.load_dataset("test")) == 2
    assert len(mgr.load_dataset("validation")) == 1


def test_missing_source_split_column_is_rejected(
    kaggle_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An override TSV without a valid source_split column fails loudly."""
    bad = tmp_path / "bad"
    bad.mkdir()
    _write_tsv(bad / "test.tsv", [_row("1.jpg", 0)])
    monkeypatch.setenv("MAMI_SPLIT_DIR", str(bad))
    with pytest.raises(ValueError, match="source_split"):
        MamiDataset(str(kaggle_root), split="test")


def test_split_tag_suffix(monkeypatch: pytest.MonkeyPatch) -> None:
    """MAMI_SPLIT_TAG yields an underscore-prefixed suffix; unsafe tags are rejected."""
    monkeypatch.setenv("MAMI_SPLIT_TAG", "tdisjoint")
    assert split_tag_suffix() == "_tdisjoint"
    monkeypatch.setenv("MAMI_SPLIT_TAG", "../x")
    with pytest.raises(ValueError):
        split_tag_suffix()


def _load(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_benchmark_output_name_tag_after_seed(monkeypatch: pytest.MonkeyPatch) -> None:
    """benchmark_qwen2vl output names carry _<tag> after _seedN, and match sciama_checks."""
    pytest.importorskip("tqdm")
    bench = _load(ROOT / "scripts" / "benchmark_qwen2vl.py", "_bench_qwen2vl_under_test")
    checks = _load(ROOT / "sciama" / "sciama_checks.py", "_sciama_checks_under_test")
    slug = "qwen2_vl_7b_instruct"
    assert (
        bench.output_file_name("test", slug, "_finetuned_seed1")
        == f"qwen2vl_test_{slug}_finetuned_seed1.json"
    )
    monkeypatch.setenv("MAMI_SPLIT_TAG", "tdisjoint")
    name = bench.output_file_name("test", slug, "_multiclass_finetuned_seed1")
    assert name == f"qwen2vl_test_{slug}_multiclass_finetuned_seed1_tdisjoint.json"
    expected = checks.bench_file(
        "qwen2vl",
        "test",
        "Qwen/Qwen2-VL-7B-Instruct",
        "multiclass",
        finetuned=True,
        seed=1,
        tag="tdisjoint",
    )
    assert expected.name == name
    assert bench.split_provenance() == {}
    monkeypatch.setenv("MAMI_SPLIT_DIR", "data/splits/template_disjoint")
    assert bench.split_provenance()["split_tag"] == "tdisjoint"


def test_train_vlm_adapter_dir_uses_tag() -> None:
    """train_vlm.py builds the adapter directory with the split tag after _seedN."""
    src = (ROOT / "scripts" / "train_vlm.py").read_text(encoding="utf-8")
    assert "_seed{args.seed}{split_tag_suffix()}" in src
    assert "MAMI_SPLIT_TAG" in src


def test_sciama_truth_reads_template_disjoint_tsv(
    split_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """sciama_checks._truth reads ids/labels from the split TSVs when a split dir is in use."""
    checks = _load(ROOT / "sciama" / "sciama_checks.py", "_sciama_checks_truth")
    ids, lab, sub = checks._truth("test", split_dir)
    assert ids == ["1", "2"]
    assert lab == {"1": 1, "2": 1}
    assert sub["1"] == {"shaming": 0, "stereotype": 0, "objectification": 0, "violence": 0}
    monkeypatch.setenv("MAMI_SPLIT_DIR", str(split_dir))
    assert checks._truth("validation")[0] == ["3"]
    monkeypatch.delenv("MAMI_SPLIT_DIR")
    monkeypatch.setattr(sys, "argv", ["sciama_checks.py", "verify", "--job", "S4"])
    committed = ROOT / "data" / "splits" / "template_disjoint" / "test.tsv"
    if committed.exists():
        s4_ids = checks._truth("test")[0]
        assert len(s4_ids) == 1000 and len(set(s4_ids)) == 1000


def test_committed_template_split_is_disjoint() -> None:
    """The committed template-disjoint split has the expected sizes and no shared template."""
    d = ROOT / "data" / "splits" / "template_disjoint"
    if not (d / "train.tsv").exists():
        pytest.skip("template-disjoint split not generated")
    tmpl: dict[str, set[str]] = {}
    for sp, size in (("train", 9000), ("validation", 1000), ("test", 1000)):
        with (d / f"{sp}.tsv").open(encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh, delimiter="\t"))
        assert len(rows) == size
        assert sum(int(r["label"]) for r in rows) == size // 2
        assert {r[SPLIT_SOURCE_COLUMN] for r in rows} <= {"train", "validation", "test"}
        tmpl[sp] = {r["template_id"] for r in rows}
    assert not tmpl["train"] & tmpl["validation"]
    assert not tmpl["train"] & tmpl["test"]
    assert not tmpl["validation"] & tmpl["test"]
