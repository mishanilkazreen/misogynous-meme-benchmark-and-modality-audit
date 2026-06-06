# Detection of Embedded Hateful Content in Images

Benchmarking standard YOLO detectors against text-prompted
vision-language detectors (VLMs) for content moderation. The milestone
goal is a paper comparing YOLO (v8/v10/v11/v12/v26) against
vision-language detectors (YOLO-World, CLIP-YOLO, optionally
CLIP-YOLO, optionally YOLO-UniOW) on the HatefulIllusion dataset, then
extending to a hate-symbol catalogue pipeline and VLM-generated
explanations for moderators.

The full task plan is in
[`.kiro/specs/vlm-content-moderation/tasks.md`](.kiro/specs/vlm-content-moderation/tasks.md).
Reference papers are catalogued in [`papers/README.md`](papers/README.md).

## Prerequisites

- Python 3.10 (pinned by `.python-version`)
- [`uv`](https://docs.astral.sh/uv/getting-started/installation/) for
  environments and packages
- [`gh`](https://cli.github.com/) for branch/PR management
- VS Code with extensions from `.vscode/extensions.json`
  (ruff, mypy, markdownlint, TOML)

## Setup

```bash
git clone https://github.com/mishanilkazreen/content-moderation.git
cd content-moderation
uv sync --dev
uv run pre-commit install
```

## Dataset

2,160 steganographic images across three subsets, each with a visibility score
1–5:

| Subset | Images | Labels |
|---|---|---|
| `digits` | 300 | 0–9 (10 classes) |
| `hate_slangs` | 690 | slang terms |
| `hate_symbols` | 1,170 | hate symbols (32 classes) |

Use `digits` during development; switch to `hate_slangs`/`hate_symbols` only
for final benchmark runs.

## Task 4 — VLM Benchmark

### Model overview

| Model | Size | Download | VRAM (fp16) |
|---|---|---|---|
| CLIP (ViT-B/32) | ~600 MB | auto on first run | ~1 GB |
| LLaVA 1.5 | 7B | auto on first run (~14 GB) | ~14 GB |
| Qwen2-VL | 7B | auto on first run (~15 GB) | ~14 GB |

### CLIP

```bash
# Verify (10 samples, no preprocessing)
uv run python scripts/benchmark_clip.py --subset digits --limit 10 --device cuda --filters none

# Full run — all three subsets
uv run python scripts/benchmark_clip.py --subset digits --device cuda
uv run python scripts/benchmark_clip.py --subset hate_slangs --device cuda
uv run python scripts/benchmark_clip.py --subset hate_symbols --device cuda
```

Output: `results/clip_{subset}.json`

### LLaVA

```bash
# Verify (10 samples, no preprocessing)
uv run python scripts/benchmark_llava.py --subset digits --limit 10 --device cuda --filters none

# Full run — all three subsets
uv run python scripts/benchmark_llava.py --subset digits --device cuda
uv run python scripts/benchmark_llava.py --subset hate_slangs --device cuda
uv run python scripts/benchmark_llava.py --subset hate_symbols --device cuda
```

Output: `results/llava_{subset}.json`

### Qwen2-VL

```bash
# Verify (10 samples, no preprocessing)
uv run python scripts/benchmark_qwen2vl.py --subset digits --limit 10 --device cuda --filters none

# Full run — all three subsets
uv run python scripts/benchmark_qwen2vl.py --subset digits --device cuda
uv run python scripts/benchmark_qwen2vl.py --subset hate_slangs --device cuda
uv run python scripts/benchmark_qwen2vl.py --subset hate_symbols --device cuda
```

Output: `results/qwen2vl_{subset}.json`

### Options common to all three scripts

| Flag | Default | Description |
|---|---|---|
| `--subset` | `digits` | `digits`, `hate_slangs`, `hate_symbols`, or `all` |
| `--filters` | all | Comma-separated preprocessing filters, e.g. `none,blur` |
| `--limit` | none | Cap number of images (useful for quick checks) |
| `--device` | `cpu` | `cuda` or `cpu` |
| `--batch-size` | `4` | Images per forward pass (LLaVA/Qwen2-VL only) |

## Code quality

```bash
uv run ruff check --fix .
uv run ruff format .
uv run mypy models/ utils/ scripts/
uv run pytest
uv run python scripts/lint_markdown.py
uv run pre-commit run --all-files
```

## Project layout

```text
content-moderation/
├── models/
│   ├── yolo/           # Ultralytics YOLO wrappers (task 3)
│   └── vlm/            # CLIP, LLaVA, Qwen2-VL classifiers (task 4)
├── utils/              # dataset, preprocessing, augmentation, OCR
├── scripts/
│   ├── benchmark_clip.py
│   ├── benchmark_llava.py
│   ├── benchmark_qwen2vl.py
│   ├── benchmark_gpt4omini.py   # optional cloud upper-bound
│   ├── benchmark_gemini.py      # optional cloud upper-bound
│   └── benchmark_vlm_classification.py  # orchestrator (all models)
├── results/            # JSON outputs (gitignored)
├── task_4/             # plan, requirements, how_to_run
└── tests/
    ├── unit/
    └── tasks/          # acceptance-gate tests (one per task)
```
