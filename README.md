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

## Quick Start

```bash
git clone https://github.com/mishanilkazreen/content-moderation.git
cd content-moderation
uv venv --python 3.10
source .venv/bin/activate
uv sync --dev
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
uv run pre-commit install
```

GPU: swap the index URL for `cu118` or `cu121`.

## Dataset

[HatefulIllusion](https://huggingface.co/datasets/yiting/HatefulIllusion_Dataset)
(2,160 images): 300 digits, 690 hate slangs, 1,170 hate symbols. Each
image carries a visibility score (1–5). See
[Qu et al. 2025](https://arxiv.org/pdf/2507.22617).

A Hugging Face account and read token are required. Add it to `.env` in
the project root (gitignored):

```
HF_TOKEN=hf_your_token_here
```

Get a token at **huggingface.co → Settings → Access Tokens** (read
permission is sufficient). Optionally set `HF_HOME` in `.env` to redirect
the cache to a larger drive (default: `~/.cache/huggingface`).

**Download the dataset** (run once before benchmarking):

```bash
uv run python scripts/download_dataset.py                          # all three subsets
uv run python scripts/download_dataset.py --subsets digits         # digits only
uv run python scripts/download_dataset.py --cache-dir D:\hf_cache  # custom cache
```

**Verify the download:**

```bash
uv run python scripts/verify_dataset.py          # full check (metadata + images)
uv run python scripts/verify_dataset.py --fast   # metadata only
uv run python scripts/verify_dataset.py --subsets digits
```

## Working on a task

Every task in `tasks.md` has a pre-created branch. Work one task per
branch.

```bash
git fetch origin
git checkout task-3-yolo-benchmark          # pick the task branch
source .venv/bin/activate
uv sync --dev
# ... implement ...
uv run ruff check --fix .
uv run ruff format .
uv run mypy models/ utils/ scripts/
uv run pytest
uv run python scripts/check_tasks.py --task 3
git push -u origin task-3-yolo-benchmark
gh pr create --fill --assignee LouisFIP27 --reviewer Mishanil
```

`scripts/check_tasks.py` runs the task-marker tests under
`tests/tasks/`. A task is "done" when its marker test passes.

## Benchmarking (task 3)

**Pretrained YOLO** (no fine-tuning, fastest):

```bash
uv run python scripts/benchmark_yolo.py --mode pretrained --model yolov8n.pt --subset digits
uv run python scripts/benchmark_yolo.py --mode pretrained --all   # all five models
```

**With a preprocessing filter** (task 3 step 4):

```bash
uv run python scripts/benchmark_yolo.py --mode pretrained --model yolov8n.pt --subset digits --preprocess blur
```

Available filters: `blur` `downscale` `grid` `gradient` `canny` `grayscale` `histogram` `gamma`
`histogram_blur` `gamma_blur` `blur_gradient` `blur_histogram`

**Fine-tune first, then benchmark:**

```bash
uv run python models/yolo/trainer.py --model yolov8n.pt --subset digits
uv run python scripts/benchmark_yolo.py --mode trained --model yolov8n.pt --subset digits
```

Results are written to `results/yolo_benchmark.json` (gitignored).

## Code quality

```bash
uv run ruff check --fix .
uv run ruff format .
uv run mypy models/ utils/ scripts/
uv run pytest
uv run python scripts/lint_markdown.py
uv run pre-commit run --all-files
```

Standards: see
[`.kiro/steering/linting-standards.md`](.kiro/steering/linting-standards.md).

## Project layout

```text
content-moderation/
├── .kiro/specs/vlm-content-moderation/   # requirements, design, tasks
├── models/
│   ├── yolo/          # Standard Ultralytics YOLO wrappers (task 3)
│   ├── vlm/           # YOLO-World, CLIP-YOLO, explainer (tasks 4, 6)
│   └── explainability/
├── utils/             # dataset, preprocessing, augmentation, OCR
├── scripts/
│   ├── download_dataset.py                # download HatefulIllusion from HF Hub
│   ├── verify_dataset.py                  # verify cached dataset integrity
│   ├── benchmark_yolo.py                  # task 3 — YOLO benchmark
│   ├── train_yolo.py                      # fine-tune YOLO on HatefulIllusion
│   ├── benchmark_vlm.py                   # task 4
│   ├── benchmark_with_symbol_catalog.py   # task 5
│   ├── explain_with_vlm.py                # task 6
│   ├── check_tasks.py                     # task tracker
│   └── visualize_transformations.py
├── tests/
│   ├── unit/          # utils tests
│   ├── property/      # hypothesis tests for utils
│   └── tasks/         # one file per top-level task in tasks.md
├── papers/            # reference PDFs
└── pyproject.toml
```

## Licences

All current dependencies use permissive or weak-copyleft licences.
See [`.kiro/steering/dependency-licenses.md`](.kiro/steering/dependency-licenses.md).
