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

Use `digits` during development; switch to `hate_slangs`/`hate_symbols` only
for final benchmark runs.

## VLM Benchmark

### Hardware and quantization

Local generative models (LLaVA, Qwen2-VL) run with **4-bit NF4
quantization** by default (via `bitsandbytes`), reducing VRAM from
~14 GB to ~5 GB so they fit on 12 GB consumer GPUs. Pass
`--quantize none` for full fp16 (requires 24+ GB VRAM).

> **CUDA note:** torch is pinned to the **CUDA 12.8** build
> (`pytorch-cu128` index in `pyproject.toml`) because `bitsandbytes`
> ships native binaries up to CUDA 13.0 only. Do not bump torch to a
> CUDA 13.2 build — `bitsandbytes` 4-bit kernels will crash at
> runtime against a 13.2 runtime. CUDA 12.8 works on all RTX 30/40
> series and data-centre Ampere/Hopper GPUs.

### Model overview

| Model | Size | Download | VRAM (4-bit) | VRAM (fp16) |
|---|---|---|---|---|
| CLIP (ViT-B/32) | ~600 MB | auto on first run | n/a | ~1 GB |
| LLaVA 1.5 | 7B | auto on first run (~14 GB) | ~5 GB | ~14 GB |
| Qwen2-VL | 7B | auto on first run (~15 GB) | ~5 GB | ~14 GB |

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

### Gemini

Cloud API. Requires a Gemini API key.

Add to `.env` (gitignored):

```
GEMINI_API_KEY=your_key_here
```

Then export it before running (PowerShell):

```powershell
$env:GEMINI_API_KEY = "your_key_here"
```

```bash
# Sanity-check (text prompt only)
uv run python scripts/test_gemini.py

# Verify (5 samples, no preprocessing)
uv run python scripts/benchmark_gemini.py --subset digits --limit 5 --filters none

# Full run — all three subsets
uv run python scripts/benchmark_gemini.py --subset digits
uv run python scripts/benchmark_gemini.py --subset hate_slangs
uv run python scripts/benchmark_gemini.py --subset hate_symbols

# Increase parallelism (default 20 threads)
uv run python scripts/benchmark_gemini.py --subset digits --workers 40
```

Output: `results/gemini_{subset}.json`

### Options common to all per-model scripts

| Flag | Default | Description |
|---|---|---|
| `--subset` | `digits` | `digits`, `hate_slangs`, `hate_symbols`, or `all` |
| `--filters` | all | Comma-separated preprocessing filters, e.g. `none,blur` |
| `--limit` | none | Cap number of images (useful for quick checks) |
| `--device` | `cpu` | `cuda` or `cpu` |
| `--batch-size` | `4` | Images per forward pass (LLaVA/Qwen2-VL only) |
| `--quantize` | `4bit` | `4bit`, `8bit`, or `none` (LLaVA/Qwen2-VL only) |
| `--workers` | `20` | Parallel API threads (Gemini only) |
| `--binary` | off | Binary yes/no mode (see below) |

### Binary classification mode

Pass `--binary` to any per-model script or to the orchestrator to run **binary
content moderation** instead of closed-set label identification.

In binary mode the prompt becomes:

> *"Does this image contain any hateful, offensive, or inappropriate content?
> Answer yes or no."*

Output files get a `_binary` suffix: `gemini_digits_binary.json`, etc.

```bash
# Quick binary sanity-check (5 images, no preprocessing)
uv run python scripts/benchmark_gemini.py --subset digits --limit 5 --filters none --binary
uv run python scripts/benchmark_gpt4omini.py --subset digits --limit 5 --filters none --binary

# Full binary run via the orchestrator
uv run python scripts/benchmark_vlm_classification.py --model gemini --subset digits --binary
uv run python scripts/benchmark_vlm_classification.py --model all --subset digits --binary
```

Output: `results/vlm_classification_binary.json`

> **Note:** CLIP does not take generative prompts. In binary mode it uses the
> text labels `["hateful", "not hateful"]` with ground truth `"hateful"` instead
> of the yes/no scheme used by generative models.

## Benchmarking YOLO

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
