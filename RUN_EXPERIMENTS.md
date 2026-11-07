# Guide to Running the MAMI Experiments

Production pipeline for MAMI experiment execution. Every step below assumes:

- Working branch is `main`.
- GPU work runs on the SCIAMA HPC cluster (NVIDIA L40, 48 GB VRAM) via
  SLURM. The Gemini API call is the only stage that needs an
  internet-enabled machine.

For each stage below the reported compute cost assumes a single L40
seed. Multi-seed reporting (recommended for every fine-tuned system)
multiplies the cost by the number of seeds.

## 0. One-off setup

```bash
uv venv --python 3.10
uv sync --group vlm-gpu
uv run pre-commit install --hook-type pre-commit --hook-type commit-msg --hook-type pre-push

# Kaggle credentials in .env, then:
uv run python scripts/download_dataset.py
```

## Task-name aliases

Every training and benchmark script accepts both the paper-facing
names (``binary`` / ``multilabel``) and the legacy pipeline names
(``singleclass`` / ``multiclass``). The two families are interchangeable
in every command below; pick whichever you find easier to read.

## 1. Extract embeddings

The extract step supports three text sources. The paper reports the
`provided` variant as the primary row and the `paddleocr` / `combined`
variants as ablations.

```bash
# Provided text (MAMI's Text Transcription column, manually verified)
uv run python scripts/extract_embeddings.py \
    --split train,validation,test \
    --model ViT-L-14-quickgelu \
    --text-source provided

# PaddleOCR-only (kept as an ablation for the OCR robustness evaluation)
uv run python scripts/extract_embeddings.py \
    --split train,validation,test \
    --model ViT-L-14-quickgelu \
    --text-source ocr \
    --ocr-engine paddleocr

# Union of both (combining provided transcript and OCR text)
uv run python scripts/extract_embeddings.py \
    --split train,validation,test \
    --model ViT-L-14-quickgelu \
    --text-source combined \
    --ocr-engine paddleocr
```

Filenames follow the convention
`{split}_{model}[_ocr_paddleocr|_combined_paddleocr].npz`. Runs at
different text sources coexist in `results/embeddings/`.

Higher-resolution CLIP variant: replace `ViT-L-14-quickgelu`
with `ViT-L-14-336-quickgelu` in any of the above.

Compute: ~30 minutes per (model, text_source) triple on SCIAMA.

## 2. Tabular sweep

Trains 20+ classical ML models on top of CLIP + text embeddings. The
sweep applies `class_weight="balanced"`, wraps non-tree
models in a `StandardScaler` pipeline, and reports the MAMI
official Task B metric.

```bash
# Validation split (hyperparameter tuning + threshold pick)
uv run autobench train --model auto_benchmark/config/model/mami_tabular_model.yaml

# Test split (final evaluation, same seed)
uv run autobench train --model auto_benchmark/config/model/mami_tabular_model_test.yaml
```

For each of `--text-source` in `{provided, paddleocr, combined}`,
create a matching data config in `auto_benchmark/config/data/`
pointing at the corresponding NPZ file.

For multi-seed reporting, iterate `random_state` in
`auto_benchmark/config/model/*.yaml` (default is 4711) over `{1, 2, 3}`
and average.

Compute: ~1 hour per (task, text_source, seed) triple on SCIAMA CPU.

## 3. Top-level XGBoost head (feeds SHAP + concept projection)

`scripts/train_classifier.py` fits a single XGBoost model on top of the
pre-extracted embeddings and saves a `.pkl`. This is the model the XAI
scripts (`run_shap_modality.py`, `run_clip_concept_projection.py`) load.

```bash
# Task A binary (previously ``--task singleclass``)
uv run python scripts/train_classifier.py \
    --model-name ViT-L-14-quickgelu \
    --task binary \
    --classifier xgboost \
    --text-source provided \
    --threshold-calibrate \
    --seed 42

# Task B multi-label (previously ``--task multiclass``)
uv run python scripts/train_classifier.py \
    --model-name ViT-L-14-quickgelu \
    --task multilabel \
    --classifier xgboost \
    --text-source provided \
    --seed 42
```

`--threshold-calibrate` (default on for binary) scans decision
thresholds on validation and picks the one that maximises macro F1.
`--calibrate-isotonic` is an optional flag that wraps the fit
in `CalibratedClassifierCV` for downstream ensembling.

Compute: ~15 seconds per task on CPU.

## 4. CLIP zero-shot benchmark

```bash
uv run python scripts/benchmark_clip.py \
    --split test \
    --task binary \
    --model-name ViT-L-14-quickgelu \
    --text-source provided \
    --prompt-ensemble \
    --tta \
    --device cuda
```

`--prompt-ensemble` (default on) averages 5-8 phrase embeddings per
class. `--tta` enables horizontal-flip test-time augmentation. Both are inference-only wins.

Compute: ~1 minute per (model, task, split).

## 5. CLIP head fine-tune

The `train_clip.py` script:

- Freezes both towers by default (`--freeze-image --freeze-text` are on
  by default; pass `--no-freeze-image` / `--no-freeze-text` for
  full-tower fine-tuning).
- Adds an MLP classification head (`--head-hidden-dim 512` default).
- Applies training-time augmentation (`--augment`; disable with
  `--no-augment` for an ablation).
- LR warmup for 100 steps + cosine decay + gradient clipping = 1.0.
- Label smoothing 0.1 on both cross-entropy and BCE.
- `pos_weight` per sub-type for Task B BCE.
- Best-val checkpoint selection.
- Reproducible with `--seed`.

```bash
for SEED in 1 2 3; do
  uv run python scripts/train_clip.py \
    --model ViT-L-14-quickgelu \
    --loss-mode classification \
    --task binary \
    --epochs 10 \
    --batch-size 32 \
    --text-source provided \
    --seed $SEED \
    --device cuda
done
```

Change `--task binary` to `--task multilabel` for Task B.

Compute: ~5 minutes per epoch on frozen ViT-L-14 = ~50 minutes per
seed. Full-tower fine-tune (`--no-freeze-image --no-freeze-text`) is
~10x that.

## 6. VLM QLoRA fine-tune

The `train_vlm.py` script:

- LoRA `target_modules="all-linear"` and `task_type=None`.
- `--gradient-accumulation-steps 8` -> effective batch size 16.
- Forces `padding_side="right"` before every collation and asserts on
  the first batch.
- JSON schema targets for Task B and joint mode.
- `--sampler balanced` for rare-class oversampling.
- Reproducible with `--seed`.

```bash
# Qwen2-VL-7B, Task A
uv run python scripts/train_vlm.py \
    --model-id Qwen/Qwen2-VL-7B-Instruct \
    --epochs 3 \
    --batch-size 2 \
    --gradient-accumulation-steps 8 \
    --quantize 4bit \
    --task binary \
    --text-source provided \
    --seed 1

# Joint Task A + Task B (one adapter, both tasks)
uv run python scripts/train_vlm.py \
    --model-id Qwen/Qwen2-VL-7B-Instruct \
    --epochs 5 \
    --batch-size 2 \
    --gradient-accumulation-steps 16 \
    --quantize 4bit \
    --task joint \
    --sampler balanced \
    --text-source provided \
    --seed 1
```

Compute per seed:

- Qwen2-VL-2B: ~1h 10m per epoch = ~3.5h for 3 epochs.
- Qwen2-VL-7B: ~2h 30m per epoch = ~7.5h for 3 epochs.
- Joint at 5 epochs: 5/3 x the single-task time.

## 7. VLM inference variants

Standard multi-label inference (JSON schema):

```bash
uv run python scripts/benchmark_qwen2vl.py \
    --model-id Qwen/Qwen2-VL-7B-Instruct \
    --split test \
    --task multilabel \
    --quantize 4bit \
    --text-source provided \
    --lora-path results/models/lora_qwen2_vl_7b_instruct_multiclass_seed1
```

Per-category binary prompting: four yes/no questions per meme,
one per sub-type. Slower but avoids format drift and gives per-class
confidence:

```bash
uv run python scripts/benchmark_qwen2vl.py \
    --model-id Qwen/Qwen2-VL-7B-Instruct \
    --split test \
    --task per_category \
    --quantize 4bit \
    --text-source provided
```

Compute: standard inference ~6 min; per-category ~24 min.

## 8. Two-stage A -> B inference

Runs Task B only on memes predicted misogynous by Task A. Reuses two
existing benchmark result JSONs:

```bash
uv run python scripts/benchmark_two_stage.py \
    --task-a results/test/xgboost_test_xgboost_binary.json \
    --task-b results/test/qwen2vl_test_qwen2_vl_7b_instruct_multiclass_finetuned.json \
    --output results/test/two_stage_test_multiclass.json
```

Compute: ~30 seconds (pure JSON merging + metric recompute).

## 9. XAI regeneration

After the top-level XGBoost head (section 3) is retrained:

```bash
sbatch scripts/submit_89_xai.slurm
```

Produces `results/shap_modality_importance.csv` (SHAP) and
`results/concept_projection_similarity.csv` +
`results/concept_projection_global_presence.csv` (concept projection).

## 10. Consolidated report

Regenerates the paper-ready comparison table from every result JSON:

```bash
uv run python scripts/generate_consolidated_table.py
```

The MAMI-official Task B metric (`mami_score_b`) is the headline
column; older result JSONs that predate the metric show `N/A` and
should be regenerated.

## 11. Second dataset: HarMeme (Harm-C)

Binary task (harmful or harmless) on the official Harm-C split (3,013 train, 177 validation, 354 test memes).
Run every command from the repository root.

Download the annotations and images (about 590 MB):

```bash
uv run python -m scripts.revision.harmeme_eval download
```

Classical classifiers on frozen CLIP ViT-L-14 features. No API key is needed, and a GPU speeds up `extract`:

```bash
uv run python -m scripts.revision.harmeme_eval extract
uv run python -m scripts.revision.harmeme_eval classify
```

Zero-shot Qwen2-VL-7B (needs a CUDA GPU; 4-bit):

```bash
uv run python -m scripts.revision.harmeme_eval qwen --device cuda
```

On SCIAMA, `extract`, `classify`, `qwen` and `summarize` run together as job S6 (`bash sciama/submit.sh S6`, see
`sciama/JOBS.md`). Compute: ~7 min on one L40 GPU.

Zero-shot Gemini (needs `GEMINI_API_KEY` in `.env`; 531 API calls):

```bash
uv run python -m scripts.revision.harmeme_eval gemini
```

Compute: ~10 min with one call at a time (about 220 s for validation and 390 s for test).

Summary table and statistics (no model is run):

```bash
uv run python -m scripts.revision.harmeme_eval summarize
uv run python -m scripts.revision.harmeme_analysis
```

These write `metrics.json`, `table_rows.tex`, `summary.md` and `analysis.json` (confidence intervals, McNemar tests, the
single-word rule) to `results/revision/dis01_harmeme/`.

Prompt sensitivity and repeat runs. The first Gemini run is repeated to check that it reproduces, and a second prompt
based on the dataset paper's definition of harm is tried (both prompts are in `docs/PROMPTS.md`):

```bash
# Repeat the original Gemini run (new file names; existing files are never overwritten)
uv run python -m scripts.revision.harmeme_eval gemini --output-tag _rerun
uv run python -m scripts.revision.harmeme_eval compare --tag _rerun

# Definition prompt: validation first. Run it on test only if `compare` prints "clears validation bar: yes".
uv run python -m scripts.revision.harmeme_eval gemini --prompt-variant definition --output-tag _definition --splits validation
uv run python -m scripts.revision.harmeme_eval compare --tag _definition
uv run python -m scripts.revision.harmeme_eval gemini --prompt-variant definition --output-tag _definition --splits test

# Same prompt on Qwen2-VL-7B (CUDA GPU; job S6-qwen-def on SCIAMA)
uv run python -m scripts.revision.harmeme_eval qwen --device cuda --prompt-variant definition --output-tag _definition
uv run python -m scripts.revision.harmeme_eval compare --model qwen2_vl_7b_instruct --tag _definition
```

`summarize` reads only the untagged result files, so these extra runs never change the main table.

## Deprecated flags kept for backward compatibility

- `--use-ocr` on any script equals `--text-source ocr`.
- `--task singleclass` equals `--task binary`.
- `--task multiclass` equals `--task multilabel`.

Old SLURM scripts and notebooks keep working without edits.
