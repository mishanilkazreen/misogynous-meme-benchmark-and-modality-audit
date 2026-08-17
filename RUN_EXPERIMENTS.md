# Guide to Running the MAMI Experiments

- All Phase A–E fixes are applied. Prior single-seed runs are treated
  as exploratory and are NOT reportable.
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

## 1. Extract embeddings (Phase 0 of the rerun plan)

The extract step now supports three text sources. The paper reports the
`provided` variant as the primary row and the `paddleocr` / `combined`
variants as ablations.

```bash
# Provided text (MAMI's Text Transcription column, manually verified)
uv run python scripts/extract_embeddings.py \
    --split train,validation,test \
    --model ViT-L-14-quickgelu \
    --text-source provided

# PaddleOCR-only (kept as an ablation for the "OCR robustness" story)
uv run python scripts/extract_embeddings.py \
    --split train,validation,test \
    --model ViT-L-14-quickgelu \
    --text-source ocr \
    --ocr-engine paddleocr

# Union of both (best signal in practice; §7.8)
uv run python scripts/extract_embeddings.py \
    --split train,validation,test \
    --model ViT-L-14-quickgelu \
    --text-source combined \
    --ocr-engine paddleocr
```

Filenames follow the convention
`{split}_{model}[_ocr_paddleocr|_combined_paddleocr].npz`. Runs at
different text sources coexist in `results/embeddings/`.

Higher-resolution CLIP variant (§7.6): replace `ViT-L-14-quickgelu`
with `ViT-L-14-336-quickgelu` in any of the above.

Compute: ~30 minutes per (model, text_source) triple on SCIAMA.

## 2. Tabular sweep (Phase 1.1 and 2.1)

Trains 20+ classical ML models on top of CLIP + text embeddings. The
sweep now applies `class_weight="balanced"` (§7.3), wraps non-tree
models in a `StandardScaler` pipeline (§7.4), and reports the MAMI
official Task B metric (§1.4).

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
thresholds on validation and picks the one that maximises macro F1
(§2.7). `--calibrate-isotonic` is an optional flag that wraps the fit
in `CalibratedClassifierCV` for downstream ensembling (§7.9); it has
no meaningful effect on standalone accuracy.

Compute: ~15 seconds per task on CPU.

## 4. CLIP zero-shot benchmark (Phase 1.3, 2.4)

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
class (§7.1). `--tta` enables horizontal-flip test-time augmentation
(§7.2). Both are inference-only wins.

Compute: ~1 minute per (model, task, split).

## 5. CLIP head fine-tune (Phase 1.4, 2.2)

The refactored `train_clip.py`:

- Freezes both towers by default (`--freeze-image --freeze-text` are on
  by default; pass `--no-freeze-image` / `--no-freeze-text` for
  full-tower fine-tuning per §2.1 Recipe B).
- Adds an MLP classification head (`--head-hidden-dim 512` default).
- Applies training-time augmentation (`--augment`; disable with
  `--no-augment` for an ablation).
- LR warmup for 100 steps + cosine decay + gradient clipping = 1.0.
- Label smoothing 0.1 on both cross-entropy and BCE.
- `pos_weight` per sub-type for Task B BCE (§1.2).
- Best-val checkpoint selection (§2.4).
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

## 6. VLM QLoRA fine-tune (Phase 1.7, 2.3)

The refactored `train_vlm.py`:

- LoRA `target_modules="all-linear"` and `task_type=None` (§2.6).
- `--gradient-accumulation-steps 8` -> effective batch size 16 (§2.5).
- Forces `padding_side="right"` before every collation and asserts on
  the first batch (§1.1).
- JSON schema targets for Task B and joint mode (§6.1, §6.3).
- `--sampler balanced` for rare-class oversampling (§6.4).
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

# Joint Task A + Task B (one adapter, both tasks; §6.3)
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

## 7. VLM inference variants (Phase 1.6, 2b.4)

Standard multi-label inference (JSON schema, §6.1):

```bash
uv run python scripts/benchmark_qwen2vl.py \
    --model-id Qwen/Qwen2-VL-7B-Instruct \
    --split test \
    --task multilabel \
    --quantize 4bit \
    --text-source provided \
    --lora-path results/models/lora_qwen2_vl_7b_instruct_multiclass_seed1
```

Per-category binary prompting (§6.5): four yes/no questions per meme,
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

## 8. Two-stage A -> B inference (Phase 2b.3)

Runs Task B only on memes predicted misogynous by Task A. Reuses two
existing benchmark result JSONs (§6.2):

```bash
uv run python scripts/benchmark_two_stage.py \
    --task-a results/test/xgboost_test_xgboost_binary.json \
    --task-b results/test/qwen2vl_test_qwen2_vl_7b_instruct_multiclass_finetuned.json \
    --output results/test/two_stage_test_multiclass.json
```

Compute: ~30 seconds (pure JSON merging + metric recompute).

## 9. XAI regeneration (Phase 3)

After the top-level XGBoost head (§3) is retrained:

```bash
sbatch scripts/submit_89_xai.slurm
```

Produces `results/shap_modality_importance.csv` (§SHAP) and
`results/concept_projection_similarity.csv` +
`results/concept_projection_global_presence.csv` (§concept projection).
Note the file rename from the old `cav_*.csv` scheme (§5.1).

## 10. Consolidated report

Regenerates the paper-ready comparison table from every result JSON:

```bash
uv run python scripts/generate_consolidated_table.py
```

The MAMI-official Task B metric (`mami_score_b`) is the headline
column; older result JSONs that predate the metric show `N/A` and
should be regenerated. See `results/comparison_report.md`.

## Deprecated flags kept for backward compatibility

- `--use-ocr` on any script equals `--text-source ocr`.
- `--task singleclass` equals `--task binary`.
- `--task multiclass` equals `--task multilabel`.

Old SLURM scripts and notebooks keep working without edits.

## Compute-budget summary for a full paper-quality rerun

For a one-week wall-clock plan hitting Task A + Task B on all reported
systems with three seeds where appropriate, budget ~63 GPU-hours and
~15 CPU-hours on SCIAMA.
