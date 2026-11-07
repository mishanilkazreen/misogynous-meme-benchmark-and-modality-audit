# SCIAMA jobs

Read [README.md](README.md) first. Each job below lists why it exists, what the prerequisite check requires, the
exact submission, the expected outputs and the verification command. Job scripts: [jobs/](jobs/) (all source
[jobs/_env.sh](jobs/_env.sh)). Checks: [sciama_checks.py](sciama_checks.py). Submission: [submit.sh](submit.sh).

Fine-tuning settings shared by every job (identical so seeds and models are comparable):
QLoRA NF4 4-bit, `LoraConfig(r=16, lora_alpha=32, target_modules="all-linear", lora_dropout=0.05)`
(`scripts/train_vlm.py`), 3 epochs, batch size 4, gradient accumulation 8, learning rate 2e-5, provided MAMI text,
best-of-epoch checkpoint on 200 validation memes.

Every prerequisite check (`sciama_checks.py prereqs`) also requires: code at `origin/main` with no modified tracked
code files, `results/sciama/logs/` present, the MAMI dataset loadable (1000 validation and 1000 test memes, sample
images readable), a CUDA GPU inside the job, and for training jobs at least 50 GB free on Lustre. It refuses to
queue a job whose outputs or adapter already exist (except S1-zeroshot, which intentionally replaces older files).

---

## S1. LLaVA-1.5-7B seeds and zero-shot re-evaluation (EXP-07, EXP-01)

Why: the committed LLaVA files are single-run and predate the per-split evaluation fix. Multiple seeds give a
seed-variance estimate. Qwen2-VL and CLIP already have seeds 1-3.

Check requires: `llava-hf/llava-1.5-7b-hf` in the Hugging Face cache; for zero-shot also both PaddleOCR NPZ files.

```bash
for T in singleclass multiclass; do for S in 1 2 3; do
  bash sciama/submit.sh S1-seed TASK=$T SEED=$S        # 6 jobs, up to 12 h each
done; done
bash sciama/submit.sh S1-zeroshot                        # 1 job, up to 6 h
```

Outputs: `results/{validation,test}/llava_<split>_llava_1_5_7b_hf[_multiclass]_finetuned_seed{1,2,3}.json`
(12 files); zero-shot `results/{validation,test}/llava_<split>_llava_1_5_7b_hf[_ocr_paddleocr][_multiclass].json`
(8 files, replacing older versions that stay in git history); adapters on Lustre
`results/models/lora_llava_1.5_7b_hf_{singleclass,multiclass}_seed{1,2,3}/`.

Verify (locally, after sync), one command per submission:

```bash
uv run python sciama/sciama_checks.py verify --job S1-seed --task singleclass --seed 1   # ... all 6
uv run python sciama/sciama_checks.py verify --job S1-zeroshot
```

---

## S2. OCR text on the fine-tuned Qwen2-VL adapters (EXP-05)

Why: extends the OCR ablation to the fine-tuned VLMs. Zero-shot OCR runs exist; fine-tuned adapters were only
evaluated on provided text.

Check requires: `Qwen/Qwen2-VL-{2B,7B}-Instruct` weights; adapters
`results/models/lora_qwen2_vl_{2b,7b}_instruct_{singleclass,multiclass}_seed{1,2,3}/` with the matching base
model; both PaddleOCR NPZ files (1000 unique ids, at most 20% empty transcripts).

```bash
bash sciama/submit.sh S2 MODEL=2b
bash sciama/submit.sh S2 MODEL=7b
```

Outputs: `results/{validation,test}/qwen2vl_<split>_qwen2_vl_{2b,7b}_instruct_ocr_paddleocr[_multiclass]_finetuned_seed{1,2,3}.json`
(12 per model; new names, nothing overwritten).

Verify: `uv run python sciama/sciama_checks.py verify --job S2 --model 2b` (then `--model 7b`).
CER/WER of the PaddleOCR text against the gold transcripts is local CPU work for EXP-05, not a SCIAMA job.

---

## S3. Rationales on the first 200 test memes (EXP-08)

Why: enlarges the 30-sample rationale set. N=200 is decided (about 10 min per model at the measured
3 s per meme). The first 200 test memes in dataset order are used, so nothing is hand-picked.

Check requires: the three model weights and the singleclass adapters used for the committed 30-sample rationales:
`results/models/lora_qwen2_vl_7b_instruct_singleclass/`, `lora_qwen2_vl_2b_instruct_singleclass/`,
`lora_llava_1.5_7b_hf_singleclass/` (each with the expected base model).

```bash
bash sciama/submit.sh S3 N=200
```

Outputs: `results/rationales/{qwen7b,qwen2b,llava7b}_test_n200.jsonl` (the 30-sample files are kept unchanged).

Verify: `uv run python sciama/sciama_checks.py verify --job S3 --n 200`. Then, locally:

```bash
uv run python scripts/evaluate_vlm_explanations.py --input results/rationales/qwen7b_test_n200.jsonl \
  --output results/rationales/metrics_qwen7b_test_n200.json          # repeat for qwen2b and llava7b
uv run python scripts/explain_with_gemini.py --split test --limit 200 \
  --output results/rationales/gemini_test_n200.jsonl                  # local, needs GEMINI_API_KEY
```

---

## S4. Template-disjoint fine-tuning (EXP-02) - blocked until L1 and C2

Why: measures template overlap across splits and evaluates on a template-disjoint split.

L1 (local, CPU): cluster meme templates across train, validation and test (perceptual hashing of the images and
clustering of CLIP image embeddings), report the overlap, and write `data/splits/template_disjoint/{train,validation,test}.tsv`
with the MAMI TSV columns plus `template_id`. No `template_id` may appear in both train and validation/test (the
check enforces this). Commit the split, the clustering script and the overlap report.

C2 (code, local, with unit tests): `utils/dataset.py` reads split TSVs from `MAMI_SPLIT_DIR` when set (images still
from the Kaggle cache); `utils/dataset.py`, `scripts/train_vlm.py` and `scripts/benchmark_qwen2vl.py` append
`_<MAMI_SPLIT_TAG>` to the adapter directory and to every output file name (after any `_seedN`), so random-split
results can never be overwritten. The check requires the text `MAMI_SPLIT_DIR`/`MAMI_SPLIT_TAG` in those files.

```bash
for M in 2b 7b; do for T in singleclass multiclass; do
  bash sciama/submit.sh S4 MODEL=$M TASK=$T SEED=1
done; done
```

Outputs: adapters `results/models/lora_qwen2_vl_{2b,7b}_instruct_<task>_seed1_tdisjoint/`; predictions
`results/{validation,test}/qwen2vl_<split>_qwen2_vl_{2b,7b}_instruct[_multiclass]_finetuned_seed1_tdisjoint.json`.

Verify: `uv run python sciama/sciama_checks.py verify --job S4 --model 2b --task singleclass --seed 1` (each job).
Note: verification compares ids with the random test split; update `_truth` in `sciama_checks.py` to read the
template-disjoint TSVs as part of C2 before verifying S4.

---

## S5. InternVL scaling study (EXP-03, P2) - code (C3) done, awaiting S5 execution

Why: studies scaling across several sizes and versions within one model family.

C3 (code, local, with unit tests) is in place: `scripts/benchmark_internvl.py` with the same flags and
naming as `scripts/benchmark_qwen2vl.py` (prefix `internvl_`, including `--lora-path` and `--seed`), and InternVL
prompt and collate support in `scripts/train_vlm.py`. The verified native-transformers Hugging Face ids (InternVL3
and InternVL3.5 `-hf` ports from OpenGVLab) are already added to `scripts/predownload_models.sh`; run it on the
login node to pre-cache the weights. The remaining step is submitting S5 on the GPU cluster.

```bash
bash sciama/submit.sh S5 MODEL_ID=<verified-hf-id> TASK=singleclass SEED=1
```

Outputs: `results/{validation,test}/internvl_<split>_<slug>[_multiclass].json` (zero-shot) and
`..._finetuned_seed1.json`; adapters on Lustre. Verify: `... verify --job S5 --model-id <id> --task ... --seed 1`.

---

## S6. HarMeme (Harm-C) second-dataset evaluation (DIS-01, P2)

Why: empirical validation on a second harmful-meme dataset (`Harm-C`, COVID-19 subset of HarMeme, 3,544 memes:
3,013 train / 177 validation / 354 test).

Prerequisites (login node, once before submitting):

```bash
uv run python -m scripts.revision.harmeme_eval download
bash scripts/predownload_models.sh
```

Submit:

```bash
bash sciama/submit.sh S6
```

(To submit from a checkout that is not at `origin/main`, prefix with `SCIAMA_ALLOW_BRANCH=1`.)

What S6 runs (`sciama/jobs/dis01_harmeme.slurm`):

1. Extracts `CLIP ViT-L/14` (`quickgelu`) 768-D image + text embeddings across `train`, `validation`, and `test`
   splits (`results/embeddings/harmeme_{train,validation,test}_vit_l_14_quickgelu.npz`).
2. Evaluates `SVM-RBF` and `XGBoost` on concatenated `1,536`-D CLIP features, writing per-sample predictions
   (`results/{validation,test}/harmeme_<split>_clip_{svm_rbf,xgboost}.json`).
3. Evaluates zero-shot `Qwen/Qwen2-VL-7B-Instruct` on `validation` and `test`
   (`results/{validation,test}/harmeme_<split>_qwen2_vl_7b_instruct.json`).

After syncing back to a machine with internet access (or on the SCIAMA login node if `GEMINI_API_KEY` is set in `.env`),
run zero-shot Gemini and generate the LaTeX summary table rows:

```bash
uv run python -m scripts.revision.harmeme_eval gemini
uv run python -m scripts.revision.harmeme_eval summarize
```

Verify: `uv run python sciama/sciama_checks.py verify --job S6`.

### S6-qwen-raw. Qwen2-VL-7B re-run that keeps the raw replies

Why: S6 stored only parsed predictions, and Qwen2-VL-7B answered "no" for 343 of 354 test memes. This job re-runs
the same zero-shot evaluation and stores each raw reply (`raw_response`) so the parsing can be audited. It writes
new files, so the S6 outputs are kept.

```bash
bash sciama/submit.sh S6-qwen-raw
```

Outputs: `results/{validation,test}/harmeme_<split>_qwen2_vl_7b_instruct_raw_responses.json`.
Verify: `uv run python sciama/sciama_checks.py verify --job S6-qwen-raw`.

### S6-qwen-def. Qwen2-VL-7B with the dataset paper's definition of harm

Why: prompt sensitivity check. Run only if the zero-shot Gemini run clears its validation bar. It uses
`--prompt-variant definition` and writes new files, so the S6 outputs are kept.

```bash
bash sciama/submit.sh S6-qwen-def
```

Outputs: `results/{validation,test}/harmeme_<split>_qwen2_vl_7b_instruct_definition.json`.
Verify: `uv run python sciama/sciama_checks.py verify --job S6-qwen-def`.

---

## After every job batch

1. `bash sciama/make_manifest.sh` on the login node; update the ledger states.
2. Sync, `git diff --stat`, run `verify` for every submission (README section 6).
3. Commit results, logs and verification reports on `main`; `bash scripts/preflight.sh`; push.
