# SCIAMA runbook

Everything an agent needs to run the project's GPU jobs on SCIAMA, bring every raw output back, verify it and
keep it. Per-job instructions: [JOBS.md](JOBS.md).

## 0. Rules for the agent (non-negotiable)

1. Connect with `ssh sciama` (alias in `~/.ssh/config`, needs the University VPN). If it fails, stop and report.
2. Queue jobs ONLY with `bash sciama/submit.sh ...`. It runs `sciama/sciama_checks.py prereqs` and refuses to call
   `sbatch` unless every prerequisite passes. Never call `sbatch` directly and never edit a check to make it pass.
   Every job re-runs the same check on the GPU node (`--in-job`) and stops if anything is missing.
3. A `PREREQ FAIL` is fixed at its cause (download the data, copy the file, implement the code change), never
   bypassed. If the cause cannot be fixed, stop and report which check failed and why.
4. Never report, summarise or write a number that did not come from a file produced by a job and checked by
   `sciama/sciama_checks.py verify`. No estimates, no placeholders, no "expected" values. If a job failed or its
   output did not verify, say so plainly.
5. Never delete or overwrite anything under `results/` on Lustre. Never use `rsync --delete`, `rm -r results`,
   `git clean`, `git checkout -- results` or `git reset --hard`. Partial outputs of a failed job are renamed to
   `<name>.failed_<JOBID>` (kept, not deleted) before resubmitting.
6. Never commit on SCIAMA. Results come back with the sync script, are verified, then committed locally on `main`.
7. Compute nodes have no internet: data and weights are downloaded on the login node first.
8. Wait at least 60 s between status checks. Do not poll in a loop.

## 1. Order of work

Submit S1-S3 first; they run while local work proceeds. S4 and S5 are blocked by local work and code changes, and
their prerequisite checks fail until that work exists.

| Order | Job (JOBS.md) | Task | Ready? |
| --- | --- | --- | --- |
| 1 | S1 LLaVA-1.5-7B seeds 1-3 + zero-shot re-evaluation | EXP-07, EXP-01 | Yes |
| 1 | S2 OCR text on the fine-tuned Qwen2-VL adapters | EXP-05 | Yes |
| 1 | S3 Rationales on the first 200 test memes (N=200, decided) | EXP-08 | Yes |
| 2 | S4 Template-disjoint fine-tuning | EXP-02 | After local split L1 and code change C2 |
| 3 | S5 InternVL scaling (P2) | EXP-03 | Code change C3 done; awaiting S5 GPU run |
| 3 | S6 HarMeme (`Harm-C`) second-dataset evaluation (P2) | DIS-01 | After `uv run python -m scripts.revision.harmeme_eval download` on login node |

EXP-01 needs no new GPU work for Qwen2-VL and CLIP: their seeds 1-3 from the fixed evaluation exist.

## 2. Connect and prepare (login node, every session)

```bash
ssh sciama
module purge
module load system/sciama services/slurm anaconda3 uv git
export PROJECT_ROOT=<repo checkout on Lustre>   # read by submit.sh and every job (sbatch --export=ALL)
cd "$PROJECT_ROOT"
git status            # modified tracked CODE files? stop and report. Results changes are expected.
git pull origin main  # submit.sh refuses to run unless HEAD == origin/main
mkdir -p results/sciama/logs results/sciama/manifests runs/slurm_logs
```

## 3. Dataset (login node, once)

The MAMI dataset comes from Kaggle through `kagglehub` (`utils/dataset.py`), cached on Lustre. The prerequisite
check loads every split, checks sizes (validation and test: 1000 memes) and opens sample images.

```bash
export KAGGLEHUB_CACHE="$HOME/.cache/kagglehub"
conda activate "${CONDA_ENV:-vlm-moderation}"
ls "$KAGGLEHUB_CACHE/datasets/chukwuebukaanulunko"   # missing? then download:
# put KAGGLE_USERNAME and KAGGLE_KEY in .env on Lustre (never commit .env), then
uv run python scripts/download_dataset.py
uv run python scripts/verify_dataset.py --fast
```

PaddleOCR transcripts live in `results/embeddings/{validation,test}_vit_l_14_quickgelu_ocr_paddleocr.npz` (S1 and
S2 need them; the check verifies 1000 unique ids each). If missing on Lustre, copy the local ones up:

```powershell
scp results/embeddings/validation_vit_l_14_quickgelu_ocr_paddleocr.npz "sciama:$env:SCIAMA_REMOTE_ROOT/results/embeddings/"
scp results/embeddings/test_vit_l_14_quickgelu_ocr_paddleocr.npz "sciama:$env:SCIAMA_REMOTE_ROOT/results/embeddings/"
```

## 4. Model weights and adapters (login node)

```bash
export HF_HOME="$HOME/.cache/huggingface"
bash scripts/predownload_models.sh    # Qwen2-VL-2B/7B, LLaVA-1.5-7B, CLIP ViT-B-32/L-14
ls results/models/                    # adapters needed by S2 and S3 (JOBS.md); the check verifies each one
bash sciama/make_manifest.sh          # size + SHA-256 of every adapter/embedding, before any job
```

The check confirms each model is fully in the Hugging Face cache and each adapter has its config, its weights and
the expected `base_model_name_or_path`.

## 5. Submit and monitor

```bash
bash sciama/submit.sh <JOB> [KEY=VALUE ...]   # see JOBS.md; prints PREREQS OK and the job id, or NOT SUBMITTED
squeue -u "$USER"
sacct -j <ID> --format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS
tail -n 50 results/sciama/logs/<job-name>_<ID>.log
tail -n 50 results/sciama/logs/<job-name>_<ID>.err
```

`submit.sh` appends the submission to `results/sciama/job_ledger.tsv`. When the job ends, edit that line's status to
the `sacct` state (`COMPLETED 0:0`, `FAILED`, `TIMEOUT`, `OUT_OF_MEMORY`, `CANCELLED`).

| Log pattern | Meaning | Action |
| --- | --- | --- |
| `PREREQ FAIL` / `FATAL:` | A prerequisite is missing | Fix the cause, resubmit with `submit.sh` |
| `Traceback` / `Error` in `.err` | Python failure; the job exits non-zero | Read it; rename partial outputs; never use them |
| `CUDA out of memory` | Batch too large | Halve `--batch-size` in the job, note it in the ledger |
| `refusals=` above 0.05 | Refused or unparsable outputs | Report the rate; do not hide it |
| `=== DONE` | Every step finished | Required before syncing the outputs |

## 6. Bring results back, verify, keep (local machine)

Set `SCIAMA_REMOTE_ROOT` to the repository checkout on SCIAMA. Optionally set `SCIAMA_BACKUP_DIR` to an
off-cluster directory for models and embeddings; without it they go to `.sciama_backup/results/`.

```powershell
powershell -ExecutionPolicy Bypass -File sciama/sync_from_sciama.ps1   # macOS/Linux: bash sciama/sync_from_sciama.sh
git status
git diff --stat       # a modified tracked result = an overwrite: expected only for S1-zeroshot; otherwise stop
uv run python sciama/sciama_checks.py verify --job <JOB> [same parameters as the submission]
```

The sync never loses a local file: before a pulled file replaces a local file with different content, the local
copy is moved to `results/sciama/archive/sync_<UTC time>/<same path>`. Commit that folder with the sync. Models
and embeddings in the backup directory are only added, never replaced.

`verify` checks every expected output: it exists; split and record count are right; image ids equal the real split;
ground-truth labels equal the dataset; the reported accuracy equals the value recomputed from the per-sample
predictions; rationale files have N error-free, non-mock records for the first N test memes. It writes
`results/sciama/verification/<job>_<params>_<time>.json` with `"passed"`. A failed report is committed too.

| Output | Location | In git? |
| --- | --- | --- |
| Per-sample predictions and metrics (JSON) | `results/test/`, `results/validation/` | Yes |
| Rationale generations and metrics | `results/rationales/` | Yes |
| Logs, ledger, manifests, verification reports | `results/sciama/`, `runs/slurm_logs/` | Yes (forced by `.gitignore`) |
| LoRA adapters, CLIP checkpoints, embeddings | Lustre `results/models/`, `results/embeddings/`, `.sciama_backup/` | No (size); SHA-256 in manifests |

Do not add ignore rules for these locations. Files over 100 MB (GitHub limit) are gzipped, never deleted.

Commit on `main`: `git add results/test results/validation results/rationales results/sciama runs/slurm_logs`,
`git commit -m "results(exp07): LLaVA seeds 1-3, Task A and B"`, `bash scripts/preflight.sh`, `git push`.
