# Experimental Results Directory Structure

This directory contains experimental results, benchmark predictions, evaluation metrics, and provenance
records for the reported experiments.

## Data Provenance and Immutability Standards

All result files in this directory are authoritative empirical artifacts. In accordance with the project's
provenance policy:

- Never alter model predictions, explanations, or metric values after execution.
- Automated code formatters, whitespace trimmers, and end-of-file fixers are explicitly excluded from `results/`
  in `.pre-commit-config.yaml`.
- Spell checking via cspell is explicitly ignored across `results/` in `cspell.json`.
- Every reported figure and table number must trace directly to a committed artifact in this directory.

## Directory Taxonomy

| Directory | Scope and Contents |
|-----------|--------------------|
| `results/test/` | Benchmark metrics and per-sample predictions on the held-out MAMI test set (1,000 samples). Used for final reporting. |
| `results/validation/` | Benchmark metrics and per-sample predictions on the MAMI validation set (1,000 samples). Used for tuning and threshold calibration. |
| `results/rationales/` | Natural language explanations (`.jsonl`) and quantitative evaluation metrics (`.json`) for explainability analysis. |
| `results/sciama/` | SCIAMA HPC execution logs, job ledgers, manifests, verification reports, and timestamped synchronization snapshots (`sync_YYYYMMDD/`). |
| `results/revision/` | Dedicated follow-up experiment suites (EXP-01 through EXP-07, ANA-01, DIS-01). |
| `results/figures/` | Publication plots and visual comparisons. |
| `results/models/` | Catalog of fine-tuned model checkpoints and LoRA adapters. Large binary files (`.pth`, `.pkl`, `.safetensors`) are not committed. |
| `results/embeddings/` | Extracted feature representations (`.npz`). Regenerated with `scripts/extract_embeddings.py`; not committed. |

## Structural Invariance Policy

The subdirectories `results/test/` and `results/validation/` must remain directly under `results/`.
They must not be nested into deeper subfolders because active automation pipelines, verification tools,
and test suites depend on their canonical paths:

- Verification harness: `sciama/sciama_checks.py` checks `RESULTS / split / f"{name}.json"`.
- HPC synchronization: `sciama/sync_from_sciama.sh` and `sciama/sync_from_sciama.ps1` route outputs by split.
- Table generation: `scripts/generate_consolidated_table.py` and `scripts/recompute_paper_metrics.py`.
- Analysis and visualization: `scripts/plot_ocr_impact.py` and the evaluation scripts in `scripts/revision/`.
- Unit and integration tests: `tests/unit/test_tabular_mami_metrics.py` and `tests/integration/test_consolidated_table.py`.
