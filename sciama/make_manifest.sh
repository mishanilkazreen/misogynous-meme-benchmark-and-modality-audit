#!/bin/bash
# Record size and SHA-256 of every large SCIAMA artefact that cannot live in git (LoRA adapters,
# CLIP checkpoints, embeddings, pickled classifiers). Run on the SCIAMA LOGIN node after each batch
# of jobs; the manifest is small and IS committed, so the artefacts are always traceable.
# Usage: bash sciama/make_manifest.sh
set -euo pipefail
cd "${PROJECT_ROOT:?set PROJECT_ROOT to the repo checkout}"
mkdir -p results/sciama/manifests
OUT="results/sciama/manifests/artefacts_$(date +%Y%m%d_%H%M%S).tsv"
{
  printf 'sha256\tbytes\tmodified_utc\tpath\n'
  find results/models results/embeddings -type f 2>/dev/null | sort | while read -r f; do
    printf '%s\t%s\t%s\t%s\n' "$(sha256sum "$f" | cut -d' ' -f1)" "$(stat -c %s "$f")" \
      "$(date -u -r "$f" +%Y-%m-%dT%H:%M:%SZ)" "$f"
  done
} > "$OUT"
echo "Wrote $OUT ($(($(wc -l < "$OUT") - 1)) files)"
