#!/bin/bash
# Pull every raw SCIAMA output that belongs in git into the local clone (macOS/Linux).
# Copies only; never deletes anything on either side. Run from the repository root.
# Windows equivalent: sciama/sync_from_sciama.ps1
#
# Nothing is lost on overwrite: if a pulled file would replace a local file with different
# content, the local version is first moved to results/sciama/archive/sync_<UTC time>/<same path>
# (tracked in git). Large artefacts (models, embeddings) are only added, never replaced.
set -euo pipefail
REMOTE="sciama:${SCIAMA_REMOTE_ROOT:?set SCIAMA_REMOTE_ROOT to the repo checkout on SCIAMA}"
RSYNC_REMOTE=/opt/apps/pkgs/rsync/3.4.1/bin/rsync
STAMP="sync_$(date -u +%Y%m%dT%H%M%SZ)"
ARCHIVE="$PWD/results/sciama/archive/$STAMP"
for d in results/test results/validation results/rationales results/revision results/sciama runs/slurm_logs; do
  mkdir -p "$d"
  # No --delete: local files are never removed. --checksum: compare content, not timestamps.
  # --backup-dir: a local file that differs is moved into the archive before being replaced.
  rsync -av --checksum --exclude='/archive/' --backup --backup-dir="$ARCHIVE/$d" \
    --rsync-path="$RSYNC_REMOTE" "$REMOTE/$d/" "$d/"
done
if [ -d "$ARCHIVE" ]; then
  echo "Archived replaced local files under $ARCHIVE (commit them with the sync)."
fi
# Large artefacts are NOT copied into git; back them up to $SCIAMA_BACKUP_DIR if set, else locally.
BACKUP_DIR="${SCIAMA_BACKUP_DIR:-}"
if [ -n "$BACKUP_DIR" ] && [ -d "$BACKUP_DIR" ]; then
  BACKUP_ROOT="$BACKUP_DIR"
else
  BACKUP_ROOT="$PWD/.sciama_backup/results"
fi
mkdir -p "$BACKUP_ROOT/models" "$BACKUP_ROOT/embeddings"
for d in models embeddings; do
  # --ignore-existing: copy new artefacts only; an existing backup copy is never replaced.
  # (Jobs refuse to overwrite an existing adapter, so a same-named file is the same artefact.)
  rsync -av --ignore-existing --rsync-path="$RSYNC_REMOTE" "$REMOTE/results/$d/" "$BACKUP_ROOT/$d/"
done
echo "Sync done. Review with: git status && git diff --stat"
