#!/usr/bin/env bash
#
# Preflight: run the EXACT checks GitHub CI runs, locally, before a push.
#
# CI (.github/workflows/ci.yml) has two gates:
#   1. pre-commit job  -> `pre-commit run --all-files`
#   2. lint-and-test   -> ruff check, ruff format --check, mypy, pytest
#
# `pre-commit run --all-files` already covers ruff, ruff-format, mypy, markdown,
# cspell and pylint, so running it plus pytest reproduces CI exactly. This
# script is wired into the git pre-push hook (see .pre-commit-config.yaml), so a
# push is aborted unless every check passes. It can also be run by hand:
#
#     bash scripts/preflight.sh
#
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

echo "==> [1/2] pre-commit (all hooks, all files) - mirrors the CI pre-commit job"
# --hook-stage pre-commit keeps this to the lint/format/test-config hooks and
# avoids recursing into this very pre-push hook.
uv run pre-commit run --all-files --hook-stage pre-commit

echo ""
echo "==> [2/2] pytest - mirrors the CI lint-and-test job"
uv run pytest tests/ -q

echo ""
echo "==> All CI-equivalent checks passed. Safe to push."
