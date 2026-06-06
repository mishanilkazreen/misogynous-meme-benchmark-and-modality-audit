"""
Run markdownlint-cli on every tracked markdown file in the repo,
including files under `.kiro/` which `.markdownlintignore` excludes
by default. Fixes issues in place when run with --fix.

Usage:
    uv run python scripts/lint_markdown.py          # check
    uv run python scripts/lint_markdown.py --fix    # auto-fix
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent


def find_markdown_files() -> list[str]:
    """Return tracked markdown paths relative to the repo root."""
    out = subprocess.check_output(["git", "ls-files", "*.md"], cwd=ROOT, text=True)
    return sorted(line for line in out.splitlines() if line)


def pick_cli() -> list[str]:
    """Return the command prefix for markdownlint-cli."""
    if markdownlint := shutil.which("markdownlint"):
        return [markdownlint]
    if npx := shutil.which("npx"):
        return [npx, "--yes", "markdownlint-cli@0.42.0"]
    raise SystemExit(
        "markdownlint-cli not found. Install it with `brew install markdownlint-cli` "
        "or `npm install -g markdownlint-cli`."
    )


def main() -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fix", action="store_true", help="Auto-fix issues")
    args = parser.parse_args()

    files = find_markdown_files()
    if not files:
        print("No markdown files tracked.")
        return 0

    cmd = [*pick_cli(), "-c", ".markdownlint.json"]
    if args.fix:
        cmd.append("--fix")
    cmd.extend(files)

    print(" ".join(cmd))
    return subprocess.call(cmd, cwd=ROOT)


if __name__ == "__main__":
    sys.exit(main())
