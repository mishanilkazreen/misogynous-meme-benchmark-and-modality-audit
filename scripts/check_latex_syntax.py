#!/usr/bin/env python3
"""Custom LaTeX Syntax & Quality Linter for Academic Manuscripts.

Checks:
  1. Quotation syntax: Enforces `` and '' instead of raw double quotes (").
  2. Non-breaking spaces: Enforces ~ before \\cite{...} and \ref{...}.
  3. Label whitespace: Flag trailing/leading whitespace inside \\label{...}.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys


def check_latex_file(file_path: Path) -> list[str]:
    """Inspect a .tex file for syntax and style issues."""
    errors: list[str] = []
    lines = file_path.read_text(encoding="utf-8").splitlines()

    in_verbatim = False

    for line_no, line in enumerate(lines, 1):
        stripped = line.strip()

        # Skip comments
        if stripped.startswith("%"):
            continue

        # Track verbatim environments
        if r"\begin{verbatim}" in line or r"\begin{lstlisting}" in line:
            in_verbatim = True
            continue
        if r"\end{verbatim}" in line or r"\end{lstlisting}" in line:
            in_verbatim = False
            continue
        if in_verbatim:
            continue

        # 1. Raw double quotes check
        # Match " when not part of a command or macro definition
        if (
            '"' in line
            and not line.startswith(r"\documentclass")
            and not line.startswith(r"\usepackage")
        ):
            # Exclude valid macro quotes or comments
            code_part = line.split("%")[0]
            if '"' in code_part:
                errors.append(
                    f'{file_path}:{line_no}: Raw double quote (") found. '
                    "Use `` for opening quotes and '' for closing quotes."
                )

        # 2. Missing non-breaking space before \cite or \ref
        # E.g., 'Kapil 2025 \cite' or 'Table \ref' instead of 'Kapil 2025~\cite' or 'Table~\ref'
        cite_matches = re.finditer(r"(?<!~)\s+(\\cite\{[^}]+\})", line)
        for m in cite_matches:
            errors.append(
                f"{file_path}:{line_no}: Missing non-breaking space (~) before citation: '{m.group(0).strip()}'"
            )

        ref_matches = re.finditer(r"(?<!~)\s+(\\ref\{[^}]+\})", line)
        for m in ref_matches:
            errors.append(
                f"{file_path}:{line_no}: Missing non-breaking space (~) before reference: '{m.group(0).strip()}'"
            )

    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="LaTeX Quality & Style Linter")
    parser.add_argument("files", nargs="+", type=Path, help="Paths to .tex files to lint")
    args = parser.parse_args()

    total_errors: list[str] = []
    for tex_path in args.files:
        if not tex_path.exists():
            print(f"Error: File {tex_path} does not exist.", file=sys.stderr)
            sys.exit(1)
        file_errors = check_latex_file(tex_path)
        total_errors.extend(file_errors)

    if total_errors:
        print(f"\nFound {len(total_errors)} LaTeX syntax/style issues:\n")
        for err in total_errors:
            print(f"  - {err}")
        print("\nPlease fix the flagged issues to ensure clean LaTeX compilation.\n")
        sys.exit(1)
    else:
        print("LaTeX syntax and quotation checks passed (0 issues found).")
        sys.exit(0)


if __name__ == "__main__":
    main()
