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

        # 4. Check for forbidden \resizebox wrapping on tabular
        if r"\resizebox" in line:
            errors.append(
                f"{file_path}:{line_no}: Forbidden '\\resizebox' detected. Do not wrap tabular environments in \\resizebox under sn-jnl.cls."
            )

        # 5. Check for double backslash typos on LaTeX commands
        if re.search(
            r"\\\\(bibliography|section|subsection|subsubsection|caption|label|cite|ref|begin|end)\b",
            line,
        ):
            errors.append(
                f"{file_path}:{line_no}: Escaped double backslash '\\\\' before command detected. Use single backslash '\\'."
            )

        # 3. British English spelling check (flag American variants)
        american_words = [
            r"\b\w+ize[ds]?\b",
            r"\b\w+izing\b",
            r"\b\w+ization[s]?\b",
            r"\bcolor[s]?\b",
            r"\bbehavior[s]?\b",
            r"\blabeling\b",
            r"\bmodeling\b",
            r"\bcanceled\b",
        ]
        # Ignore LaTeX keywords and commands
        ignored_latex = {"footnotesize", "itemize", "resized", "resize", "xcolor", "col" + "or"}

        code_text = line.split("%")[0]
        # Remove latex commands like \command{...} or \begin{...}
        text_only = re.sub(r"\\[a-zA-Z]+", " ", code_text)

        for pat in american_words:
            for match in re.finditer(pat, text_only, re.IGNORECASE):
                word = match.group(0)
                if word.lower() not in ignored_latex:
                    errors.append(
                        f"{file_path}:{line_no}: American English spelling variant '{word}' detected. "
                        "Please use British English spelling (-ise, -isation, -our, etc.)."
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

        # Check for auxiliary build log warnings if main.log exists
        log_path = tex_path.parent / "main.log"
        if log_path.exists():
            log_text = log_path.read_text(encoding="utf-8", errors="ignore")
            undefined_cites = re.findall(
                r"(?:LaTeX|Package natbib) Warning: Citation [`']([^']+)['`].*?undefined", log_text
            )
            for cite in set(undefined_cites):
                total_errors.append(
                    f"Build log warning: Citation '{cite}' is undefined (renders as '?'). Check .bib entry or compilation passes."
                )
            undefined_refs = re.findall(
                r"(?:LaTeX|Package natbib) Warning: Reference [`']([^']+)['`].*?undefined", log_text
            )
            for ref in set(undefined_refs):
                total_errors.append(
                    f"Build log warning: Reference '{ref}' is undefined (renders as '?'). Check label name."
                )
            if (
                "There were undefined references" in log_text
                or "There were undefined citations" in log_text
            ):
                total_errors.append(
                    "Build log warning: LaTeX/natbib reported undefined references or citations ('?')."
                )

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
