#!/usr/bin/env python3
"""Linter to check for subjective, hyperbolic, or unquantified modifiers in LaTeX manuscripts."""

from pathlib import Path
import re
import sys

# List of subjective adjectives, adverbs, and hyperbolic fillers forbidden in academic writing
FORBIDDEN_WORDS = [
    "crucial",
    "crucially",
    "substantial",
    "substantially",
    "essentially",
    "essential",
    "great",
    "greatly",
    "remarkable",
    "remarkably",
    "massive",
    "massively",
    "severe",
    "severely",
    "critical",
    "critically",
    "dramatic",
    "dramatically",
    "vital",
    "paramount",
    "profound",
    "profoundly",
    "game-changing",
    "groundbreaking",
    "dominant",
]


def check_file(path: Path) -> list[str]:
    errors = []
    lines = path.read_text(encoding="utf-8").splitlines()
    in_verbatim = False

    for line_no, line in enumerate(lines, 1):
        stripped = line.strip()
        if stripped.startswith("%"):
            continue

        if r"\begin{verbatim}" in line or r"\begin{lstlisting}" in line:
            in_verbatim = True
            continue
        if r"\end{verbatim}" in line or r"\end{lstlisting}" in line:
            in_verbatim = False
            continue
        if in_verbatim:
            continue

        code_part = line.split("%")[0]
        # Ignore latex command macro names like \critical or \essential if any
        text_only = re.sub(r"\\[a-zA-Z]+", " ", code_part)

        for word in FORBIDDEN_WORDS:
            pattern = r"\b" + re.escape(word) + r"\b"
            for match in re.finditer(pattern, text_only, re.IGNORECASE):
                errors.append(
                    f"{path}:{line_no}: Subjective/hyperbolic word detected: '{match.group(0)}'. "
                    "Please replace with objective, technical, and quantitative phrasing."
                )

    return errors


def main() -> None:
    if len(sys.argv) < 2:
        tex_files = list(Path("submission").glob("*.tex"))
    else:
        tex_files = [Path(p) for p in sys.argv[1:] if p.endswith(".tex")]

    total_errors = []
    for f in tex_files:
        if f.exists():
            total_errors.extend(check_file(f))

    if total_errors:
        print(f"\nFound {len(total_errors)} subjective/hyperbolic wording issues:\n")
        for err in total_errors:
            print(f"  - {err}")
        print(
            "\nPlease remove subjective adjectives and adverbs to maintain objective academic tone.\n"
        )
        sys.exit(1)
    else:
        print("Subjective wording check passed (0 subjective modifiers found).")
        sys.exit(0)


if __name__ == "__main__":
    main()
