#!/usr/bin/env python3
"""
Strip abstract fields from a BibTeX file.
Usage: python3 strip_abstracts.py <input.bib> [output.bib]
If output is omitted, overwrites the input file.
"""

import re
import sys

ABSTRACT_PATTERN = re.compile(
    r"\s*abstract\s*=\s*\{.*?\},?\s*\n",
    re.DOTALL | re.IGNORECASE,
)


def strip_abstracts(text: str) -> str:
    """Remove all abstract = {...} fields from BibTeX text."""
    return ABSTRACT_PATTERN.sub("", text)


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: strip_abstracts.py <input.bib> [output.bib]", file=sys.stderr)
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else input_path

    with open(input_path, encoding="utf-8") as fh:
        original = fh.read()

    cleaned = strip_abstracts(original)
    removed = original.count("abstract   =") + original.count("abstract =")

    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write(cleaned)

    print(f"Stripped {removed} abstract field(s) from {input_path} → {output_path}")


if __name__ == "__main__":
    main()
