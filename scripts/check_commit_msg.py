"""Commit message format checker.

Enforces conventional commit format:
  type(scope): description

Valid types: feat, fix, docs, style, refactor, test, chore, ci, perf, build, revert
First line must be 72 characters or fewer.
"""

import re
import sys

VALID_TYPES = [
    "feat",
    "fix",
    "docs",
    "style",
    "refactor",
    "test",
    "chore",
    "ci",
    "perf",
    "build",
    "revert",
]

PATTERN = re.compile(
    r"^(" + "|".join(VALID_TYPES) + r")(\(.+\))?!?:\s.{1,72}$"
)


def main() -> int:
    """Validate commit message format."""
    if len(sys.argv) < 2:
        print("Usage: check_commit_msg.py <commit-msg-file>")
        return 1

    with open(sys.argv[1], encoding="utf-8") as f:
        message = f.read().strip()

    first_line = message.split("\n")[0]

    if PATTERN.match(first_line):
        return 0

    types_str = "|".join(VALID_TYPES)
    print(f"Bad commit message format.")
    print(f"  Got:      {first_line}")
    print(f"  Expected: type(scope): description (max 72 chars)")
    print(f"  Types:    {types_str}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
