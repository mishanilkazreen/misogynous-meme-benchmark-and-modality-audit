"""
Check which paper-plan tasks have been implemented.

Runs the task-marker test files under tests/tasks/. Each test file
corresponds to one top-level task in .kiro/specs/vlm-content-moderation/
tasks.md and is expected to fail with NotImplementedError (or be marked
xfail) until the task is complete. A task is considered "done" when its
test file passes.

Usage:
    uv run python scripts/check_tasks.py
    uv run python scripts/check_tasks.py --task 3   # single task
"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

TASK_DIR = Path(__file__).resolve().parent.parent / "tests" / "tasks"


def run_task(task_id: str | None) -> int:
    """Run the task marker test(s) and return the exit code."""
    if task_id is None:
        targets = [str(TASK_DIR)]
    else:
        match = sorted(TASK_DIR.glob(f"test_task_{task_id.zfill(2)}_*.py"))
        if not match:
            print(f"No task test file found for task {task_id}", file=sys.stderr)
            return 2
        targets = [str(p) for p in match]
    cmd = ["uv", "run", "pytest", "-v", "--no-header", *targets]
    print(" ".join(cmd))
    return subprocess.call(cmd)


def main() -> None:
    """Entry point for the task tracker."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", help="Run only the given task id (e.g. 3)")
    args = parser.parse_args()
    sys.exit(run_task(args.task))


if __name__ == "__main__":
    main()
