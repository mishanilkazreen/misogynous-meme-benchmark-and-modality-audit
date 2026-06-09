# Contributing

Short, opinionated workflow. Follow this literally the first time.

## 1. Install tools (once per machine, macOS)

```bash
# CLI
brew install uv gh git node
brew install markdownlint-cli

# VS Code extensions
code --install-extension charliermarsh.ruff
code --install-extension ms-python.python
code --install-extension ms-python.mypy-type-checker
code --install-extension davidanson.vscode-markdownlint
code --install-extension tamasfe.even-better-toml
```

## 2. Set up the repo (once per clone)

```bash
git clone https://github.com/mishanilkazreen/content-moderation.git
cd content-moderation
uv venv --python 3.10
uv sync --dev
uv run pre-commit install
```

> **Note:** torch is managed by `pyproject.toml` (cu128 index).
> `uv sync --dev` handles everything — no separate torch install needed.

## 3. Pick a task and switch to its branch

The branches are already pushed. Don't create new ones, don't push to
`main`. Open tasks are in
[GitHub issues](https://github.com/mishanilkazreen/content-moderation/issues).

Start with **Task 3** (issue #53, branch `task-3-yolo-benchmark`):

```bash
git fetch origin
git checkout task-3-yolo-benchmark
source .venv/bin/activate
uv sync --dev
```

## 4. Implement, verify, push

Prompt the AI agent with something like:

> Implement task 3 per the instructions in issue #53 and
> `.kiro/specs/vlm-content-moderation/tasks.md`. Only commit files
> that are actually needed. Run the verify commands before asking me
> to review.

Verify every change locally before committing:

```bash
uv run ruff check --fix .
uv run ruff format .
uv run mypy models/ utils/ scripts/
uv run pytest
uv run python scripts/lint_markdown.py --fix
uv run python scripts/check_tasks.py --task 3
```

A task is "done" when `check_tasks.py --task N` passes (no `xfail`).

Push and open a PR:

```bash
git push -u origin task-3-yolo-benchmark
gh pr create --fill --assignee LouisFIP27 --reviewer Mishanil --base main
```

## 5. Rules

- Never push to `main` directly.
- Never force-push.
- One task per branch. If a task grows, split it into a follow-up
  issue, don't hide extra work in the PR.
- If pre-commit complains, fix it. Don't pass `--no-verify`.
- If stuck for more than 30 minutes, comment on the issue or ping
  @Mishanil.
