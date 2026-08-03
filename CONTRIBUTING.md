# Contributing

## 1. Install tools (once per machine, macOS)

```bash
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
git clone https://github.com/mishanilkazreen/misogynous-meme-benchmark-and-modality-audit.git
cd content-moderation
uv venv --python 3.10
uv sync --dev
uv run pre-commit install
```

> **Note:** torch is managed by `pyproject.toml` (cu128 index).
> `uv sync --dev` handles everything - no separate torch install needed.

## 3. Workflow

All work happens on `main`. Open tasks are tracked in
[GitHub issues](https://github.com/mishanilkazreen/misogynous-meme-benchmark-and-modality-audit/issues)
with verification commands and closing criteria in each issue body.

See the project task tracker in [`README.md`](README.md) for current
status and assignments.

## 4. Before committing

```bash
uv run ruff check --fix .
uv run ruff format .
uv run mypy models/ utils/ scripts/
uv run pytest
uv run python scripts/lint_markdown.py
uv run pre-commit run --all-files
```

## 5. Rules

- All pre-commit hooks must pass before pushing.
- If pre-commit complains, fix it. Do not pass `--no-verify`.
- Keep commits focused. One logical change per commit.
- If stuck for more than 30 minutes, comment on the issue.
