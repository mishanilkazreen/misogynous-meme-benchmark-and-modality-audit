# Development Setup Guide

## Python Version Management

This project requires Python 3.10 or 3.11 for compatibility with ML libraries.

### Setting Up the Environment

1. **Check available Python versions:**
   ```bash
   uv python list
   ```

2. **Create virtual environment with specific Python version:**
   ```bash
   # Remove any existing environments
   rm -rf venv .venv

   # Create new environment with Python 3.10
   uv venv --python 3.10 venv

   # Activate environment
   # Windows:
   venv\Scripts\activate
   # Linux/Mac:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   uv pip install -e ".[dev]"
   uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   ```

4. **Install pre-commit hooks:**
   ```bash
   pre-commit install
   ```

### Python Version Constraints

- **Supported:** Python 3.10, 3.11
- **Not supported:** Python 3.12+ (PyTorch compatibility issues)
- **Not supported:** Python 3.9- (missing features)

The `.python-version` file ensures consistent Python version usage across the project.

### Troubleshooting

**Issue: uv uses wrong Python version**
```bash
# Force specific Python version
uv venv --python 3.10 venv
```

**Issue: Virtual environment conflicts**
```bash
# Clean slate approach
rm -rf venv .venv
uv venv --python 3.10 venv
```

**Issue: PyTorch not working**
```bash
# Reinstall with CUDA support
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## CI/CD Pipeline

### GitHub Actions Workflows

1. **CI Pipeline** (`.github/workflows/ci.yml`):
   - Runs on push/PR to main
   - Tests Python 3.10 and 3.11
   - Linting with ruff
   - Type checking with mypy
   - Full test suite with coverage

2. **CD Pipeline** (`.github/workflows/cd.yml`):
   - Builds package distributions
   - Publishes to PyPI on tags
   - Creates GitHub releases

### Pre-commit Hooks

Automatically run on each commit:
- Code formatting (ruff)
- Linting (ruff)
- Type checking (mypy)
- YAML/JSON validation
- Trailing whitespace removal

### Local Development Commands

```bash
# Run all checks locally
venv\Scripts\python -m ruff check .
venv\Scripts\python -m ruff format --check .
venv\Scripts\python -m mypy models/ utils/ --ignore-missing-imports
venv\Scripts\python -m pytest tests/ -v

# Run pre-commit on all files
pre-commit run --all-files
```

## Package Management

This project uses `uv` for fast Python package management:

```bash
# Add new dependency
uv add package-name

# Add development dependency
uv add --dev package-name

# Install from lock file
uv sync

# Update dependencies
uv lock --upgrade
```

## Environment Variables

Set these for optimal development experience:

```bash
# Windows
set PYTHONPATH=%CD%
set CUDA_VISIBLE_DEVICES=0

# Linux/Mac
export PYTHONPATH=$(pwd)
export CUDA_VISIBLE_DEVICES=0
```
