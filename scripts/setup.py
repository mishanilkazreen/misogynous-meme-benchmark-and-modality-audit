#!/usr/bin/env python3
"""
Setup script for VLM Content Moderation project.
Automates environment setup, dependency installation, and pre-commit hooks.
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd: list[str], description: str) -> bool:
    """Run a command and return success status."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e.stderr}")
        return False


def check_uv_installed() -> bool:
    """Check if uv is installed."""
    try:
        subprocess.run(["uv", "--version"], check=True, capture_output=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def main() -> None:
    """Main setup function."""
    print("Setting up project")
    print("=" * 50)

    # Check if uv is installed
    if not check_uv_installed():
        print("❌ uv is not installed. Please install it first:")
        print("   curl -LsSf https://astral.sh/uv/install.sh | sh")
        sys.exit(1)

    project_root = Path(__file__).parent.parent
    print(f"📁 Project root: {project_root}")

    # Change to project directory
    import os
    os.chdir(project_root)

    steps = [
        (["uv", "venv", "--python", "3.10"], "Creating virtual environment"),
        (["uv", "pip", "install", "-e", ".[dev]"], "Installing dependencies"),
        (["uv", "pip", "install", "torch", "torchvision", "--index-url",
          "https://download.pytorch.org/whl/cpu"], "Installing PyTorch (CPU)"),
        (["uv", "run", "pre-commit", "install"], "Setting up pre-commit hooks"),
        (["uv", "run", "pytest", "--version"], "Verifying pytest installation"),
        (["uv", "run", "ruff", "--version"], "Verifying ruff installation"),
    ]

    failed_steps = []
    for cmd, description in steps:
        if not run_command(cmd, description):
            failed_steps.append(description)

    print("\n" + "=" * 50)
    if failed_steps:
        print("❌ Setup completed with errors:")
        for step in failed_steps:
            print(f"   - {step}")
        print("\nPlease fix the errors and run setup again.")
    else:
        print("✅ Setup completed successfully!")
        print("\nNext steps:")
        print("   1. Activate virtual environment: source .venv/bin/activate")
        print("   2. Run tests: uv run pytest")
        print("   3. Check code quality: uv run pre-commit run --all-files")


if __name__ == "__main__":
    main()
