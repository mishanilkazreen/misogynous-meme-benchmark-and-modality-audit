"""Unified Literature Review and Grant Proposal Automation Tool.

Usage:
    python scripts/litreview.py clean
    python scripts/litreview.py sync-zotero
"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

SCRIPT_DIR = Path(__file__).resolve().parent


def run_script(script_name: str, args_list: list[str] | None = None) -> int:
    """Execute python sub-script with arguments."""
    script_path = SCRIPT_DIR / script_name
    cmd = ["python3", str(script_path)]
    if args_list:
        cmd.extend(args_list)

    result = subprocess.run(cmd, check=False)
    return result.returncode


def setup_grant_environment(grant_arg: str) -> Path:
    """Resolve grant path and configure environment variables."""
    grant_path = Path(grant_arg)
    if not grant_path.exists() and not grant_path.is_absolute():
        candidate = SCRIPT_DIR.parent / grant_arg
        if candidate.exists():
            grant_path = candidate
        else:
            candidate_app = SCRIPT_DIR.parent / "application" / grant_arg
            if candidate_app.exists():
                grant_path = candidate_app

    grant_path = grant_path.resolve()
    import os

    os.environ["GRANT_DIR"] = str(grant_path)
    os.environ["BIB_PATH"] = str(grant_path / "references.bib")
    os.environ["PAPERS_DIR"] = str(grant_path / "downloaded_papers")
    os.environ["DIGEST_PATH"] = str(grant_path / "literature_digest.md")
    os.environ["REPORT_PATH"] = str(grant_path / "missing_papers_report.md")

    return grant_path


def handle_init_grant(grant_name: str) -> None:
    """Initialize new grant proposal directory structure."""
    new_grant_dir = (SCRIPT_DIR.parent / "application" / grant_name).resolve()
    if new_grant_dir.exists():
        print(
            f"Error: Grant directory '{grant_name}' already exists at {new_grant_dir}",
            file=sys.stderr,
        )
        sys.exit(1)

    new_grant_dir.mkdir(parents=True, exist_ok=True)
    (new_grant_dir / "papers").mkdir(parents=True, exist_ok=True)

    bib_file = new_grant_dir / "references.bib"
    with bib_file.open("w", encoding="utf-8") as f:
        f.write(f"% References bibliography for {grant_name}\n")

    readme_file = new_grant_dir / "README.md"
    with readme_file.open("w", encoding="utf-8") as f:
        f.write(
            f"# {grant_name.replace('_', ' ').title()} Proposal\n\n"
            "This directory contains application drafts, tasks, and literature review references.\n"
        )

    tasks_file = new_grant_dir / "TASKS.md"
    with tasks_file.open("w", encoding="utf-8") as f:
        f.write(
            f"# Tasks for {grant_name.replace('_', ' ').title()}\n\n"
            "- [ ] Literature review\n"
            "- [ ] Research proposal draft\n"
            "- [ ] Budget planning\n"
        )

    print(f"Successfully initialised new grant at {new_grant_dir}")
    sys.exit(0)


def build_arg_parser() -> argparse.ArgumentParser:
    """Construct CLI argument parser for litreview tool."""
    parser = argparse.ArgumentParser(
        description="Unified Literature Review and Grant Proposal Automation Tool."
    )
    parser.add_argument(
        "-g", "--grant", type=str, default="papers", help="Path or folder name of the grant"
    )
    subparsers = parser.add_subparsers(dest="command", help="Subcommand to run")

    subparsers.add_parser("rename", help="Rename PDF files in papers/ to match BibTeX keys")
    subparsers.add_parser("digest", help="Generate literature_digest.md for NotebookLM upload")
    subparsers.add_parser("zotero", help="Import and match PDFs from Zotero Exported Items folder")
    subparsers.add_parser("sync-zotero", help="Sync bibliography directly from Zotero Web API")

    open_parser = subparsers.add_parser(
        "open-missing", help="Open URLs of missing papers in web browser"
    )
    open_parser.add_argument("--batch", type=int, default=1, help="Batch number")
    open_parser.add_argument("--size", type=int, default=10, help="Batch size")

    search_parser = subparsers.add_parser("search", help="Search Google Scholar via SerpAPI")
    search_parser.add_argument("query", type=str, help="Search query")
    search_parser.add_argument("--year-start", type=int, default=None, help="Start year")
    search_parser.add_argument("--num-results", type=int, default=10, help="Number of results")

    s2_search_parser = subparsers.add_parser("s2-search", help="Search Semantic Scholar by keyword")
    s2_search_parser.add_argument("query", type=str, help="Search query")
    s2_search_parser.add_argument("--limit", type=int, default=15, help="Number of results")

    s2_rec_parser = subparsers.add_parser(
        "s2-recommend", help="Get Semantic Scholar recommendations"
    )
    s2_rec_parser.add_argument("seed", type=str, help="Seed paper DOI or S2 id")
    s2_rec_parser.add_argument("--limit", type=int, default=15, help="Number of recommendations")

    init_parser = subparsers.add_parser(
        "init-grant", help="Initialise a new grant proposal directory structure"
    )
    init_parser.add_argument("name", type=str, help="Name of the new grant folder")

    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    setup_grant_environment(args.grant)

    commands_map = {
        "rename": lambda: sys.exit(run_script("rename_pdfs.py")),
        "digest": lambda: sys.exit(run_script("generate_digest.py")),
        "zotero": lambda: sys.exit(run_script("process_zotero_export.py")),
        "sync-zotero": lambda: sys.exit(run_script("sync_zotero_api.py")),
        "s2-search": lambda: sys.exit(
            run_script("semantic_scholar.py", ["search", args.query, str(args.limit)])
        ),
        "s2-recommend": lambda: sys.exit(
            run_script("semantic_scholar.py", ["recommend", args.seed, str(args.limit)])
        ),
    }

    if args.command == "init-grant":
        handle_init_grant(args.name)
    elif args.command in commands_map:
        commands_map[args.command]()
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
