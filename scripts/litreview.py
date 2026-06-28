# pylint: disable=too-many-statements,broad-exception-caught,too-many-branches
import argparse
import os
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def run_script(script_name, args_list=None):
    script_path = os.path.join(SCRIPT_DIR, script_name)
    cmd = ["python3", script_path]
    if args_list:
        cmd.extend(args_list)
    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_name}: {e}")
        return e.returncode


def main():
    parser = argparse.ArgumentParser(description="Unified Literature Review and Grant Proposal Automation Tool.")
    parser.add_argument(
        "-g",
        "--grant",
        type=str,
        default="papers",
        help="Path or folder name of the grant (default: papers)",
    )
    subparsers = parser.add_subparsers(dest="command", help="Subcommand to run")

    # clean
    subparsers.add_parser("clean", help="Format, lint, deduplicate, and sort references.bib")

    # rename
    subparsers.add_parser("rename", help="Rename PDF files in papers/ to match BibTeX keys")

    # digest
    subparsers.add_parser("digest", help="Generate literature_digest.md for NotebookLM upload")

    # missing
    subparsers.add_parser("missing", help="Check for missing PDFs and generate missing_papers_report.md")

    # zotero
    subparsers.add_parser("zotero", help="Import, match, and rename PDFs from Zotero Exported Items folder")

    # sync-zotero
    subparsers.add_parser("sync-zotero", help="Sync bibliography directly from Zotero Web API using credentials")

    # open-missing
    open_parser = subparsers.add_parser(
        "open-missing", help="Open URLs/DOIs of missing papers in web browser in batches"
    )
    open_parser.add_argument("--batch", type=int, default=1, help="Batch number (1-indexed)")
    open_parser.add_argument("--size", type=int, default=10, help="Batch size (number of links to open)")

    # search
    search_parser = subparsers.add_parser("search", help="Search Google Scholar via SerpAPI")
    search_parser.add_argument("query", type=str, help="Search query")
    search_parser.add_argument("--year-start", type=int, default=None, help="Start year of publication")
    search_parser.add_argument("--num-results", type=int, default=10, help="Number of results to retrieve")

    # s2-search
    s2_search_parser = subparsers.add_parser("s2-search", help="Search Semantic Scholar by keyword")
    s2_search_parser.add_argument("query", type=str, help="Search query")
    s2_search_parser.add_argument("--limit", type=int, default=15, help="Number of results to retrieve")

    # s2-recommend
    s2_rec_parser = subparsers.add_parser(
        "s2-recommend", help="Get Semantic Scholar recommendations for a seed paper (DOI or S2 id)"
    )
    s2_rec_parser.add_argument("seed", type=str, help="Seed paper DOI or Semantic Scholar id")
    s2_rec_parser.add_argument("--limit", type=int, default=15, help="Number of recommendations to retrieve")

    # process-csv
    subparsers.add_parser(
        "process-csv", help="Filter, query Crossref, and add Springer Nature CSV search results to references.bib"
    )

    # add-papers
    subparsers.add_parser("add-papers", help="Append new manual papers and run cleaning")

    # count-words
    count_parser = subparsers.add_parser("count-words", help="Count words in grant proposal markdown sections")
    count_parser.add_argument("file", type=str, help="Path to markdown proposal file")
    count_parser.add_argument("args", nargs=argparse.REMAINDER, help="Optional section heading or line numbers")

    # init-grant
    init_parser = subparsers.add_parser("init-grant", help="Initialise a new grant proposal directory structure")
    init_parser.add_argument("name", type=str, help="Name of the new grant folder (e.g. royal_society_2026)")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    # Resolve grant directory
    grant_dir = args.grant
    if not os.path.exists(grant_dir) and not os.path.isabs(grant_dir):
        candidate = os.path.join(SCRIPT_DIR, "..", grant_dir)
        if os.path.exists(candidate):
            grant_dir = candidate
        else:
            candidate_app = os.path.join(SCRIPT_DIR, "..", "application", grant_dir)
            if os.path.exists(candidate_app):
                grant_dir = candidate_app

    grant_dir = os.path.abspath(grant_dir)
    os.environ["GRANT_DIR"] = grant_dir
    os.environ["BIB_PATH"] = os.path.join(grant_dir, "references.bib")
    os.environ["PAPERS_DIR"] = os.path.join(grant_dir, "downloaded_papers")
    os.environ["DIGEST_PATH"] = os.path.join(grant_dir, "literature_digest.md")
    os.environ["REPORT_PATH"] = os.path.join(grant_dir, "missing_papers_report.md")

    if args.command == "clean":
        sys.exit(run_script("clean_references.py"))
    elif args.command == "rename":
        sys.exit(run_script("rename_pdfs.py"))
    elif args.command == "digest":
        sys.exit(run_script("generate_digest.py"))
    elif args.command == "missing":
        sys.exit(run_script("list_missing_pdfs.py"))
    elif args.command == "zotero":
        sys.exit(run_script("process_zotero_export.py"))
    elif args.command == "sync-zotero":
        sys.exit(run_script("sync_zotero_api.py"))
    elif args.command == "open-missing":
        sys.exit(run_script("open_missing_papers.py", ["--batch", str(args.batch), "--size", str(args.size)]))
    elif args.command == "search":
        sub_args = [args.query]
        if args.year_start is not None:
            sub_args.append(str(args.year_start))
        if args.num_results is not None:
            if args.year_start is None:
                sub_args.insert(1, "")  # placeholder
            sub_args.append(str(args.num_results))
        sys.exit(run_script("search_scholar.py", sub_args))
    elif args.command == "s2-search":
        sys.exit(run_script("semantic_scholar.py", ["search", args.query, str(args.limit)]))
    elif args.command == "s2-recommend":
        sys.exit(run_script("semantic_scholar.py", ["recommend", args.seed, str(args.limit)]))
    elif args.command == "process-csv":
        sys.exit(run_script("process_springer_csv.py"))
    elif args.command == "add-papers":
        sys.exit(run_script("add_new_papers.py"))
    elif args.command == "count-words":
        sub_args = [args.file]
        if args.args:
            sub_args.extend(args.args)
        sys.exit(run_script("count_words.py", sub_args))
    elif args.command == "init-grant":
        new_grant_dir = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "application", args.name))
        if os.path.exists(new_grant_dir):
            print(f"Error: Grant directory '{args.name}' already exists at {new_grant_dir}")
            sys.exit(1)
        os.makedirs(new_grant_dir, exist_ok=True)
        os.makedirs(os.path.join(new_grant_dir, "papers"), exist_ok=True)
        with open(os.path.join(new_grant_dir, "references.bib"), "w", encoding="utf-8") as f:
            f.write(f"% References bibliography for {args.name}\n")
        with open(os.path.join(new_grant_dir, "README.md"), "w", encoding="utf-8") as f:
            f.write(
                f"# {args.name.replace('_', ' ').title()} Proposal\n\n"
                "This directory contains application drafts, tasks, and literature review references.\n"
            )
        with open(os.path.join(new_grant_dir, "TASKS.md"), "w", encoding="utf-8") as f:
            f.write(
                f"# Tasks for {args.name.replace('_', ' ').title()}\n\n"
                "- [ ] Literature review\n"
                "- [ ] Research proposal draft\n"
                "- [ ] Budget planning\n"
            )
        print(f"Successfully initialised new grant at {new_grant_dir}")
        sys.exit(0)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
