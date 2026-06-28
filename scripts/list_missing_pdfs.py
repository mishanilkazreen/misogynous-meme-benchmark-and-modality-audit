# pylint: disable=too-many-locals,too-many-statements,broad-exception-caught
import os
from urllib.parse import urlparse

from clean_references import parse_bibtex

# Path setup
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_GRANT = os.path.join(PROJECT_ROOT, "application", "horizon_drs_2026")
grant_dir = os.environ.get("GRANT_DIR", DEFAULT_GRANT)
bib_path = os.environ.get("BIB_PATH", os.path.join(grant_dir, "references.bib"))
papers_dir = os.environ.get("PAPERS_DIR", os.path.join(grant_dir, "papers"))
report_path = os.environ.get("REPORT_PATH", os.path.join(grant_dir, "missing_papers_report.md"))


def main():
    if not os.path.exists(bib_path):
        print(f"Error: Bibliography not found at {bib_path}")
        return
    if not os.path.exists(papers_dir):
        print(f"Error: Papers directory not found at {papers_dir}")
        return

    with open(bib_path, encoding="utf-8") as f:
        content = f.read()
    entries = parse_bibtex(content)

    # Get all PDF filenames in papers/
    pdf_files = {os.path.splitext(f)[0] for f in os.listdir(papers_dir) if f.endswith(".pdf")}

    missing_entries = []
    for entry in entries:
        key = entry["original_key"]
        if key not in pdf_files:
            missing_entries.append(entry)

    print(f"Total bibliography entries: {len(entries)}")
    print(f"Total PDFs present: {len(pdf_files)}")
    print(f"Missing PDFs: {len(missing_entries)}")

    # Write a Markdown report
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Missing Papers Report - Horizon Europe CL3 DRS-03\n\n")
        f.write(
            f"This report lists the **{len(missing_entries)}** papers present in "
            f"our [references.bib](file://{bib_path}) database that do not have "
            f"a corresponding PDF in the `papers/` folder. "
        )
        f.write(
            "You can download them using the links below and save them directly "
            "to the `papers/` folder with the target filename.\n\n"
        )

        f.write("| # | Target Filename | Paper Title | Authors & Year | Download Link / DOI |\n")
        f.write("|---|-----------------|-------------|----------------|---------------------|\n")

        for idx, entry in enumerate(missing_entries):
            fields = entry["fields"]
            title = fields.get("title", "").strip("{}")
            author = fields.get("author", "").strip("{}")
            year = fields.get("year", "").strip("{}")
            url = fields.get("url", "").strip("{}")
            doi = fields.get("doi", "").strip("{}")

            # Shorten authors
            authors_short = author
            if " and " in author:
                parts = author.split(" and ")
                authors_short = parts[0] + " et al." if len(parts) > 2 else " & ".join(parts)

            target_filename = f"`{entry['original_key']}.pdf`"

            if doi:
                link = f"[DOI: {doi}](https://doi.org/{doi})"
            elif url:
                domain = urlparse(url).netloc
                link = f"[{domain} Article]({url})"
            else:
                link = "N/A"

            f.write(f"| {idx + 1} | {target_filename} | {title} | {authors_short} ({year}) | {link} |\n")

    print(f"Report successfully written to {report_path}")


if __name__ == "__main__":
    main()
