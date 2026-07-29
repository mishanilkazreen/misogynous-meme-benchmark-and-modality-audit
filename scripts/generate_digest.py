# pylint: disable=too-many-locals,too-many-statements,broad-exception-caught
import os
import sys
import textwrap

from clean_references import parse_bibtex

# Path setup
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_GRANT = os.path.join(PROJECT_ROOT, "application", "horizon_drs_2026")
grant_dir = os.environ.get("GRANT_DIR", DEFAULT_GRANT)
bib_path = os.environ.get("BIB_PATH", os.path.join(grant_dir, "references.bib"))
papers_dir = os.environ.get("PAPERS_DIR", os.path.join(grant_dir, "papers"))
digest_path = os.environ.get("DIGEST_PATH", os.path.join(grant_dir, "literature_digest.md"))


def clean_value(val):
    if not val:
        return ""
    val = val.strip()
    if val.startswith("{") and val.endswith("}"):
        val = val[1:-1]
    return val


def main():
    if not os.path.exists(bib_path):
        print(f"Error: Bibliography not found at {bib_path}")
        sys.exit(1)
    if not os.path.exists(papers_dir):
        print(f"Error: Papers directory not found at {papers_dir}")
        sys.exit(1)

    print("Parsing references.bib...")
    with open(bib_path, encoding="utf-8") as f:
        content = f.read()
    entries = parse_bibtex(content)

    # Get all PDF filenames in papers/ (without .pdf extension)
    pdf_files = {os.path.splitext(f)[0] for f in os.listdir(papers_dir) if f.endswith(".pdf")}

    print(f"Generating literature digest at {digest_path}...")

    with open(digest_path, "w", encoding="utf-8") as f:
        f.write("# Project Literature Digest - Horizon Europe CL3 DRS-03\n\n")
        f.write(
            "This document compiles the metadata and abstracts of all references in our project bibliography.\n"
        )
        f.write(
            "Upload this file to NotebookLM to provide it with a searchable index of the literature,\n"
        )
        f.write("citation keys, and research abstracts.\n\n")

        f.write(f"**Total References:** {len(entries)}\n")
        f.write(f"**Local PDFs Available:** {len(pdf_files)} / {len(entries)}\n\n")
        f.write("---\n\n")

        for idx, entry in enumerate(entries):
            key = entry["original_key"]
            fields = entry["fields"]

            title = clean_value(fields.get("title", ""))
            author = clean_value(fields.get("author", ""))
            year = clean_value(fields.get("year", ""))
            journal = clean_value(fields.get("journal", fields.get("booktitle", "")))
            doi = clean_value(fields.get("doi", ""))
            url = clean_value(fields.get("url", ""))
            abstract = clean_value(fields.get("abstract", ""))

            pdf_status = (
                "✅ PDF Available" if key in pdf_files else "❌ PDF Missing (Needs Download)"
            )

            wrapped_title = textwrap.fill(title, width=100)
            f.write(f"## [{idx + 1}] {wrapped_title}\n\n")
            f.write(f"- **Citation Key:** `{key}`\n")

            wrapped_author = textwrap.fill(author, width=100)
            # Indent wrapped lines for list formatting
            wrapped_author_indented = wrapped_author.replace("\n", "\n  ")
            f.write(f"- **Authors:** {wrapped_author_indented}\n")
            f.write(f"- **Year:** {year}\n")

            if journal:
                wrapped_journal = textwrap.fill(journal, width=100).replace("\n", "\n  ")
                f.write(f"- **Source:** *{wrapped_journal}*\n")

            links = []
            if doi:
                links.append(f"[DOI (https://doi.org/{doi})](https://doi.org/{doi})")
            if url:
                links.append(f"[URL]({url})")
            if links:
                # Wrap links carefully, but since URLs shouldn't contain spaces,
                # textwrap is fine or we keep them on one line
                joined_links = ", ".join(links)
                # Keep URLs intact (not split by textwrap if possible) but if very long, let's write it out
                f.write(f"- **Links:** {joined_links}\n")

            f.write(f"- **Status:** {pdf_status}\n\n")

            if abstract:
                f.write("### Abstract\n")
                wrapped_abstract = textwrap.fill(abstract, width=100)
                f.write(f"{wrapped_abstract}\n\n")
            else:
                f.write("### Abstract\n*No abstract available in bibliography.*\n\n")

            f.write("---\n\n")

    print("Literature digest generated successfully!")


if __name__ == "__main__":
    main()
