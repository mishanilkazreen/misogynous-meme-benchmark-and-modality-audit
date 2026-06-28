# pylint: disable=too-many-locals,too-many-statements,broad-exception-caught
import argparse
import os
import sys
import webbrowser

from clean_references import parse_bibtex

# Path setup
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_GRANT = os.path.join(PROJECT_ROOT, "application", "horizon_drs_2026")
grant_dir = os.environ.get("GRANT_DIR", DEFAULT_GRANT)
bib_path = os.environ.get("BIB_PATH", os.path.join(grant_dir, "references.bib"))
papers_dir = os.environ.get("PAPERS_DIR", os.path.join(grant_dir, "papers"))


def main():
    parser = argparse.ArgumentParser(description="Open URLs of missing PDFs in web browser in batches.")
    parser.add_argument("--batch", type=int, default=1, help="Batch number (1-indexed) to open.")
    parser.add_argument("--size", type=int, default=10, help="Number of URLs to open in this batch.")
    args = parser.parse_args()

    if not os.path.exists(bib_path):
        print(f"Error: Bibliography not found at {bib_path}")
        sys.exit(1)
    if not os.path.exists(papers_dir):
        print(f"Error: Papers directory not found at {papers_dir}")
        sys.exit(1)

    with open(bib_path, encoding="utf-8") as f:
        content = f.read()
    entries = parse_bibtex(content)

    # Get all PDF filenames in papers/ (without .pdf extension)
    pdf_files = {os.path.splitext(f)[0] for f in os.listdir(papers_dir) if f.endswith(".pdf")}

    missing_entries = []
    for entry in entries:
        key = entry["original_key"]
        if key not in pdf_files:
            missing_entries.append(entry)

    total_missing = len(missing_entries)
    print(f"Total bibliography entries: {len(entries)}")
    print(f"Total PDFs present: {len(pdf_files)}")
    print(f"Missing PDFs: {total_missing}")

    if total_missing == 0:
        print("No missing papers to download!")
        return

    # Calculate slices
    start_idx = (args.batch - 1) * args.size
    end_idx = min(start_idx + args.size, total_missing)

    if start_idx >= total_missing or start_idx < 0:
        max_batch = (total_missing + args.size - 1) // args.size
        print(f"Error: Batch {args.batch} is out of range. Max batch is {max_batch}.")
        return

    print(f"\n--- Opening Batch {args.batch} (papers {start_idx + 1} to {end_idx} of {total_missing}) ---")

    for idx in range(start_idx, end_idx):
        entry = missing_entries[idx]
        fields = entry["fields"]
        key = entry["original_key"]
        title = fields.get("title", "").strip("{}")
        url = fields.get("url", "").strip("{}")
        doi = fields.get("doi", "").strip("{}")

        # Fallback to DOI URL if url field is missing or empty
        download_url = url
        if not download_url and doi:
            download_url = f"https://doi.org/{doi}"

        if not download_url:
            print(f"[{idx + 1}] Key: {key} - No URL or DOI found for: '{title[:50]}...'")
            continue

        print(f"[{idx + 1}] Key: {key}")
        print(f"    Title: {title}")
        print(f"    Target Filename: {key}.pdf")
        print(f"    Opening: {download_url}")

        webbrowser.open(download_url)

    print("\nBatch URLs opened in your default browser.")
    print("Please download the PDFs and save them to the papers/ folder.")
    print("Then run `python3 scripts/rename_pdfs.py` to match the filenames to citation keys.")


if __name__ == "__main__":
    main()
