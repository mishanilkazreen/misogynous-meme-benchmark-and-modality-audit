# pylint: disable=too-many-locals,too-many-statements,broad-exception-caught,too-many-branches
import os
import re
import shutil
import sys

from clean_references import STOP_WORDS, get_primary_author_surname, parse_bibtex

# Path setup
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_GRANT = os.path.join(PROJECT_ROOT, "application", "horizon_drs_2026")
grant_dir = os.environ.get("GRANT_DIR", DEFAULT_GRANT)
bib_path = os.environ.get("BIB_PATH", os.path.join(grant_dir, "references.bib"))
papers_dir = os.environ.get("PAPERS_DIR", os.path.join(grant_dir, "papers"))
zotero_export_dir = os.environ.get("ZOTERO_EXPORT_DIR", os.path.join(papers_dir, "Exported Items"))


def clean_text_to_tokens(text):
    if not text:
        return set()
    text = text.replace("{", "").replace("}", "").lower()
    text = re.sub(r"[^a-z0-9\s\-]", " ", text)
    tokens = []
    for token in re.split(r"[\s\_\-]", text):
        clean_tok = re.sub(r"[^a-z0-9]", "", token)
        if clean_tok and len(clean_tok) > 1 and clean_tok not in STOP_WORDS:
            tokens.append(clean_tok)
    return set(tokens)


def get_year_from_text(text):
    match = re.search(r"\b(20\d{2}|19\d{2})\b", text)
    return match.group(0) if match else None


def main():
    if not os.path.exists(zotero_export_dir):
        print(f"Error: Zotero export directory not found at {zotero_export_dir}")
        sys.exit(1)
    if not os.path.exists(bib_path):
        print(f"Error: Bibliography file not found at {bib_path}")
        sys.exit(1)

    # 1. Parse bibliography references
    with open(bib_path, encoding="utf-8") as f:
        content = f.read()
    entries = parse_bibtex(content)
    print(f"Loaded {len(entries)} reference entries from {bib_path}.")

    bib_data = []
    for entry in entries:
        fields = entry["fields"]
        title = fields.get("title", "").strip("{}")
        author = fields.get("author", "").strip("{}")
        year = fields.get("year", "").strip("{}")

        title_tokens = clean_text_to_tokens(title)
        bib_data.append(
            {
                "key": entry["original_key"],
                "title": title,
                "author": author,
                "year": year,
                "title_tokens": title_tokens,
            }
        )

    # 2. Find all PDF files in Zotero export folder recursively
    zotero_pdfs = []
    for root, _dirs, files in os.walk(zotero_export_dir):
        for file in files:
            if file.endswith(".pdf"):
                zotero_pdfs.append(os.path.join(root, file))

    print(f"Found {len(zotero_pdfs)} PDF files inside Zotero export directory.")

    # 3. Match each Zotero PDF and copy it to papers/
    copied_count = 0
    skipped_count = 0
    unmatched_files = []

    for pdf_path in zotero_pdfs:
        filename = os.path.basename(pdf_path)
        name_without_ext = os.path.splitext(filename)[0]
        pdf_tokens = clean_text_to_tokens(name_without_ext)
        pdf_year = get_year_from_text(name_without_ext)

        best_entry = None
        best_score = -1.0
        best_overlap_count = 0

        for entry in bib_data:
            intersection = pdf_tokens.intersection(entry["title_tokens"])
            overlap_count = len(intersection)
            if overlap_count == 0:
                continue

            union = pdf_tokens.union(entry["title_tokens"])
            jaccard = overlap_count / len(union) if union else 0.0

            # Year match bonus
            if pdf_year and entry["year"] == pdf_year:
                jaccard += 0.2

            # Author match bonus
            author_surname = get_primary_author_surname(entry["author"])
            if author_surname and author_surname in name_without_ext.lower():
                jaccard += 0.2

            if jaccard > best_score:
                best_score = jaccard
                best_entry = entry
                best_overlap_count = overlap_count

        threshold = 0.15
        # Enforce threshold
        is_match = False
        if best_entry and best_score >= threshold and best_overlap_count >= 2:
            is_match = True
        elif best_entry and best_score >= 0.1:
            # Fallback if both author and year match
            intersection = pdf_tokens.intersection(best_entry["title_tokens"])
            if len(intersection) >= 1:
                is_match = True

        if is_match:
            dest_filename = f"{best_entry['key']}.pdf"
            dest_path = os.path.join(papers_dir, dest_filename)

            # Copy and rename to papers/
            if os.path.exists(dest_path):
                print(
                    f"Warning: PDF for {best_entry['key']} already exists in papers/. Skipping copy of '{filename}'."
                )
                skipped_count += 1
            else:
                shutil.copy2(pdf_path, dest_path)
                print(f"Matched and Moved: '{filename}' -> '{dest_filename}'")
                copied_count += 1
        else:
            print(f"Could not match PDF: '{filename}'")
            unmatched_files.append(pdf_path)

    print(
        f"\nCompleted Zotero Ingestion: Moved {copied_count} files, skipped {skipped_count} existing files."
    )
    if unmatched_files:
        print(f"Total unmatched files: {len(unmatched_files)}")
        # Copy unmatched files to papers/ with their original names for manual checking
        print("Moving unmatched files to papers/ with original names for manual review...")
        for p in unmatched_files:
            fn = os.path.basename(p)
            dest = os.path.join(papers_dir, fn)
            if not os.path.exists(dest):
                shutil.copy2(p, dest)
                print(f"  Moved unmatched: {fn}")

    # 4. Clean up Zotero Export folder and RDF file
    print("\nCleaning up temporary Zotero export directory...")
    try:
        shutil.rmtree(zotero_export_dir)
        rdf_file = os.path.join(zotero_export_dir + ".rdf")
        if os.path.exists(rdf_file):
            os.remove(rdf_file)
        print("Cleanup completed successfully.")
    except Exception as e:
        print(f"Warning: Cleanup failed: {e}")


if __name__ == "__main__":
    main()
