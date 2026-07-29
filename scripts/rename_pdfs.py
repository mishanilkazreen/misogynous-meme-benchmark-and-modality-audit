# pylint: disable=too-many-locals,too-many-statements,broad-exception-caught,too-many-branches,duplicate-code
import os
import re
import sys

from clean_references import STOP_WORDS, get_primary_author_surname, parse_bibtex

# Path setup
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_GRANT = os.path.join(PROJECT_ROOT, "application", "horizon_drs_2026")
grant_dir = os.environ.get("GRANT_DIR", DEFAULT_GRANT)
bib_path = os.environ.get("BIB_PATH", os.path.join(grant_dir, "references.bib"))
papers_dir = os.environ.get("PAPERS_DIR", os.path.join(grant_dir, "papers"))


def clean_text_to_tokens(text):
    if not text:
        return set()
    # Remove braces and lowercase
    text = text.replace("{", "").replace("}", "").lower()
    # Replace non-alphanumeric with space
    text = re.sub(r"[^a-z0-9\s\-]", " ", text)
    # Tokenize
    tokens = []
    for token in re.split(r"[\s\_\-]", text):
        clean_tok = re.sub(r"[^a-z0-9]", "", token)
        # Filter short tokens or stop words, but keep numbers (years, yolov8, etc.)
        if clean_tok and len(clean_tok) > 1 and clean_tok not in STOP_WORDS:
            tokens.append(clean_tok)
    return set(tokens)


def get_year_from_text(text):
    match = re.search(r"\b(20\d{2}|19\d{2})\b", text)
    return match.group(0) if match else None


def main():
    if not os.path.exists(papers_dir):
        print(f"Error: Papers directory not found at {papers_dir}")
        sys.exit(1)
    if not os.path.exists(bib_path):
        print(f"Error: Bibliography file not found at {bib_path}")
        sys.exit(1)

    # Parse cleaned references
    with open(bib_path, encoding="utf-8") as f:
        content = f.read()
    entries = parse_bibtex(content)
    print(f"Loaded {len(entries)} reference entries from {bib_path}.")

    # Pre-tokenize all bib titles
    bib_data = []
    for entry in entries:
        fields = entry["fields"]
        title = fields.get("title", "").strip("{}")
        author = fields.get("author", "").strip("{}")
        year = fields.get("year", "").strip("{}")
        doi = fields.get("doi", "").strip("{}")

        title_tokens = clean_text_to_tokens(title)

        bib_data.append(
            {
                "key": entry["original_key"],  # the keys are already updated in the cleaned bib
                "title": title,
                "author": author,
                "year": year,
                "doi": doi,
                "title_tokens": title_tokens,
            }
        )

    # Get all PDF files
    all_files = [f for f in os.listdir(papers_dir) if f.endswith(".pdf")]
    print(f"Found {len(all_files)} PDF files in {papers_dir}.")

    matches = []
    unmatched = []

    for filename in all_files:
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

            # Score formula: Jaccard similarity weighted by token overlap
            union = pdf_tokens.union(entry["title_tokens"])
            jaccard = overlap_count / len(union) if union else 0.0

            # Year match bonus
            if pdf_year and entry["year"] == pdf_year:
                jaccard += 0.2  # significant bonus

            # If the filename contains author's surname
            author_surname = get_primary_author_surname(entry["author"])
            if author_surname and author_surname in name_without_ext.lower():
                jaccard += 0.2  # author bonus

            if jaccard > best_score:
                best_score = jaccard
                best_entry = entry
                best_overlap_count = overlap_count

        # We enforce a threshold of similarity to avoid false matches
        # At least 2 tokens must overlap, or 1 token if the name is very short
        threshold = 0.15
        if best_entry and best_score >= threshold and best_overlap_count >= 2:
            matches.append(
                {
                    "filename": filename,
                    "key": best_entry["key"],
                    "title": best_entry["title"],
                    "score": best_score,
                }
            )
        else:
            # Let's try one more fallback: substring matching or lower threshold if year and author match
            fallback_found = False
            if best_entry and best_score >= 0.1:
                # If year and author match, or title tokens have high containment
                intersection = pdf_tokens.intersection(best_entry["title_tokens"])
                if len(intersection) >= 1:
                    matches.append(
                        {
                            "filename": filename,
                            "key": best_entry["key"],
                            "title": best_entry["title"],
                            "score": best_score,
                        }
                    )
                    fallback_found = True
            if not fallback_found:
                unmatched.append(filename)

    print("\n--- MATCHING RESULTS (DRY RUN) ---")
    print(f"Matched: {len(matches)} / {len(all_files)}")
    print(f"Unmatched: {len(unmatched)} / {len(all_files)}")

    # If there are unmatched, print them
    if unmatched:
        print("\nUnmatched files:")
        for idx, f in enumerate(unmatched):
            print(f"  {idx + 1}. {f}")

    # Rename the matched files
    print("\nRenaming matched files...")
    renamed_count = 0
    skipped_count = 0
    for match in matches:
        old_path = os.path.join(papers_dir, match["filename"])
        new_filename = f"{match['key']}.pdf"
        new_path = os.path.join(papers_dir, new_filename)

        if match["filename"] == new_filename:
            skipped_count += 1
            continue

        # Check if destination already exists
        if os.path.exists(new_path):
            # If the destination already exists, we might have multiple PDFs matching the same reference
            # Or we already renamed it in a previous run.
            # To be safe, we don't overwrite if files are different, but here let's append a suffix or skip
            print(
                f"Warning: Destination {new_filename} already exists. Skipping rename of {match['filename']}."
            )
            skipped_count += 1
            continue

        try:
            os.rename(old_path, new_path)
            print(f"Renamed: '{match['filename']}' -> '{new_filename}'")
            renamed_count += 1
        except Exception as e:
            print(f"Error renaming {match['filename']}: {e}")

    print(f"\nDone! Renamed {renamed_count} files, skipped {skipped_count} files.")


if __name__ == "__main__":
    main()
