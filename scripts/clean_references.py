# pylint: disable=too-many-locals,too-many-statements,too-many-branches,duplicate-code,broad-exception-caught
import os
import re
import sys
import tempfile

# Path setup
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_GRANT = os.path.join(PROJECT_ROOT, "application", "horizon_drs_2026")
grant_dir = os.environ.get("GRANT_DIR", DEFAULT_GRANT)
bib_path = os.environ.get("BIB_PATH", os.path.join(grant_dir, "references.bib"))


STOP_WORDS = {
    "a",
    "an",
    "the",
    "of",
    "on",
    "for",
    "in",
    "with",
    "at",
    "by",
    "to",
    "from",
    "and",
    "or",
    "is",
    "are",
    "using",
    "based",
    "via",
    "through",
    "about",
    "into",
    "its",
    "as",
    "our",
    "it",
    "under",
    "among",
    "toward",
    "towards",
    "during",
    "against",
    "between",
    "within",
    "over",
    "above",
    "below",
    "behind",
    "beside",
    "beyond",
    "near",
    "off",
    "onto",
    "out",
    "up",
    "down",
    "throughout",
    "how",
    "why",
    "what",
    "where",
    "who",
    "when",
    "which",
}

CAPITALIZATION_MAP = {
    "yolo": "YOLO",
    "yolov3": "YOLOv3",
    "yolov4": "YOLOv4",
    "yolov5": "YOLOv5",
    "yolov5s": "YOLOv5s",
    "yolov7": "YOLOv7",
    "yolov8": "YOLOv8",
    "yolov8x": "YOLOv8x",
    "yolov9": "YOLOv9",
    "yolov10": "YOLOv10",
    "yolov11": "YOLOv11",
    "yolov12": "YOLOv12",
    "uav": "UAV",
    "uavs": "UAVs",
    "sar": "SAR",
    "vlm": "VLM",
    "vlms": "VLMs",
    "clip": "CLIP",
    "llava": "LLaVA",
    "qwen": "Qwen",
    "paligemma": "PaliGemma",
    "rpi": "RPi",
    "raspberry": "Raspberry",
    "pi": "Pi",
    "opencv": "OpenCV",
    "fpga": "FPGA",
    "fpgas": "FPGAs",
    "lidar": "LiDAR",
    "lidars": "LiDARs",
    "adas": "ADAS",
    "its": "ITS",
    "lora": "LoRa",
    "mami": "MAMI",
    "rgbt": "RGBT",
    "gps": "GPS",
    "cnn": "CNN",
    "cnns": "CNNs",
    "ann": "ANN",
    "rnn": "RNN",
    "rnns": "RNNs",
    "ai": "AI",
    "agi": "AGI",
    "5g": "5G",
    "6g": "6G",
    "ugv": "UGV",
    "ugvs": "UGVs",
    "ryze": "Ryze",
    "tello": "Tello",
    "dji": "DJI",
    "imu": "IMU",
    "ros": "ROS",
    "fps": "FPS",
    "map": "mAP",
    "ssd": "SSD",
    "coco": "COCO",
    "macvi": "MaCVi",
    "coxnet": "COXNet",
    "mvdnet": "MVDNet",
    "detr": "DETR",
    "owrt": "OWRT",
    "efoe": "EFOE",
    "devit": "DEViT",
    "ugen": "UGEN",
    "oran": "ORAN",
    "gan": "GAN",
    "vsdrl": "VSDRL",
    "yoloow": "YoloOW",
    "aigc": "AIGC",
    "uas": "UAS",
    "uam": "UAM",
}


def clean_author_name(name):
    # Strip LaTeX accents and brackets
    name = re.sub(r'\\[\'"`^~=]{.*?}', "", name)
    name = re.sub(r'\\[\'"`^~=]\w', "", name)
    name = name.replace("{", "").replace("}", "")
    # Keep only alphanumeric and spaces
    name = re.sub(r"[^a-zA-Z\s\-]", "", name)
    return name.strip()


def get_primary_author_surname(author_field):
    if not author_field:
        return "unknown"
    # Split by 'and' to get authors
    authors = [a.strip() for a in re.split(r"\band\b", author_field, flags=re.IGNORECASE)]
    if not authors:
        return "unknown"
    first_author = authors[0]

    # If comma exists, it is Surname, Firstname
    if "," in first_author:
        surname = first_author.split(",")[0]
    else:
        # Western name order: Firstname Middle... Surname
        parts = first_author.split()
        if len(parts) > 1 and parts[-1].lower() == "others":
            parts = parts[:-1]
        surname = parts[-1] if parts else "unknown"

    surname = clean_author_name(surname)
    return surname.lower() if surname else "unknown"


def clean_title_for_key(title_field):
    if not title_field:
        return "paper"
    # Remove LaTeX commands and braces
    title = re.sub(r"\\[a-zA-Z]+", "", title_field)
    title = title.replace("{", "").replace("}", "")
    # Replace non-alphanumeric (except space/hyphen) with spaces
    title = re.sub(r"[^a-zA-Z0-9\s\-]", " ", title)
    # Tokenize
    tokens = [t.strip("-").lower() for t in title.split() if t.strip("-")]
    for token in tokens:
        # Check if it's alphanumeric and not a stop word
        clean_tok = re.sub(r"[^a-z0-9]", "", token)
        if clean_tok and clean_tok not in STOP_WORDS:
            return clean_tok
    return tokens[0] if tokens else "paper"


def protect_acronyms_in_text(text):
    if not text:
        return text

    # We want to find words, check if they match our CAPITALIZATION_MAP, and wrap them in braces.
    # To avoid wrapping words that are already wrapped (like {YOLO} or {{YOLO}}), we first strip
    # single-word braces for known words.

    # Let's write a replacement function that processes words
    # Split by space but preserve structure. We can tokenize by words and keep punctuation.
    def replace_word(match):
        word = match.group(0)
        # Strip outer braces if they exist around the word
        clean_word = word.strip("{}")
        clean_lower = clean_word.lower()

        # Check for hyphenated words like YOLO-based or CNN-driven
        parts = clean_lower.split("-")
        new_parts = []
        changed = False
        for part in parts:
            if part in CAPITALIZATION_MAP:
                new_parts.append("{" + CAPITALIZATION_MAP[part] + "}")
                changed = True
            else:
                # Keep original case of the part
                # Find original part by index
                new_parts.append(part)

        if changed:
            # Reconstruct with original casing for non-mapped parts
            # For simplicity, if the whole word matches a capitalised term, return it wrapped
            if clean_lower in CAPITALIZATION_MAP:
                return "{" + CAPITALIZATION_MAP[clean_lower] + "}"
            # Otherwise, reconstruct hyphenated parts
            # Let's map original parts
            orig_parts = word.strip("{}").split("-")
            reconstruct = []
            for i, part in enumerate(parts):
                if part in CAPITALIZATION_MAP:
                    reconstruct.append("{" + CAPITALIZATION_MAP[part] + "}")
                else:
                    reconstruct.append(orig_parts[i])
            return "-".join(reconstruct)

        if clean_lower in CAPITALIZATION_MAP:
            return "{" + CAPITALIZATION_MAP[clean_lower] + "}"
        return word

    # Match alphanumeric words, possibly wrapped in braces, or hyphenated
    pattern = r"\{?[a-zA-Z0-9\-]+\}?"
    return re.sub(pattern, replace_word, text)


def parse_fields(entry_text):
    # Extract fields from entry text.
    # A robust way is to scan for key = value, paying attention to braces.
    # We find fields inside the main braces of the entry.
    # Find the first { which starts the entry content, and trace fields.
    first_brace = entry_text.find("{")
    if first_brace == -1:
        return {}

    content = entry_text[first_brace + 1 :].strip()
    # The last character should be } (or we strip it)
    if content.endswith("}"):
        content = content[:-1].strip()

    # Split by comma but respect nested braces/quotes
    fields = {}

    # First, let's extract the key (which is before the first comma)
    comma_pos = content.find(",")
    if comma_pos == -1:
        return {}

    # Rest of content contains fields
    fields_content = content[comma_pos + 1 :].strip()

    # We parse field = value pairs
    pos = 0
    length = len(fields_content)
    while pos < length:
        # Find next '='
        eq_pos = fields_content.find("=", pos)
        if eq_pos == -1:
            break

        field_name = fields_content[pos:eq_pos].strip().lower()
        # Find value starting after '='
        val_start = eq_pos + 1
        while val_start < length and fields_content[val_start].isspace():
            val_start += 1

        if val_start >= length:
            break

        # Parse value based on starting char (brace { or quote " or raw text)
        val_end = val_start
        if fields_content[val_start] == "{":
            depth = 1
            val_end += 1
            while val_end < length and depth > 0:
                if fields_content[val_end] == "{":
                    depth += 1
                elif fields_content[val_end] == "}":
                    depth -= 1
                val_end += 1
            value = fields_content[val_start:val_end]
        elif fields_content[val_start] == '"':
            # Find next non-escaped quote
            val_end += 1
            while val_end < length:
                if fields_content[val_end] == '"' and fields_content[val_end - 1] != "\\":
                    val_end += 1
                    break
                val_end += 1
            value = fields_content[val_start:val_end]
        else:
            # Raw text value until next comma (respecting nested braces/quotes just in case)
            depth = 0
            in_quote = False
            while val_end < length:
                char = fields_content[val_end]
                if char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                elif char == '"' and fields_content[val_end - 1] != "\\":
                    in_quote = not in_quote
                elif char == "," and depth == 0 and not in_quote:
                    break
                val_end += 1
            value = fields_content[val_start:val_end].strip()

        fields[field_name] = value

        # Advance pos past value and comma
        pos = val_end
        while pos < length and (fields_content[pos].isspace() or fields_content[pos] == ","):
            pos += 1

    return fields


def parse_bibtex(text):
    entries = []
    pos = 0
    length = len(text)
    while True:
        pos = text.find("@", pos)
        if pos == -1:
            break

        match = re.match(r"@(\w+)\s*\{\s*([^,\s]+)\s*,", text[pos:])
        if not match:
            pos += 1
            continue

        entry_type = match.group(1).lower()
        entry_key = match.group(2)

        # Find matching closing brace
        start_idx = pos + match.end()
        depth = 1
        i = start_idx
        while i < length and depth > 0:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1

        entry_text = text[pos:i]

        # Parse fields
        fields = parse_fields(entry_text)

        entries.append({"type": entry_type, "original_key": entry_key, "fields": fields, "original_text": entry_text})
        pos = i

    return entries


def main():
    if not os.path.exists(bib_path):
        print(f"Error: Bibliography file not found at {bib_path}")
        sys.exit(1)

    with open(bib_path, encoding="utf-8") as f:
        content = f.read()

    print(f"Reading {bib_path}...")
    entries = parse_bibtex(content)
    print(f"Parsed {len(entries)} entries.")

    # Clean and normalise entries
    cleaned_entries = []
    seen_identifiers = {}  # DOI -> entry, Title -> entry to deduplicate

    for entry in entries:
        fields = entry["fields"]

        # Get title and clean it
        title_val = fields.get("title", "")

        # Strip outer braces/quotes from values for processing
        def unwrap_val(v):
            if v.startswith("{") and v.endswith("}"):
                return v[1:-1].strip()
            if v.startswith('"') and v.endswith('"'):
                return v[1:-1].strip()
            return v.strip()

        title = unwrap_val(title_val)
        doi = unwrap_val(fields.get("doi", ""))
        url = unwrap_val(fields.get("url", ""))
        author = unwrap_val(fields.get("author", ""))
        year_str = unwrap_val(fields.get("year", ""))

        # Try to find a 4-digit year in the year field or key or title
        year_match = re.search(r"\b\d{4}\b", year_str)
        if year_match:
            year = year_match.group(0)
        else:
            # Fall back to original key year or 2025
            key_year_match = re.search(r"\b(20\d{2}|19\d{2})\b", entry["original_key"])
            year = key_year_match.group(0) if key_year_match else "2025"

        # Clean title for deduplication comparison
        norm_title = re.sub(r"[^a-z0-9]", "", title.lower())
        norm_doi = doi.lower().replace("https://doi.org/", "").replace("http://doi.org/", "").strip("/")

        # Deduplicate check
        is_duplicate = False
        dup_reason = ""
        matched_key = None

        if norm_doi and norm_doi in seen_identifiers:
            is_duplicate = True
            dup_reason = f"DOI {doi}"
            matched_key = norm_doi
        elif norm_title and norm_title in seen_identifiers:
            is_duplicate = True
            dup_reason = f"Title similarity: '{title[:40]}...'"
            matched_key = norm_title
        elif url and url in seen_identifiers:
            is_duplicate = True
            dup_reason = f"URL {url}"
            matched_key = url

        if is_duplicate:
            print(f"Skipping duplicate: {entry['original_key']} (Reason: {dup_reason})")
            # Update fields of the existing entry if this one has more info
            existing_entry = seen_identifiers[matched_key]
            for k, v in fields.items():
                if k not in existing_entry["fields"] or not unwrap_val(existing_entry["fields"][k]):
                    existing_entry["fields"][k] = v
            continue

        # Standardise fields: casing, spacing, and protect acronyms in Title
        new_fields = {}

        # Fields to always strip (not useful in citations, inflate file size)
        strip_fields = {"abstract", "note", "annote", "file", "mendeley-tags", "rating"}

        for k, v in fields.items():
            k_clean = k.strip().lower()
            v_clean = v.strip()

            # Remove empty or placeholders
            if not v_clean or v_clean == "{}" or v_clean == '""':
                continue

            # Strip unwanted fields (abstracts, notes, file paths, etc.)
            if k_clean in strip_fields:
                continue

            # If title, protect acronyms
            if k_clean == "title" or k_clean in ["journal", "booktitle"]:
                unwrapped = unwrap_val(v_clean)
                protected = protect_acronyms_in_text(unwrapped)
                v_clean = "{" + protected + "}"
            elif k_clean == "author":
                # Just make sure author names are wrapped in {}
                unwrapped = unwrap_val(v_clean)
                # Normalize spacing inside authors
                unwrapped = re.sub(r"\s+", " ", unwrapped)
                v_clean = "{" + unwrapped + "}"
            elif k_clean == "year":
                v_clean = "{" + year + "}"
            elif not (v_clean.startswith("{") and v_clean.endswith("}")) and not (
                v_clean.startswith('"') and v_clean.endswith('"')
            ):
                # Wrap in braces if not wrapped
                v_clean = "{" + v_clean + "}"

            new_fields[k_clean] = v_clean

        # Ensure year is in fields
        if "year" not in new_fields:
            new_fields["year"] = "{" + year + "}"

        # Store for processing keys later
        new_entry = {
            "type": entry["type"],
            "original_key": entry["original_key"],
            "fields": new_fields,
            "author_surname": get_primary_author_surname(author),
            "year": year,
            "title_keyword": clean_title_for_key(title),
        }

        cleaned_entries.append(new_entry)

        # Track identifier
        if norm_doi:
            seen_identifiers[norm_doi] = new_entry
        if norm_title:
            seen_identifiers[norm_title] = new_entry
        if url:
            seen_identifiers[url] = new_entry

    # Now generate unique citation keys
    generated_keys = {}
    for entry in cleaned_entries:
        base_key = f"{entry['author_surname']}{entry['year']}{entry['title_keyword']}"
        # Remove any non-alphanumeric chars from the key itself just in case
        base_key = re.sub(r"[^a-zA-Z0-9]", "", base_key)

        # Check uniqueness
        key = base_key
        suffix_char = ord("a")
        while key in generated_keys:
            key = base_key + chr(suffix_char)
            suffix_char += 1
            if suffix_char > ord("z"):
                key = base_key + str(suffix_char - ord("z"))

        generated_keys[key] = entry
        entry["key"] = key

    # Sort entries by key
    cleaned_entries.sort(key=lambda x: x["key"])

    # Write entries to file
    with open(bib_path, "w", encoding="utf-8") as f:
        f.write("% =============================================================================\n")
        f.write("% Horizon Europe CL3 DRS-03 - Project Literature References\n")
        f.write(f"% Total clean sources: {len(cleaned_entries)}\n")
        f.write("% Generated and formatted automatically\n")
        f.write("% =============================================================================\n\n")

        for entry in cleaned_entries:
            f.write(f"@{entry['type']}{{{entry['key']},\n")

            # Print standard fields first, then others, sorted
            std_fields = [
                "author",
                "title",
                "journal",
                "booktitle",
                "year",
                "volume",
                "number",
                "pages",
                "publisher",
                "doi",
                "url",
            ]
            other_fields = sorted([k for k in entry["fields"] if k not in std_fields])

            ordered_keys = [k for k in std_fields if k in entry["fields"]] + other_fields

            for i, k in enumerate(ordered_keys):
                val = entry["fields"][k]
                comma = "," if i < len(ordered_keys) - 1 else ""
                # We align field names
                f.write(f"  {k:<10} = {val}{comma}\n")

            f.write("}\n\n")

    print(f"Successfully cleaned and wrote {len(cleaned_entries)} references to {bib_path}!")

    # Write a mapping of original keys to new keys (useful for renaming PDFs).
    # Use a platform-neutral location and never let this optional step break cleaning.
    try:
        mapping_path = os.environ.get(
            "KEY_MAPPING_PATH", os.path.join(tempfile.gettempdir(), "references_key_mapping.txt")
        )
        with open(mapping_path, "w", encoding="utf-8") as f:
            for entry in cleaned_entries:
                f.write(
                    f"{entry['original_key']} -> {entry['key']} | "
                    f"Title: {unwrap_val(entry['fields'].get('title', ''))}\n"
                )
        print(f"Key mapping written to {mapping_path}")
    except Exception as e:
        print(f"Note: could not write key mapping file ({e}); continuing without it.")


if __name__ == "__main__":
    main()
