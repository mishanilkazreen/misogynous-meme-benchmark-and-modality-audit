import re
import sys

with open("submission/main.tex") as f:
    tex = f.read()

# Ignore standard terms that do not require acronym usage rules
IGNORED = {
    "AI",
    "API",
    "CPU",
    "GPU",
    "JSON",
    "LGBTQ",
    "ML",
    "OCR",
    "PDF",
    "RAM",
    "RGB",
    "SVM",
    "TSV",
    "URL",
    "USA",
    "VLM",
}

# Find definitions like "Full Name (ACRONYM)"
matches = re.findall(r"([A-Z][a-zA-Z\s\-]+)\s*\(([A-Z0-9]{2,10})\)", tex)

issues = []
for full, acr in matches:
    if acr in IGNORED:
        continue
    # Count occurrences of exact word ACRONYM
    occurrences = len(re.findall(r"\b" + re.escape(acr) + r"\b", tex))
    if occurrences < 2:
        issues.append(
            f'Acronym "{acr}" ({full.strip()}) defined but only appears {occurrences} time(s).'
        )

if issues:
    print("Acronym check failed:")
    for issue in set(issues):
        print(" -", issue)
    sys.exit(1)
else:
    print("Acronym check passed (all defined acronyms are used at least twice).")
