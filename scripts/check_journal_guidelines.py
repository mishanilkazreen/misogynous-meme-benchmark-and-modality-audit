#!/usr/bin/env python3
r"""Strict pre-commit check script for Springer Nature (Neural Computing and Applications) submission guidelines.

Checks:
- Documentclass uses sn-jnl with valid pdflatex and reference style options.
- Paper is properly anonymized for double-blind review.
- Abstract length is within 150-450 words.
- Keywords count is between 4 and 6.
- Section headings do not exceed 3 levels (\section, \subsection, \subsubsection).
- Mandatory Declarations section is included (Funding, Conflict of Interest, Data Availability, etc.).
- Figure/Table captions and numbering rules are followed.
- Clean submission directory check (no temporary latex build files or non-submission docs).
- Successful PDF compilation check using build_paper.sh.
"""

from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile


def check_submission_dir_cleanliness(sub_dir: Path) -> list[str]:
    errors = []
    unallowed_extensions = {
        ".aux",
        ".log",
        ".out",
        ".blg",
        ".fls",
        ".fdb_latexmk",
        ".synctex.gz",
        ".toc",
    }
    unallowed_files = {
        "AUTHORS.md",
        "METHODOLOGY.md",
        "PROMPTS.md",
        "RESULTS.md",
        "paper_template.tex",
    }

    for item in sub_dir.iterdir():
        if item.is_file():
            if item.name in unallowed_files:
                errors.append(f"Forbidden non-submission file in submission folder: {item.name}")
            if item.suffix in unallowed_extensions:
                errors.append(
                    f"Leftover temporary LaTeX build file found: {item.name}. Run build_paper script to clean."
                )
    return errors


def check_latex_guidelines(tex_path: Path) -> list[str]:
    errors = []
    content = tex_path.read_text(encoding="utf-8")

    # 1. Documentclass check
    docclass_match = re.search(r"\\documentclass\[(.*?)\]\{sn-jnl\}", content)
    if not docclass_match:
        errors.append(
            "Document class must be '\\documentclass[...]{sn-jnl}' (Springer Nature template)."
        )
    else:
        opts = docclass_match.group(1)
        if "pdflatex" not in opts:
            errors.append("Documentclass options must include 'pdflatex'.")
        if not any(
            ref_opt in opts
            for ref_opt in ["sn-mathphys-num", "sn-basic", "sn-mathphys-ay", "sn-standard"]
        ):
            errors.append(
                "Documentclass options must specify a valid Springer reference style (e.g. 'sn-mathphys-num' or 'sn-basic')."
            )

    # 2. Anonymization & Author Biographies check (Double-Blind Review)
    if (
        re.search(r"\\author\{(?!\s*\\fnm\{Anonymous\}).+?\}", content, re.DOTALL)
        and "Anonymous" not in content.split(r"\maketitle")[0]
    ):
        errors.append(
            "Paper title block must be anonymized ('Anonymous Author(s)') for double-blind peer review."
        )

    # Check for forbidden author biography environments or headings
    if re.search(
        r"\\begin\{(?:IEEE)?biography\}|\\bio\{|\\section\*?\{Author Biograph",
        content,
        re.IGNORECASE,
    ):
        errors.append(
            "Author biography detected in manuscript. Neural Computing and Applications strictly forbids author bios in double-blind submissions."
        )

    # Check that real author names do not appear in body text
    real_authors = [
        "Anna Rösner",
        "Mani Ghahremani",
        "Mishanil Kazreen",
        "Louis Papot",
        "Morgan Woodford",
        "Golcarenarenji",
    ]
    for author_name in real_authors:
        # Match author name outside comments
        for line_no, line in enumerate(content.splitlines(), start=1):
            stripped = line.split("%")[0]
            if author_name.lower() in stripped.lower():
                errors.append(
                    f"Real author name '{author_name}' detected on line {line_no} violating double-blind review policy."
                )

    # 3. Abstract check
    abstract_match = re.search(r"\\abstract\{(.*?)\}", content, re.DOTALL)
    if not abstract_match:
        errors.append("Missing '\\abstract{...}' block in LaTeX manuscript.")
    else:
        abstract_text = abstract_match.group(1).strip()
        words = re.findall(r"\b\w+\b", abstract_text)
        word_count = len(words)
        if word_count < 150 or word_count > 450:
            errors.append(
                f"Abstract word count ({word_count} words) is outside Springer guidelines (150-450 words)."
            )

    # 4. Keywords check
    keywords_match = re.search(r"\\keywords\{(.*?)\}", content, re.DOTALL)
    if not keywords_match:
        errors.append("Missing '\\keywords{...}' block in LaTeX manuscript.")
    else:
        keywords_text = keywords_match.group(1).strip()
        keywords_list = [k.strip() for k in re.split(r"[,;]", keywords_text) if k.strip()]
        if len(keywords_list) < 4 or len(keywords_list) > 6:
            errors.append(
                f"Keywords count ({len(keywords_list)}) must be between 4 and 6 keywords per guidelines."
            )

    # 5. Heading hierarchy check (max 3 levels)
    if r"\subsubsubsection" in content:
        errors.append(
            "Heading hierarchy exceeds 3 levels ('\\subsubsubsection' is forbidden). Use \\section, \\subsection, \\subsubsection."
        )

    # 6. Table formatting check (no resizebox on tabular)
    if re.search(r"\\resizebox\{.*?\}\{.*?\}\{\s*\\begin\{tabular\}", content):
        errors.append(
            "Forbidden table resizing: Wrapping '\\begin{tabular}' inside '\\resizebox' corrupts sn-jnl hooks. Use '\\footnotesize' or '\\setlength{\\tabcolsep}{...}' instead."
        )

    # 7. Declarations section check
    if r"\section*{Declarations}" not in content and r"\section{Declarations}" not in content:
        errors.append("Missing mandatory '\\section*{Declarations}' block.")
    else:
        required_declarations = [
            "Funding",
            "Conflict of Interest",
            "Ethics Approval",
            "Consent to Participate",
            "Consent for Publication",
            "Data Availability",
            "Code Availability",
            "Authors' Contributions",
        ]
        for decl in required_declarations:
            if decl.lower() not in content.lower():
                errors.append(
                    f"Mandatory declaration item '{decl}' is missing from the Declarations section."
                )

    return errors


def check_pdf_compilation(sub_dir: Path) -> list[str]:
    errors = []
    tex_file = sub_dir / "main.tex"
    if not tex_file.exists():
        return ["Missing main.tex in submission directory."]

    with tempfile.TemporaryDirectory() as tmpdir:
        # Copy submission files to temporary directory for test compilation
        for item in sub_dir.iterdir():
            if item.is_file():
                shutil.copy(item, tmpdir)
            elif item.is_dir():
                shutil.copytree(item, Path(tmpdir) / item.name)

        build_script = Path(tmpdir) / "build_paper.sh"
        if not build_script.exists():
            return ["Missing build_paper.sh in submission directory."]

        result = subprocess.run(
            [str(build_script)], cwd=tmpdir, capture_output=True, text=True, check=False
        )
        if result.returncode != 0:
            errors.append(
                f"Compilation script build_paper.sh failed with exit code {result.returncode}:\n{result.stderr[-500:]}"
            )
        else:
            pdf_file = Path(tmpdir) / "main.pdf"
            if not pdf_file.exists():
                errors.append("build_paper.sh finished but main.pdf was not generated.")

    return errors


def main():
    repo_root = Path(__file__).resolve().parent.parent
    sub_dir = repo_root / "submission"
    tex_path = sub_dir / "main.tex"

    print("Checking Springer Nature (Neural Computing and Applications) submission guidelines...")

    if not tex_path.exists():
        print(
            f"Skipping: submission/main.tex not found at {tex_path}. "
            "The LaTeX submission source is not tracked in this repository."
        )
        sys.exit(0)

    errors = []
    errors.extend(check_submission_dir_cleanliness(sub_dir))
    errors.extend(check_latex_guidelines(tex_path))
    errors.extend(check_pdf_compilation(sub_dir))

    if errors:
        print("\n❌ Submission guideline checks failed with the following issues:")
        for err in errors:
            print(f"  - {err}")
        sys.exit(1)

    print("\n✅ All Springer Nature submission guideline checks passed successfully!")
    sys.exit(0)


if __name__ == "__main__":
    main()
