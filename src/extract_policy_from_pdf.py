"""One-off helper to extract and clean policy text from a PDF.

This script was used to convert the public overseas travel inconvenience
insurance PDF into `data/policy_clean/policy_clean.txt` for the RAG index.

Usage (from project root, after installing pdfplumber into a venv):

    python -m src.extract_policy_from_pdf \
        --pdf "path/to/policy.pdf" \
        --out "data/policy_clean/policy_clean.txt"

The cleaning here is intentionally simple:
- concatenate page texts with blank lines between pages
- drop completely empty lines
- drop lines that are only digits (likely page numbers)

You can adjust the rules if you work with a different policy format.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pdfplumber


def extract_and_clean(pdf_path: Path, out_path: Path) -> None:
    pdf_path = pdf_path.expanduser().resolve()
    out_path = out_path.expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            lines.append(text)

    raw = "\n\n".join(lines)

    clean_lines: list[str] = []
    for line in raw.splitlines():
        line = line.rstrip()
        if not line.strip():
            # drop empty lines
            continue
        if line.strip().isdigit():
            # drop pure page numbers
            continue
        clean_lines.append(line)

    clean = "\n".join(clean_lines)
    out_path.write_text(clean, encoding="utf-8")
    print(f"[INFO] Wrote cleaned policy to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract & clean policy text from PDF")
    parser.add_argument("--pdf", type=str, required=True, help="Input PDF path")
    parser.add_argument("--out", type=str, required=True, help="Output cleaned .txt path")
    args = parser.parse_args()

    extract_and_clean(Path(args.pdf), Path(args.out))


if __name__ == "__main__":
    main()
