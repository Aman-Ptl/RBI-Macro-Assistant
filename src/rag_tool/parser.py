"""
Extracts page-wise text from RBI PDFs and infers document metadata (type,
date) from the filename, based on the naming convention used in
data/raw_pdfs/ (e.g. mpc_minutes_2026-08-19.pdf, bulletin_2026-07.pdf).

This is a pure text-extraction step -- no LLM calls, so it can be built and
tested fully offline.
"""
import re
from pathlib import Path

import pymupdf

# Maps a filename prefix to a human-readable document type label.
DOC_TYPE_PREFIXES = {
    "mpc_minutes": "MPC Minutes",
    "bulletin": "RBI Bulletin",
    "soe": "State of the Economy",
    "bop": "Balance of Payments",
    "mpr": "Monetary Policy Report",
    "fsr": "Financial Stability Report",
}

# Matches a full date (YYYY-MM-DD) or a year-month (YYYY-MM) in the filename.
# Accepts either "-" or "_" as the separator, since real-world downloaded
# filenames aren't always consistent (e.g. "soe_2026_03", "soe-2026_04").
DATE_PATTERN = re.compile(r"(\d{4})[-_](\d{2})(?:[-_](\d{2}))?")


def infer_metadata_from_filename(pdf_path: Path) -> dict:
    """
    Infers doc_type and date from a filename like 'mpc_minutes_2026-08-19.pdf'
    or 'bulletin_2026-07.pdf'. Falls back to sensible defaults if the name
    doesn't match the expected convention.
    """
    stem = pdf_path.stem  # filename without extension

    doc_type = "unknown"
    for prefix, label in DOC_TYPE_PREFIXES.items():
        if stem.startswith(prefix):
            doc_type = label
            break

    date_match = DATE_PATTERN.search(stem)
    if date_match:
        year, month, day = date_match.groups()
        date_str = f"{year}-{month}-{day}" if day else f"{year}-{month}"
    else:
        # Fall back to a standalone 4-digit year (e.g. "mpr_2025.pdf",
        # which only has a year, no month) instead of giving up entirely.
        year_only_match = re.search(r"(19|20)\d{2}", stem)
        date_str = year_only_match.group(0) if year_only_match else "unknown"

    return {
        "source_file": pdf_path.name,
        "doc_type": doc_type,
        "date": date_str,
    }


def extract_pdf_pages(pdf_path: Path) -> list[dict]:
    """
    Returns a list of {"page": page_number, "text": raw_text} for every
    page in the PDF. page_number is 1-indexed to match how humans refer
    to pages.
    """
    pages = []
    with pymupdf.open(pdf_path) as doc:
        for i, page in enumerate(doc, start=1):
            pages.append({"page": i, "text": page.get_text()})
    return pages


def parse_pdf(pdf_path: Path) -> dict:
    """
    Full parse of one PDF: metadata + page-wise raw text.

    Returns:
      {
        "source_file": "...", "doc_type": "...", "date": "...",
        "pages": [{"page": 1, "text": "..."}, ...]
      }
    """
    pdf_path = Path(pdf_path)
    metadata = infer_metadata_from_filename(pdf_path)
    pages = extract_pdf_pages(pdf_path)
    return {**metadata, "pages": pages}


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m src.rag_tool.parser <pdf_path>")
        sys.exit(1)

    result = parse_pdf(Path(sys.argv[1]))
    print(f"Source: {result['source_file']}")
    print(f"Doc type: {result['doc_type']}")
    print(f"Date: {result['date']}")
    print(f"Pages: {len(result['pages'])}")
    print("\n--- First page preview ---")
    print(result["pages"][0]["text"][:500])
