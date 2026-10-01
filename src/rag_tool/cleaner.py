"""
Cleans raw text extracted from RBI PDFs: strips the recurring bilingual
letterhead block, standalone page-number lines, the footer signature line,
and any running headers/footers that repeat across most pages of a
document (e.g. "ARTICLE RBI Bulletin March 2026 State of the Economy" on
every page of a Bulletin article). The running-header detection is
document-agnostic -- it works by frequency, not by hardcoded text, so it
applies to any RBI document type without per-type rules.

Pure string processing -- no LLM calls, fully offline.
"""
import re
import unicodedata
from collections import Counter

# Lines containing any of these substrings are boilerplate and get dropped.
BOILERPLATE_MARKERS = (
    "PRESS RELEASE",
    "RESERVE BANK OF INDIA",
    "Department of Communication",
    "Website :",
    "Central Office",
    "Shahid Bhagat Singh Marg",
    "Mumbai - 400 001",
    "फोन/Phone",
    "ई-मेल/email",
    "helpdoc@rbi.org.in",
)

# Footer signature line, e.g. "Press Release: 2026-2027/925    Chief General Manager"
FOOTER_SIGNATURE_PATTERN = re.compile(r"Press Release:\s*\d{4}-\d{4}/\d+")

# A line that is only a page number (e.g. "2", "11"), possibly with whitespace.
PAGE_NUMBER_LINE_PATTERN = re.compile(r"^\s*\d{1,3}\s*$")


def _contains_devanagari(line: str) -> bool:
    """True if the line has a meaningful amount of Devanagari script text."""
    devanagari_chars = sum(
        1 for ch in line if "DEVANAGARI" in unicodedata.name(ch, "")
    )
    return devanagari_chars > 3  # a couple of stray marks don't count


def clean_page_text(raw_text: str) -> str:
    """Cleans the raw text of a single page."""
    lines = raw_text.split("\n")
    kept_lines = []

    for line in lines:
        stripped = line.strip()

        if not stripped:
            kept_lines.append("")  # keep blank lines for now, collapse later
            continue

        if PAGE_NUMBER_LINE_PATTERN.match(stripped):
            continue

        if FOOTER_SIGNATURE_PATTERN.search(stripped):
            continue

        if any(marker in stripped for marker in BOILERPLATE_MARKERS):
            continue

        if _contains_devanagari(stripped):
            continue

        kept_lines.append(stripped)

    cleaned = "\n".join(kept_lines)
    # Collapse 3+ consecutive blank lines down to a single blank line.
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def strip_repeated_lines(
    pages: list[dict], min_pages: int = 3, min_repeat_fraction: float = 0.4
) -> list[dict]:
    """
    Removes lines that repeat identically across a large fraction of pages
    -- these are running headers/footers (document title, section name,
    etc. printed on every page) rather than actual content.

    This is document-agnostic: it works by counting line frequency across
    the whole document instead of matching hardcoded strings, so it
    catches per-document-type headers without needing a rule per type.

    Skipped for very short documents (< min_pages), where a line repeating
    on 2 of 2 pages is more likely to be genuine content than a header.
    """
    if len(pages) < min_pages:
        return pages

    line_counts = Counter()
    for page in pages:
        # Count each distinct line once per page, so a line repeated
        # within one page doesn't inflate its cross-page frequency.
        lines_on_page = {ln.strip() for ln in page["text"].split("\n") if ln.strip()}
        line_counts.update(lines_on_page)

    threshold = max(3, int(len(pages) * min_repeat_fraction))
    repeated_lines = {line for line, count in line_counts.items() if count >= threshold}

    if not repeated_lines:
        return pages

    cleaned_pages = []
    for page in pages:
        kept = [
            ln for ln in page["text"].split("\n")
            if not ln.strip() or ln.strip() not in repeated_lines
        ]
        text = "\n".join(kept)
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        cleaned_pages.append({"page": page["page"], "text": text})

    return cleaned_pages


def clean_document_pages(pages: list[dict]) -> list[dict]:
    """
    Applies clean_page_text to every page, then a second pass to strip
    running headers/footers that repeat across the document.
    """
    cleaned = [{"page": p["page"], "text": clean_page_text(p["text"])} for p in pages]
    cleaned = strip_repeated_lines(cleaned)
    return cleaned


if __name__ == "__main__":
    import sys
    from pathlib import Path

    from src.rag_tool.parser import parse_pdf

    if len(sys.argv) < 2:
        print("Usage: python -m src.rag_tool.cleaner <pdf_path>")
        sys.exit(1)

    result = parse_pdf(Path(sys.argv[1]))
    cleaned_pages = clean_document_pages(result["pages"])

    print("--- Page 1 before ---")
    print(result["pages"][0]["text"][:600])
    print("\n--- Page 1 after ---")
    print(cleaned_pages[0]["text"][:600])