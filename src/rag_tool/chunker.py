"""
Splits cleaned document pages into chunks for embedding.

RBI documents like MPC minutes are numbered-paragraph documents with
section headers ("Resolution", "Statement by Dr. X", etc.). Splitting on
blank lines naturally separates these into paragraph-sized blocks (each
numbered paragraph is one block, since the paragraph number and its text
have no blank line between them). This function groups consecutive
paragraph blocks into chunks close to the target token size, tracking
which section and paragraph range each chunk covers, and rolls that
metadata into every chunk so retrieval results carry it forward.

Pure string/list processing -- no LLM calls, fully offline.
"""
import re

from src.config import CHUNK_SIZE_TOKENS, CHUNK_OVERLAP_TOKENS

PARA_NUMBER_PATTERN = re.compile(r"^(\d{1,3})\.\s")
DATE_LINE_PATTERN = re.compile(r"^[A-Z][a-z]+ \d{1,2},? \d{4}$")  # e.g. "August 19, 2026"


# A block is treated as a section header if it's a single short line with
# no leading paragraph number, no sentence-ending punctuation, and isn't
# just a standalone date (the document's issue date, not a real heading).
def _is_section_header(block: str) -> bool:
    lines = [ln for ln in block.split("\n") if ln.strip()]
    if len(lines) != 1:
        return False
    line = lines[0].strip()
    if PARA_NUMBER_PATTERN.match(line):
        return False
    if DATE_LINE_PATTERN.match(line):
        return False
    if len(line) > 70:
        return False
    if line.endswith((".", ":", ";")):
        return False
    return True


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~1.3 tokens per word for English prose."""
    return int(len(text.split()) * 1.3)


def _blocks_from_pages(cleaned_pages: list[dict]) -> list[dict]:
    """
    Splits each page's cleaned text into blank-line-separated blocks,
    tagging each with its source page number.
    """
    blocks = []
    for page in cleaned_pages:
        for raw_block in page["text"].split("\n\n"):
            block = raw_block.strip()
            if block:
                blocks.append({"page": page["page"], "text": block})
    return blocks


def chunk_document(cleaned_pages: list[dict], doc_metadata: dict) -> list[dict]:
    """
    Chunks a document into ~CHUNK_SIZE_TOKENS pieces, grouped by section,
    with CHUNK_OVERLAP_TOKENS of overlap carried into the next chunk.

    Returns a list of:
      {
        "text": "...",
        "doc_type": "...", "date": "...", "source_file": "...",
        "section": "...", "page_range": "3-4", "para_range": "11-13",
      }
    """
    blocks = _blocks_from_pages(cleaned_pages)

    chunks = []
    current_section = "Preamble"
    current_paras: list[dict] = []  # blocks accumulated for the current chunk
    current_tokens = 0

    def flush_chunk():
        if not current_paras:
            return
        text = "\n\n".join(b["text"] for b in current_paras)
        pages = [b["page"] for b in current_paras]
        para_nums = [
            m.group(1) for b in current_paras
            if (m := PARA_NUMBER_PATTERN.match(b["text"]))
        ]
        chunks.append({
            "text": text,
            "doc_type": doc_metadata["doc_type"],
            "date": doc_metadata["date"],
            "source_file": doc_metadata["source_file"],
            "section": current_section,
            "page_range": f"{min(pages)}-{max(pages)}" if pages else "",
            "para_range": f"{para_nums[0]}-{para_nums[-1]}" if para_nums else "",
        })

    for block in blocks:
        if _is_section_header(block["text"]):
            # A new section starts: flush what we have so the previous
            # chunk doesn't span two sections, then start fresh.
            flush_chunk()
            current_paras = []
            current_tokens = 0
            current_section = block["text"]
            continue

        block_tokens = _estimate_tokens(block["text"])

        if current_tokens + block_tokens > CHUNK_SIZE_TOKENS and current_paras:
            flush_chunk()
            # Carry the last paragraph forward for overlap/continuity.
            overlap_block = current_paras[-1]
            current_paras = [overlap_block]
            current_tokens = _estimate_tokens(overlap_block["text"])

        current_paras.append(block)
        current_tokens += block_tokens

    flush_chunk()
    return chunks


if __name__ == "__main__":
    import sys
    from pathlib import Path

    from src.rag_tool.parser import parse_pdf
    from src.rag_tool.cleaner import clean_document_pages

    if len(sys.argv) < 2:
        print("Usage: python -m src.rag_tool.chunker <pdf_path>")
        sys.exit(1)

    parsed = parse_pdf(Path(sys.argv[1]))
    cleaned = clean_document_pages(parsed["pages"])
    doc_meta = {
        "doc_type": parsed["doc_type"],
        "date": parsed["date"],
        "source_file": parsed["source_file"],
    }
    chunks = chunk_document(cleaned, doc_meta)

    print(f"Generated {len(chunks)} chunks\n")
    for c in chunks[:3]:
        print(f"[{c['section']}] pages {c['page_range']}, paras {c['para_range']}")
        print(c["text"][:200].replace("\n", " "))
        print("---")
