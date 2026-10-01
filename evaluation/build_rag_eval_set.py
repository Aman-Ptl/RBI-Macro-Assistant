"""
Auto-generates a RAG retrieval evaluation set by sampling real chunks from
the vector store (not inventing questions by hand), similar in spirit to
build_eval_set.py for the SQL tool.

Unlike the SQL eval (which checks an exact numeric value), there's no
single "correct number" for a document chunk, so this measures retrieval
hit-rate instead: for a question built from a chunk's own metadata, does
hybrid_retrieve() actually surface that chunk (or at least the right
source document) in its top-k results?

This needs no LLM calls at all -- it tests the retriever directly, so it
runs in seconds and costs zero API quota.

Run: python -m evaluation.build_rag_eval_set
"""
import json
import random

from src.config import EVAL_DIR
from src.rag_tool.vectorstore import get_all_chunks

RANDOM_SEED = 42
CHUNKS_PER_DOCUMENT = 3  # how many eval questions to generate per source file

MONTH_NUM_TO_NAME = {
    "01": "January", "02": "February", "03": "March", "04": "April",
    "05": "May", "06": "June", "07": "July", "08": "August",
    "09": "September", "10": "October", "11": "November", "12": "December",
}


def _natural_date(iso_date: str) -> str:
    """
    Converts a metadata date ("2026-04", "2026-08-19", or "2025") into
    the "Month YYYY" phrasing extract_date_hint() actually recognizes
    (e.g. "April 2026"). Without this, questions built straight from raw
    ISO dates ("2026-04") never trigger the date filter, since the
    retriever's date-hint parser expects month names, not numbers -- the
    same "wrong month surfaces above the right one" problem the SQL tool
    hit before the unit/date fixes there.
    Falls back to the raw string for year-only dates (no month to name).
    """
    parts = iso_date.split("-")
    if len(parts) >= 2:
        year, month = parts[0], parts[1]
        month_name = MONTH_NUM_TO_NAME.get(month)
        if month_name:
            return f"{month_name} {year}"
    return iso_date


def _build_question(chunk: dict) -> str:
    """
    Turns a chunk's metadata into a natural-language question whose answer
    should live in that chunk. A few common MPC Minutes sections get a
    more natural phrasing; everything else falls back to a generic template
    that still works for any document type/section.
    """
    meta = chunk["metadata"]
    doc_type = meta["doc_type"]
    date = _natural_date(meta["date"])
    section = meta["section"]

    if section == "Resolution":
        return f"What did the MPC decide at its {date} meeting?"

    if section == "Rationale for Monetary Policy Decisions":
        return f"Why did the MPC make its policy decision in {date}?"

    if section.startswith("Statement by "):
        member = section.replace("Statement by ", "")
        return f"What did {member} say in the {date} MPC Minutes?"

    return f"What does the {date} {doc_type} say in its '{section}' section?"


def build_rag_eval_set(chunks_per_document: int = CHUNKS_PER_DOCUMENT, seed: int = RANDOM_SEED):
    random.seed(seed)

    all_chunks = get_all_chunks()
    if not all_chunks:
        print("No chunks found in the vector store. Run scripts.ingest_pdfs first.")
        return []

    # Group chunks by source file so every document gets fair representation,
    # not just whichever document happened to produce the most chunks.
    chunks_by_file = {}
    for chunk in all_chunks:
        chunks_by_file.setdefault(chunk["metadata"]["source_file"], []).append(chunk)

    eval_items = []
    item_id = 1

    for source_file, chunks in sorted(chunks_by_file.items()):
        sample = random.sample(chunks, min(chunks_per_document, len(chunks)))

        for chunk in sample:
            question = _build_question(chunk)
            eval_items.append({
                "id": f"r{item_id:03d}",
                "question": question,
                "expected_chunk_id": chunk["id"],
                "expected_source_file": source_file,
                "expected_section": chunk["metadata"]["section"],
            })
            item_id += 1

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    out_path = EVAL_DIR / "rag_eval_set.json"
    with open(out_path, "w") as f:
        json.dump(eval_items, f, indent=2)

    print(f"{len(eval_items)} RAG eval questions generated across {len(chunks_by_file)} documents -> {out_path}")
    return eval_items


if __name__ == "__main__":
    build_rag_eval_set()