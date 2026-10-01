"""
Runs the full ingestion pipeline over every PDF in data/raw_pdfs/:
parse -> clean -> chunk -> embed -> store in the vector database.

Safe to re-run: add_chunks() upserts by a stable chunk ID (source file +
chunk index), so re-ingesting a document overwrites its old chunks
instead of duplicating them.

Usage:
  python -m scripts.ingest_pdfs
"""
from pathlib import Path

from src.config import RAW_PDF_DIR
from src.rag_tool.parser import parse_pdf
from src.rag_tool.cleaner import clean_document_pages
from src.rag_tool.chunker import chunk_document
from src.rag_tool.embedder import embed_chunks
from src.rag_tool.vectorstore import add_chunks, get_collection


def ingest_all_pdfs():
    pdf_paths = sorted(RAW_PDF_DIR.glob("*.pdf"))
    if not pdf_paths:
        print(f"No PDFs found in {RAW_PDF_DIR}")
        return

    collection = get_collection()
    total_chunks = 0

    for pdf_path in pdf_paths:
        print(f"Processing {pdf_path.name}...")

        parsed = parse_pdf(pdf_path)
        cleaned_pages = clean_document_pages(parsed["pages"])

        doc_metadata = {
            "doc_type": parsed["doc_type"],
            "date": parsed["date"],
            "source_file": parsed["source_file"],
        }
        chunks = chunk_document(cleaned_pages, doc_metadata)

        if not chunks:
            print(f"  Warning: no chunks generated for {pdf_path.name}")
            continue

        chunks_with_embeddings = embed_chunks(chunks)
        added = add_chunks(chunks_with_embeddings, collection=collection)

        print(f"  {len(parsed['pages'])} pages -> {added} chunks stored")
        total_chunks += added

    print(f"\nDone. {len(pdf_paths)} PDFs processed, {total_chunks} chunks in the vector store.")


if __name__ == "__main__":
    ingest_all_pdfs()
