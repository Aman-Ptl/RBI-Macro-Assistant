"""
Diagnostic tool: lists every chunk stored for a given document (matched by
a substring of its filename), in order, so you can directly inspect
whether a document's content was chunked and labeled as expected --
without going through search/ranking at all.

Usage:
  python -m scripts.inspect_chunks mpc_minutes_2026-08-19
"""
import sys

from src.rag_tool.vectorstore import get_all_chunks


def inspect_document(filename_substring: str):
    all_chunks = get_all_chunks()
    matches = [c for c in all_chunks if filename_substring in c["metadata"]["source_file"]]

    if not matches:
        print(f"No chunks found with source_file containing '{filename_substring}'")
        return

    # Chunk IDs look like "<source_file>::chunkN" -- sort by N so the
    # output reads in document order.
    def chunk_index(chunk):
        return int(chunk["id"].rsplit("chunk", 1)[-1])

    matches.sort(key=chunk_index)

    print(f"{len(matches)} chunks found for '{filename_substring}':\n")
    for c in matches:
        meta = c["metadata"]
        print(f"[{chunk_index(c)}] section='{meta['section']}' "
              f"pages={meta['page_range']} paras={meta['para_range']}")
        print(f"    {c['text'][:150].replace(chr(10), ' ')}")
        print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m scripts.inspect_chunks <filename_substring>")
        sys.exit(1)

    inspect_document(sys.argv[1])
    