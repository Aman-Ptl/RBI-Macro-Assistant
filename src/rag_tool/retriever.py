"""
Retrieves the most relevant chunks for a question using hybrid search:
dense (embedding similarity) + BM25 (keyword overlap), combined with
Reciprocal Rank Fusion (RRF). Optionally narrows the search to a
specific date first, when the question mentions one.

Why hybrid: a small embedding model (MiniLM-L6) can miss the most
relevant chunk when the query and the answer use different wording, even
though the topic is the same. BM25 catches the cases where the query's
exact words appear directly in the target chunk.

Why date filtering: with many documents covering similar recurring topics
(MPC minutes from different months, RBI Bulletins, Monetary Policy
Reports all discussing inflation/growth in similar language), a generic
query like "why did the MPC keep rates unchanged" can rank chunks from
the WRONG month above the right one -- the corpus has too many similar-
sounding candidates for search alone to disambiguate. If the question
names a date, narrowing the candidate pool to that date first removes
that ambiguity entirely, the same way the SQL tool narrows to a specific
row with a WHERE clause instead of scanning the whole table.

No LLM calls anywhere in this file -- only a local embedding model and
local keyword ranking, so it's unaffected by Groq's rate limits.
"""
import re

import numpy as np
from rank_bm25 import BM25Okapi

from src.config import TOP_K
from src.rag_tool.embedder import embed_texts
from src.rag_tool.vectorstore import query as dense_query
from src.rag_tool.vectorstore import get_all_chunks

RRF_K = 60  # standard smoothing constant for Reciprocal Rank Fusion
DENSE_CANDIDATES = 20  # how many candidates each method contributes before fusion
BM25_CANDIDATES = 20

MONTH_NAMES = {
    "january": "01", "february": "02", "march": "03", "april": "04",
    "may": "05", "june": "06", "july": "07", "august": "08",
    "september": "09", "october": "10", "november": "11", "december": "12",
}
# Matches "August 2026" style mentions in a question.
MONTH_YEAR_PATTERN = re.compile(
    r"\b(" + "|".join(MONTH_NAMES) + r")\s+(\d{4})\b", re.IGNORECASE
)


def _tokenize(text: str) -> list[str]:
    """Simple lowercase word tokenizer -- good enough for BM25 over English text."""
    return re.findall(r"[a-z0-9]+", text.lower())


def extract_date_hint(query_text: str) -> str | None:
    """
    Looks for a "Month YYYY" mention in the question and converts it to a
    "YYYY-MM" prefix that can be matched against chunk metadata dates.
    Returns None if no such mention is found -- callers should then
    search the whole corpus rather than filtering.
    """
    match = MONTH_YEAR_PATTERN.search(query_text)
    if not match:
        return None
    month_name, year = match.groups()
    return f"{year}-{MONTH_NAMES[month_name.lower()]}"


def _cosine_similarity_rank(query_embedding: list[float], chunks: list[dict], top_k: int) -> list[dict]:
    """Ranks a (usually small, pre-filtered) set of chunks by cosine similarity."""
    if not chunks:
        return []

    query_vec = np.array(query_embedding)
    chunk_vecs = np.array([c["embedding"] for c in chunks])

    # Cosine similarity: dot product over the product of norms.
    similarities = chunk_vecs @ query_vec / (
        np.linalg.norm(chunk_vecs, axis=1) * np.linalg.norm(query_vec) + 1e-8
    )
    ranked_indices = np.argsort(-similarities)[:top_k]
    return [chunks[i] for i in ranked_indices]


def dense_retrieve(query_text: str, top_k: int = DENSE_CANDIDATES, date_prefix: str | None = None) -> list[dict]:
    """Embedding-similarity search, optionally restricted to chunks whose date starts with date_prefix."""
    query_embedding = embed_texts([query_text])[0]

    if date_prefix:
        candidates = [c for c in get_all_chunks() if c["metadata"]["date"].startswith(date_prefix)]
        return _cosine_similarity_rank(query_embedding, candidates, top_k)

    return dense_query(query_embedding, top_k=top_k)


def bm25_retrieve(query_text: str, top_k: int = BM25_CANDIDATES, date_prefix: str | None = None) -> list[dict]:
    """
    Keyword search, optionally restricted to chunks whose date starts
    with date_prefix. Builds the BM25 index fresh from the (possibly
    filtered) corpus on every call -- fine at this project's scale.
    """
    all_chunks = get_all_chunks()
    if date_prefix:
        all_chunks = [c for c in all_chunks if c["metadata"]["date"].startswith(date_prefix)]
    if not all_chunks:
        return []

    tokenized_corpus = [_tokenize(c["text"]) for c in all_chunks]
    bm25 = BM25Okapi(tokenized_corpus)

    scores = bm25.get_scores(_tokenize(query_text))
    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

    return [all_chunks[i] for i in ranked_indices[:top_k]]


def hybrid_retrieve(query_text: str, top_k: int = TOP_K, date_filter: str | None = None) -> list[dict]:
    """
    Combines dense and BM25 rankings via Reciprocal Rank Fusion.

    date_filter: an explicit "YYYY-MM" (or "YYYY") prefix to restrict the
    search to. If not given, it's auto-detected from the question text
    (e.g. "in August 2026" -> "2026-08"); if none is found either, the
    whole corpus is searched.
    """
    if date_filter is None:
        date_filter = extract_date_hint(query_text)

    dense_results = dense_retrieve(query_text, date_prefix=date_filter)
    bm25_results = bm25_retrieve(query_text, date_prefix=date_filter)

    scores: dict[str, float] = {}
    chunk_by_id: dict[str, dict] = {}

    for rank, chunk in enumerate(dense_results, start=1):
        scores[chunk["id"]] = scores.get(chunk["id"], 0.0) + 1 / (RRF_K + rank)
        chunk_by_id[chunk["id"]] = chunk

    for rank, chunk in enumerate(bm25_results, start=1):
        scores[chunk["id"]] = scores.get(chunk["id"], 0.0) + 1 / (RRF_K + rank)
        chunk_by_id.setdefault(chunk["id"], chunk)

    ranked_ids = sorted(scores, key=lambda i: scores[i], reverse=True)[:top_k]
    return [chunk_by_id[i] for i in ranked_ids]


# Public API used by the agent's RAG tool later.
retrieve = hybrid_retrieve


if __name__ == "__main__":
    import sys

    question = sys.argv[1] if len(sys.argv) > 1 else "Why did the MPC keep the repo rate unchanged in August 2026?"
    detected_date = extract_date_hint(question)
    if detected_date:
        print(f"(detected date filter: {detected_date})\n")

    results = hybrid_retrieve(question)

    print(f"Query: {question}\n")
    for i, r in enumerate(results, 1):
        meta = r["metadata"]
        print(f"[{i}] {meta['doc_type']} ({meta['date']}) | {meta['section']}")
        print(f"    {r['text'][:200]}")
        print()