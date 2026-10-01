"""
Embeds text chunks using a local sentence-transformers model.

This runs fully on-device once the model weights are downloaded (a
one-time download on first use, then no network calls needed). No LLM
API involved, so it's unaffected by Groq's rate limits.

Model choice: all-MiniLM-L6-v2 -- small (~80MB), fast, and a solid
general-purpose default for English text. Good enough for this project's
scale (a few hundred chunks); a larger model like bge-small-en-v1.5 would
give slightly better retrieval quality at the cost of speed, if needed
later.
"""
from functools import lru_cache

from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    """
    Loads the embedding model once and caches it -- downloading it on
    first call (one-time, needs internet), then reusing the same
    in-memory model for every subsequent call in this process.
    """
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embeds a list of strings, returning one embedding vector per string."""
    model = get_embedding_model()
    embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    return embeddings.tolist()


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Adds an "embedding" field to each chunk dict, computed from its "text".
    Batches all chunks into a single encode() call for efficiency.
    """
    texts = [c["text"] for c in chunks]
    embeddings = embed_texts(texts)
    return [{**chunk, "embedding": emb} for chunk, emb in zip(chunks, embeddings)]


if __name__ == "__main__":
    sample_texts = [
        "The MPC voted to keep the repo rate unchanged at 5.25 per cent.",
        "CPI inflation for 2026-27 is projected to be 5.0 per cent.",
    ]
    vectors = embed_texts(sample_texts)
    print(f"Embedded {len(vectors)} texts, dimension: {len(vectors[0])}")
    print(f"First 5 values of embedding 1: {vectors[0][:5]}")