"""
Thin wrapper around ChromaDB for storing chunk embeddings and running
similarity search. ChromaDB persists to disk locally -- no network or
API calls, so this is unaffected by Groq's rate limits.

Embeddings are computed separately (via embedder.py) and passed in
explicitly here, rather than letting Chroma compute them internally.
This keeps the embedding step visible and swappable.
"""
import chromadb

from src.config import CHROMA_DB_DIR

COLLECTION_NAME = "rbi_docs"


def get_client():
    """Returns a persistent Chroma client backed by a local directory."""
    CHROMA_DB_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(CHROMA_DB_DIR))


def get_collection(client=None):
    """Gets (or creates) the collection chunks are stored in."""
    client = client or get_client()
    return client.get_or_create_collection(name=COLLECTION_NAME)


def _chunk_id(chunk: dict, index: int) -> str:
    """
    Builds a stable, unique ID for a chunk so re-ingesting the same
    document overwrites its old chunks instead of duplicating them.
    """
    return f"{chunk['source_file']}::chunk{index}"


def add_chunks(chunks: list[dict], collection=None) -> int:
    """
    Adds embedded chunks (each must have "embedding" and "text") to the
    vector store, along with their metadata for filtering/display later.

    Returns the number of chunks added.
    """
    if not chunks:
        return 0

    collection = collection or get_collection()

    ids = [_chunk_id(c, i) for i, c in enumerate(chunks)]
    documents = [c["text"] for c in chunks]
    embeddings = [c["embedding"] for c in chunks]
    metadatas = [
        {
            "doc_type": c["doc_type"],
            "date": c["date"],
            "source_file": c["source_file"],
            "section": c["section"],
            "page_range": c["page_range"],
            "para_range": c["para_range"],
        }
        for c in chunks
    ]

    collection.upsert(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)
    return len(chunks)


def query(query_embedding: list[float], top_k: int = 5, collection=None) -> list[dict]:
    """
    Runs similarity search and returns the top_k matches as a list of
    {"id": ..., "text": ..., "metadata": {...}, "distance": float} -- lower
    distance means more similar. The id is included so callers (like the
    hybrid retriever) can merge these results with other ranking methods.
    """
    collection = collection or get_collection()

    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)

    matches = []
    for doc_id, text, metadata, distance in zip(
        results["ids"][0], results["documents"][0],
        results["metadatas"][0], results["distances"][0]
    ):
        matches.append({"id": doc_id, "text": text, "metadata": metadata, "distance": distance})
    return matches


def get_all_chunks(collection=None) -> list[dict]:
    """
    Fetches every chunk in the store as {"id", "text", "metadata",
    "embedding"}. Used by BM25 (needs the full corpus for its keyword
    index) and by metadata-filtered dense search (needs embeddings to
    rank a restricted subset manually). Cheap at this project's scale --
    a few hundred chunks.
    """
    collection = collection or get_collection()
    results = collection.get(include=["documents", "metadatas", "embeddings"])

    return [
        {"id": doc_id, "text": text, "metadata": metadata, "embedding": embedding}
        for doc_id, text, metadata, embedding in zip(
            results["ids"], results["documents"],
            results["metadatas"], results["embeddings"]
        )
    ]


if __name__ == "__main__":
    # Structural test with dummy chunks/embeddings (no embedding model
    # needed) to verify the store/query round trip.
    dummy_chunks = [
        {
            "text": "The MPC kept the repo rate unchanged at 5.25 per cent.",
            "doc_type": "MPC Minutes", "date": "2026-08-19",
            "source_file": "mpc_minutes_2026-08-19.pdf",
            "section": "Resolution", "page_range": "1-2", "para_range": "5-6",
            "embedding": [1.0, 0.0, 0.0],
        },
        {
            "text": "CPI inflation is projected at 5.0 per cent for 2026-27.",
            "doc_type": "MPC Minutes", "date": "2026-08-19",
            "source_file": "mpc_minutes_2026-08-19.pdf",
            "section": "Rationale", "page_range": "3-3", "para_range": "13-13",
            "embedding": [0.0, 1.0, 0.0],
        },
    ]

    client = chromadb.EphemeralClient()  # in-memory, for this test only
    collection = client.get_or_create_collection(name="test")

    added = add_chunks(dummy_chunks, collection=collection)
    print(f"Added {added} chunks")

    results = query([0.9, 0.1, 0.0], top_k=2, collection=collection)
    for r in results:
        print(f"distance={r['distance']:.4f} | {r['metadata']['section']} | {r['text'][:60]}")