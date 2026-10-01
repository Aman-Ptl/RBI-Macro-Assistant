"""
Runs the RAG eval set against hybrid_retrieve() and measures retrieval
quality -- no LLM calls involved, this tests the retriever directly, so
it costs zero API quota and runs in seconds.

Two metrics, at two strictness levels:
  - Chunk hit-rate@5:   did the exact expected chunk appear in the top 5?
  - Document hit-rate@5: did ANY chunk from the expected source document
                          appear in the top 5? (looser -- a different
                          chunk from the same document is often still a
                          useful result, so this is the more realistic
                          "did it find the right document" measure)
  - MRR (document-level): rewards ranking the right document's chunk
                          higher, not just getting it in the top 5 somewhere

Run: python -m evaluation.run_rag_eval
"""
import json

from src.config import EVAL_DIR
from src.rag_tool.retriever import hybrid_retrieve

RESULTS_PATH = EVAL_DIR.parent.parent / "evaluation" / "results" / "rag_eval_results.json"
TOP_K = 5


def run_eval(top_k: int = TOP_K):
    eval_path = EVAL_DIR / "rag_eval_set.json"
    if not eval_path.exists():
        print(f"Eval set not found: {eval_path}")
        print("Run this first: python -m evaluation.build_rag_eval_set")
        return

    with open(eval_path) as f:
        eval_items = json.load(f)

    results = []
    chunk_hits = 0
    doc_hits = 0
    reciprocal_ranks = []

    for i, item in enumerate(eval_items, 1):
        print(f"[{i}/{len(eval_items)}] {item['question']}")

        retrieved = hybrid_retrieve(item["question"], top_k=top_k)
        retrieved_ids = [r["id"] for r in retrieved]
        retrieved_files = [r["metadata"]["source_file"] for r in retrieved]

        chunk_hit = item["expected_chunk_id"] in retrieved_ids
        doc_hit = item["expected_source_file"] in retrieved_files

        # Document-level reciprocal rank: 1/rank of the first chunk from
        # the expected document, 0 if it never appears.
        rr = 0.0
        for rank, source_file in enumerate(retrieved_files, start=1):
            if source_file == item["expected_source_file"]:
                rr = 1 / rank
                break
        reciprocal_ranks.append(rr)

        if chunk_hit:
            chunk_hits += 1
        if doc_hit:
            doc_hits += 1

        results.append({
            "id": item["id"],
            "question": item["question"],
            "expected_source_file": item["expected_source_file"],
            "expected_section": item["expected_section"],
            "retrieved_sections": [r["metadata"]["section"] for r in retrieved],
            "chunk_hit": chunk_hit,
            "doc_hit": doc_hit,
            "reciprocal_rank": rr,
        })

        status = "CHUNK HIT" if chunk_hit else ("DOC HIT" if doc_hit else "MISS")
        print(f"  [{status}]")

    n = len(eval_items)
    chunk_hit_rate = chunk_hits / n * 100
    doc_hit_rate = doc_hits / n * 100
    mrr = sum(reciprocal_ranks) / n

    print("\n=== RAG EVAL SUMMARY ===")
    print(f"Total questions: {n}")
    print(f"Chunk hit-rate@{top_k}: {chunk_hits}/{n} ({chunk_hit_rate:.1f}%)")
    print(f"Document hit-rate@{top_k}: {doc_hits}/{n} ({doc_hit_rate:.1f}%)")
    print(f"Document-level MRR: {mrr:.3f}")

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump({
            "chunk_hit_rate_pct": round(chunk_hit_rate, 2),
            "doc_hit_rate_pct": round(doc_hit_rate, 2),
            "mrr": round(mrr, 3),
            "top_k": top_k,
            "total_questions": n,
            "results": results,
        }, f, indent=2)
    print(f"\nDetailed results saved -> {RESULTS_PATH}")


if __name__ == "__main__":
    run_eval()