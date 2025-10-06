from rag.settings import settings
from rag.weaviate_manager import WeaviateManager
from rag.bge_reranker import Reranker


def expand_queries(query: str, client, n=3):
    """Sinh thêm truy vấn phụ bằng generative module."""
    coll = client.collections.get(settings.COLLECTION_NAME)
    task = f"Generate {n} alternative search queries for: '{query}'. Return one per line."
    res = coll.generate.near_text(query=query, grouped_task=task)
    queries = [q.strip("-• ") for q in res.generated.split("\n") if q.strip()]
    return [query] + queries[:n]


def rag_pipeline(query: str):
    print(f"\n🔹 Query: {query}")
    with WeaviateManager(settings.WEAVIATE_URL) as mgr:
        # 1️⃣ Sinh multi-query
        subqueries = expand_queries(query, mgr.client, settings.MULTI_QUERY_N)
        print(f"🧩 Expanded queries: {subqueries}")

        # 2️⃣ Hybrid retrieval
        candidates = []
        for q in subqueries:
            results = mgr.hybrid_search(
                settings.COLLECTION_NAME,
                q,
                limit=settings.CANDIDATE_POOL,
                alpha=settings.HYBRID_ALPHA,
            )
            candidates.extend(results)

        # Loại trùng uuid
        seen = {}
        for c in candidates:
            seen[c["uuid"]] = c
        candidates = list(seen.values())
        print(f"🔍 Retrieved {len(candidates)} docs")

        # 3️⃣ Rerank
        reranker = Reranker(settings.RERANKER_MODEL)
        ranked = reranker.rerank(query, candidates)
        top_docs = ranked[:settings.RERANK_TOPK]
        print(f"🏆 Top {settings.RERANK_TOPK} results after rerank:")
        for d in top_docs:
            print(f" - {d['properties']['title']} ({d['rerank_score']:.4f})")

        # 4️⃣ Sinh câu trả lời bằng Weaviate Generative
        answer = mgr.generate_answer(
            settings.COLLECTION_NAME,
            query,
            task="Summarize the key insights from the retrieved documents.",
        )
        return {"query": query, "answer": answer, "contexts": [d["properties"] for d in top_docs]}


if __name__ == "__main__":
    query = "What is the Transformer architecture?"
    result = rag_pipeline(query)
    print("\n✅ FINAL ANSWER:\n", result["answer"])
