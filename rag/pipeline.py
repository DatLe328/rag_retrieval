from .settings import settings
from .weaviate_manager import WeaviateManager
from .bge_reranker import BGEReranker


def expand_queries(query: str, client, n=3):
    """Sinh nhiều query phụ bằng Weaviate Generative module."""
    coll = client.collections.get(settings.COLLECTION_NAME)
    task = f"Generate {n} alternative search queries for: '{query}'. Return one per line."
    res = coll.generate.near_text(query=query, grouped_task=task)
    queries = [q.strip("-• ") for q in res.generated.split("\n") if q.strip()]
    return [query] + queries[:n]


def rag_pipeline(user_query: str):
    print(f"\n🔹 Query: {user_query}")

    with WeaviateManager(settings.WEAVIATE_URL) as mgr:
        # 1️⃣ Multi-query expansion
        subqueries = expand_queries(user_query, mgr.client, n=settings.MULTI_QUERY_N)
        print(f"🧩 Expanded queries: {subqueries}")

        # 2️⃣ Hybrid retrieval
        candidates = []
        for q in subqueries:
            results = mgr.hybrid_search(settings.COLLECTION_NAME, q, limit=settings.CANDIDATE_POOL, alpha=settings.HYBRID_ALPHA)
            candidates.extend(results)

        # loại trùng uuid
        seen = {}
        for c in candidates:
            seen[c["uuid"]] = c
        candidates = list(seen.values())

        print(f"🔍 Retrieved {len(candidates)} candidates")

        # 3️⃣ Rerank
        reranker = BGEReranker(settings.RERANKER_MODEL)
        ranked = reranker.rerank(user_query, candidates)
        top_docs = ranked[:settings.RERANK_TOPK]
        print(f"🏆 Top {settings.RERANK_TOPK}:")
        for d in top_docs:
            print(f" - {d['properties']['title']} ({d['rerank_score']:.4f})")

        # 4️⃣ Generative answer
        answer = mgr.generate_answer(settings.COLLECTION_NAME, user_query, task="Summarize the main ideas of the retrieved documents.")
        return {"query": user_query, "answer": answer, "contexts": [d["properties"] for d in top_docs]}
