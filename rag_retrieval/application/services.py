import json, math
from typing import List, Dict, Any
from rag_retrieval.db.weaviate_db import WeaviateManager
from rag_retrieval.model.model_factory import get_embedder, get_chat_model, get_reranker

_embedder, _chat_model, _reranker = None, None, None

def get_embedder_instance(provider, model):
    global _embedder
    if _embedder is None:
        _embedder = get_embedder(provider=provider, model_name=model)
    return _embedder

def get_chat_instance(provider, model):
    global _chat_model
    if _chat_model is None:
        _chat_model = get_chat_model(provider=provider, model_name=model)
    return _chat_model

def get_reranker_instance(provider, model):
    global _reranker
    if _reranker is None:
        _reranker = get_reranker(provider=provider, model_name=model)
    return _reranker

# --- helpers ---
def generate_multi_queries(chat_model, user_query: str, n: int = 5) -> List[str]:
    system_prompt = (
        "You are a query rewriting assistant. Given a user query, produce multiple alternative, "
        "concise search queries that help retrieve diverse relevant documents. "
        "Output exactly the queries, one per line."
    )
    user_prompt = f"User query: {user_query}\n\nGenerate {n} alternative queries:"
    out = chat_model.generate(user_prompt, system_prompt=system_prompt)

    lines = []
    for line in out.splitlines():
        line = line.strip(" \t\n\r \"'-")
        if not line:
            continue
        if line[0].isdigit() and (line[1] in ['.', ')']):
            line = line.split('.', 1)[-1].strip()
        if line.startswith('- '):
            line = line[2:].strip()
        lines.append(line)
        if len(lines) >= n:
            break
    if not lines:
        lines = [user_query] + [user_query + f" {i}" for i in range(1, n)]
    return lines[:n]

def combine_scores(vector_score, bm25_score, bm25_max, alpha: float) -> float:
    bm25_norm = (bm25_score / bm25_max) if bm25_max > 0 else 0.0
    return alpha * (vector_score or 0.0) + (1 - alpha) * bm25_norm

def candidate_to_doc_obj(hit: Dict[str, Any]) -> Dict[str, Any]:
    uid = hit.get("uuid")
    props = hit.get("properties", {}) or {}
    return {
        "id": uid,
        "title": props.get("title"),
        "abstract": props.get("abstract"),
        "keywords": props.get("keywords"),
        "text": props.get("text"),
        "bm25_score": hit.get("score"),
        "vector_score": 1 - hit["distance"] if "distance" in hit else None,
    }

# --- main RAG pipeline ---
def rag_pipeline(user_query: str, multi_n: int, top_k: int, alpha: float,
                 weav_host: str, weav_port: int, weav_grpc: int,
                 embedder_conf: tuple, chat_conf: tuple, reranker_conf: tuple) -> Dict[str, Any]:

    chat = get_chat_instance(*chat_conf)
    embedder = get_embedder_instance(*embedder_conf)
    reranker = get_reranker_instance(*reranker_conf)

    multi_queries = generate_multi_queries(chat, user_query, n=multi_n)

    candidates: Dict[str, Dict[str, Any]] = {}
    bm25_scores, vector_scores = [], []

    with WeaviateManager(host=weav_host, port=weav_port, grpc_port=weav_grpc) as mgr:
        for q in multi_queries:
            # BM25
            try:
                hits_bm25 = mgr.search("Papers", q, "bm25", limit=50)
            except:
                hits_bm25 = []
            # Vector
            try:
                q_vec = embedder.get_embedding(q)
                hits_vec = mgr.search("Papers", q_vec, "vector", limit=50)
            except:
                hits_vec = []

            for h in hits_bm25 + hits_vec:
                doc = candidate_to_doc_obj(h)
                if not doc["id"]:
                    doc["id"] = f"hash_{hash(json.dumps(doc))}"
                existing = candidates.get(doc["id"])
                if existing:
                    if doc["bm25_score"]:
                        existing["bm25_score"] = max(existing.get("bm25_score", 0), doc["bm25_score"])
                    if doc["vector_score"]:
                        existing["vector_score"] = max(existing.get("vector_score", 0), doc["vector_score"])
                else:
                    candidates[doc["id"]] = doc
                if doc["bm25_score"]: bm25_scores.append(doc["bm25_score"])
                if doc["vector_score"]: vector_scores.append(doc["vector_score"])

    max_bm25 = max(bm25_scores) if bm25_scores else 0
    for d in candidates.values():
        d["bm25_score"] = d.get("bm25_score") or 0
        d["vector_score"] = max(0.0, min(1.0, d.get("vector_score") or 0))
        d["combined_score"] = combine_scores(d["vector_score"], d["bm25_score"], max_bm25, alpha)

    top_candidates = sorted(candidates.values(), key=lambda x: x["combined_score"], reverse=True)[:200]

    docs_text = []
    for doc in top_candidates:
        parts = [doc.get("title") or "", doc.get("abstract") or "", (doc.get("text") or "")[:4000]]
        docs_text.append("\n\n".join([p for p in parts if p.strip()]))

    try:
        rerank_results = reranker.rerank(user_query, docs_text, top_k=top_k)
    except:
        rerank_results = [(i, d["combined_score"], docs_text[i]) for i, d in enumerate(top_candidates[:top_k])]

    final = []
    for idx, score, txt in rerank_results:
        meta = top_candidates[idx]
        final.append({
            "id": meta["id"],
            "title": meta["title"],
            "abstract": meta["abstract"],
            "keywords": meta["keywords"],
            "bm25_score": meta["bm25_score"],
            "vector_score": meta["vector_score"],
            "combined_score": meta["combined_score"],
            "reranker_score": score,
            "snippet": txt[:500] + "..." if txt and len(txt) > 500 else txt,
        })

    return {"query": user_query, "multi_queries": multi_queries, "results": final}
