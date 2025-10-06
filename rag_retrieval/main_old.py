# rag_app.py
import os
import json
from typing import List, Dict, Any, Tuple
from flask import Flask, request, jsonify
from datetime import datetime
import math

# import your modules (đảm bảo PYTHONPATH chứa project của bạn hoặc cùng thư mục)
from rag_retrieval.model.model_factory import get_embedder, get_chat_model, get_reranker
from rag_retrieval.db.weaviate_db import WeaviateManager, Property, DataType  # assuming weaviate_db.py exports these

# Config mặc định — override bằng env vars
OLLAMA_PROVIDER = os.getenv("EMBEDDER_PROVIDER", "ollama")
OLLAMA_EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
OLLAMA_CHAT_PROVIDER = os.getenv("CHAT_PROVIDER", "ollama")
OLLAMA_CHAT_MODEL = os.getenv("CHAT_MODEL", "llama3.2:3b")  # thay theo config của bạn

RERANKER_PROVIDER = os.getenv("RERANKER_PROVIDER", "bge")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")

WEAVIATE_HOST = os.getenv("WEAVIATE_HOST", "localhost")
WEAVIATE_PORT = int(os.getenv("WEAVIATE_PORT", 8080))
WEAVIATE_GRPC = int(os.getenv("WEAVIATE_GRPC_PORT", 50051))

# Hyperparams
MULTI_QUERY_N = int(os.getenv("MULTI_QUERY_N", 5))
HYBRID_ALPHA = float(os.getenv("HYBRID_ALPHA", 0.5))  # weight of vector score vs bm25 (0..1)
CANDIDATE_POOL = int(os.getenv("CANDIDATE_POOL", 200))  # số candidate lấy trước khi rerank
RERANK_TOPK = int(os.getenv("RERANK_TOPK", 5))

app = Flask(__name__)

# tạo model instances (lazy init)
_embedder = None
_chat_model = None
_reranker = None

def get_embedder_instance():
    global _embedder
    if _embedder is None:
        _embedder = get_embedder(OLLAMA_PROVIDER, OLLAMA_EMBED_MODEL)
    return _embedder

def get_chat_instance():
    global _chat_model
    if _chat_model is None:
        _chat_model = get_chat_model(OLLAMA_CHAT_PROVIDER, OLLAMA_CHAT_MODEL)
    return _chat_model

def get_reranker_instance():
    global _reranker
    if _reranker is None:
        _reranker = get_reranker(RERANKER_PROVIDER, RERANKER_MODEL)
    return _reranker

# --- helper functions ---

def generate_multi_queries(chat_model, user_query: str, n: int = 5) -> List[str]:
    """
    Dùng LLM để generate N alternative/sub-queries.
    Yêu cầu LLM trả về mỗi query trên 1 dòng (newline separated).
    """
    system_prompt = (
        "You are a query rewriting assistant. Given a user query, produce multiple alternative, "
        "concise search queries that help retrieve diverse relevant documents. "
        "Output exactly the queries, one per line, without extra commentary."
    )
    user_prompt = f"User query: {user_query}\n\nGenerate {n} alternative concise search queries (one per line):"
    # gọi model
    out = chat_model.generate(user_prompt, system_prompt=system_prompt)
    # normalise: split into lines, strip empty, limit to n
    lines = []
    for line in out.splitlines():
        line = line.strip(" \t\n\r \"'-")
        if not line:
            continue
        # sometimes model enumerates "1. ..." or "- ..."
        if line and (line[0].isdigit() and (line[1:3] == '.' or line[1] == '.')):
            # remove leading "1." or "1) "
            parts = line.split('.', 1)
            if len(parts) > 1:
                line = parts[1].strip()
        if line.startswith('- '):
            line = line[2:].strip()
        lines.append(line)
        if len(lines) >= n:
            break
    # fallback: if model returned nothing, return original query + simple variations
    if not lines:
        lines = [user_query] + [user_query + f" {i}" for i in range(1, n)]
        lines = lines[:n]
    return lines

def normalize_scores(scores: List[float]) -> List[float]:
    """Min-max normalize list to [0,1]. If all equal, return same list of 1s or 0s (avoid divide 0)."""
    if not scores:
        return []
    mn = min(scores)
    mx = max(scores)
    if math.isclose(mx, mn):
        # if all zeros: return zeros, else ones
        if mx == 0:
            return [0.0 for _ in scores]
        return [1.0 for _ in scores]
    return [(s - mn) / (mx - mn) for s in scores]

def combine_scores(vector_score: float, bm25_score: float, bm25_scale: float, alpha: float) -> float:
    """Combine vector_score (0..1) and bm25_score (raw) into unified score using scaling and alpha."""
    # bm25_scale is max_bm25; scale to [0,1]
    bm25_norm = (bm25_score / bm25_scale) if bm25_scale > 0 else 0.0
    return alpha * (vector_score if vector_score is not None else 0.0) + (1-alpha) * bm25_norm

def candidate_to_doc_obj(hit: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize a hit/dict from manager.search to a doc object with id, text, title, abstract, keywords, vector_score, bm25_score.
    Expect hit contains 'uuid', 'properties', 'score' or 'distance' depending on call.
    """
    uid = hit.get("uuid")
    props = hit.get("properties", {}) or {}
    # ensure fields
    title = props.get("title")
    abstract = props.get("abstract")
    keywords = props.get("keywords")
    text = props.get("text")
    # scores might be in 'score' or 'distance' fields
    score = hit.get("score")
    distance = hit.get("distance")
    # interpret: if distance present, compute vector_score = 1-distance
    vector_score = None
    bm25_score = None
    if distance is not None:
        try:
            vector_score = 1.0 - float(distance)
        except Exception:
            vector_score = None
    if score is not None:
        # ambiguous: could be bm25 score or vector score depending on query_type; caller should pass correctly
        bm25_score = float(score)
    return {
        "id": uid,
        "title": title,
        "abstract": abstract,
        "keywords": keywords,
        "text": text,
        "vector_score": vector_score,
        "bm25_score": bm25_score
    }

# --- main pipeline endpoint ---

@app.route("/query", methods=["POST"])
def rag_query():
    """
    JSON body:
    {
      "query": "your question",
      "multi_n": 5,      # optional
      "top_k": 5,        # optional final top k to return
      "alpha": 0.6       # optional hybrid alpha for vector vs bm25
    }
    """
    data = request.get_json(force=True)
    user_query = data.get("query")
    if not user_query:
        return jsonify({"error": "Missing 'query' parameter"}), 400
    multi_n = int(data.get("multi_n", MULTI_QUERY_N))
    top_k = int(data.get("top_k", RERANK_TOPK))
    alpha = float(data.get("alpha", HYBRID_ALPHA))

    chat = get_chat_instance()
    embedder = get_embedder_instance()
    reranker = get_reranker_instance()

    # 1) Generate multi queries via LLM
    try:
        multi_queries = generate_multi_queries(chat, user_query, n=multi_n)
    except Exception as e:
        return jsonify({"error": f"Failed to generate multi queries: {e}"}), 500

    # 2) For each generated query, run BM25 (on title/abstract/keywords) and vector search
    candidates: Dict[str, Dict[str, Any]] = {}  # map id -> candidate doc object
    bm25_scores_list = []
    vector_scores_list = []

    with WeaviateManager(host=WEAVIATE_HOST, port=WEAVIATE_PORT, grpc_port=WEAVIATE_GRPC) as mgr:
        for q in multi_queries:
            # BM25 (manager.search should return list of hits with 'score' field for bm25)
            try:
                hits_bm25 = mgr.search(collection_name="Papers", query=q, search_type="bm25", limit=50)
            except Exception as e:
                hits_bm25 = []
            # Vector: embed the query
            try:
                q_vec = embedder.get_embedding(q)
                hits_vector = mgr.search(collection_name="Papers", query=q_vec, search_type="vector", limit=50)
            except Exception as e:
                hits_vector = []

            # Normalize and accumulate candidates
            for h in hits_bm25:
                # h expected to have uuid, properties, score
                doc = candidate_to_doc_obj(h)
                # set bm25_score
                if doc["id"] is None:
                    # create pseudo id via content hash
                    doc_id = f"hash_{hash(json.dumps(doc['title'] or '') + str(doc['abstract'] or ''))}"
                    doc["id"] = doc_id
                if doc["bm25_score"] is None and h.get("score") is not None:
                    doc["bm25_score"] = float(h.get("score"))
                # store or update
                existing = candidates.get(doc["id"])
                if existing is None:
                    candidates[doc["id"]] = doc
                else:
                    # keep best bm25_score
                    if doc["bm25_score"] is not None:
                        existing["bm25_score"] = max(existing.get("bm25_score") or 0.0, doc["bm25_score"])
                if doc["bm25_score"] is not None:
                    bm25_scores_list.append(doc["bm25_score"])

            for h in hits_vector:
                doc = candidate_to_doc_obj(h)
                if doc["id"] is None:
                    doc_id = f"hash_{hash(json.dumps(doc['title'] or '') + str(doc['abstract'] or ''))}"
                    doc["id"] = doc_id
                # if vector_score present in hit
                if doc["vector_score"] is None and h.get("score") is not None:
                    # some implementations placed vector score into 'score'
                    try:
                        doc["vector_score"] = float(h.get("score"))
                    except Exception:
                        pass
                existing = candidates.get(doc["id"])
                if existing is None:
                    candidates[doc["id"]] = doc
                else:
                    # keep best vector_score
                    if doc["vector_score"] is not None:
                        prev = existing.get("vector_score")
                        if prev is None or doc["vector_score"] > prev:
                            existing["vector_score"] = doc["vector_score"]
                if doc["vector_score"] is not None:
                    vector_scores_list.append(doc["vector_score"])

    # 3) Combine scores to get unified score (0..1)
    # normalize bm25
    max_bm25 = max(bm25_scores_list) if bm25_scores_list else 0.0
    # normalize vector scores if none in [0,1] then clip
    # For safety, clip vector scores to [0,1]
    for cid, doc in candidates.items():
        if doc.get("vector_score") is not None:
            doc["vector_score"] = max(0.0, min(1.0, doc["vector_score"]))
        if doc.get("bm25_score") is None:
            doc["bm25_score"] = 0.0
        doc["combined_score"] = combine_scores(doc.get("vector_score") or 0.0, doc.get("bm25_score") or 0.0, max_bm25, alpha)

    # select candidate pool top by combined_score
    all_candidates = list(candidates.values())
    all_candidates.sort(key=lambda d: d.get("combined_score", 0.0), reverse=True)
    top_candidates = all_candidates[:CANDIDATE_POOL]

    # 4) Prepare documents list for reranker (string content)
    documents_texts = []
    docs_meta = []
    for doc in top_candidates:
        content_pieces = []
        if doc.get("title"):
            content_pieces.append(doc["title"])
        if doc.get("abstract"):
            content_pieces.append(doc["abstract"])
        if doc.get("text"):
            content_pieces.append(doc["text"][:4000])  # truncate to avoid too long
        full_text = "\n\n".join(content_pieces)
        documents_texts.append(full_text)
        docs_meta.append(doc)

    # 5) Rerank using BGE reranker; returns list of tuples (original_index, score, document_text)
    try:
        rerank_results = reranker.rerank(user_query, documents_texts, top_k=top_k)
    except Exception as e:
        # if reranker fails, fallback to combined_score order
        rerank_results = [(i, docs_meta[i].get("combined_score", 0.0), documents_texts[i]) for i in range(min(len(documents_texts), top_k))]

    # 6) Format final output
    final = []
    for orig_idx, r_score, doc_text in rerank_results:
        meta = docs_meta[orig_idx]
        final.append({
            "id": meta.get("id"),
            "title": meta.get("title"),
            "abstract": meta.get("abstract"),
            "keywords": meta.get("keywords"),
            "bm25_score": meta.get("bm25_score"),
            "vector_score": meta.get("vector_score"),
            "combined_score": meta.get("combined_score"),
            "reranker_score": r_score,
            "snippet": (doc_text[:500] + "...") if doc_text and len(doc_text) > 500 else doc_text
        })

    return jsonify({
        "query": user_query,
        "multi_queries": multi_queries,
        "results": final
    }), 200


if __name__ == "__main__":
    # chạy flask app
    app.run(host="0.0.0.0", port=int(os.getenv("RAG_FLASK_PORT", 5000)), debug=True)
