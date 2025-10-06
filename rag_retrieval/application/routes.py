from flask import Blueprint, request, jsonify
from .services import rag_pipeline
from rag_retrieval.config.settings import Settings

bp = Blueprint("rag", __name__)

@bp.route("/query", methods=["POST"])
def query():
    data = request.get_json(force=True)
    user_query = data.get("query")
    if not user_query:
        return jsonify({"error": "Missing 'query'"}), 400

    result = rag_pipeline(
        user_query,
        multi_n=int(data.get("multi_n", Settings.MULTI_QUERY_N)),
        top_k=int(data.get("top_k", Settings.RERANK_TOPK)),
        alpha=float(data.get("alpha", Settings.HYBRID_ALPHA)),
        weav_host=Settings.WEAVIATE_HOST, weav_port=Settings.WEAVIATE_PORT, weav_grpc=Settings.WEAVIATE_GRPC,
        embedder_conf=(Settings.EMBEDDER_PROVIDER, Settings.EMBED_MODEL),
        chat_conf=(Settings.CHAT_PROVIDER, Settings.CHAT_MODEL),
        reranker_conf=(Settings.RERANKER_PROVIDER, Settings.RERANKER_MODEL)
    )
    return jsonify(result)