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

    multi_n = int(data["multi_n"]) if "multi_n" in data else Settings.MULTI_QUERY_N
    top_k = int(data["top_k"]) if "top_k" in data else Settings.RERANK_TOPK
    alpha = float(data["alpha"]) if "alpha" in data else Settings.HYBRID_ALPHA

    result = rag_pipeline(
        user_query,
        multi_n=multi_n,
        top_k=top_k,
        alpha=alpha,
        weav_host=Settings.WEAVIATE_HOST,
        weav_port=Settings.WEAVIATE_PORT,
        weav_grpc=Settings.WEAVIATE_GRPC,
        weav_collection=Settings.WEAVIATE_COLLECTION_NAME,
        chat_conf=(Settings.CHAT_PROVIDER, Settings.CHAT_MODEL),
        reranker_conf=(Settings.RERANKER_PROVIDER, Settings.RERANKER_MODEL)
    )

    return jsonify(result)
