from flask import Blueprint, request, jsonify
from .services import rag_pipeline, process_conversational_query, chat_history_manager, get_chat_instance
from rag_retrieval.config.settings import Settings

bp = Blueprint("rag", __name__)

@bp.route("/query", methods=["POST"])
def query():
    data = request.get_json(force=True)
    user_query = data.get("query")
    user_id = data.get("user_id") # ✅ Nhận user_id từ request

    if not user_query:
        return jsonify({"error": "Missing 'query'"}), 400
    if not user_id:
        return jsonify({"error": "Missing 'user_id'"}), 400

    # ✅ BƯỚC MỚI: Xử lý ngữ cảnh hội thoại
    chat_model = get_chat_instance(Settings.CHAT_PROVIDER, Settings.CHAT_MODEL)
    standalone_query = process_conversational_query(user_id, user_query, chat_model)
    print("DEBUGGGGGG")
    print(standalone_query)

    # Chạy pipeline với truy vấn đã được xử lý
    result = rag_pipeline(
        user_query=standalone_query, # ✅ Sử dụng truy vấn độc lập
        multi_n=int(data.get("multi_n", Settings.MULTI_QUERY_N)),
        top_k=int(data.get("top_k", Settings.RERANK_TOPK)),
        alpha=float(data.get("alpha", Settings.HYBRID_ALPHA)),
        weav_host=Settings.WEAVIATE_HOST, weav_port=Settings.WEAVIATE_PORT, weav_grpc=Settings.WEAVIATE_GRPC,
        weav_collection=Settings.WEAVIATE_COLLECTION_NAME,
        chat_conf=(Settings.CHAT_PROVIDER, Settings.CHAT_MODEL),
        reranker_conf=(Settings.RERANKER_PROVIDER, Settings.RERANKER_MODEL)
    )

    # ✅ BƯỚC MỚI: Cập nhật lịch sử chat
    chat_history_manager.add_message(user_id, "user", user_query) # Lưu câu hỏi gốc
    generated_answer = result.get("generated_answer", "Không có câu trả lời.")
    chat_history_manager.add_message(user_id, "assistant", generated_answer)

    return jsonify(result)
# from flask import Blueprint, request, jsonify
# from .services import rag_pipeline
# from rag_retrieval.config.settings import Settings

# bp = Blueprint("rag", __name__)

# @bp.route("/query", methods=["POST"])
# def query():
#     data = request.get_json(force=True)
#     user_query = data.get("query")
#     if not user_query:
#         return jsonify({"error": "Missing 'query'"}), 400
    

#     result = rag_pipeline(
#         user_query,
#         multi_n=int(data.get("multi_n", Settings.MULTI_QUERY_N)),
#         top_k=int(data.get("top_k", Settings.RERANK_TOPK)),
#         alpha=float(data.get("alpha", Settings.HYBRID_ALPHA)),
#         weav_host=Settings.WEAVIATE_HOST, weav_port=Settings.WEAVIATE_PORT, weav_grpc=Settings.WEAVIATE_GRPC,
#         weav_collection=Settings.WEAVIATE_COLLECTION_NAME,
#         chat_conf=(Settings.CHAT_PROVIDER, Settings.CHAT_MODEL),
#         reranker_conf=(Settings.RERANKER_PROVIDER, Settings.RERANKER_MODEL)
#     )
#     return jsonify(result)