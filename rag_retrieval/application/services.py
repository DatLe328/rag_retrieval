from typing import List, Dict, Any
from rag_retrieval.db.weaviate_db import WeaviateManager
from rag_retrieval.model.model_factory import get_chat_model, get_reranker
import time
import warnings
import requests
warnings.filterwarnings("ignore", message=".*XLMRobertaTokenizerFast.*")
warnings.filterwarnings("ignore", message=".*swigvarlink.*")
warnings.filterwarnings("ignore", message=".*torch_dtype.*deprecated.*")

_chat_model, _reranker = None, None


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

def get_embedding_ollama(text: str, model_name: str = "nomic-embed-text") -> List[float]:
    try:
        response = requests.post(
            "http://10.1.1.237:11434/api/embeddings", # Thay bằng URL Ollama của bạn
            json={"model": model_name, "prompt": text},
        )
        response.raise_for_status()
        return response.json()["embedding"]
    except Exception:
        return []
        
def generate_multi_queries(chat_model, user_query: str, n: int = 5) -> List[str]:
    system_prompt = (
        "You are a query rewriting assistant. Given a user query, produce multiple alternative, "
        "concise search queries that help retrieve diverse relevant documents. "
        "Consider technical, practical, comparative, and conceptual perspectives."
        "Output exactly one query per line with no additional text, only query."
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
    
    if n == 1:
        res = [user_query]
    else:
        res = lines[:n - 1]
        res.append(user_query)

    return res


def rag_pipeline(user_query: str, multi_n: int, top_k: int, alpha: float,
                 weav_host: str, weav_port: int, weav_grpc: int,
                 weav_collection: str,
                 chat_conf: tuple, reranker_conf: tuple) -> Dict[str, Any]:

    # --- 1. KHỞI TẠO BÁO CÁO VÀ BẮT ĐẦU ĐO THỜI GIAN ---
    start_time = time.monotonic()
    report = {
        "timings_ms": {},
        "statistics": {},
        "parameters": {
            "user_query": user_query,
            "multi_n": multi_n,
            "top_k": top_k,
            "alpha": alpha,
            "reranker_model": reranker_conf[1] # Lưu lại URL/tên model reranker
        },
        "intermediate_steps": {}
    }

    chat = get_chat_instance(*chat_conf)
    reranker = get_reranker_instance(*reranker_conf)

    # --- 2. BƯỚC TẠO TRUY VẤN CON (QUERY GENERATION) ---
    print("Generating quries...")
    query_gen_start = time.monotonic()
    multi_queries = generate_multi_queries(chat, user_query, n=multi_n)
    query_gen_end = time.monotonic()
    report["timings_ms"]["query_generation"] = round((query_gen_end - query_gen_start) * 1000)
    report["statistics"]["num_generated_queries"] = len(multi_queries)
    report["intermediate_steps"]["generated_queries"] = multi_queries
    print("Generate completed")

    # --- 3. BƯỚC TRUY XUẤT ỨNG VIÊN (CANDIDATE RETRIEVAL) ---
    print("Start query...")
    retrieval_start = time.monotonic()
    candidates: Dict[str, Dict[str, Any]] = {}
    initial_candidate_count = 0
    with WeaviateManager(host=weav_host, http_port=weav_port) as mgr:
        all_queries = [user_query] + multi_queries
        for q in list(set(all_queries)): 
            # 1. EMBEDDING TRUY VẤN TRƯỚC KHI TÌM KIẾM
            q_vector = get_embedding_ollama(q)
            if not q_vector:
                print(f"Bỏ qua truy vấn '{q}' vì không thể tạo embedding.")
                continue

            # 2. GỌI hybrid_search VỚI CẢ TEXT VÀ VECTOR
            hits = mgr.hybrid_search(
                collection_name=weav_collection, 
                query_text=q, # Dùng cho BM25
                query_vector=q_vector, # Dùng cho vector search
                alpha=alpha, 
                limit=50
            )
            
            initial_candidate_count += len(hits)
            for h in hits:
                # ... (phần còn lại giữ nguyên)
                doc_id = h["id"]
                if doc_id not in candidates or h["combined_score"] > candidates[doc_id]["combined_score"]:
                    props = h.get("properties", {})
                    candidates[doc_id] = { "id": h["id"], "properties": props, "combined_score": h["combined_score"] }

    retrieval_end = time.monotonic()
    report["timings_ms"]["candidate_retrieval"] = round((retrieval_end - retrieval_start) * 1000)
    report["statistics"]["num_initial_candidates"] = initial_candidate_count
    report["statistics"]["num_deduplicated_candidates"] = len(candidates)
    
    top_candidates = sorted(candidates.values(), key=lambda x: x["combined_score"], reverse=True)
    print("Query completed")

    # --- 4. BƯỚC SẮP XẾP LẠI (RERANKING) ---
    print("Start reranking")
    docs_to_rerank = []
    for doc in top_candidates:
        props = doc["properties"]
        parts = [props.get("title") or "", props.get("abstract") or "", (props.get("content") or "")[:4000]]
        docs_to_rerank.append("\n\n".join(p for p in parts if p.strip()))
    
    report["statistics"]["num_docs_sent_to_reranker"] = len(docs_to_rerank)
    
    rerank_start = time.monotonic()
    reranked_results = reranker.rerank(user_query, docs_to_rerank, top_k=top_k)
    rerank_end = time.monotonic()
    report["timings_ms"]["reranking"] = round((rerank_end - rerank_start) * 1000)
    print("Finished rerank")

    # --- 5. TỔNG HỢP KẾT QUẢ CUỐI CÙNG ---
    final_retrieved_docs = []
    for original_index, score, text in reranked_results:
        meta = top_candidates[original_index]
        props = meta["properties"]
        final_retrieved_docs.append({
            "id": meta["id"],
            "title": props.get("title"),
            "abstract": props.get("abstract"),
            "keywords": props.get("keywords"),
            "combined_score": meta.get("combined_score"),
            "reranker_score": score,
            "snippet": text[:500] + "..." if len(text) > 500 else text,
            "content": text
        })
    
    report["statistics"]["num_final_results"] = len(final_retrieved_docs)

    end_time = time.monotonic()
    report["timings_ms"]["total_pipeline_duration"] = round((end_time - start_time) * 1000)


    #  6: SINH NỘI DUNG TÓM TẮT (GENERATION)
    print("Generating summary...")
    generation_start = time.monotonic()
    generated_summary = ""

    # Chỉ sinh tóm tắt nếu có tài liệu liên quan được tìm thấy
    if not final_retrieved_docs:
        generated_summary = ""
    else:
        # 1. Chuẩn bị bối cảnh (Context)
        context_string = ""
        for i, doc in enumerate(final_retrieved_docs):
            context_string += f"--- Nguồn tài liệu {i+1}: {doc['title']} ---\n"
            context_string += doc['content']
            context_string += f"Từ khóa: {doc['keywords']}"
            context_string += "\n\n"
        
        # 2. TẠO PROMPT YÊU CẦU TÓM TẮT VÀ ĐỊNH DẠNG MARKDOWN
        summarizer_prompt = f"""Bạn là một chuyên gia trình bày thông tin. Nhiệm vụ của bạn là đọc [Kiến thức] và tóm tắt lại các thông tin liên quan đến [Câu hỏi], sau đó trình bày kết quả bằng cú pháp Markdown.

    ### QUY TẮC ĐỊNH DẠNG:
    - Sử dụng tiêu đề (ví dụ: `## Tiêu đề chính`, `### Tiêu đề phụ`) cho các mục lớn.
    - Sử dụng dấu gạch đầu dòng (`-`) để liệt kê các đặc điểm, điều kiện, hoặc danh sách.
    - Sử dụng `**in đậm**` để nhấn mạnh các thuật ngữ quan trọng, các con số hoặc các nhãn dữ liệu (ví dụ: `**Lãi suất:** 15.9%/năm`).

    ### QUY TẮC NỘI DUNG:
    - Đầu ra của bạn CHỈ ĐƯỢC PHÉP là phần kiến thức đã được tóm tắt và định dạng Markdown.
    - Tuyệt đối không thêm bất kỳ lời chào, câu dẫn, lời giải thích hay kết luận nào.
    - Nếu [Kiến thức] không chứa thông tin nào liên quan đến [Câu hỏi], hãy trả về một chuỗi rỗng duy nhất.

    [Câu hỏi]
    {user_query}

    [Kiến thức]
    {context_string}
    """
        
        # 3. Gọi LLM để sinh nội dung tóm tắt
        if summarizer_prompt:
            generated_summary = chat.generate(summarizer_prompt)
        else:
            generated_summary = "Lỗi: Không thể tải được prompt. Vui lòng kiểm tra lại file cấu hình."

    generation_end = time.monotonic()
    report["timings_ms"]["summary_generation"] = round((generation_end - generation_start) * 1000)
    print("Summary generated.")

    print(generated_summary)

    # --- BƯỚC MỚI: 6.5 KIỂM TRA SỰ PHÙ HỢP CỦA NỘI DUNG TÓM TẮT ---
    final_answer = generated_summary # Mặc định câu trả lời cuối cùng là bản tóm tắt

    if generated_summary and generated_summary.strip(): # Chỉ kiểm tra nếu bản tóm tắt không rỗng
        print("Verifying summary relevance...")
        verification_start = time.monotonic()
        
        verifier_prompt = f"""Bạn là một AI chuyên đánh giá sự liên quan. Hãy đọc [Câu hỏi] và [Nội dung tóm tắt] dưới đây. 
    Nhiệm vụ của bạn là đưa ra kết luận xem [Nội dung tóm tắt] có chứa thông tin trực tiếp để trả lời [Câu hỏi] hay không.

    Hãy trả lời bằng MỘT TỪ DUY NHẤT:
    - "LIÊN QUAN" nếu nội dung tóm tắt trả lời được câu hỏi.
    - "KHÔNG LIÊN QUAN" nếu nội dung tóm tắt chỉ nói về chủ đề chung chung nhưng không có thông tin cụ thể để trả lời câu hỏi.

    [Câu hỏi]:
    {user_query}

    [Nội dung tóm tắt]:
    {generated_summary}
    """
        
        verification_result = chat.generate(verifier_prompt).strip().upper()
        
        # Nếu kết quả kiểm tra là KHÔNG LIÊN QUAN, ghi đè câu trả lời cuối cùng
        if "KHÔNG LIÊN QUAN" in verification_result:
            final_answer = "Không có nội dung phù hợp với câu hỏi."
            print("Verification result: NOT RELEVANT. Overwriting answer.")
        else:
            print("Verification result: RELEVANT.")
            
        verification_end = time.monotonic()
        report["timings_ms"]["answer_verification"] = round((verification_end - verification_start) * 1000)

    # Cập nhật tên biến để trả về cho response
    report["generated_answer"] = final_answer

    # --- 7. HOÀN TẤT VÀ TRẢ VỀ RESPONSE CUỐI CÙNG ---
    # ... (Phần này giữ nguyên, nhưng đảm bảo bạn trả về biến `final_answer` thay vì `generated_summary` nếu đã đổi tên)
    end_time = time.monotonic()
    report["timings_ms"]["total_pipeline_duration"] = round((end_time - start_time) * 1000)

    final_docs_for_response = []
    for doc in final_retrieved_docs:
        full_text = doc.get("content") or ""
        final_docs_for_response.append({
            "id": doc["id"],
            "title": doc["title"],
            "abstract": doc["abstract"],
            "keywords": doc["keywords"],
            "combined_score": doc["combined_score"],
            "reranker_score": doc["reranker_score"],
            "content": full_text[:500] + "..." if len(full_text) > 500 else full_text,
        })

    # Đóng gói mọi thứ vào một response duy nhất
    return {
        "report": report,
        "generated_answer": final_answer, # TRẢ VỀ BIẾN NÀY
        "results":final_docs_for_response 
    }