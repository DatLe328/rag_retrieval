from rag_retrieval.db.weaviate_db import WeaviateManager
from datetime import datetime, timezone
import os, asyncio
from dotenv import load_dotenv
from rag_retrieval.goldenverba.load_data import load_files
from weaviate.classes.config import Property, DataType
from goldenverba.components.chunking.MarkdownChunker import MarkdownChunker
from goldenverba.components.document import Document
import requests  # dùng cho Ollama summary API
from model.wrapper.llm_ollama import OllamaChatModel


def get_embedding_ollama(text: str, model_name: str = "nomic-embed-text"):
    """
    Gọi API của Ollama để tạo vector embedding cho một đoạn text.
    """
    try:
        # Giả sử Ollama đang chạy tại địa chỉ này
        response = requests.post(
            "http://10.1.1.237:11434/api/embeddings",
            json={"model": model_name, "prompt": text},
        )
        response.raise_for_status()
        return response.json()["embedding"]
    except Exception as e:
        print(f"❌ Lỗi khi tạo embedding cho text: {e}")
        return []



# ==============================================================================
# GOM FILE CHÍNH VÀ FILE KEYWORD
# ==============================================================================
def merge_files(products_data):
    md_files = [f for f in products_data if f["ext"] == ".md"]
    merged = []

    for f in md_files:
        fname = f["filename"].lower()
        if fname.endswith("_kw.md"):
            continue  # bỏ file keyword, sẽ gán vào file chính

        # tìm file keyword tương ứng
        base = fname.replace(".md", "")
        kw_file = next((k for k in md_files if k["filename"].lower() == f"{base}_kw.md"), None)

        merged.append({
            "filename": f["filename"],
            "text": f["text"],
            "kw_text": kw_file["text"] if kw_file else "",
        })
    return merged

# ==============================================================================
# CHUNK + EMBEDDING FULL TEXT + SUMMARY + KW
# ==============================================================================
# async def chunk_and_add(manager, merged_files):
#     collection_name = "Papers"
#     print(f"Sẽ thêm {len(merged_files)} file vào collection '{collection_name}'\n")
#     llm = OllamaChatModel(model_name="llama3.2:3b")
#     for item in merged_files:
#         title = item["filename"]
#         text = item["text"]
#         kw_text = item["kw_text"]

#         print(f"→ Đang xử lý file: {title}")

#         # Tạo Document để chunk
#         document = Document(
#             title=title,
#             content=text,
#             extension=".md",
#             fileSize=0,
#             labels=[],
#             source="",
#             meta={},
#             metadata=""
#         )
#         chunker = MarkdownChunker()
#         try:
#             chunks = await chunker.chunk(chunker.config, [document])
#         except TypeError:
#             chunks = await chunker.chunk([document])

#         # Sinh abstract bằng Ollama
#         abstract = summarize_text_ollama(text,llm)

#         # Ghi từng chunk vào Weaviate
#         for idx, chunk in enumerate(document.chunks if hasattr(document, "chunks") and document.chunks else chunks):
#             chunk_data = {
#                 "title": title,
#                 "abstract": abstract,
#                 "text": getattr(chunk, "content", str(chunk)),
#                 "keywords": [kw_text] if kw_text else [],
#                 "created_date": datetime.now(timezone.utc).isoformat(),
#             }
#             try:
#                 manager.add(collection_name=collection_name, properties=chunk_data)
#                 print(f"   ✅ Chunk {idx+1} added.")
#             except Exception as e:
#                 print(f"   ❌ Lỗi chunk {idx+1}: {e}")
# run_add_products.py

async def chunk_and_add(manager, merged_files):
    collection_name = "Papers"
    print(f"Sẽ thêm {len(merged_files)} file vào collection '{collection_name}'\n")
    llm = OllamaChatModel(model_name="llama3.2:3b") # Giữ lại để tạo summary
    
    for item in merged_files:
        title = item["filename"]
        text = item["text"]
        kw_text = item["kw_text"]

        print(f"→ Đang xử lý file: {title}")

        # Tạo Document để chunk (giữ nguyên)
        document = Document(title=title, content=text, extension=".md", fileSize=0, labels=[], source="", meta={}, metadata="")
        chunker = MarkdownChunker()
        try:
            chunks = await chunker.chunk(chunker.config, [document])
        except TypeError:
            chunks = await chunker.chunk([document])

        # Sinh abstract bằng Ollama (giữ nguyên)
        abstract = summarize_text_ollama(text, llm)

        # Ghi từng chunk vào Weaviate
        for idx, chunk in enumerate(document.chunks if hasattr(document, "chunks") and document.chunks else chunks):
            chunk_text = getattr(chunk, "content", str(chunk))
            
            # 1. TẠO EMBEDDING TỪ PHÍA CLIENT
            # Kết hợp title và nội dung chunk để embedding tốt hơn
            text_to_embed = f"Tiêu đề: {title}\nNội dung: {chunk_text}"
            embedding_vector = get_embedding_ollama(text_to_embed)

            if not embedding_vector:
                print(f"   ⚠️ Bỏ qua chunk {idx+1} vì không thể tạo embedding.")
                continue

            # 2. CHUẨN BỊ DỮ LIỆU
            chunk_data = {
                "title": title,
                "abstract": abstract,
                "text": chunk_text,
                "keywords": [kw_text] if kw_text else [],
                "created_date": datetime.now(timezone.utc).isoformat(),
            }

            # 3. GỬI DỮ LIỆU KÈM VECTOR
            try:
                # Gọi hàm add đã được sửa đổi, truyền cả vector vào
                manager.add(
                    collection_name=collection_name, 
                    properties=chunk_data, 
                    vector=embedding_vector
                )
                print(f"   ✅ Chunk {idx+1} added with its own vector.")
            except Exception as e:
                print(f"   ❌ Lỗi chunk {idx+1}: {e}")

# ==============================================================================
# MAIN
# ==============================================================================
if __name__ == "__main__":
    load_dotenv()
    print("--- Bắt đầu quá trình embedding full text ---")

    md_folder = "data"
    products_data = load_files(md_folder)
    merged_files = merge_files(products_data)

    try:
        with WeaviateManager(host="localhost") as manager:
            collection_name = "Papers"
            properties = [
                Property(name="title", data_type=DataType.TEXT),
                Property(name="abstract", data_type=DataType.TEXT),
                Property(name="keywords", data_type=DataType.TEXT_ARRAY),
                Property(name="content", data_type=DataType.TEXT),
                Property(name="created_date", data_type=DataType.DATE),
            ]
            print(f"[1/2] Tạo lại collection '{collection_name}' ...")
            manager.create_collection(name=collection_name, properties=properties, force_recreate=True)
            print(f"✅ Collection '{collection_name}' sẵn sàng.\n")

            asyncio.run(chunk_and_add(manager, merged_files))

    except ConnectionError as ce:
        print("Lỗi kết nối Weaviate:", ce)
    except Exception as e:
        print("Lỗi không mong muốn:", e)

    print("\n--- ✅ Hoàn tất quá trình embedding ---")
