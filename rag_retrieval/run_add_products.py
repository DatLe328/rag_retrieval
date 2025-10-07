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

def summarize_text_ollama(text: str, model_name: str = "llama3.2:3b") -> str:
    """
    Gọi OllamaChatModel để tạo tóm tắt cực ngắn (≈20 từ).
    """
    try:
        llm = OllamaChatModel(model_name=model_name)
        system_prompt = (
            "Bạn là trợ lý tóm tắt chính xác. "
            "Tạo bản tóm tắt ngắn gọn khoảng 20 từ, không mở đầu hay kết luận dư."
        )
        user_prompt = f"Tóm tắt nội dung sau trong khoảng 20 từ:\n\n{text[:4000]}"
        summary = llm.generate(user_prompt=user_prompt, system_prompt=system_prompt)
        return summary.strip()
    except Exception as e:
        print(f"❌ Lỗi khi gọi OllamaChatModel: {e}")
        return ""

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
async def chunk_and_add(manager, merged_files):
    collection_name = "Papers"
    print(f"Sẽ thêm {len(merged_files)} file vào collection '{collection_name}'\n")

    for item in merged_files:
        title = item["filename"]
        text = item["text"]
        kw_text = item["kw_text"]

        print(f"→ Đang xử lý file: {title}")

        # Tạo Document để chunk
        document = Document(
            title=title,
            content=text,
            extension=".md",
            fileSize=0,
            labels=[],
            source="",
            meta={},
            metadata=""
        )
        chunker = MarkdownChunker()
        try:
            chunks = await chunker.chunk(chunker.config, [document])
        except TypeError:
            chunks = await chunker.chunk([document])

        # Sinh abstract bằng Ollama
        abstract = summarize_text_ollama(text)

        # Ghi từng chunk vào Weaviate
        for idx, chunk in enumerate(document.chunks if hasattr(document, "chunks") and document.chunks else chunks):
            chunk_data = {
                "title": title,
                "abstract": abstract,
                "text": getattr(chunk, "content", str(chunk)),
                "keywords": [kw_text] if kw_text else [],
                "created_date": datetime.now(timezone.utc).isoformat(),
            }
            try:
                manager.add(collection_name=collection_name, properties=chunk_data)
                print(f"   ✅ Chunk {idx+1} added.")
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
        with WeaviateManager() as manager:
            collection_name = "Papers"
            properties = [
                Property(name="title", data_type=DataType.TEXT),
                Property(name="abstract", data_type=DataType.TEXT),
                Property(name="keywords", data_type=DataType.TEXT_ARRAY),
                Property(name="text", data_type=DataType.TEXT),
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
