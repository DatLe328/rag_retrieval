from rag_retrieval.db.weaviate_db import WeaviateManager
from datetime import datetime, timezone
import os
import asyncio
from dotenv import load_dotenv
from rag_retrieval.goldenverba.load_data import load_files

# Thêm import chunker và Document
from goldenverba.components.chunking.MarkdownChunker import MarkdownChunker
from goldenverba.components.chunking.SentenceChunker import SentenceChunker
from goldenverba.components.document import Document

# ============================================================================== 
# ĐỌC DỮ LIỆU TỪ FILE (md, txt, pdf, docx) 
# ==============================================================================

md_folder = "data"  # <-- Đổi thành đường dẫn folder chứa file của bạn
products_data = load_files(md_folder)

# ============================================================================== 
# LOGIC CHUNK VÀ THÊM DỮ LIỆU VÀO WEAVIATE 
# ==============================================================================

async def chunk_and_add(manager, products_data):
    collection_name = "Papers"
    print(f"Sẽ thêm dữ liệu vào collection: '{collection_name}'")

    for product in products_data:
        product_title = product.get("filename", "Không có tiêu đề")
        ext = product.get("ext", "")
        text = product.get("text", "")
        print(f"  -> Đang chuẩn bị chunk và thêm: {product_title}")

        # Chọn chunker phù hợp
        if ext == ".md":
            chunker = MarkdownChunker()
        else:
            chunker = SentenceChunker()

        # Tạo Document object từ text
        document = Document(
            title=product_title,
            content=text,
            extension=ext,
            fileSize=0,
            labels=[],
            source="",
            meta={},
            metadata=""
        )                
        documents = [document]

        # Chunk nội dung (dùng config mặc định của chunker)
        config = chunker.config
        try:
            chunks = await chunker.chunk(config, documents)
        except TypeError:
            chunks = await chunker.chunk(documents)

        # Thêm từng chunk vào DB
        for idx, chunk in enumerate(document.chunks if hasattr(document, "chunks") and document.chunks else chunks):
            chunk_data = {
                "filename": product_title,
                "chunk_index": idx,
                "content": chunk.content if hasattr(chunk, "content") else str(chunk),
                "ext": ext,
            }
            try:
                manager.add(
                    collection_name=collection_name,
                    properties=chunk_data
                )
                print(f"     ✅ Thêm chunk {idx+1}: {product_title}")
            except Exception as e:
                print(f"     ❌ Lỗi khi thêm chunk {idx+1}: {e}")

if __name__ == "__main__":
    load_dotenv()
    print("--- Bắt đầu quá trình chunk và thêm dữ liệu vào Weaviate ---")

    try:
        with WeaviateManager() as manager:
            asyncio.run(chunk_and_add(manager, products_data))

    except ConnectionError as ce:
        print(f"\nLỖI KẾT NỐI: Không thể kết nối tới Weaviate. Vui lòng kiểm tra lại dịch vụ.")
        print(f"Chi tiết: {ce}")
    except Exception as e:
        print(f"\nĐã xảy ra lỗi không mong muốn: {e}")

    print("\n--- ✅ Quá trình chunk và thêm dữ liệu đã hoàn tất. ---")