from rag_retrieval.db.weaviate_db import WeaviateManager
from datetime import datetime, timezone
import os
import asyncio
from dotenv import load_dotenv
from rag_retrieval.goldenverba.load_data import load_files

from goldenverba.components.chunking.MarkdownChunker import MarkdownChunker
from goldenverba.components.chunking.SentenceChunker import SentenceChunker
from goldenverba.components.document import Document

# ==============================================================================
# ĐỌC DỮ LIỆU TỪ FILE (md, txt, pdf, docx)
# ==============================================================================
md_folder = "data"  # <-- Đường dẫn folder chứa file .md của bạn
products_data = load_files(md_folder)

# ==============================================================================
# HÀM LỌC FILE: phân loại file thường và file chứa 'kw'
# ==============================================================================
def filter_files(products_data, kw_mode=False):
    if kw_mode:
        # Luồng 2: file có 'kw' trong tên
        return [f for f in products_data if f["ext"] == ".md" and "kw" in f["filename"].lower()]
    else:
        # Luồng 1: file thường (không có 'kw')
        return [f for f in products_data if f["ext"] == ".md" and "kw" not in f["filename"].lower()]

# ==============================================================================
# LOGIC CHUNK, EMBEDDING NỘI DUNG VÀ EMBEDDING KEYWORD VÀO WEAVIATE
# ==============================================================================
async def chunk_and_add(manager, products_data):
    collection_name = "Papers"
    print(f"Sẽ thêm dữ liệu vào collection: '{collection_name}'")

    # Kiểm tra danh sách file nạp
    print("\n--- Kiểm tra danh sách file ---")
    all_files = [f["filename"] for f in products_data]
    print("Tổng số file đọc được:", len(all_files))
    print("Danh sách file:", all_files)

    normal_files = filter_files(products_data, kw_mode=False)
    kw_files = filter_files(products_data, kw_mode=True)
    print("Normal files:", [f["filename"] for f in normal_files])
    print("Keyword files:", [f["filename"] for f in kw_files])

    # --------------------------------------------------------------------------
    # LUỒNG 1: Chunk và embedding nội dung từ file thường
    # --------------------------------------------------------------------------
    print("\n--- Luồng 1: Chunk và embedding nội dung ---")
    for product in normal_files:
        product_title = product.get("filename", "Không có tiêu đề")
        ext = product.get("ext", "")
        text = product.get("text", "")
        print(f"  -> Đang chunk và thêm: {product_title}")

        chunker = MarkdownChunker() if ext == ".md" else SentenceChunker()

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

        # Chunk nội dung
        config = chunker.config
        try:
            chunks = await chunker.chunk(config, documents)
        except TypeError:
            chunks = await chunker.chunk(documents)

        # Ghi từng chunk vào Weaviate
        for idx, chunk in enumerate(
            document.chunks if hasattr(document, "chunks") and document.chunks else chunks
        ):
            chunk_data = {
                "filename": product_title,
                "chunk_index": idx,
                "text": getattr(chunk, "content", str(chunk)),  # Dùng 'text' để vector hóa
                "ext": ext,
            }
            try:
                manager.add(collection_name=collection_name, properties=chunk_data)
                print(f"     ✅ Thêm chunk {idx+1}: {product_title}")
            except Exception as e:
                print(f"     ❌ Lỗi khi thêm chunk {idx+1}: {e}")

    # --------------------------------------------------------------------------
    # LUỒNG 2: Embedding keyword từ file có 'kw'
    # --------------------------------------------------------------------------
    print("\n--- Luồng 2: Embedding keyword từ file có 'kw' ---")
    for product in kw_files:
        product_title = product.get("filename", "Không có tiêu đề")
        ext = product.get("ext", "")
        text = product.get("text", "")
        print(f"  -> Đang embedding nguyên nội dung keyword file: {product_title}")

        # Thêm toàn bộ nội dung file keyword làm một entry duy nhất
        keyword_data = {
            "filename": product_title,
            "chunk_index": 0,
            "text": text,  # Dùng 'text' để vector hóa
            "ext": ext,
        }

        try:
            manager.add(collection_name=collection_name, properties=keyword_data)
            print(f"     ✅ Thêm file keyword {product_title} vào Weaviate")
        except Exception as e:
            print(f"     ❌ Lỗi khi thêm file keyword {product_title}: {e}")

# ==============================================================================
# CHẠY CHÍNH
# ==============================================================================
if __name__ == "__main__":
    load_dotenv()
    print("--- Bắt đầu quá trình chunk và thêm dữ liệu vào Weaviate ---")

    try:
        with WeaviateManager() as manager:
            asyncio.run(chunk_and_add(manager, products_data))
    except ConnectionError as ce:
        print("\nLỖI KẾT NỐI: Không thể kết nối tới Weaviate.")
        print(f"Chi tiết: {ce}")
    except Exception as e:
        print(f"\nĐã xảy ra lỗi không mong muốn: {e}")

    print("\n--- ✅ Quá trình chunk và thêm dữ liệu đã hoàn tất. ---")
