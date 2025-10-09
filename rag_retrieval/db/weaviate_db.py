import weaviate
import weaviate.classes as wvc
from weaviate.classes.config import Configure, Property, DataType
from weaviate.classes.query import MetadataQuery
from weaviate.classes.init import AdditionalConfig
from typing import List, Dict, Optional, Any
from datetime import datetime

class WeaviateManager:
    def __init__(self, host="10.1.1.237", http_port=3000):
        self.host = host
        self.http_port = http_port
        self.client = None

    def __enter__(self):
        try:
            # ✅ Kết nối REST-only, bỏ toàn bộ gRPC
            self.client = weaviate.connect_to_local(
                host=self.host,
                port=self.http_port,
                skip_init_checks=True,  # bỏ health-check gRPC
                additional_config=AdditionalConfig(timeout=(10, 60)),
            )
            return self
        except Exception as e:
            raise ConnectionError(f"Không thể kết nối tới Weaviate: {e}")

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.client:
            self.client.close()

    def create_collection(self, name: str, properties: List[Property], force_recreate: bool = False):
        if force_recreate and self.client.collections.exists(name):
            self.client.collections.delete(name)

        if not self.client.collections.exists(name):
            self.client.collections.create(
                name=name,
                properties=properties,
                # vectorizer_config=Configure.Vectorizer.text2vec_ollama(
                #     model="nomic-embed-text",
                #     api_endpoint="http://10.1.1.237:11434",
                # ),
                vector_index_config=Configure.VectorIndex.hnsw(
                    distance_metric=wvc.config.VectorDistances.COSINE
                )
            )

    # def add(self, collection_name: str, properties: Dict[str, Any]):
    #     collection = self.client.collections.get(collection_name)
    #     collection.data.insert(properties=properties)
    def add(self, collection_name: str, properties: Dict[str, Any], vector: List[float] = None):
        collection = self.client.collections.get(collection_name)
        # Gửi cả properties và vector (nếu có)
        collection.data.insert(properties=properties, vector=vector)

    # def search(self, collection_name: str, query: str, search_type: str, limit: int = 5, properties: List[str] = None) -> List[Dict]:
    #     collection = self.client.collections.get(collection_name)
        
    #     if search_type == "vector":
    #         response = collection.query.near_text(
    #             query=query, limit=limit, return_metadata=["distance"]
    #         )
    #         results = []
    #         for obj in response.objects:
    #             distance = getattr(obj.metadata, "distance", None)
    #             score = 1 - distance if distance is not None else None
    #             results.append({"uuid": str(obj.uuid), "properties": obj.properties, "score": score, "distance": distance})
    #         return results

    #     elif search_type == "bm25":
    #         response = collection.query.bm25(
    #             query=query, query_properties=properties, limit=limit, return_metadata=MetadataQuery(score=True)
    #         )
    #         return [{"uuid": str(obj.uuid), "properties": obj.properties, "score": getattr(obj.metadata, "score", None)} for obj in response.objects]
    #     else:
    #         raise ValueError(f"Loại tìm kiếm '{search_type}' không được hỗ trợ.")

    def search(
        self,
        collection_name: str,
        search_type: str,
        limit: int = 5,
        query_text: str = None, # Dùng cho bm25
        query_vector: List[float] = None, # Dùng cho vector search
        properties: List[str] = None
    ) -> List[Dict]:
        collection = self.client.collections.get(collection_name)
        
        if search_type == "vector":
            if not query_vector:
                raise ValueError("Tìm kiếm vector yêu cầu một 'query_vector'.")
            
            # SỬ DỤNG near_vector thay cho near_text
            response = collection.query.near_vector(
                near_vector=query_vector, 
                limit=limit, 
                return_metadata=["distance"]
            )
            results = []
            for obj in response.objects:
                distance = getattr(obj.metadata, "distance", None)
                score = 1 - distance if distance is not None else None
                results.append({"uuid": str(obj.uuid), "properties": obj.properties, "score": score, "distance": distance})
            return results

        elif search_type == "bm25":
            if not query_text:
                raise ValueError("Tìm kiếm BM25 yêu cầu một 'query_text'.")
            
            response = collection.query.bm25(
                query=query_text, query_properties=properties, limit=limit, return_metadata=MetadataQuery(score=True)
            )
            return [{"uuid": str(obj.uuid), "properties": obj.properties, "score": getattr(obj.metadata, "score", None)} for obj in response.objects]
        else:
            raise ValueError(f"Loại tìm kiếm '{search_type}' không được hỗ trợ.")


    def hybrid_search(
        self,
        collection_name: str,
        query_text: str,
        query_vector: List[float],
        alpha: float,
        limit: int = 50,
        properties: List[str] = ["title", "abstract", "keywords", "text"]
    ) -> List[Dict[str, Any]]:
        try:
            hits_bm25 = self.search(collection_name, "bm25", limit, query_text=query_text, properties=properties)
        except Exception:
            hits_bm25 = []

        try:
            # Truyền query_vector vào đây
            hits_vec = self.search(collection_name, "vector", limit, query_vector=query_vector)
        except Exception:
            hits_vec = []
        # try:
        #     hits_bm25 = self.search(collection_name, query, "bm25", limit, properties)
        # except Exception:
        #     hits_bm25 = []

        # try:
        #     hits_vec = self.search(collection_name, query, "vector", limit, properties)
        # except Exception:
        #     hits_vec = []

        candidates: Dict[str, Dict[str, Any]] = {}
        for h in hits_bm25 + hits_vec:
            doc_id = h.get("uuid")
            if not doc_id:
                continue

            existing = candidates.get(doc_id)
            if not existing:
                candidates[doc_id] = {
                    "id": doc_id,
                    "properties": h.get("properties", {}),
                    "bm25_score": 0.0,
                    "vector_score": 0.0,
                }
            
            # Cập nhật điểm số cao nhất cho từng loại
            # Giả định h['score'] là bm25_score cho hits_bm25 và vector_score cho hits_vec
            if "distance" in h: # Đây là kết quả vector search
                 vec_score = 1 - h["distance"] if h.get("distance") is not None else 0.0
                 candidates[doc_id]["vector_score"] = max(candidates[doc_id]["vector_score"], vec_score)
            else: # Đây là kết quả bm25
                 bm25_score = h.get("score") or 0.0
                 candidates[doc_id]["bm25_score"] = max(candidates[doc_id]["bm25_score"], bm25_score)

        if not candidates:
            return []

        # 3. Tính toán điểm kết hợp (logic y hệt code cũ)
        all_bm25_scores = [d["bm25_score"] for d in candidates.values() if d["bm25_score"] > 0]
        max_bm25 = max(all_bm25_scores) if all_bm25_scores else 0

        for doc in candidates.values():
            bm25_score = doc.get("bm25_score", 0.0)
            vector_score = doc.get("vector_score", 0.0)
            
            # Chuẩn hóa BM25
            bm25_norm = (bm25_score / max_bm25) if max_bm25 > 0 else 0.0
            
            # Kết hợp điểm
            combined_score = alpha * vector_score + (1 - alpha) * bm25_norm
            doc["combined_score"] = combined_score

        # 4. Sắp xếp và trả về kết quả
        sorted_results = sorted(candidates.values(), key=lambda x: x["combined_score"], reverse=True)
        return sorted_results




def gen():
    """
    Tạo dữ liệu mẫu đa dạng, dựa hoàn toàn vào khả năng embedding của Weaviate.
    Hàm này sẽ xóa collection cũ (nếu có) và tạo lại từ đầu.
    """
    from datetime import timezone
    from dotenv import load_dotenv
    import os

    print("--- Bắt đầu quá trình tạo dữ liệu (Weaviate tự embedding) ---")

    # Tải biến môi trường (ví dụ: WEAVIATE_HOST) từ file .env
    load_dotenv()

    # Dữ liệu mẫu đa dạng (giữ nguyên)
    papers = [
        {
            "title": "Attention Is All You Need",
            "abstract": "The Transformer, based solely on attention mechanisms, is introduced. This new architecture eschews recurrence and convolutions entirely.",
            "keywords": ["AI", "Transformer", "NLP", "Attention Mechanism"],
            "text": "The full text explains the encoder-decoder structure, scaled dot-product attention, multi-head attention, and positional encodings which are fundamental to the Transformer model's success in sequence transduction tasks.",
            "created_date": datetime(2017, 6, 12, tzinfo=timezone.utc),
        },
        {
            "title": "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
            "abstract": "BERT stands for Bidirectional Encoder Representations from Transformers. It is designed to pre-train deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context.",
            "keywords": ["AI", "NLP", "BERT", "Language Model"],
            "text": "This paper demonstrates that BERT achieves state-of-the-art results on a wide array of natural language processing tasks. The core innovation is the application of bidirectional training of Transformers, which was not possible with previous language models.",
            "created_date": datetime(2018, 10, 11, tzinfo=timezone.utc),
        },
        # ... (Thêm các bài báo khác nếu cần)
        {
            "title": "Renewable Energy Sources and Grid Integration Challenges",
            "abstract": "This paper explores the technical challenges of integrating variable renewable energy sources like solar and wind into traditional power grids.",
            "keywords": ["Energy", "Renewable Energy", "Solar Power", "Wind Power", "Power Grid"],
            "text": "The primary challenges discussed are intermittency and grid stability. Solutions like energy storage systems (batteries), smart grid technologies, and improved forecasting models are evaluated for their effectiveness in creating a reliable and sustainable energy future.",
            "created_date": datetime(2020, 8, 5, tzinfo=timezone.utc),
        },
    ]

    # Sử dụng WeaviateManager để thực hiện các thao tác
    try:
        # ✅ KHÔNG CẦN KHỞI TẠO EMBEDDER CỤC BỘ NỮA
        with WeaviateManager() as manager: # <-- Gọi mà không truyền embedder
            collection_name = "Papers"
            properties = [
                Property(name="title", data_type=DataType.TEXT),
                Property(name="abstract", data_type=DataType.TEXT),
                Property(name="keywords", data_type=DataType.TEXT_ARRAY),
                Property(name="text", data_type=DataType.TEXT),
                Property(name="created_date", data_type=DataType.DATE),
            ]
            
            print(f"\n[1/3] Đang tạo collection '{collection_name}' (sẽ xóa nếu đã tồn tại)...")
            # Hàm create_collection đã được cấu hình để dùng text2vec-ollama
            manager.create_collection(name=collection_name, properties=properties, force_recreate=True)
            print(f"✅ Collection '{collection_name}' đã được tạo thành công.")

            print(f"\n[2/3] Đang thêm {len(papers)} tài liệu vào collection...")
            for i, paper in enumerate(papers):
                # Gọi hàm add đã được đơn giản hóa, chỉ truyền properties
                manager.add(collection_name=collection_name, properties=paper)
                print(f"  -> Đã thêm tài liệu {i+1}/{len(papers)}: '{paper['title']}'")
            
            print("✅ Thêm dữ liệu hoàn tất.")

            print("\n[3/3] Kiểm tra nhanh bằng một truy vấn tìm kiếm...")
            query_text = "What are the challenges of renewable energy?"
            # Hàm search đã được cập nhật để dùng near_text
            search_results = manager.search(collection_name, query=query_text, search_type="vector", limit=2)
            
            print(f"\n🔍 Kết quả tìm kiếm vector cho: '{query_text}'")
            if search_results:
                for res in search_results:
                    print(f"  - Tiêu đề: {res['properties'].get('title')} (Score: {res['score']:.4f})")
            else:
                print("  - Không tìm thấy kết quả.")

    except ConnectionError as ce:
        print(f"\nLỖI KẾT NỐI: {ce}")
        print("Vui lòng đảm bảo Weaviate đang chạy và cấu hình host/port là chính xác.")
    except Exception as e:
        print(f"\nĐã xảy ra lỗi không mong muốn: {e}")

    print("\n--- ✅ Quá trình tạo dữ liệu đã hoàn tất. ---")


if __name__ == "__main__":
    # Chạy hàm này để bắt đầu tạo dữ liệu
    gen()