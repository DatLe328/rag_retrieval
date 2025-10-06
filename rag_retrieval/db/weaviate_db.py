import weaviate
import weaviate.classes as wvc
from weaviate.classes.config import Configure, Property, DataType
from weaviate.classes.query import MetadataQuery
from weaviate.classes.init import AdditionalConfig
from typing import List, Dict, Optional, Any
from datetime import datetime


class WeaviateManager:
    def __init__(self, host="localhost", http_port=8080, grpc_port=50051, embedder=None):
        self.host = host
        self.http_port = http_port
        self.grpc_port = grpc_port
        self.embedder = embedder
        self.client = None

    def __enter__(self):
        try:
            self.client = weaviate.connect_to_custom(
                http_host=self.host,
                http_port=self.http_port,
                http_secure=False,
                grpc_host=self.host,
                grpc_port=self.grpc_port,
                grpc_secure=False,
                additional_config=AdditionalConfig(timeout=(10, 60)),
            )
            return self
        except Exception as e:
            raise ConnectionError(f"Không thể kết nối tới Weaviate: {e}")

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.client:
            self.client.close()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.client:
            self.client.close()

    def create_collection(self, name: str, properties: List[Property], vector_config: Any = None, force_recreate: bool = False):
        """Tạo collection, mặc định dùng self_provided vectors."""
        if force_recreate and self.client.collections.exists(name):
            self.client.collections.delete(name)

        if not self.client.collections.exists(name):
            if vector_config is None:
                vector_config = Configure.Vectors.self_provided(
                    vector_index_config=Configure.VectorIndex.hnsw(
                        distance_metric=wvc.config.VectorDistances.COSINE
                    )
                )
            self.client.collections.create(name=name, properties=properties, vector_config=vector_config)

    def add(
        self,
        collection_name: str,
        title: str,
        abstract: str,
        keywords: list[str],
        text: str,
        created_date: datetime,
        vector: Optional[List[float]] = None,
    ):
        """Thêm tài liệu — tự động sinh vector nếu chưa có."""
        if vector is None:
            if not self.embedder:
                raise ValueError("Không có embedder để sinh vector.")
            vector = self.embedder.get_embedding(title)

        collection = self.client.collections.get(collection_name)
        collection.data.insert(
            properties={
                "title": title,
                "abstract": abstract,
                "keywords": keywords,
                "text": text,
                "created_date": created_date,
            },
            vector=vector,
        )

    def search(
        self,
        collection_name: str,
        query: Any,
        search_type: str = "vector",
        limit: int = 5,
        properties: List[str] = ["title", "abstract", "keywords", "text"]
    ) -> List[Dict]:
        """Trả về list gồm uuid, properties, score (càng cao càng tốt)."""
        collection = self.client.collections.get(collection_name)

        # ✅ Nếu là vector search, mà query là text → tự embed
        if search_type == "vector":
            if isinstance(query, str):
                if not self.embedder:
                    raise ValueError("Không có embedder để vector hóa query.")
                query = self.embedder.get_embedding(query)

            response = collection.query.near_vector(
                near_vector=query, limit=limit, return_metadata=["distance"]
            )
            results = []
            for obj in response.objects:
                distance = getattr(obj.metadata, "distance", None)
                score = 1 - distance if distance is not None else None
                results.append({"uuid": str(obj.uuid), "properties": obj.properties, "score": score, "distance": distance })
            return results

        elif search_type == "bm25":
            response = collection.query.bm25(
                query=query,
                query_properties=properties,
                limit=limit,
                return_metadata=MetadataQuery(score=True)
            )
            return [{"uuid": str(obj.uuid), "properties": obj.properties, "score": getattr(obj.metadata, "score", None)} for obj in response.objects]

        else:
            raise ValueError(f"Loại tìm kiếm '{search_type}' không được hỗ trợ.")

    def hybrid_search(
        self,
        collection_name: str,
        query: str,
        alpha: float,
        limit: int = 50,
        properties: List[str] = ["title", "abstract", "keywords", "text"]
    ) -> List[Dict[str, Any]]:
        """
        Thực hiện tìm kiếm BM25 và Vector riêng biệt, sau đó kết hợp điểm số
        theo logic tùy chỉnh (chuẩn hóa BM25 và dùng trọng số alpha).
        Đây là phiên bản đóng gói của logic cũ trong rag_pipeline.
        """
        # 1. Thực hiện cả hai loại tìm kiếm
        try:
            hits_bm25 = self.search(collection_name, query, "bm25", limit, properties)
        except Exception:
            hits_bm25 = []

        try:
            hits_vec = self.search(collection_name, query, "vector", limit, properties)
        except Exception:
            hits_vec = []

        # 2. Tập hợp và khử trùng lặp kết quả
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



def test():
    """Kiểm tra search (vector + BM25) trên dữ liệu nhỏ."""
    from datetime import timezone
    from rag_retrieval.model.ollama_models import OllamaEmbedder
    from dotenv import load_dotenv

    load_dotenv()
    embedder = OllamaEmbedder()

    with WeaviateManager(embedder=embedder) as manager:
        collection_name = "Papers"
        properties = [
            Property(name="title", data_type=DataType.TEXT),
            Property(name="abstract", data_type=DataType.TEXT),
            Property(name="keywords", data_type=DataType.TEXT_ARRAY),
            Property(name="text", data_type=DataType.TEXT),
            Property(name="created_date", data_type=DataType.DATE),
        ]
        manager.create_collection(name=collection_name, properties=properties, force_recreate=True)

        # Thêm vài bài thử
        papers = [
            {
                "title": "Attention Is All You Need",
                "abstract": "This paper introduces the Transformer architecture...",
                "keywords": ["AI", "Transformer"],
                "text": "Full text about the Transformer architecture...",
                "created_date": datetime(2017, 6, 12, tzinfo=timezone.utc),
            },
            {
                "title": "BERT: Pre-training of Deep Bidirectional Transformers",
                "abstract": "BERT is designed to pre-train deep bidirectional representations...",
                "keywords": ["AI", "NLP", "BERT"],
                "text": "Full text about BERT...",
                "created_date": datetime(2018, 10, 11, tzinfo=timezone.utc),
            },
        ]

        for paper in papers:
            manager.add(
                collection_name=collection_name,
                title=paper["title"],
                abstract=paper["abstract"],
                keywords=paper["keywords"],
                text=paper["text"],
                created_date=paper["created_date"],
            )

        # ============================
        # 1️⃣ Vector search
        # ============================
        query_text = "What is a Transformer architecture?"
        search_results = manager.search(collection_name, query=query_text, search_type="vector", limit=2)

        print(f"\n🔍 Vector search cho: '{query_text}'")
        for res in search_results:
            print(f"  - {res['properties'].get('title')} (Score: {res['score']})")

        # ============================
        # 2️⃣ BM25 search
        # ============================
        query_text = "Transformer architecture"
        search_results = manager.search(collection_name, query=query_text, search_type="bm25", limit=2)

        print(f"\n🔍 BM25 search cho: '{query_text}'")
        for res in search_results:
            print(f"  - {res['properties'].get('title')} (Score: {res['score']})")

    print("\n✅ Test hoàn tất.")


def gen():
    """
    Tạo dữ liệu mẫu đa dạng để kiểm tra hệ thống RAG.
    Hàm này sẽ xóa collection cũ (nếu có) và tạo lại từ đầu.
    """
    from datetime import timezone
    # Giả định bạn đã có embedder trong thư mục model và đã cài dotenv
    from rag_retrieval.model.wrapper.embedder_ollama import OllamaEmbedder
    from dotenv import load_dotenv
    import os

    print("--- Bắt đầu quá trình tạo dữ liệu thử nghiệm ---")

    # Tải biến môi trường (ví dụ: OLLAMA_BASE_URL) từ file .env
    load_dotenv()
    
    try:
        # Khởi tạo embedder
        # Hãy chắc chắn rằng model embed của bạn (ví dụ: nomic-embed-text) đã được pull về Ollama
        embedder = OllamaEmbedder(model_name=os.getenv("EMBED_MODEL", "nomic-embed-text"))
        print(f"Đã khởi tạo Embedder với model '{embedder.model_name}'")
    except Exception as e:
        print(f"Lỗi khi khởi tạo Embedder: {e}")
        print("Vui lòng đảm bảo Ollama đang chạy và đã pull model embedding.")
        return

    # Dữ liệu mẫu đa dạng
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
        {
            "title": "The Impact of Climate Change on Global Food Security",
            "abstract": "This study analyzes the effects of rising global temperatures and changing precipitation patterns on agricultural yields worldwide.",
            "keywords": ["Climate Change", "Agriculture", "Food Security", "Environment"],
            "text": "Detailed analysis shows that staple crops like wheat, rice, and maize are particularly vulnerable to climate stressors. The paper discusses mitigation strategies, including developing climate-resilient crops and improving water management techniques to ensure future food security.",
            "created_date": datetime(2021, 3, 22, tzinfo=timezone.utc),
        },
        {
            "title": "CRISPR-Cas9: A Revolutionary Tool for Genome Editing",
            "abstract": "An overview of the CRISPR-Cas9 system, a powerful and precise tool for making changes to the DNA of organisms.",
            "keywords": ["Biology", "Genetics", "CRISPR", "Biotechnology"],
            "text": "The text delves into the mechanisms of CRISPR-Cas9, its applications in treating genetic disorders, agricultural advancements, and the ethical considerations surrounding its use in humans. It compares CRISPR to older gene-editing techniques like ZFNs and TALENs.",
            "created_date": datetime(2014, 1, 15, tzinfo=timezone.utc),
        },
        {
            "title": "Renewable Energy Sources and Grid Integration Challenges",
            "abstract": "This paper explores the technical challenges of integrating variable renewable energy sources like solar and wind into traditional power grids.",
            "keywords": ["Energy", "Renewable Energy", "Solar Power", "Wind Power", "Power Grid"],
            "text": "The primary challenges discussed are intermittency and grid stability. Solutions like energy storage systems (batteries), smart grid technologies, and improved forecasting models are evaluated for their effectiveness in creating a reliable and sustainable energy future.",
            "created_date": datetime(2020, 8, 5, tzinfo=timezone.utc),
        },
         {
            "title": "A Study on Deep Reinforcement Learning for Robotic Manipulation",
            "abstract": "This work presents a novel deep reinforcement learning (DRL) framework that enables robots to learn complex manipulation skills from raw pixel inputs.",
            "keywords": ["AI", "Robotics", "Reinforcement Learning", "Deep Learning"],
            "text": "The framework utilizes a combination of convolutional neural networks for vision and recurrent neural networks for temporal understanding. We demonstrate its effectiveness on tasks such as object grasping and stacking, showing superior performance compared to traditional control methods.",
            "created_date": datetime(2019, 5, 30, tzinfo=timezone.utc),
        },
    ]

    # Sử dụng WeaviateManager để thực hiện các thao tác
    try:
        with WeaviateManager(embedder=embedder) as manager:
            collection_name = "Papers"
            properties = [
                Property(name="title", data_type=DataType.TEXT),
                Property(name="abstract", data_type=DataType.TEXT),
                Property(name="keywords", data_type=DataType.TEXT_ARRAY),
                Property(name="text", data_type=DataType.TEXT),
                Property(name="created_date", data_type=DataType.DATE),
            ]
            
            print(f"\n[1/3] Đang tạo collection '{collection_name}' (sẽ xóa nếu đã tồn tại)...")
            manager.create_collection(name=collection_name, properties=properties, force_recreate=True)
            print(f"✅ Collection '{collection_name}' đã được tạo thành công.")

            print(f"\n[2/3] Đang thêm {len(papers)} tài liệu vào collection...")
            for i, paper in enumerate(papers):
                manager.add(
                    collection_name=collection_name,
                    title=paper["title"],
                    abstract=paper["abstract"],
                    keywords=paper["keywords"],
                    text=paper["text"],
                    created_date=paper["created_date"],
                )
                print(f"  -> Đã thêm tài liệu {i+1}/{len(papers)}: '{paper['title']}'")
            
            print("✅ Thêm dữ liệu hoàn tất.")

            print("\n[3/3] Kiểm tra nhanh bằng một truy vấn tìm kiếm...")
            query_text = "What are the challenges of renewable energy?"
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