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
                results.append({"uuid": str(obj.uuid), "properties": obj.properties, "score": score})
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



if __name__ == "__main__":
    gen()