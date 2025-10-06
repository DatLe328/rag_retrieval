import weaviate
import weaviate.classes as wvc
from weaviate.classes.config import Configure, Property, DataType
from typing import List, Dict, Optional, Any
from datetime import datetime

class WeaviateManager:
    """Quản lý gọn cho Weaviate v4."""

    def __init__(self, host="localhost", port=8080, grpc_port=50051, headers=None):
        self.connection_params = {
            "host": host,
            "port": port,
            "grpc_port": grpc_port,
            "headers": headers or {}
        }
        self.client = None


    # def __enter__(self):
    #     try:
    #         self.client = weaviate.connect_to_local(**self.connection_params)
    #     except Exception as e:
    #         raise ConnectionError(f"Không thể kết nối tới Weaviate: {e}")
    #     return self
    def __enter__(self):
        try:
            http_host = f"http://{self.connection_params['host']}:{self.connection_params['port']}"
            self.client = weaviate.connect_to_custom(
                http_host=http_host,
                grpc_host=f"{self.connection_params['host']}:{self.connection_params['grpc_port']}",
                headers=self.connection_params.get("headers", {}),
            )
            print(f"✅ Đã kết nối Weaviate tại {http_host}")
        except Exception as e:
            raise ConnectionError(f"Không thể kết nối tới Weaviate: {e}")
        return self

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

    def add_paper_object(
        self,
        collection_name: str,
        title: str,
        abstract: str,
        keywords: list[str],
        text: str,
        created_date: datetime,
        vector: Optional[List[float]] = None,
    ):
        """Thêm một paper object vào collection."""
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

    def search(self, collection_name: str, query: Any, search_type: str = "vector", limit: int = 5) -> List[Dict]:
        """Trả về list gồm uuid, properties, score (càng cao càng tốt)."""
        collection = self.client.collections.get(collection_name)

        if search_type == "vector":
            response = collection.query.near_vector(near_vector=query, limit=limit, return_metadata=["distance"])
            results = []
            for obj in response.objects:
                distance = getattr(obj.metadata, "distance", None)
                score = 1 - distance if distance is not None else None
                results.append({"uuid": str(obj.uuid), "properties": obj.properties, "score": score})
            return results

        elif search_type == "text":
            response = collection.query.near_text(query=query, limit=limit, return_metadata=["score"])
        elif search_type == "bm25":
            response = collection.query.bm25(query=query, limit=limit, return_metadata=["score"])
        elif search_type == "hybrid":
            response = collection.query.hybrid(query=query, limit=limit, return_metadata=["score"])
        else:
            raise ValueError(f"Loại tìm kiếm '{search_type}' không được hỗ trợ.")

        return [{"uuid": str(obj.uuid), "properties": obj.properties, "score": getattr(obj.metadata, "score", None)} for obj in response.objects]


def gen():
    from datetime import timezone
    from rag_retrieval.model.ollama_models import OllamaEmbedder
    from faker import Faker
    import random

    fake = Faker()
    embedder = OllamaEmbedder()

    with WeaviateManager() as manager:
        collection_name = "Papers"
        properties = [
            Property(name="title", data_type=DataType.TEXT),
            Property(name="abstract", data_type=DataType.TEXT),
            Property(name="keywords", data_type=DataType.TEXT_ARRAY),
            Property(name="text", data_type=DataType.TEXT),
            Property(name="created_date", data_type=DataType.DATE),
        ]
        manager.create_collection(
            name=collection_name,
            properties=properties,
            force_recreate=True
        )

        # ============================
        # 1. Paper nổi tiếng (có thật)
        # ============================
        base_papers = [
            {
                "title": "Attention Is All You Need",
                "abstract": "This paper introduces the Transformer architecture, a novel approach replacing recurrence with self-attention.",
                "keywords": ["AI", "Transformer", "Deep Learning"],
                "text": "Full paper text describing the Transformer model, encoder-decoder structure, and scaled dot-product attention.",
                "created_date": datetime(2017, 6, 12, tzinfo=timezone.utc)
            },
            {
                "title": "BERT: Pre-training of Deep Bidirectional Transformers",
                "abstract": "BERT is designed to pre-train deep bidirectional representations using masked language modeling.",
                "keywords": ["AI", "NLP", "BERT"],
                "text": "Full paper text about BERT, bidirectional transformers, masked language modeling and next sentence prediction.",
                "created_date": datetime(2018, 10, 11, tzinfo=timezone.utc)
            },
            {
                "title": "GPT-3: Language Models are Few-Shot Learners",
                "abstract": "GPT-3 demonstrates that large language models can perform tasks with few-shot learning.",
                "keywords": ["AI", "NLP", "GPT"],
                "text": "Details about GPT-3's architecture, 175 billion parameters, and its ability to perform in-context learning.",
                "created_date": datetime(2020, 5, 28, tzinfo=timezone.utc)
            },
            {
                "title": "Scaling Laws for Neural Language Models",
                "abstract": "This paper explores how performance scales with model size, dataset size, and compute.",
                "keywords": ["AI", "Scaling", "Language Model"],
                "text": "Empirical scaling laws for deep learning, showing predictable improvement with increased scale.",
                "created_date": datetime(2020, 1, 1, tzinfo=timezone.utc)
            },
        ]

        # ============================
        # 2. Sinh thêm dữ liệu fake
        # ============================
        extra_titles = [
            "Improving Neural Machine Translation with Transformers",
            "Large Language Models: Opportunities and Risks",
            "Graph Neural Networks: A Review",
            "Advances in Reinforcement Learning",
            "Efficient Training of Deep Neural Networks",
            "Evaluation Methods for Text Generation Models",
            "Applications of AI in Healthcare",
            "Natural Language Understanding Benchmarks",
            "Self-Supervised Learning in Vision",
            "AI for Scientific Discovery"
        ]

        fake_papers = []
        for title in extra_titles:
            fake_papers.append({
                "title": title,
                "abstract": fake.text(max_nb_chars=200),
                "keywords": random.sample(
                    ["AI", "NLP", "Transformer", "Deep Learning", "Graph", "RL", "Scaling", "Healthcare", "Vision"], 3
                ),
                "text": fake.text(max_nb_chars=1000),
                "created_date": fake.date_time_between(start_date="-6y", end_date="now", tzinfo=timezone.utc)
            })

        # thêm nhiều paper giả nữa
        for i in range(20):
            fake_papers.append({
                "title": f"Research Paper #{i+1}",
                "abstract": fake.text(max_nb_chars=250),
                "keywords": random.sample(
                    ["AI", "NLP", "Transformer", "Deep Learning", "Graph", "RL", "Scaling", "Healthcare", "Vision"], 3
                ),
                "text": fake.text(max_nb_chars=1500),
                "created_date": fake.date_time_between(start_date="-10y", end_date="now", tzinfo=timezone.utc)
            })

        # gộp tất cả
        all_papers = base_papers + fake_papers

        # ============================
        # 3. Thêm vào DB
        # ============================
        vectors = embedder.get_embeddings([p["title"] for p in all_papers])

        for paper, vec in zip(all_papers, vectors):
            manager.add_paper_object(
                collection_name=collection_name,
                title=paper["title"],
                abstract=paper["abstract"],
                keywords=paper["keywords"],
                text=paper["text"],
                created_date=paper["created_date"],
                vector=vec
            )

        print(f"✅ Đã thêm {len(all_papers)} papers vào collection {collection_name}")


def test():
    from datetime import timezone
    from rag_retrieval.model.ollama_models import OllamaEmbedder
    from dotenv import load_dotenv

    load_dotenv()

    papers_data = [
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

    try:
        embedder = OllamaEmbedder()
        paper_vectors = embedder.get_embeddings([p["title"] for p in papers_data])

        with WeaviateManager() as manager:
            collection_name = "Papers"
            properties = [
                Property(name="title", data_type=DataType.TEXT),
                Property(name="abstract", data_type=DataType.TEXT),
                Property(name="keywords", data_type=DataType.TEXT_ARRAY),
                Property(name="text", data_type=DataType.TEXT),
                Property(name="created_date", data_type=DataType.DATE),
            ]
            manager.create_collection(name=collection_name, properties=properties, force_recreate=True)

            for paper, vec in zip(papers_data, paper_vectors):
                manager.add_paper_object(
                    collection_name=collection_name,
                    title=paper["title"],
                    abstract=paper["abstract"],
                    keywords=paper["keywords"],
                    text=paper["text"],
                    created_date=paper["created_date"],
                    vector=vec,
                )

            # Vector search
            query_text = "What is a Transformer architecture?"
            query_vector = embedder.get_embedding(query_text)
            search_results = manager.search(collection_name, query_vector, search_type="vector", limit=2)

            print(f"\n🔍 Vector search cho: '{query_text}'")
            for res in search_results:
                print(f"  - {res['properties'].get('title')} (Score: {res['score']})")

            # BM25 search
            query_text = "Transformer architecture"
            search_results = manager.search(collection_name, query_text, search_type="bm25", limit=2)

            print(f"\n🔍 BM25 search cho: '{query_text}'")
            for res in search_results:
                print(f"  - {res['properties'].get('title')} (Score: {res['score']})")

    except Exception as e:
        print(f"Lỗi trong Kịch bản 1: {e}")


if __name__ == "__main__":
    gen()