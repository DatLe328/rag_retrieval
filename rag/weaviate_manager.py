import weaviate
from weaviate.classes.config import Property, DataType, Configure
from typing import List, Dict


class WeaviateManager:
    """Weaviate client (chỉ dùng model provider integrations, ví dụ OpenAI)."""

    def __init__(self, url: str = "http://localhost:8080"):
        self.url = url
        self.client = None

    def __enter__(self):
        self.client = weaviate.connect_to_local(host=self.url.replace("http://", ""))
        print(f"✅ Connected to Weaviate at {self.url}")
        return self

    def __exit__(self, *args):
        if self.client:
            self.client.close()

    def create_collection(self, name: str, force_recreate: bool = False):
        """Tạo collection với text2vec-openai."""
        if force_recreate and self.client.collections.exists(name):
            self.client.collections.delete(name)

        if not self.client.collections.exists(name):
            self.client.collections.create(
                name=name,
                properties=[
                    Property(name="title", data_type=DataType.TEXT),
                    Property(name="abstract", data_type=DataType.TEXT),
                    Property(name="keywords", data_type=DataType.TEXT_ARRAY),
                    Property(name="text", data_type=DataType.TEXT),
                    Property(name="created_date", data_type=DataType.DATE),
                ],
                vector_config=Configure.Vectors.text2vec_openai(
                    model="text-embedding-3-small",
                    vectorize_properties=["title", "abstract", "text"]
                ),
            )
            print(f"✅ Created collection '{name}'")

    def insert(self, name: str, data: Dict):
        """Chèn dữ liệu (Weaviate tự embed)."""
        collection = self.client.collections.get(name)
        collection.data.insert(data)

    def hybrid_search(self, name: str, query: str, limit=5, alpha=0.5) -> List[Dict]:
        """Tìm kiếm bằng hybrid (BM25 + vector)."""
        collection = self.client.collections.get(name)
        response = collection.query.hybrid(
            query=query, alpha=alpha, limit=limit, return_metadata=["score"]
        )
        return [
            {"uuid": str(o.uuid), "properties": o.properties, "score": o.metadata.score}
            for o in response.objects
        ]

    def generate_answer(self, name: str, query: str, task: str = "Summarize context") -> str:
        """Sinh câu trả lời bằng generative-openai."""
        collection = self.client.collections.get(name)
        response = collection.generate.hybrid(query=query, grouped_task=task)
        return response.generated
