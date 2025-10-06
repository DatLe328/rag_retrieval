from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Weaviate
    WEAVIATE_URL: str = "http://localhost:8080"
    COLLECTION_NAME: str = "Papers"

    # Pipeline
    MULTI_QUERY_N: int = 3
    CANDIDATE_POOL: int = 50
    RERANK_TOPK: int = 5
    HYBRID_ALPHA: float = 0.6

    # Reranker
    RERANKER_MODEL: str = "BAAI/bge-reranker-v2-m3"

    class Config:
        env_file = ".env"

settings = Settings()
