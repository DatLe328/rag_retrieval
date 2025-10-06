import os


class Settings:
    # Weaviate
    WEAVIATE_HOST = os.getenv("WEAVIATE_HOST", "localhost")
    WEAVIATE_PORT = int(os.getenv("WEAVIATE_PORT", 8080))
    WEAVIATE_GRPC = int(os.getenv("WEAVIATE_GRPC_PORT", 50051))

    # Ollama
    EMBEDDER_PROVIDER = os.getenv("EMBEDDER_PROVIDER", "ollama")
    EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
    CHAT_PROVIDER = os.getenv("CHAT_PROVIDER", "ollama")
    CHAT_MODEL = os.getenv("CHAT_MODEL", "llama3.2:3b")

    # Reranker
    RERANKER_PROVIDER = os.getenv("RERANKER_PROVIDER", "bge")
    RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")

    # Hyperparams
    MULTI_QUERY_N = int(os.getenv("MULTI_QUERY_N", 5))
    HYBRID_ALPHA = float(os.getenv("HYBRID_ALPHA", 0.6))
    CANDIDATE_POOL = int(os.getenv("CANDIDATE_POOL", 200))
    RERANK_TOPK = int(os.getenv("RERANK_TOPK", 5))

    # Flask
    RAG_FLASK_PORT = int(os.getenv("RAG_FLASK_PORT", 5000))
    RAG_FLASK_DEBUG = bool(os.getenv("RAG_FLASK_DEBUG", False))