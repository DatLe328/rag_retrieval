# rag_retrieval

- Windows:

```shell
# Activate virtual environment
.\.venv\Scripts\activate
# Assign PYTHONPATH 
$env:PYTHONPATH = (Get-Location).Path

```

- Linux

```
# Activate virtual environment
source venv/bin/activate
# Assign PYTHONPATH 
export PYTHONPATH=$(pwd)
```

- `.env`

```
# Weaviate
WEAVIATE_HOST=localhost
WEAVIATE_PORT=8080
WEAVIATE_GRPC_PORT=50051
WEAVIATE_COLLECTION_NAME=Papers

# Ollama
EMBEDDER_PROVIDER=ollama
EMBED_MODEL=nomic-embed-text
CHAT_PROVIDER=ollama
CHAT_MODEL=llama3.2:3b

# Reranker
# RERANKER_PROVIDER=bge
# RERANKER_MODEL=BAAI/bge-reranker-v2-m3
RERANKER_PROVIDER=jina
RERANKER_MODEL=jinaai/jina-reranker-v2-base-multilingual


# Hyperparams
MULTI_QUERY_N=5
HYBRID_ALPHA=0.6
CANDIDATE_POOL=200
RERANK_TOPK=5

# Flask
RAG_FLASK_PORT=5000
RAG_FLASK_DEBUG=True
```