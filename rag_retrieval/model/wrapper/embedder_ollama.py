import requests
from typing import List, Optional
import os
from ..base_models import BaseEmbedder, BaseLLMModel

class OllamaEmbedder(BaseEmbedder):

    def __init__(self, model_name: str = "nomic-embed-text"):
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://10.1.1.237:11434")
        self.model_name = model_name

    def get_embedding(self, text: str):
        resp = requests.post(
            f"{self.base_url}/api/embeddings",
            json={"model": self.model_name, "prompt": text}
        )
        resp.raise_for_status()
        return resp.json()["embedding"]

    def get_embeddings(self, texts: list[str]):
        return [self.get_embedding(t) for t in texts]