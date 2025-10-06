import requests
from typing import List, Optional
from tqdm import tqdm
import os
from .base_models import BaseEmbedder, BaseLLMModel

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


class OllamaChatModel(BaseLLMModel):
    """Triển khai cụ thể của BaseChatModel cho Ollama."""
    
    def __init__(self, model_name: str, ollama_base_url: str = "http://10.1.1.237:11434"):
        super().__init__(model_name=model_name)
        self.base_url = ollama_base_url
        self.chat_url = f"{self.base_url}/api/chat"
        self._check_server()

    def _check_server(self):
        try:
            requests.get(self.base_url)
        except requests.exceptions.RequestException:
            raise ConnectionError(f"Không thể kết nối tới Ollama tại {self.base_url}")

    def generate(self, user_prompt: str, system_prompt: Optional[str] = None) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": False
        }

        try:
            response = requests.post(self.chat_url, json=payload)
            response.raise_for_status()
            content = response.json().get("message", {}).get("content", "").strip()
            # Dọn dẹp output
            return content.strip('"')
        except requests.exceptions.RequestException as e:
            print(f"Lỗi khi giao tiếp với Ollama Chat API: {e}")
            return f"Lỗi: Không thể nhận phản hồi từ model {self.model_name}."