from abc import ABC, abstractmethod
from typing import List, Optional, Tuple


class BaseLLMModel(ABC):
    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        print(f"Khởi tạo Chat Model: {self.__class__.__name__} với model '{self.model_name}'")
    
    @abstractmethod
    def generate(self, user_prompt: str, system_prompt: Optional[str] = None) -> str:
        pass


class BaseReranker(ABC):
    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        print(f"Khởi tạo Reranker: {self.__class__.__name__} với model '{self.model_name}'")

    @abstractmethod
    def rerank(self, query: str, documents: List[str], top_k: int = 5) -> List[Tuple[int, float, str]]:
        pass