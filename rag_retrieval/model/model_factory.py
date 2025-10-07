from .base_models import BaseLLMModel
from .wrapper.llm_ollama import OllamaChatModel
from .base_models import BaseReranker
from .wrapper.reranker_bge import BGEReranker
from .wrapper.reranker_jina import JinaReranker


def get_chat_model(provider: str, model_name: str) -> BaseLLMModel:
    if provider.lower() == "ollama":
        return OllamaChatModel(model_name=model_name)
    else:
        raise ValueError(f"Nhà cung cấp chat model '{provider}' không được hỗ trợ.")


def get_reranker(provider: str, model_name: str) -> BaseReranker:
    if provider.lower() == "bge":
        return BGEReranker(model_name=model_name)
    elif provider.lower() == "jina":
        return JinaReranker(model_name=model_name)
    else:
        raise ValueError(f"Nhà cung cấp reranker '{provider}' không được hỗ trợ.")