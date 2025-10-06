from .base_models import BaseEmbedder, BaseLLMModel
from .ollama_models import OllamaEmbedder, OllamaChatModel
from .base_models import BaseReranker
from .bge_reranker import BGEReranker
# from openai_models import OpenAIEmbedder, OpenAIChatModel 

def get_embedder(provider: str, model_name: str) -> BaseEmbedder:
    if provider.lower() == "ollama":
        return OllamaEmbedder(model_name=model_name)
    # elif provider.lower() == "openai":
    #     return OpenAIEmbedder(model_name=model_name)
    else:
        raise ValueError(f"Nhà cung cấp embedder '{provider}' không được hỗ trợ.")

def get_chat_model(provider: str, model_name: str) -> BaseLLMModel:
    if provider.lower() == "ollama":
        return OllamaChatModel(model_name=model_name)
    # elif provider.lower() == "openai":
    #     return OpenAIChatModel(model_name=model_name)
    else:
        raise ValueError(f"Nhà cung cấp chat model '{provider}' không được hỗ trợ.")


def get_reranker(provider: str, model_name: str) -> BaseReranker:
    if provider.lower() == "bge":
        return BGEReranker(model_name=model_name)
    # elif provider.lower() == "cohere":
    #     return CohereReranker(...)
    else:
        raise ValueError(f"Nhà cung cấp reranker '{provider}' không được hỗ trợ.")