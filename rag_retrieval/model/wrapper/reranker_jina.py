from typing import List, Tuple
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from ..base_models import BaseReranker


class JinaReranker(BaseReranker):

    def __init__(self, model_name: str = "jinaai/jina-reranker-v2-base-multilingual", device: str = None):
        super().__init__(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, trust_remote_code=True)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        print(f"Loaded JinaReranker ({model_name}) on {self.device}")

    @torch.inference_mode()
    def rerank(self, query: str, documents: List[str], top_k: int = 5) -> List[Tuple[int, float, str]]:
        """Trả về danh sách (index, score, text) sắp xếp theo độ liên quan giảm dần."""
        pairs = [(query, doc) for doc in documents]
        batch = self.tokenizer(
            [q for q, _ in pairs],
            [d for _, d in pairs],
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)

        scores = self.model(**batch).logits.view(-1).float()
        sorted_indices = torch.argsort(scores, descending=True)

        results = [(int(i), float(scores[i]), documents[int(i)]) for i in sorted_indices[:top_k]]
        return results
