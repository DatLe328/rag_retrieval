from FlagEmbedding import FlagReranker
from typing import List, Tuple
import torch
from ..base_models import BaseReranker


class BGEReranker(BaseReranker):
    def __init__(self, model_name: str = 'BAAI/bge-reranker-v2-m3', use_fp16: bool = True, device: str = "cuda:0"):
        super().__init__(model_name=model_name)
        
        if not torch.cuda.is_available():
            print("Cảnh báo: Không tìm thấy GPU. Reranker sẽ chạy trên CPU và có thể chậm.")
            use_fp16 = False
       
        try:
            use_fp16=True
            self.model = FlagReranker(model_name, use_fp16=use_fp16, device=device)
            print(f"Đã tải thành công model reranker '{self.model_name}'.")
        except Exception as e:
            raise RuntimeError(f"Lỗi khi tải model BGE Reranker: {e}")

    def rerank(self, query: str, documents: List[str], top_k: int = 5) -> List[Tuple[int, float, str]]:
        if not documents:
            return []

        print(f"Đang rerank {len(documents)} tài liệu cho truy vấn: '{query}'...")

        pairs = [[query, doc] for doc in documents]

        with torch.no_grad():
            scores = self.model.compute_score(pairs)

        results_with_scores = []
        for i, doc in enumerate(documents):
            results_with_scores.append({
                "original_index": i,
                "score": scores[i],
                "document": doc
            })
        
        sorted_results = sorted(results_with_scores, key=lambda x: x['score'], reverse=True)
        
        top_results = sorted_results[:top_k]

        formatted_output = [
            (res["original_index"], res["score"], res["document"])
            for res in top_results
        ]

        print(f"Rerank hoàn tất. Trả về {len(formatted_output)} kết quả hàng đầu.")
        return formatted_output