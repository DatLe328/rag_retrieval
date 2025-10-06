from FlagEmbedding import FlagReranker
from typing import List, Tuple
import torch

from .base_models import BaseReranker

class BGEReranker(BaseReranker):
    def __init__(self, model_name: str = 'BAAI/bge-reranker-v2-m3', use_fp16: bool = True, device: str = "cuda:0"):
        super().__init__(model_name=model_name)
        
        if not torch.cuda.is_available():
            print("Cảnh báo: Không tìm thấy GPU. Reranker sẽ chạy trên CPU và có thể chậm.")
            use_fp16 = False
       
        try:
            use_fp16=True
            self.model = FlagReranker(model_name, use_fp16=use_fp16, device=device)
            print(f"✅ Đã tải thành công model reranker '{self.model_name}'.")
        except Exception as e:
            raise RuntimeError(f"Lỗi khi tải model BGE Reranker: {e}")

    # Refer speed
    def rerank(self, query: str, documents: List[str], top_k: int = 5) -> List[Tuple[int, float, str]]:
        """
        Tính toán điểm tương đồng, sắp xếp lại và trả về top_k kết quả tốt nhất.
        """
        if not documents:
            return []

        print(f"Đang rerank {len(documents)} tài liệu cho truy vấn: '{query}'...")

        # Tạo các cặp [query, document] để tính điểm
        pairs = [[query, doc] for doc in documents]

        # Tính toán điểm số cho tất cả các cặp
        # Model sẽ trả về một mảng các điểm số
        with torch.no_grad():
            scores = self.model.compute_score(pairs)

        # Kết hợp điểm số với tài liệu và chỉ số gốc của chúng
        # (chỉ số gốc rất quan trọng để biết tài liệu nào được chọn)
        results_with_scores = []
        for i, doc in enumerate(documents):
            results_with_scores.append({
                "original_index": i,
                "score": scores[i],
                "document": doc
            })
        
        # Sắp xếp các kết quả dựa trên điểm số từ cao đến thấp
        sorted_results = sorted(results_with_scores, key=lambda x: x['score'], reverse=True)
        
        # Chỉ lấy top_k kết quả hàng đầu
        top_results = sorted_results[:top_k]

        # Định dạng lại output theo yêu cầu của lớp cơ sở
        formatted_output = [
            (res["original_index"], res["score"], res["document"])
            for res in top_results
        ]

        print(f"✅ Rerank hoàn tất. Trả về {len(formatted_output)} kết quả hàng đầu.")
        return formatted_output

    def rerank_slow(self, query: str, documents: List[str], top_k: int = 5) -> List[Tuple[int, float, str]]:
        """
        Tính toán điểm tương đồng bằng batch encode, tránh warning HuggingFace.
        """
        if not documents:
            return []

        print(f"Đang rerank {len(documents)} tài liệu cho truy vấn: '{query}'...")

        # Tạo cặp [query, document]
        pairs = [(query, doc) for doc in documents]

        # Batch encode
        enc = self.model.tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.model.model.device)

        with torch.inference_mode():
            outputs = self.model.model(**enc)
            scores = outputs.logits.squeeze()

        # Convert tensor -> list of float
        if scores.dim() == 0:  # chỉ 1 giá trị
            scores = [scores.item()]
        else:
            scores = scores.tolist()

        results_with_scores = [
            {"original_index": i, "score": float(scores[i]), "document": documents[i]}
            for i in range(len(documents))
        ]

        sorted_results = sorted(results_with_scores, key=lambda x: x['score'], reverse=True)
        top_results = sorted_results[:top_k]

        formatted_output = [
            (res["original_index"], res["score"], res["document"])
            for res in top_results
        ]

        print(f"✅ Rerank hoàn tất. Trả về {len(formatted_output)} kết quả hàng đầu.")
        return formatted_output
