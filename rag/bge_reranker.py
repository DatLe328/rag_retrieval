from sentence_transformers import CrossEncoder

class Reranker:
    def __init__(self, model_name="BAAI/bge-reranker-v2-m3"):
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, docs):
        pairs = [[query, d["properties"]["abstract"]] for d in docs]
        scores = self.model.predict(pairs)
        for d, s in zip(docs, scores):
            d["rerank_score"] = float(s)
        return sorted(docs, key=lambda x: x["rerank_score"], reverse=True)
