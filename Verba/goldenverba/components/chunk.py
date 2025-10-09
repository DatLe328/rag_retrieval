from spacy.tokens import Doc, Span


class Chunk:
    def __init__(
        self,
        content: str = "",
        content_without_overlap: str = "",
        chunk_id: str = "",
        start_i: int = 0,
        end_i: int = 0,
        abstract: str = "",
        keywords: list[str] = None,
        ingestion_date: str = "",
    ):
        self.content = content
        self.title = ""
        self.chunk_id = chunk_id
        self.vector = None
        self.doc_uuid = None
        self.pca = [0, 0, 0]
        self.start_i = start_i
        self.end_i = end_i
        self.content_without_overlap = content_without_overlap
        self.labels = []
        self.abstract = abstract
        self.keywords = keywords if keywords is not None else []
        self.ingestion_date = ingestion_date

    def to_json(self) -> dict:
        """Convert the Chunk object to a dictionary."""
        return {
            "content": self.content,
            "chunk_id": self.chunk_id,
            "doc_uuid": self.doc_uuid,
            "title": self.title,
            "abstract": self.abstract,
            "keywords": self.keywords,
            "ingestion_date": self.ingestion_date,
            "pca": self.pca,
            "start_i": self.start_i,
            "end_i": self.end_i,
            "content_without_overlap": self.content_without_overlap,
            "labels": self.labels,
        }

    @classmethod
    def from_json(cls, data: dict):
        """Construct a Chunk object from a dictionary."""
        chunk = cls(
            content=data.get("content", ""),
            chunk_id=data.get("chunk_id", 0),
            start_i=data.get("start_i", 0),
            end_i=data.get("end_i", 0),
            content_without_overlap=data.get("content_without_overlap", ""),
            abstract=data.get("abstract", ""),
            keywords=data.get("keywords", []),
            ingestion_date=data.get("ingestion_date", ""),
        )
        chunk.doc_uuid = data.get("doc_uuid", "")
        return chunk
