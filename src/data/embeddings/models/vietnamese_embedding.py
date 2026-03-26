"""
AITeamVN Vietnamese Embedding — SentenceTransformer model optimized for Vietnamese.

Model: AITeamVN/Vietnamese_Embedding
Dimensions: 1024
Max Sequence Length: 2048
Use case: Tiếng Việt, RAG, semantic search — chạy local, không cần API key.

Usage::

    from src.data.embeddings.vietnamese_embedding import VietnameseEmbedding

    embedding = VietnameseEmbedding()
    vectors = embedding.embed_documents(["xin chào", "tạm biệt"])
    query_vec = embedding.embed_query("tìm kiếm món ăn")
"""

from __future__ import annotations

from sentence_transformers import SentenceTransformer

from src.data.embeddings.base import BaseEmbedding
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# ------------------------------------------------------------------

class VietnameseEmbedding(BaseEmbedding):
    """
    AITeamVN Vietnamese Embedding — Local embedding model for Vietnamese.

    Đặc điểm:
        - Base model: BAAI/bge-m3.
        - Chạy local via SentenceTransformers (không cần API key).
        - 1024 dimensions.
        - Độ dài chuỗi tối đa: 2048 tokens.
    """

    def __init__(self, model_name: str, dimensions: int) -> None:
        super().__init__(
            model_name,
            dimensions,
        )
        
        # Load model và set thông số theo đúng tài liệu HuggingFace
        self._model = SentenceTransformer(
            self.model_name,
            truncate_dim=self.dimensions
        )
        self._model.max_seq_length = 2048

        logger.info("VietnameseEmbedding ready — model=%s", self.model_name)

    def _embed(self, texts: list[str] | str) -> list[list[float]] | list[float]:
        """
        Thực thi core embedding logic.
        Dù là query hay document, model này đều dùng hàm encode.
        """
        # SentenceTransformer.encode() tự detect list string hay single string
        embeddings = self._model.encode(texts)
        
        return embeddings.tolist()