"""
Voyage 4 Nano Embedding — Local embedding model via SentenceTransformers.

Model: voyageai/voyage-4-nano
Dimensions: 2048 (truncated)
Use case: RAG, semantic search — chạy local, không cần API key.

Usage::

    from data.embeddings.voyage_4_nano import Voyage4NanoEmbedding

    embedding = Voyage4NanoEmbedding()
    vectors = embedding.embed_documents(["xin chào", "hello"])
    query_vec = embedding.embed_query("tìm kiếm")
"""

from __future__ import annotations

from sentence_transformers import SentenceTransformer

from data.embeddings.base_embedding import BaseEmbedding
from utils.logger import setup_logger

logger = setup_logger(__name__)

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------

MODEL_NAME = "voyageai/voyage-4-nano"
DIMENSIONS = 2048


class Voyage4NanoEmbedding(BaseEmbedding):
    """
    Voyage 4 Nano — lightweight local embedding model.

    Đặc điểm:
        - Chạy local via SentenceTransformers (không cần API key)
        - 2048 dimensions (truncated)
        - Phân biệt query vs document prompt tự động
          qua encode_query() / encode_document()
    """

    def __init__(self) -> None:
        super().__init__(
            model_name=MODEL_NAME,
            dimensions=DIMENSIONS,
        )
        self._model = SentenceTransformer(
            MODEL_NAME,
            trust_remote_code=True,
            truncate_dim=DIMENSIONS,
        )

        logger.info("Voyage4NanoEmbedding ready — model=%s", MODEL_NAME)

    def _embed(self, texts: list[str] | str) -> list[list[float]] | list[float]:
        """Embed texts dưới dạng documents."""
        if isinstance(texts, str):
            embeddings = self._model.encode_query(texts)
        else:
            embeddings = self._model.encode_document(texts)
        return embeddings.tolist()
