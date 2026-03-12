"""
Base Embedding — Abstract base class cho tất cả embedding models.

Tương thích với LangChain Embeddings interface:
    - embed_documents(texts) → list[list[float]]
    - embed_query(text) → list[float]

Subclass chỉ cần implement `_embed(texts)` — core embedding logic.
Các convenience methods (embed_query, batching, async) được base class xử lý.

Usage::

    class MyEmbedding(BaseEmbedding):
        def __init__(self):
            super().__init__(model_name="my-model", dimensions=512)

        def _embed(self, texts: list[str]) -> list[list[float]]:
            return call_my_api(texts)

    embedding = MyEmbedding()
    vectors = embedding.embed_documents(["hello", "world"])
    query_vec = embedding.embed_query("search term")
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from langchain_core.embeddings import Embeddings

from utils.logger import setup_logger

logger = setup_logger(__name__)


class BaseEmbedding(Embeddings, ABC):
    """
    Abstract base class cho embedding models.

    Kế thừa LangChain `Embeddings` để tương thích với:
    - VectorStore (ChromaDB, FAISS, Pinecone, ...)
    - LangGraph InMemoryStore (index={"embed": ...})
    - Retrievers, RAG chains, etc.

    Subclass CHỈ CẦN implement:
        - `_embed(texts: list[str]) -> list[list[float]]`

    Base class cung cấp:
        - embed_documents() — batch processing + logging
        - embed_query()     — single query embedding
        - aembed_documents() / aembed_query() — async versions
        - Validation, error handling, metadata
    """

    def __init__(
        self,
        model_name: str,
        dimensions: int,
    ) -> None:
        """
        Args:
            model_name: Tên model embedding (e.g., "voyage-3-lite", "text-embedding-3-small")
            dimensions: Số chiều output vector
        """
        self._model_name = model_name
        self._dimensions = dimensions

        logger.info(
            "Embedding initialized — model=%s, dims=%d",
            model_name, dimensions,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def model_name(self) -> str:
        """Tên model embedding."""
        return self._model_name

    @property
    def dimensions(self) -> int:
        """Số chiều của embedding vector."""
        return self._dimensions

    # ------------------------------------------------------------------
    # Abstract — subclass PHẢI implement
    # ------------------------------------------------------------------

    @abstractmethod
    def _embed(self, texts: list[str]|str) -> list[list[float]]:
        """
        Core embedding logic — gọi model/API để embed texts.

        Args:
            texts: Danh sách texts cần embed (đã được batch sẵn)

        Returns:
            List of embedding vectors, mỗi vector có `dimensions` chiều.

        Raises:
            Exception: Nếu API call thất bại.
        """
        ...

    # ------------------------------------------------------------------
    # LangChain Embeddings Interface
    # ------------------------------------------------------------------

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Embed danh sách documents.

        Đây là method chính cho indexing (lưu vào VectorStore).

        Args:
            texts: Danh sách documents cần embed.

        Returns:
            List of embedding vectors.
        """
        if not texts:
            return []

        logger.debug(
            "Embedding %d documents — model=%s",
            len(texts), self._model_name,
        )

        embeddings = self._embed(texts)

        logger.debug(
            "Embedded %d documents → %d vectors (dim=%d)",
            len(texts), len(embeddings), self._dimensions,
        )

        return embeddings

    def embed_query(self, text: str) -> list[float]:
        """
        Embed single query text.

        Đây là method cho search/retrieval.
        Một số models phân biệt query vs document embedding
        (e.g., Voyage dùng input_type="query" vs "document").

        Args:
            text: Query text cần embed.

        Returns:
            Embedding vector.
        """
        logger.debug("Embedding query — model=%s", self._model_name)
        result = self._embed(text)
        return result

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"model={self._model_name!r}, "
            f"dims={self._dimensions})"
        )
