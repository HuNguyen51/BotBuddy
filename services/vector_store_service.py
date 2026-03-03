"""
Vector Store Service — ChromaDB Wrapper.

Cung cấp 2 hàm cốt lõi:
- add_document(): Lưu document + embedding vào ChromaDB
- similarity_search(): Tìm documents liên quan theo semantic

ChromaDB tự xử lý embedding (mặc định dùng all-MiniLM-L6-v2),
nên ta chỉ cần pass text vào.
"""

from __future__ import annotations

import uuid
from typing import Any

import chromadb

from configs.settings import settings
from utils.logger import setup_logger

logger = setup_logger(__name__)


class VectorStoreService:
    """
    ChromaDB vector store cho long-term memory / RAG.

    Usage::

        vs = VectorStoreService()
        vs.add_document("Python là ngôn ngữ lập trình", {"source": "wiki"})
        results = vs.similarity_search("ngôn ngữ lập trình nào phổ biến?")
    """

    def __init__(
        self,
        collection_name: str | None = None,
        persist_directory: str | None = None,
    ) -> None:
        self._collection_name = collection_name or settings.vector_store.collection_name
        persist_dir = persist_directory or settings.vector_store.persist_directory

        self._client = chromadb.PersistentClient(path=persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
        )

        logger.info(
            "Vector store initialized — collection=%s, persist=%s, count=%d",
            self._collection_name, persist_dir, self._collection.count(),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_document(
        self,
        content: str,
        metadata: dict[str, Any] | None = None,
        doc_id: str | None = None,
    ) -> str:
        """
        Thêm document vào vector store.

        Args:
            content: Nội dung text
            metadata: Metadata bổ sung (source, category, ...)
            doc_id: Custom ID, tự generate nếu không truyền

        Returns:
            Document ID
        """
        doc_id = doc_id or str(uuid.uuid4())

        self._collection.add(
            documents=[content],
            metadatas=[metadata or {}],
            ids=[doc_id],
        )

        logger.info("Vector store add — doc_id=%s, length=%d", doc_id, len(content))
        return doc_id

    def add_documents(
        self,
        contents: list[str],
        metadatas: list[dict[str, Any]] | None = None,
        ids: list[str] | None = None,
    ) -> list[str]:
        """Batch add nhiều documents."""
        ids = ids or [str(uuid.uuid4()) for _ in contents]
        metadatas = metadatas or [{} for _ in contents]

        self._collection.add(
            documents=contents,
            metadatas=metadatas,
            ids=ids,
        )

        logger.info("Vector store batch add — count=%d", len(contents))
        return ids

    def similarity_search(
        self,
        query: str,
        top_k: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Tìm documents tương tự nhất với query.

        Returns:
            List of dicts: [{"content": ..., "metadata": ..., "distance": ...}]
        """
        kwargs: dict[str, Any] = {
            "query_texts": [query],
            "n_results": min(top_k, self._collection.count() or 1),
        }
        if where:
            kwargs["where"] = where

        results = self._collection.query(**kwargs)

        documents = []
        for i in range(len(results["ids"][0])):
            documents.append({
                "id": results["ids"][0][i],
                "content": results["documents"][0][i],
                "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                "distance": results["distances"][0][i] if results["distances"] else None,
            })

        logger.info("Vector store search — query='%s', results=%d", query[:80], len(documents))
        return documents

    def delete_document(self, doc_id: str) -> None:
        """Xóa document theo ID."""
        self._collection.delete(ids=[doc_id])
        logger.info("Vector store delete — doc_id=%s", doc_id)

    @property
    def count(self) -> int:
        """Số documents trong collection."""
        return self._collection.count()
