"""
Vector Memory — Long-term memory qua ChromaDB (Agentic RAG).

Agent tự quyết định khi nào cần retrieve thông tin từ vector store.
"""

from __future__ import annotations

from typing import Any

from services.vector_store_service import VectorStoreService
from utils.logger import setup_logger

logger = setup_logger(__name__)


class VectorMemory:
    """
    Long-term memory sử dụng vector store.

    Usage::

        vm = VectorMemory()
        vm.store("Python ra đời năm 1991", {"source": "wiki"})
        results = vm.search("Python được tạo ra khi nào?")
        context = vm.format_context(results)
    """

    def __init__(self, vector_store: VectorStoreService | None = None) -> None:
        self._store = vector_store or VectorStoreService()

    def store(
        self,
        content: str,
        metadata: dict[str, Any] | None = None,
        doc_id: str | None = None,
    ) -> str:
        """Lưu thông tin vào long-term memory."""
        return self._store.add_document(content, metadata, doc_id)

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Tìm thông tin liên quan."""
        return self._store.similarity_search(query, top_k)

    def format_context(self, results: list[dict[str, Any]]) -> str:
        """Format kết quả thành context string cho LLM prompt."""
        if not results:
            return "No relevant information found in memory."

        parts = []
        for i, r in enumerate(results, 1):
            source = r.get("metadata", {}).get("source", "unknown")
            parts.append(f"[{i}] ({source}) {r['content']}")
        return "\n\n".join(parts)

    @property
    def count(self) -> int:
        return self._store.count
