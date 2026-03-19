"""
Base Document Store — Abstract interface cho vector database.

Định nghĩa contract chung cho tất cả vector store implementations.
Bất kỳ backend nào (Qdrant, ChromaDB, Pinecone, ...) đều phải follow
interface này để đảm bảo tính nhất quán và dễ swap.

Interface gồm 3 nhóm:
    - Collection Management: create, delete, list collections
    - Document Operations: upsert (nhận points đã embed), delete documents
    - Search: vector search có/không filter

Document store KHÔNG chịu trách nhiệm embed. Embedding xảy ra ở bên ngoài
(caller tự embed rồi truyền vector vào). Điều này giúp tách biệt concerns
và linh hoạt hơn khi swap embedding model.

Usage::

    class MyStore(BaseDocumentStore):
        def upsert_documents(self, points, collection_name=None):
            ...

    store = MyStore(collection_name="fnb_menu")
    store.create_collection("fnb_menu")
    store.upsert_documents(points=[...])
    results = store.search(query_vector=[0.1, 0.2, ...], limit=5)
    store.delete_collection("fnb_menu")
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseDocumentStore(ABC):
    """
    Abstract base class cho Document Store.

    Mọi vector database implementation đều phải kế thừa class này
    và implement các abstract methods bên dưới.

    Lưu ý: Store KHÔNG embed. Caller tự embed rồi truyền points/vectors vào.
    """

    # ------------------------------------------------------------------
    # Collection Management — CRUD cho collections
    # ------------------------------------------------------------------

    @abstractmethod
    def create_collection(self, collection_name: str) -> None:
        """
        Tạo collection mới. Nếu đã tồn tại thì skip (idempotent).

        Args:
            collection_name: Tên collection cần tạo.
        """
        ...

    @abstractmethod
    def delete_collection(self, collection_name: str) -> None:
        """
        Xóa collection và toàn bộ documents bên trong.

        Args:
            collection_name: Tên collection cần xóa.
        """
        ...

    @abstractmethod
    def list_collections(self) -> list[str]:
        """
        Liệt kê tất cả collections hiện có.

        Returns:
            Danh sách tên collections.
        """
        ...

    # ------------------------------------------------------------------
    # Document Operations — upsert & delete
    # ------------------------------------------------------------------

    @abstractmethod
    def upsert_documents(
        self,
        points: list[Any],
        *,
        collection_name: str | None = None,
    ) -> None:
        """
        Upsert points (đã embed) vào vector store.

        Upsert = Insert nếu ID chưa tồn tại, Update nếu đã tồn tại.
        Points phải bao gồm vector, payload/metadata, và ID.

        Args:
            points: Danh sách points đã embed sẵn
                    (e.g., qdrant_client.models.PointStruct).
            collection_name: Tên collection đích (mặc định dùng collection đã set).
        """
        ...

    @abstractmethod
    def delete_documents(
        self,
        ids: list[int | str],
        *,
        collection_name: str | None = None,
    ) -> None:
        """
        Xóa documents theo danh sách IDs.

        Args:
            ids: Danh sách IDs cần xóa.
            collection_name: Tên collection (mặc định dùng collection đã set).
        """
        ...

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    @abstractmethod
    def search(
        self,
        query_vector: list[Any],
        *,
        collection_name: str | None = None,
        query_filter: Any = None,
        limit: int = 5,
        with_payload: bool = True,
    ) -> list[Any]:
        """
        Vector search — tìm documents gần nhất với query vector.

        Caller tự embed query thành vector rồi truyền vào.

        Args:
            query_vector: Vector đã embed của query.
            collection_name: Tên collection cần search.
            query_filter: Đối tượng Filter để lọc kết quả
                          (e.g., qdrant_client.models.Filter).
            limit: Số lượng kết quả tối đa.
            with_payload: Có trả về payload hay không.

        Returns:
            Danh sách kết quả tìm được.
        """
        ...

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
