"""
Qdrant Document Store — Qdrant vector database implementation.

Singleton client pattern — chỉ khởi tạo QdrantClient MỘT LẦN duy nhất
trong suốt vòng đời ứng dụng, tái sử dụng cho mọi operations.

Hỗ trợ:
    - Collection management: create, delete, list collections
    - Document operations: upsert (embed + lưu), delete documents
    - Search: semantic search có/không filter

Usage::

    from data.documents import QdrantDocumentStore

    store = QdrantDocumentStore(collection_name="fnb_collection")

    # Upsert documents (embed + lưu)
    store.upsert_documents(
        texts=["Sao Hỏa là hành tinh đỏ"],
        metadatas=[{"category": "science"}],
        ids=[1],
    )

    # Search
    results = store.search("hành tinh đỏ", limit=3)

    # Collection management
    store.create_collection("new_collection")
    store.delete_collection("old_collection")
    print(store.list_collections())

Filter usage::

    from qdrant_client.models import Filter, FieldCondition, MatchValue

    f = Filter(must=[FieldCondition(key="category", match=MatchValue(value="a"))])
    results = store.search_with_filter("hành tinh", query_filter=f, limit=3)
"""

from __future__ import annotations

import threading
from typing import Any, Callable, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    Filter,
    PointStruct,
    ScoredPoint,
    VectorParams,
    PointIdsList 
)

from data.documents.base_document_store import BaseDocumentStore
from utils.logger import setup_logger

logger = setup_logger(__name__)


# Đường dẫn mặc định cho local Qdrant storage
DEFAULT_QDRANT_PATH = "external_data_storage/qdrant_db"

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------

DEFAULT_VECTOR_SIZE = 2048
DEFAULT_DISTANCE = Distance.DOT

global qdrant_client
qdrant_client: QdrantClient | None = None

def get_qdrant_client(**kwargs) -> QdrantClient:
    global qdrant_client
    if qdrant_client is None:
        qdrant_client = QdrantClient(**kwargs)
    return qdrant_client

class QdrantDocumentStore(BaseDocumentStore):
    """
    Qdrant-backed document store.

    Đặc điểm:
        - Singleton QdrantClient
        - Tự động tạo collection nếu chưa tồn tại
        - Embed documents/query qua pluggable BaseEmbedding
        - search_with_filter() nhận trực tiếp Filter object
          → dễ dàng inject bất kỳ điều kiện lọc nào từ bên ngoài
    """

    _instance: Optional["QdrantDocumentStore"] = None

    @classmethod
    def get_instance(
        cls, 
        collection_name: str,
        qdrant_client: QdrantClient,
        *,
        vector_size: int = DEFAULT_VECTOR_SIZE,
        distance: Distance = DEFAULT_DISTANCE,
    ) -> "QdrantDocumentStore":
        """
        Lấy singleton QdrantClient instance.

        Returns:
            QdrantClient instance (singleton).
        """
        if cls._instance is None:
            cls._instance = cls(
                collection_name=collection_name,
                qdrant_client=qdrant_client,
                vector_size=vector_size,
                distance=distance,
            )
        return cls._instance

    def __init__(
        self,
        collection_name: str,
        qdrant_client: QdrantClient,
        *,
        vector_size: int = DEFAULT_VECTOR_SIZE,
        distance: Distance = DEFAULT_DISTANCE,
    ) -> None:
        """
        Args:
            collection_name: Tên collection mặc định.
            embedding: Embedding model instance. Mặc định: Voyage4NanoEmbedding.
            qdrant_path: Đường dẫn local Qdrant DB.
            vector_size: Số chiều vector.
            distance: Metric khoảng cách (DOT, COSINE, EUCLID).
        """
        self._collection_name = collection_name
        self._client = qdrant_client
        self._vector_size = vector_size
        self._distance = distance

        # Đảm bảo collection mặc định tồn tại
        self.create_collection(collection_name)

        logger.info(
            "QdrantDocumentStore initialized — collection=%s, vector_size=%d, distance=%s",
            collection_name, vector_size, distance,
        )

    # ------------------------------------------------------------------
    # Collection Management
    # ------------------------------------------------------------------

    def create_collection(self, collection_name: str) -> None:
        """
        Tạo collection mới. Idempotent — nếu đã tồn tại thì skip.

        Args:
            collection_name: Tên collection cần tạo.
        """
        existing = [c.name for c in self._client.get_collections().collections]

        if collection_name not in existing:
            self._client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=self._vector_size,
                    distance=self._distance,
                ),
            )
            logger.info("Created collection — %s", collection_name)
        else:
            logger.debug("Collection already exists — %s", collection_name)

    def delete_collection(self, collection_name: str) -> None:
        """
        Xóa collection và toàn bộ documents bên trong.

        Args:
            collection_name: Tên collection cần xóa.
        """
        existing = [c.name for c in self._client.get_collections().collections]

        if collection_name in existing:
            self._client.delete_collection(collection_name=collection_name)
            logger.info("Deleted collection — %s", collection_name)
        else:
            logger.warning(
                "Collection does not exist, skipping delete — %s", collection_name,
            )

    def list_collections(self) -> list[str]:
        """
        Liệt kê tất cả collections hiện có.

        Returns:
            Danh sách tên collections.
        """
        collections = [
            c.name for c in self._client.get_collections().collections
        ]
        logger.debug("Listed %d collections", len(collections))
        return collections

    # ------------------------------------------------------------------
    # Document Operations
    # ------------------------------------------------------------------

    def upsert_documents(
        self,
        points: list[PointStruct],
        *,
        collection_name: str | None = None,
    ) -> None:
        """
        Upsert = Insert nếu ID chưa tồn tại, Update nếu đã tồn tại.

        Args:
            points: Nội dung documents.
        """
        target = collection_name or self._collection_name
        self.create_collection(target)

        operation_info = self._client.upsert(
            collection_name=target,
            points=points,
        )

        logger.info(
            "Upserted %d points — collection=%s, status=%s",
            len(points), target, operation_info.status,
        )

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
        target = collection_name or self._collection_name

        n_deleted = len(ids)
        ids = PointIdsList(points=ids)

        self._client.delete(
            collection_name=target,
            points_selector=ids,
        )

        logger.info(
            "Deleted %d documents — collection=%s", n_deleted, target,
        )

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query_vector: list[Any],
        *,
        collection_name: str | None = None,
        query_filter: Filter | None = None,
        limit: int = 5,
        with_payload: bool = True,
    ) -> list[ScoredPoint]:
        """
        Semantic search có điều kiện lọc.

        Filter object được truyền trực tiếp từ caller → dễ dàng injection
        bất kỳ điều kiện lọc phức tạp nào (must, should, must_not, nested, ...).

        Args:
            query_vector: Câu truy vấn.
            collection_name: Tên collection (mặc định dùng collection đã set).
            query_filter: Qdrant Filter object để lọc kết quả.
            limit: Số lượng kết quả tối đa.
            with_payload: Có trả về payload hay không.

        Returns:
            Danh sách ScoredPoint kết quả (đã filter).

        Example::

            from qdrant_client.models import Filter, FieldCondition, MatchValue

            f = Filter(
                must=[
                    FieldCondition(key="tenant_id", match=MatchValue(value="tenant-123")),
                    FieldCondition(key="category", match=MatchValue(value="food")),
                ]
            )
            results = store.search_with_filter("phở bò", query_filter=f, limit=10)
        """
        target = collection_name or self._collection_name

        results = self._client.query_points(
            collection_name=target,
            query=query_vector,
            query_filter=query_filter,
            with_payload=with_payload,
            limit=limit,
        ).points

        logger.debug(
            "Search with filter — collection=%s, found=%d",
            target, len(results),
        )
        return results

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @property
    def client(self) -> QdrantClient:
        """Trả về raw QdrantClient instance nếu cần thao tác nâng cao."""
        return self._client

    @property
    def collection_name(self) -> str:
        """Tên collection mặc định."""
        return self._collection_name

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"collection={self._collection_name!r}, "
            f"vector_size={self._vector_size}, "
            f"distance={self._distance})"
        )
