"""
Document Store Factory — Quản lý việc khởi tạo và cung cấp Document Store.

Mục tiêu chính: Cung cấp 1 interface thống nhất để các caller lấy ra
Document Store (thường trả về BaseDocumentStore object) mà không cần quan tâm
bên dưới sử dụng Qdrant hay provider khác, đảm bảo Dependency Injection.
"""

from __future__ import annotations

from src.data.documents.base import BaseDocumentStore
from src.settings import settings
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

ALLOWLIST = ["qdrant"]

class DocumentStoreFactory:
    """
    Factory để tải và cung cấp base class cho Vector Document DB.

    Giữ trạng thái Singleton của base Document Store để dùng trong toàn app,
    tránh khởi tạo lại DB Client mỗi khi cần gọi.
    """

    _store: BaseDocumentStore | None = None

    @classmethod
    def get(cls, name: str = "qdrant", **kwargs) -> BaseDocumentStore:
        """
        Lấy instance của Document Store. Dùng Singleton pattern để chỉ khởi tạo 1 lần.

        Returns:
            Biến thể subclass tuân thủ BaseDocumentStore interface. (vd: QdrantDocumentStore)
        """

        if name not in ALLOWLIST:
            raise ValueError(f"Không hỗ trợ model: {name}, please use one of {ALLOWLIST}")

        if name == "qdrant":
            return cls.__get_qdrant(**kwargs)
        else:
            raise ValueError(f"Unknown document store type: {name}")

    @staticmethod
    def __get_qdrant(**kwargs) -> BaseDocumentStore:
        from src.data.documents.stores.qdrant_store import QdrantDocumentStore
        return QdrantDocumentStore.get_instance(**kwargs)