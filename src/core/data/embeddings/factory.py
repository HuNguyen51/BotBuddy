"""
Embedding Factory — Khởi tạo các embedding models sử dụng Lazy Load.

Giúp tránh việc import các thư viện nặng (như sentence_transformers) 
từ đầu ứng dụng nếu model đó không thực sự được gọi tới.

Usage::

    from src.data.embeddings.factory import EmbeddingFactory

    # Chỉ khi nào gọi method này thì vietnamese_embedding mới được import
    embedding = EmbeddingFactory.VietnameseEmbedding()
"""

from __future__ import annotations

from src.core.data.embeddings.base import BaseEmbedding

ALLOWLIST = [
    "AITeamVN/Vietnamese_Embedding",
    "voyageai/voyage-4-nano"
]


class EmbeddingFactory:
    """
    Factory class áp dụng Lazy Loading cho việc khởi tạo Embedding.
    """

    @classmethod
    def get(cls, name: str = "AITeamVN/Vietnamese_Embedding", **kwargs) -> BaseEmbedding:
        """
        Khởi tạo embedding object dựa trên tên model.
        Nếu không truyền params, sẽ sử dụng config mặc định từ settings.
        """

        if name not in ALLOWLIST:
            raise ValueError(f"Không hỗ trợ model: {name}, please use one of {ALLOWLIST}")

        if name == "AITeamVN/Vietnamese_Embedding":
            return cls.__get_vietnamese_embedding(**kwargs)
        elif name == "voyageai/voyage-4-nano":
            return cls.__get_voyage_4_nano(**kwargs)

    @staticmethod
    def __get_vietnamese_embedding(**kwargs) -> BaseEmbedding:
        from src.core.data.embeddings.models.vietnamese_embedding import VietnameseEmbedding
        return VietnameseEmbedding.get_instance(**kwargs)

    @staticmethod
    def __get_voyage_4_nano(**kwargs) -> BaseEmbedding:
        from src.core.data.embeddings.models.voyage_4_nano import Voyage4NanoEmbedding
        return Voyage4NanoEmbedding.get_instance(**kwargs)  