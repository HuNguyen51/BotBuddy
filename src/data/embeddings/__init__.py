"""
Embeddings Package — Embedding models cho RAG & Semantic Search.

Usage::

    from src.data.embeddings import EmbeddingFactory

    embedding = EmbeddingFactory.create()
    vectors = embedding.embed_documents(["text1", "text2"])
"""

from src.data.embeddings.base import BaseEmbedding
from src.data.embeddings.factory import EmbeddingFactory

__all__ = [
    "BaseEmbedding", 
    "EmbeddingFactory"
]
