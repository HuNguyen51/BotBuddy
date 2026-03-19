"""
Embeddings Package — Embedding models cho RAG & Semantic Search.

Usage::

    from src.data.embeddings import Voyage4NanoEmbedding

    embedding = Voyage4NanoEmbedding()
    vectors = embedding.embed_documents(["text1", "text2"])
"""

from src.data.embeddings.base_embedding import BaseEmbedding
from src.data.embeddings.voyage_4_nano import Voyage4NanoEmbedding

__all__ = ["BaseEmbedding", "Voyage4NanoEmbedding"]
