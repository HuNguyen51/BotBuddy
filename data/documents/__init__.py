"""
Documents Package — Vector database interface cho RAG & Semantic Search.

Cung cấp abstract interface (BaseDocumentStore) và Qdrant implementation
(QdrantDocumentStore) với singleton client pattern.

Store KHÔNG chịu trách nhiệm embed — caller tự embed rồi truyền points vào.

Usage::

    from data.documents import QdrantDocumentStore, get_qdrant_client
    from data.embeddings import Voyage4NanoEmbedding
    from qdrant_client.models import PointStruct

    # Init
    client = get_qdrant_client(path="external_data_storage/qdrant_db")
    store = QdrantDocumentStore(collection_name="fnb_menu", qdrant_client=client)
    embedding = Voyage4NanoEmbedding()

    # Embed + upsert
    vectors = embedding.embed_documents(["Phở bò Hà Nội", "Bún chả"])
    points = [
        PointStruct(id=1, vector=vectors[0], payload={"text": "Phở bò"}),
        PointStruct(id=2, vector=vectors[1], payload={"text": "Bún chả"}),
    ]
    store.upsert_documents(points=points)

    # Search
    query_vec = embedding.embed_query("món phở")
    results = store.search(query_vector=query_vec, limit=5)

    # Cleanup
    store.delete_documents(ids=[1])
    store.delete_collection("old_collection")
"""

from data.documents.base_document_store import BaseDocumentStore
from data.documents.qdrant_store import QdrantDocumentStore, get_qdrant_client

__all__ = ["BaseDocumentStore", "QdrantDocumentStore", "get_qdrant_client"]
