"""
Ingestion Package — Data ingestion pipeline cho F&B products.

Orchestrate toàn bộ flow: Load → Score → Filter → Build text → Upsert.
Document store và embedding được inject từ bên ngoài.

Usage::

    from data.documents import QdrantDocumentStore
    from data.embeddings import Voyage4NanoEmbedding
    from data.scoring import ProductScorer
    from data.ingestion import FnBDataIngestor

    embedding = Voyage4NanoEmbedding()
    store = QdrantDocumentStore(collection_name="fnb_menu", embedding=embedding)
    scorer = ProductScorer(threshold=60)

    ingestor = FnBDataIngestor(document_store=store, scorer=scorer)
    report = ingestor.ingest_from_file(Path("data.json"), "fnb_menu")
    print(report.summary())
"""

from data.ingestion.fnb_ingestor import FnBDataIngestor, IngestionReport

__all__ = ["FnBDataIngestor", "IngestionReport"]
