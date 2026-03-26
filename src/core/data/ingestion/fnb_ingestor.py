"""
FnB Data Ingestor — Pipeline orchestrator cho việc nạp dữ liệu F&B vào vector store.

Full flow: Load JSON → Score → Filter < threshold → Build document text
→ Embed → Build points → document_store.upsert_documents()

Document store và embedding đều được inject từ bên ngoài (Dependency Injection).

Usage::

    from src.core.data.embeddings import Voyage4NanoEmbedding
    from src.core.data.documents import QdrantDocumentStore, get_qdrant_client
    from src.core.data.scoring import ProductScorer
    from src.core.data.ingestion import FnBDataIngestor

    # Init dependencies
    embedding = Voyage4NanoEmbedding()
    client = get_qdrant_client(path="external_data_storage/qdrant_db")
    store = QdrantDocumentStore(collection_name="fnb_menu", qdrant_client=client)
    scorer = ProductScorer(threshold=60)

    # Init ingestor (inject all dependencies)
    ingestor = FnBDataIngestor(
        document_store=store,
        embedding=embedding,
        scorer=scorer,
    )

    # Run
    report = ingestor.ingest_from_file(
        file_path=Path("external_data_storage/fnb.json"),
        collection_name="fnb_menu",
    )
    print(report.scoring_report.summary())
"""

from __future__ import annotations

import uuid
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qdrant_client.models import PointStruct

from src.core.data.documents.base import BaseDocumentStore
from src.core.data.embeddings.base import BaseEmbedding
from src.core.data.scoring.product_scorer import ProductScorer, ScoringReport
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


# ------------------------------------------------------------------
# Report
# ------------------------------------------------------------------

@dataclass
class IngestionReport:
    """Báo cáo kết quả ingestion pipeline."""

    total_products: int
    """Tổng products đầu vào."""

    scored_passed: int
    """Số products đạt chất lượng (score >= threshold)."""

    scored_failed: int
    """Số products bị loại (score < threshold)."""

    upserted: int
    """Số documents đã upsert thành công vào vector store."""

    collection_name: str
    """Tên collection đích."""

    scoring_report: ScoringReport
    """Chi tiết scoring report."""

    def summary(self) -> str:
        """Summary string cho log/dashboard."""
        return (
            f"Ingestion hoàn tất: {self.upserted}/{self.total_products} products "
            f"đã nạp vào collection '{self.collection_name}'. "
            f"{self.scoring_report.summary()}"
        )


# ------------------------------------------------------------------
# FnB Data Ingestor
# ------------------------------------------------------------------

class FnBDataIngestor:
    """
    Pipeline orchestrator: JSON → Score → Filter → Embed → Upsert.

    Dependencies (all injected):
        - document_store: BaseDocumentStore — nhận points đã embed, upsert vào DB
        - embedding: BaseEmbedding — embed document text thành vectors
        - scorer: ProductScorer — chấm điểm chất lượng data
    """

    def __init__(
        self,
        document_store: BaseDocumentStore,
        embedding: BaseEmbedding,
        scorer: ProductScorer | None = None,
    ) -> None:
        """
        Args:
            document_store: Vector store implementation (chỉ lo upsert, không embed).
            embedding: Embedding model instance (lo embed text → vector).
            scorer: Product scorer instance. Mặc định: ProductScorer(threshold=60).
        """
        self._store = document_store
        self._embedding = embedding
        self._scorer = scorer or ProductScorer()

        logger.info(
            "FnBDataIngestor initialized — threshold=%d, embedding=%s",
            self._scorer.threshold, self._embedding.model_name,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ingest_from_file(
        self,
        file_path: str | Path,
        collection_name: str,
    ) -> IngestionReport:
        """
        Full pipeline từ file JSON.

        1. Load JSON
        2. Score từng product
        3. Filter products < threshold
        4. Build document text + metadata
        5. Embed document texts
        6. Build PointStruct list
        7. Upsert vào vector store
        8. Trả về report

        Args:
            file_path: Đường dẫn tới file JSON chứa danh sách products.
            collection_name: Tên collection đích trong vector store.

        Returns:
            IngestionReport với chi tiết scoring + ingestion.
        """
        file_path = Path(file_path)
        logger.info("Loading products from %s", file_path)

        with open(file_path, encoding="utf-8") as f:
            products: list[dict[str, Any]] = json.load(f)

        logger.info("Loaded %d products from file", len(products))

        return self.ingest_products(products, collection_name)

    def ingest_products(
        self,
        products: list[dict[str, Any]],
        collection_name: str,
    ) -> IngestionReport:
        """
        Full pipeline từ list[dict].

        Args:
            products: Danh sách product dicts.
            collection_name: Tên collection đích.

        Returns:
            IngestionReport.
        """
        # Step 1: Score
        scoring_report = self._scorer.score_products(products)

        logger.info(
            "Scoring done — %d passed, %d failed (threshold=%d)",
            scoring_report.passed, scoring_report.failed,
            scoring_report.threshold,
        )

        # Step 2: Early return nếu không có product nào đạt chuẩn
        if not scoring_report.passed_products:
            logger.warning("No products passed scoring — nothing to ingest")
            return IngestionReport(
                total_products=len(products),
                scored_passed=0,
                scored_failed=scoring_report.failed,
                upserted=0,
                collection_name=collection_name,
                scoring_report=scoring_report,
            )

        # Step 3: Build document texts + metadata cho products đạt chuẩn
        texts: list[str] = []
        payloads: list[dict[str, Any]] = []
        doc_ids: list[str] = []

        for scored_product in scoring_report.passed_products:
            product = scored_product.product

            doc_text = self._build_document_text(product)
            metadata = self._build_metadata(product, scored_product.score)
            doc_id = self._build_doc_id(product)

            texts.append(doc_text)
            payloads.append({**metadata, "text": doc_text})
            doc_ids.append(doc_id)

        # Step 4: Embed tất cả document texts
        logger.info("Embedding %d documents...", len(texts))
        vectors = self._embedding.embed_documents(texts)
        logger.info("Embedding complete — %d vectors generated", len(vectors))

        # Step 5: Build PointStruct list
        points = [
            PointStruct(
                id=doc_id,
                vector=vector,
                payload=payload,
            )
            for doc_id, vector, payload in zip(doc_ids, vectors, payloads)
        ]

        # Step 6: Upsert vào vector store
        logger.info(
            "Upserting %d points into collection '%s'",
            len(points), collection_name,
        )

        self._store.upsert_documents(
            points=points,
            collection_name=collection_name,
        )

        report = IngestionReport(
            total_products=len(products),
            scored_passed=scoring_report.passed,
            scored_failed=scoring_report.failed,
            upserted=len(points),
            collection_name=collection_name,
            scoring_report=scoring_report,
        )

        logger.info("Ingestion complete — %s", report.summary())

        return report

    # ------------------------------------------------------------------
    # Internal — Build document text & metadata
    # ------------------------------------------------------------------

    @staticmethod
    def _build_document_text(product: dict[str, Any]) -> str:
        """
        Tạo document text tối ưu cho semantic search.

        Kết hợp nhiều fields thành một đoạn text tự nhiên giàu ngữ nghĩa,
        giúp embedding model hiểu context tốt hơn từ nhiều góc độ.
        """
        parts = [
            f"Tên món: {product['name']}",
            f"Danh mục: {product.get('category', '')}",
            f"Mô tả: {product.get('description', '')}",
        ]

        ingredients = product.get("ingredients")
        if ingredients and isinstance(ingredients, list):
            parts.append(f"Nguyên liệu: {', '.join(ingredients)}")

        tags = product.get("tags")
        if tags and isinstance(tags, list):
            parts.append(f"Đặc điểm: {', '.join(tags)}")

        available_time = product.get("available_time")
        if available_time:
            parts.append(f"Phục vụ: {available_time}")

        price = product.get("price", 0)
        if isinstance(price, (int, float)) and price > 0:
            parts.append(f"Giá: {price:,}đ")

        return "\n".join(parts)

    @staticmethod
    def _build_metadata(
        product: dict[str, Any],
        quality_score: int,
    ) -> dict[str, Any]:
        """
        Tạo metadata structured cho vector store payload.

        Lưu ý: Hầu hết vector DBs (Qdrant, ChromaDB) payload/metadata
        hỗ trợ primitive types. Lists phải convert thành comma-separated string.
        """
        return {
            "tenant_id": product.get("tenant_id", ""),
            "item_id": product.get("id", ""),
            "name": product.get("name", ""),
            "price": product.get("price", 0),
            "category": product.get("category", ""),
            "tags": ", ".join(product.get("tags", [])),
            "available_time": product.get("available_time", ""),
            "image_url": product.get("image_url", ""),
            "quality_score": quality_score,
        }

    @staticmethod
    def _build_doc_id(product: dict[str, Any]) -> str:
        """
        Tạo document ID duy nhất.

        Dùng composite key "tenant_id::item_id" để tránh collision
        giữa các tenants. Khi re-ingest, upsert sẽ update thay vì duplicate.
        """
        tenant_id = product.get("tenant_id", "unknown")
        item_id = product.get("id", "unknown")
        doc_id = uuid.uuid5(uuid.NAMESPACE_DNS, f"{tenant_id}::{item_id}")
        return doc_id
