"""
Menu Search Tool — Semantic search sản phẩm trong menu F&B.

Tool quan trọng nhất trong RAG pipeline. Nhận query tự nhiên từ user
(vd: "tôi muốn gì đó ngọt, không quá đặc"), embed query rồi search
trong vector store, trả về top sản phẩm phù hợp nhất.

Tenant-isolated: tenant_id được inject runtime qua InjectedToolArg,
LLM không thấy và không cần điền tenant_id — hệ thống tự inject từ
config["configurable"]["tenant_id"].

Factory pattern: dependencies (document_store, embedding) được inject
lúc tạo tool qua closure.

Usage::

    from src.data.documents import QdrantDocumentStore, get_qdrant_client
    from src.data.embeddings import Voyage4NanoEmbedding
    from src.tools.menu_search_tool import create_menu_search_tool

    embedding = Voyage4NanoEmbedding()
    client = get_qdrant_client(path="external_data_storage/qdrant_db")
    store = QdrantDocumentStore(collection_name="fnb_menu", qdrant_client=client)

    menu_search = create_menu_search_tool(document_store=store, embedding=embedding)

    # LLM chỉ thấy: menu_search(query, top_k)
    # tenant_id inject tự động từ config
    agent = BaseAgent(tools=[menu_search])
    agent.invoke("tìm món cay", config={"configurable": {"tenant_id": "tenant-1"}})
"""

from __future__ import annotations

from typing import Annotated, Any

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from qdrant_client.models import FieldCondition, Filter, MatchValue, ScoredPoint

from src.data.documents.base import BaseDocumentStore
from src.data.embeddings.base import BaseEmbedding
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


# ------------------------------------------------------------------
# Format helpers
# ------------------------------------------------------------------

def _format_search_results(results: list[ScoredPoint]) -> str:
    """Format Qdrant ScoredPoint results thành readable string cho LLM."""
    if not results:
        return "Không tìm thấy sản phẩm nào phù hợp với yêu cầu."

    formatted = []
    for i, point in enumerate(results, 1):
        payload = point.payload or {}
        score = round(point.score, 4) if point.score else 0

        price = payload.get("price", "N/A")
        price_str = f"{price:,}đ" if isinstance(price, (int, float)) else str(price)

        lines = [
            f"--- Kết quả {i} (Score: {score}) ---",
            f"ID: {payload.get('item_id', 'N/A')}",
            f"Tên: {payload.get('name', 'N/A')}",
            f"Giá: {price_str}",
            f"Danh mục: {payload.get('category', 'N/A')}",
            f"Mô tả: {payload.get('text', 'N/A')}",
            f"Tags: {payload.get('tags', 'N/A')}",
            f"Thời gian phục vụ: {payload.get('available_time', 'N/A')}",
        ]
        formatted.append("\n".join(lines))

    return "\n\n".join(formatted)


# ------------------------------------------------------------------
# Factory
# ------------------------------------------------------------------

def create_menu_search_tool(
    document_store: BaseDocumentStore,
    embedding: BaseEmbedding,
    collection_name: str = "fnb_menu",
):
    """
    Factory tạo menu_search tool với dependencies injected.

    Args:
        document_store: Vector store instance (đã init).
        embedding: Embedding model (để embed query).
        collection_name: Tên collection trong vector store.

    Returns:
        StructuredTool — truyền thẳng vào Agent(tools=[...]).
        LLM chỉ thấy: query, top_k. tenant_id tự inject từ config.
    """

    @tool
    def menu_search(
        query: str,
        top_k: int = 5,
        *,
        config: RunnableConfig,
    ) -> str:
        """Search menu sản phẩm theo ngữ nghĩa (semantic search).

        Sử dụng khi user mô tả mong muốn bằng ngôn ngữ tự nhiên, ví dụ:
        - "tôi muốn gì đó ngọt, không quá đặc"
        - "có món nào cay cay, ăn kèm bia không?"
        - "tìm món cho trẻ em"
        - "đồ uống mát lạnh giải nhiệt"

        Args:
            query: Mô tả tự nhiên về sản phẩm user muốn tìm.
            top_k: Số kết quả trả về (mặc định 5, tối đa 10).
            config: cấu hình của đoạn chat.
        """
        tenant_id = config.get("configurable", {}).get("tenant_id")
        top_k = min(max(top_k, 1), 10)

        logger.info(
            "menu_search — query='%s', tenant_id=%s, top_k=%d",
            query[:80], tenant_id, top_k,
        )

        # Embed query → vector
        query_vector = embedding.embed_query(query)

        # Tenant-isolated search
        tenant_filter = Filter(
            must=[
                FieldCondition(
                    key="tenant_id",
                    match=MatchValue(value=tenant_id),
                ),
            ]
        )

        results = document_store.search(
            query_vector=query_vector,
            collection_name=collection_name,
            query_filter=tenant_filter,
            limit=top_k,
        )

        logger.info("menu_search — found %d results", len(results))

        return _format_search_results(results)

    return menu_search
