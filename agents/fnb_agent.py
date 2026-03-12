"""
F&B Agent — AI tư vấn menu nhà hàng / quán ăn.

Kế thừa BaseAgent, thêm:
    - System prompt chuyên biệt cho tư vấn F&B
    - 3 tools: menu_search, get_product_detail, get_recommendations
    - Dependencies tự khởi tạo: Voyage4NanoEmbedding + QdrantDocumentStore
    - tenant_id inject qua InjectedToolArg (LLM không thấy)

Usage::

    from agents.fnb_agent import FnBAgent

    agent = FnBAgent(model=llm)

    # tenant_id truyền qua config — LLM không biết
    response = agent.invoke(
        "tôi muốn gì đó cay cay, ăn kèm bia",
        tenant_id="tenant-1",
        thread_id="session-abc",
    )
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agents.base_agent import BaseAgent
from data.documents.qdrant_store import (
    DEFAULT_QDRANT_PATH,
    QdrantDocumentStore,
    get_qdrant_client,
)
from data.embeddings.voyage_4_nano import Voyage4NanoEmbedding
from tools.get_product_detail_tool import create_get_product_detail_tool
from tools.get_recommendations_tool import create_get_recommendations_tool
from tools.menu_search_tool import create_menu_search_tool
from utils.logger import setup_logger

logger = setup_logger(__name__)

# ------------------------------------------------------------------
# Paths & constants
# ------------------------------------------------------------------

_DATA_PATH = Path(__file__).resolve().parent.parent / "external_data_storage" / "fnb.json"
_DEFAULT_COLLECTION = "fnb_menu"

# ------------------------------------------------------------------
# System prompt
# ------------------------------------------------------------------
from prompts.fnb_prompt import FNB_SYSTEM_PROMPT

# ------------------------------------------------------------------
# FnBAgent
# ------------------------------------------------------------------

class FnBAgent(BaseAgent):
    """
    Agent chuyên tư vấn F&B — search menu, xem chi tiết, gợi ý upsell.

    Dependencies tự khởi tạo:
        - Voyage4NanoEmbedding  → embed queries cho semantic search
        - QdrantDocumentStore   → vector store chứa menu data
        - 3 F&B tools           → tạo qua factory + InjectedToolArg

    Usage::

        agent = FnBAgent(model=llm)
        response = agent.invoke(
            "có gì ngon cho buổi tối không?",
            tenant_id="tenant-1",
        )
    """

    def __init__(
        self,
        model: Any,
        *,
        qdrant_path: str = DEFAULT_QDRANT_PATH,
        collection_name: str = _DEFAULT_COLLECTION,
        data_path: str | Path = _DATA_PATH,
        extra_tools: list | None = None,
        checkpointer: Any = None,
    ) -> None:
        """
        Args:
            model: LLM model instance.
            qdrant_path: Đường dẫn local Qdrant DB.
            collection_name: Tên collection trong Qdrant.
            data_path: Đường dẫn tới fnb.json.
            extra_tools: Tools bổ sung ngoài 3 tools F&B mặc định.
            checkpointer: Memory checkpointer (mặc định MemorySaver).
        """
        # Init dependencies
        embedding = Voyage4NanoEmbedding()
        client = get_qdrant_client(path=qdrant_path)
        document_store = QdrantDocumentStore(
            collection_name=collection_name,
            qdrant_client=client,
        )

        logger.info(
            "FnBAgent dependencies initialized — "
            "embedding=%s, store=%s, collection=%s",
            embedding.model_name, document_store, collection_name,
        )

        # Tạo tools qua factory (DI + InjectedToolArg)
        menu_search = create_menu_search_tool(
            document_store=document_store,
            embedding=embedding,
            collection_name=collection_name,
        )
        get_product_detail = create_get_product_detail_tool(data_path=data_path)
        get_recommendations = create_get_recommendations_tool(data_path=data_path)

        tools = [menu_search, get_product_detail, get_recommendations]
        if extra_tools:
            tools.extend(extra_tools)

        super().__init__(
            model=model,
            tools=tools,
            system_prompt=FNB_SYSTEM_PROMPT,
            checkpointer=checkpointer,
        )

        logger.info("FnBAgent ready — %d tools loaded", len(tools))