"""
Get Product Detail Tool — Lấy thông tin chi tiết của một sản phẩm.

Khi user hỏi chi tiết về một món cụ thể (vd: "bánh mì này có cay không?",
"cà phê này phù hợp uống buổi tối không?"), tool này fetch đầy đủ thông tin
từ JSON data source để AI có context chính xác nhất.

tenant_id được inject runtime qua InjectedToolArg — LLM chỉ cần truyền
product_id, hệ thống tự biết đang serve tenant nào.

Usage::

    from src.tools.get_product_detail_tool import create_get_product_detail_tool

    get_product_detail = create_get_product_detail_tool(
        data_path=Path("external_data_storage/fnb.json")
    )
    agent = BaseAgent(tools=[get_product_detail])
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


# ------------------------------------------------------------------
# Format helper
# ------------------------------------------------------------------

def _format_product_detail(product: dict[str, Any]) -> str:
    """Format product dict thành readable string cho LLM context."""
    ingredients = ", ".join(product.get("ingredients", []))
    tags = ", ".join(product.get("tags", []))
    paired_ids = ", ".join(product.get("best_paired_with", []))

    price = product.get("price", 0)
    price_str = f"{price:,}đ" if isinstance(price, (int, float)) else str(price)

    lines = [
        "=== Chi tiết sản phẩm ===",
        f"ID: {product.get('id', 'N/A')}",
        f"Tên: {product.get('name', 'N/A')}",
        f"Giá: {price_str}",
        f"Danh mục: {product.get('category', 'N/A')}",
        f"Mô tả: {product.get('description', 'N/A')}",
        f"Nguyên liệu: {ingredients or 'Không có thông tin'}",
        f"Tags: {tags or 'Không có'}",
        f"Kết hợp tốt với (IDs): {paired_ids or 'Không có gợi ý'}",
        f"Thời gian phục vụ: {product.get('available_time', 'N/A')}",
        f"Hình ảnh: {product.get('image_url', 'N/A')}",
    ]
    return "\n".join(lines)


# ------------------------------------------------------------------
# Factory
# ------------------------------------------------------------------

def create_get_product_detail_tool(data_path: str | Path):
    """
    Factory tạo get_product_detail tool.

    Args:
        data_path: Đường dẫn tới file JSON chứa danh sách products.

    Returns:
        StructuredTool — LLM chỉ thấy product_id. tenant_id tự inject.
    """
    data_path = Path(data_path)

    # In-memory cache — closure giữ reference
    _cache: dict[str, dict[str, Any]] = {}

    def _load_products() -> dict[str, dict[str, Any]]:
        """Lazy-load toàn bộ products từ JSON, index bằng composite key."""
        if _cache:
            return _cache

        logger.info("Loading product data from %s", data_path)

        with open(data_path, encoding="utf-8") as f:
            raw_data: list[dict[str, Any]] = json.load(f)

        for item in raw_data:
            key = f"{item['tenant_id']}::{item['id']}"
            _cache[key] = item

        logger.info("Loaded %d products into cache", len(_cache))
        return _cache

    @tool
    def get_product_detail(
        product_id: str,
        config: RunnableConfig,
    ) -> str:
        """Lấy thông tin chi tiết đầy đủ của một sản phẩm theo ID.

        Sử dụng khi user hỏi cụ thể về một món, ví dụ:
        - "Món này có cay không?"
        - "Nguyên liệu của món này gồm những gì?"
        - "Món này phù hợp ăn lúc nào?"
        - "Giá của món Wagyu A5 là bao nhiêu?"

        Args:
            product_id: ID của sản phẩm (vd: "item-001", "sf-005", "dr-010").
            config: cấu hình của đoạn chat.
        """
        tenant_id = config.get("configurable", {}).get("tenant_id")
        logger.info(
            "get_product_detail — product_id=%s, tenant_id=%s",
            product_id, tenant_id,
        )

        products = _load_products()
        key = f"{tenant_id}::{product_id}"

        product = products.get(key)
        if product is None:
            logger.warning("Product not found — key=%s", key)
            return (
                f"Không tìm thấy sản phẩm với ID '{product_id}' "
                f"trong hệ thống. "
                f"Vui lòng kiểm tra lại ID hoặc dùng menu_search để tìm sản phẩm."
            )

        logger.info("Product found — name=%s", product.get("name"))
        return _format_product_detail(product)

    return get_product_detail
