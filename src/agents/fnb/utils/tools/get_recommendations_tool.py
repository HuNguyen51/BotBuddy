"""
Get Recommendations Tool — Upsell Engine.

Nhận vào list món user đang quan tâm hoặc đã chọn, dựa trên field
`best_paired_with` trong data schema để gợi ý các sản phẩm kết hợp.

AI dùng tool này để tự nhiên suggest thêm — ví dụ:
"Bạn đang chọn cà phê đen, nhiều khách hay kết hợp thêm bánh croissant
bơ để cân vị, bạn có muốn thử không?"

tenant_id được inject runtime qua InjectedToolArg — LLM chỉ cần truyền
product_ids, hệ thống tự biết đang serve tenant nào.

Usage::

    from src.tools.get_recommendations_tool import create_get_recommendations_tool

    get_recommendations = create_get_recommendations_tool(
        data_path=Path("external_data_storage/fnb.json")
    )
    agent = BaseAgent(tools=[get_recommendations])
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

def _format_recommendation(product: dict[str, Any], reason: str) -> str:
    """Format một sản phẩm gợi ý thành readable string."""
    price = product.get("price", 0)
    price_str = f"{price:,}đ" if isinstance(price, (int, float)) else str(price)
    tags = ", ".join(product.get("tags", []))

    return (
        f"• {product.get('name', 'N/A')} — {price_str}\n"
        f"  Danh mục: {product.get('category', 'N/A')}\n"
        f"  Mô tả: {product.get('description', 'N/A')}\n"
        f"  Tags: {tags}\n"
        f"  Lý do gợi ý: {reason}"
    )


# ------------------------------------------------------------------
# Factory
# ------------------------------------------------------------------

def create_get_recommendations_tool(data_path: str | Path):
    """
    Factory tạo get_recommendations tool.

    Args:
        data_path: Đường dẫn tới file JSON chứa danh sách products.

    Returns:
        StructuredTool — LLM chỉ thấy product_ids. tenant_id tự inject.
    """
    data_path = Path(data_path)

    # In-memory caches — closure giữ reference
    _product_cache: dict[str, dict[str, Any]] = {}
    _tenant_index: dict[str, dict[str, dict[str, Any]]] = {}

    def _load_and_index():
        """Lazy-load + index theo tenant."""
        if _product_cache:
            return _product_cache, _tenant_index

        logger.info("Loading product data for recommendations from %s", data_path)

        with open(data_path, encoding="utf-8") as f:
            raw_data: list[dict[str, Any]] = json.load(f)

        for item in raw_data:
            tid = item["tenant_id"]
            iid = item["id"]

            _product_cache[f"{tid}::{iid}"] = item

            if tid not in _tenant_index:
                _tenant_index[tid] = {}
            _tenant_index[tid][iid] = item

        logger.info(
            "Loaded %d products across %d tenants",
            len(_product_cache), len(_tenant_index),
        )
        return _product_cache, _tenant_index

    @tool
    def get_recommendations(
        product_ids: list[str],
        config: RunnableConfig,
    ) -> str:
        """Gợi ý sản phẩm kết hợp (upsell) dựa trên các món user đã chọn/quan tâm.

        Sử dụng khi muốn suggest thêm sản phẩm cho user:
        - User vừa chọn một món → gợi ý các món kết hợp ngon
        - User đang phân vân → gợi ý combo phổ biến
        - Tự nhiên upsell trong cuộc trò chuyện

        Ví dụ: User chọn "Thăn bò Wagyu" → gợi ý "Bia Sapporo" và "Cơm trộn Bibimbap"

        Args:
            product_ids: Danh sách ID của các sản phẩm user đang quan tâm.
                         Ví dụ: ["item-001", "item-003"]
            config: cấu hình của đoạn chat.
        """
        tenant_id = config.get("configurable", {}).get("tenant_id")

        logger.info(
            "get_recommendations — product_ids=%s, tenant_id=%s",
            product_ids, tenant_id,
        )

        _, tenant_index = _load_and_index()

        tenant_products = tenant_index.get(tenant_id)
        if not tenant_products:
            return f"Không tìm thấy dữ liệu menu cho tenant '{tenant_id}'."

        # Thu thập recommended IDs + lý do
        recommendations: dict[str, list[str]] = {}

        for pid in product_ids:
            product = tenant_products.get(pid)
            if product is None:
                logger.warning("Product %s not found in tenant %s, skipping", pid, tenant_id)
                continue

            source_name = product.get("name", pid)

            for paired_id in product.get("best_paired_with", []):
                if paired_id in product_ids:
                    continue
                if paired_id not in tenant_products:
                    continue

                if paired_id not in recommendations:
                    recommendations[paired_id] = []
                recommendations[paired_id].append(source_name)

        if not recommendations:
            return (
                "Không tìm thấy gợi ý kết hợp cho các sản phẩm đã chọn. "
                "Bạn có thể dùng menu_search để khám phá thêm các món khác."
            )

        # Sắp xếp: items được recommend nhiều nhất lên trước
        sorted_recs = sorted(
            recommendations.items(),
            key=lambda x: len(x[1]),
            reverse=True,
        )

        # Format output
        output_parts = [
            f"=== Gợi ý kết hợp cho {len(product_ids)} sản phẩm đã chọn ===\n"
        ]

        for rec_id, sources in sorted_recs:
            rec_product = tenant_products[rec_id]
            reason = f"Kết hợp tốt với: {', '.join(sources)}"
            if len(sources) > 1:
                reason += f" ({len(sources)} sản phẩm recommend)"
            output_parts.append(_format_recommendation(rec_product, reason))

        output_parts.append(f"\nTổng cộng {len(sorted_recs)} gợi ý kết hợp.")

        logger.info("get_recommendations — returning %d recommendations", len(sorted_recs))
        return "\n\n".join(output_parts)

    return get_recommendations
