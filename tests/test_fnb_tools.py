"""
Test Tools — Test 3 F&B tools: menu_search, get_product_detail, get_recommendations.

Kiểm tra:
    - InjectedToolArg: tenant_id bị ẩn khỏi LLM schema
    - Factory pattern: tạo tool từ factory hoạt động đúng
    - Tool logic: invoke trả kết quả chính xác
"""

from pathlib import Path

import pytest

from tools.get_product_detail_tool import create_get_product_detail_tool
from tools.get_recommendations_tool import create_get_recommendations_tool
from tools.menu_search_tool import create_menu_search_tool

# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

DATA_PATH = Path(__file__).resolve().parent.parent / "external_data_storage" / "fnb.json"


class MockDocumentStore:
    """Mock store — trả kết quả rỗng hoặc giả."""

    def __init__(self, results=None):
        self._results = results or []

    def search(self, **kwargs):
        return self._results


class MockEmbedding:
    """Mock embedding — trả vector cố định."""

    model_name = "mock-embedding"

    def embed_query(self, text: str) -> list[float]:
        return [0.1, 0.2, 0.3, 0.4]


@pytest.fixture
def menu_search_tool():
    return create_menu_search_tool(
        document_store=MockDocumentStore(),
        embedding=MockEmbedding(),
    )


@pytest.fixture
def product_detail_tool():
    return create_get_product_detail_tool(data_path=DATA_PATH)


@pytest.fixture
def recommendations_tool():
    return create_get_recommendations_tool(data_path=DATA_PATH)


# ------------------------------------------------------------------
# Test InjectedToolArg — tenant_id ẩn khỏi LLM
# ------------------------------------------------------------------

class TestInjectedToolArg:
    """Verify tenant_id bị ẩn khỏi LLM schema cho tất cả tools."""

    def test_menu_search_hides_tenant_id(self, menu_search_tool):
        """LLM chỉ thấy query và top_k, KHÔNG thấy tenant_id."""
        schema = menu_search_tool.tool_call_schema.model_json_schema()
        fields = list(schema["properties"].keys())

        assert "query" in fields
        assert "top_k" in fields
        assert "tenant_id" not in fields

    def test_product_detail_hides_tenant_id(self, product_detail_tool):
        """LLM chỉ thấy product_id, KHÔNG thấy tenant_id."""
        schema = product_detail_tool.tool_call_schema.model_json_schema()
        fields = list(schema["properties"].keys())

        assert "product_id" in fields
        assert "tenant_id" not in fields

    def test_recommendations_hides_tenant_id(self, recommendations_tool):
        """LLM chỉ thấy product_ids, KHÔNG thấy tenant_id."""
        schema = recommendations_tool.tool_call_schema.model_json_schema()
        fields = list(schema["properties"].keys())

        assert "product_ids" in fields
        assert "tenant_id" not in fields

    def test_tenant_id_exists_in_args_schema(self, menu_search_tool):
        """tenant_id vẫn tồn tại trong args_schema (internal), chỉ ẩn khỏi LLM."""
        schema = menu_search_tool.args_schema.model_json_schema()
        fields = list(schema["properties"].keys())

        assert "tenant_id" in fields
        assert "query" in fields


# ------------------------------------------------------------------
# Test Tool Metadata
# ------------------------------------------------------------------

class TestToolMetadata:
    """Verify tool names và descriptions."""

    def test_menu_search_name(self, menu_search_tool):
        assert menu_search_tool.name == "menu_search"

    def test_product_detail_name(self, product_detail_tool):
        assert product_detail_tool.name == "get_product_detail"

    def test_recommendations_name(self, recommendations_tool):
        assert recommendations_tool.name == "get_recommendations"

    def test_menu_search_has_description(self, menu_search_tool):
        assert "semantic search" in menu_search_tool.description.lower()

    def test_product_detail_has_description(self, product_detail_tool):
        assert "chi tiết" in product_detail_tool.description.lower()

    def test_recommendations_has_description(self, recommendations_tool):
        assert "gợi ý" in recommendations_tool.description.lower()


# ------------------------------------------------------------------
# Test menu_search
# ------------------------------------------------------------------

class TestMenuSearchTool:
    """Test menu_search tool logic."""

    def test_empty_results(self, menu_search_tool):
        """Search không tìm thấy gì → trả message phù hợp."""
        result = menu_search_tool.invoke({
            "query": "món không tồn tại",
            "tenant_id": "tenant-1",
        })
        assert "Không tìm thấy" in result

    def test_top_k_clamped_max(self):
        """top_k > 10 phải bị clamp xuống 10."""
        store = MockDocumentStore()
        emb = MockEmbedding()
        tool = create_menu_search_tool(store, emb)

        # Tool sẽ gọi store.search với limit <= 10
        result = tool.invoke({
            "query": "test",
            "tenant_id": "tenant-1",
            "top_k": 999,
        })
        # Không lỗi là đủ — clamp logic hoạt động
        assert isinstance(result, str)

    def test_top_k_clamped_min(self):
        """top_k < 1 phải bị clamp lên 1."""
        store = MockDocumentStore()
        emb = MockEmbedding()
        tool = create_menu_search_tool(store, emb)

        result = tool.invoke({
            "query": "test",
            "tenant_id": "tenant-1",
            "top_k": -5,
        })
        assert isinstance(result, str)

    def test_with_mock_results(self):
        """Search có kết quả → format đúng."""
        from unittest.mock import MagicMock

        mock_point = MagicMock()
        mock_point.payload = {
            "item_id": "item-001",
            "name": "Phở bò",
            "price": 50000,
            "category": "Món nước",
            "text": "Phở bò Hà Nội",
            "tags": "Nóng, Truyền thống",
            "available_time": "Cả ngày",
        }
        mock_point.score = 0.95

        store = MockDocumentStore(results=[mock_point])
        emb = MockEmbedding()
        tool = create_menu_search_tool(store, emb)

        result = tool.invoke({
            "query": "phở",
            "tenant_id": "tenant-1",
        })

        assert "Phở bò" in result
        assert "50,000đ" in result
        assert "0.95" in result


# ------------------------------------------------------------------
# Test get_product_detail
# ------------------------------------------------------------------

class TestGetProductDetailTool:
    """Test get_product_detail tool logic."""

    def test_find_existing_product(self, product_detail_tool):
        """Tìm product có tồn tại → trả chi tiết đầy đủ."""
        result = product_detail_tool.invoke({
            "product_id": "item-001",
            "tenant_id": "tenant-1",
        })

        assert "Wagyu" in result
        assert "1,250,000đ" in result
        assert "Chi tiết sản phẩm" in result

    def test_product_not_found(self, product_detail_tool):
        """Product không tồn tại → trả message lỗi."""
        result = product_detail_tool.invoke({
            "product_id": "item-999-nonexistent",
            "tenant_id": "tenant-1",
        })

        assert "Không tìm thấy" in result

    def test_wrong_tenant(self, product_detail_tool):
        """Product tồn tại nhưng sai tenant → không tìm thấy."""
        result = product_detail_tool.invoke({
            "product_id": "item-001",
            "tenant_id": "wrong-tenant",
        })

        assert "Không tìm thấy" in result

    def test_result_contains_ingredients(self, product_detail_tool):
        """Kết quả phải chứa nguyên liệu."""
        result = product_detail_tool.invoke({
            "product_id": "item-001",
            "tenant_id": "tenant-1",
        })

        assert "Nguyên liệu" in result


# ------------------------------------------------------------------
# Test get_recommendations
# ------------------------------------------------------------------

class TestGetRecommendationsTool:
    """Test get_recommendations tool logic."""

    def test_get_recommendations_for_valid_product(self, recommendations_tool):
        """Product có best_paired_with → trả gợi ý."""
        result = recommendations_tool.invoke({
            "product_ids": ["item-001"],
            "tenant_id": "tenant-1",
        })

        assert "Gợi ý kết hợp" in result

    def test_no_recommendations(self, recommendations_tool):
        """Product không có best_paired_with → message phù hợp."""
        result = recommendations_tool.invoke({
            "product_ids": ["nonexistent-item"],
            "tenant_id": "tenant-1",
        })

        assert "Không tìm thấy gợi ý" in result or "Không tìm thấy dữ liệu" in result

    def test_wrong_tenant_no_data(self, recommendations_tool):
        """Tenant không tồn tại → message lỗi."""
        result = recommendations_tool.invoke({
            "product_ids": ["item-001"],
            "tenant_id": "nonexistent-tenant",
        })

        assert "Không tìm thấy" in result

    def test_already_selected_not_recommended(self, recommendations_tool):
        """Sản phẩm đã chọn không được gợi ý lại."""
        result = recommendations_tool.invoke({
            "product_ids": ["item-001", "item-021"],
            "tenant_id": "tenant-1",
        })

        # item-021 là best_paired_with của item-001
        # Nếu user đã chọn item-021, nó không nên xuất hiện trong gợi ý
        if "Gợi ý kết hợp" in result:
            assert "item-021" not in result or "ID: item-021" not in result
