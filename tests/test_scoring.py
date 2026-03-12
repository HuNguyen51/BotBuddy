"""
Test Scoring — Test ProductScorer, ScoredProduct, ScoringReport.

Test pure logic — không cần external services.
"""

from data.scoring.product_scorer import ProductScorer, ScoredProduct, ScoringReport


# ------------------------------------------------------------------
# Fixtures — sample products
# ------------------------------------------------------------------

FULL_PRODUCT = {
    "tenant_id": "tenant-1",
    "id": "item-001",
    "name": "Thăn bò Wagyu A5 thượng hạng",
    "price": 1250000,
    "category": "Thịt bò nướng",
    "description": "Thịt mềm như bơ, tan ngay đầu lưỡi với vân mỡ cẩm thạch hoàn hảo.",
    "ingredients": ["Thịt bò Wagyu Nhật", "Muối hồng", "Tiêu đen"],
    "tags": ["Không cay", "Giàu đạm"],
    "best_paired_with": ["item-021", "item-015"],
    "available_time": "Cả ngày",
    "image_url": "https://example.com/images/wagyu-a5.jpg",
}

MINIMAL_PRODUCT = {
    "tenant_id": "tenant-1",
    "id": "item-999",
    "name": "Nước lọc",
    "price": 5000,
}

EMPTY_PRODUCT = {
    "tenant_id": "tenant-1",
    "id": "item-000",
}


# ------------------------------------------------------------------
# Test ProductScorer
# ------------------------------------------------------------------

class TestProductScorer:
    """Test scoring logic."""

    def test_default_threshold(self):
        scorer = ProductScorer()
        assert scorer.threshold == 60

    def test_custom_threshold(self):
        scorer = ProductScorer(threshold=80)
        assert scorer.threshold == 80

    def test_full_product_gets_100(self):
        """Product đầy đủ mọi field phải đạt 100 điểm."""
        scorer = ProductScorer()
        result = scorer.score_product(FULL_PRODUCT)

        assert result.score == 100
        assert result.passed is True
        assert result.missing_fields == []

    def test_full_product_details(self):
        """Kiểm tra điểm từng field."""
        scorer = ProductScorer()
        result = scorer.score_product(FULL_PRODUCT)

        assert result.details["name"] == 20
        assert result.details["description"] == 25
        assert result.details["price"] == 15
        assert result.details["category"] == 10
        assert result.details["ingredients"] == 12
        assert result.details["available_time"] == 7
        assert result.details["tags"] == 5
        assert result.details["best_paired_with"] == 3
        assert result.details["image_url"] == 3

    def test_minimal_product_fails(self):
        """Product quá ít field không đạt threshold 60."""
        scorer = ProductScorer()
        result = scorer.score_product(MINIMAL_PRODUCT)

        assert result.score < 60
        assert result.passed is False
        assert len(result.missing_fields) > 0

    def test_minimal_product_has_name_and_price(self):
        """Dù fail, vẫn phải ghi nhận đúng name và price."""
        scorer = ProductScorer()
        result = scorer.score_product(MINIMAL_PRODUCT)

        assert result.details["name"] == 20
        assert result.details["price"] == 15

    def test_empty_product_gets_zero(self):
        """Product rỗng hoàn toàn phải 0 điểm."""
        scorer = ProductScorer()
        result = scorer.score_product(EMPTY_PRODUCT)

        assert result.score == 0
        assert result.passed is False
        assert len(result.missing_fields) == 9  # Tất cả 9 fields

    def test_description_quality_check(self):
        """Description quá ngắn (< 20 chars) phải bị 0 điểm."""
        scorer = ProductScorer()
        product = {**FULL_PRODUCT, "description": "Ngon"}  # Chỉ 4 ký tự
        result = scorer.score_product(product)

        assert result.details["description"] == 0
        assert "description" in result.missing_fields

    def test_price_must_be_positive(self):
        """Price = 0 hoặc âm phải bị 0 điểm."""
        scorer = ProductScorer()

        result_zero = scorer.score_product({**FULL_PRODUCT, "price": 0})
        assert result_zero.details["price"] == 0

        result_neg = scorer.score_product({**FULL_PRODUCT, "price": -100})
        assert result_neg.details["price"] == 0

    def test_image_url_must_start_with_http(self):
        """image_url phải bắt đầu bằng http."""
        scorer = ProductScorer()

        result = scorer.score_product({**FULL_PRODUCT, "image_url": "not-a-url"})
        assert result.details["image_url"] == 0

    def test_ingredients_must_be_nonempty_list(self):
        """ingredients phải là list có ít nhất 1 item."""
        scorer = ProductScorer()

        result_empty = scorer.score_product({**FULL_PRODUCT, "ingredients": []})
        assert result_empty.details["ingredients"] == 0

        result_str = scorer.score_product({**FULL_PRODUCT, "ingredients": "string"})
        assert result_str.details["ingredients"] == 0


class TestScoredProduct:
    """Test ScoredProduct dataclass."""

    def test_product_id_property(self):
        scorer = ProductScorer()
        result = scorer.score_product(FULL_PRODUCT)
        assert result.product_id == "item-001"

    def test_product_name_property(self):
        scorer = ProductScorer()
        result = scorer.score_product(FULL_PRODUCT)
        assert result.product_name == "Thăn bò Wagyu A5 thượng hạng"

    def test_tenant_id_property(self):
        scorer = ProductScorer()
        result = scorer.score_product(FULL_PRODUCT)
        assert result.tenant_id == "tenant-1"


class TestScoringReport:
    """Test batch scoring + ScoringReport."""

    def test_batch_scoring(self):
        scorer = ProductScorer()
        report = scorer.score_products([FULL_PRODUCT, MINIMAL_PRODUCT, EMPTY_PRODUCT])

        assert report.total == 3
        assert report.passed == 1
        assert report.failed == 2

    def test_summary_string(self):
        scorer = ProductScorer()
        report = scorer.score_products([FULL_PRODUCT, MINIMAL_PRODUCT])

        assert "1/2" in report.summary()
        assert "sẵn sàng" in report.summary()

    def test_passed_products_list(self):
        scorer = ProductScorer()
        report = scorer.score_products([FULL_PRODUCT, EMPTY_PRODUCT])

        assert len(report.passed_products) == 1
        assert report.passed_products[0].product_id == "item-001"

    def test_failed_products_list(self):
        scorer = ProductScorer()
        report = scorer.score_products([FULL_PRODUCT, EMPTY_PRODUCT])

        assert len(report.failed_products) == 1
        assert report.failed_products[0].product_id == "item-000"

    def test_empty_input(self):
        scorer = ProductScorer()
        report = scorer.score_products([])

        assert report.total == 0
        assert report.passed == 0
        assert report.failed == 0
