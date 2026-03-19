"""
Product Scorer — Weighted scoring cho chất lượng dữ liệu sản phẩm F&B.

Chấm điểm 0–100 cho mỗi sản phẩm dựa trên số field đã điền đầy đủ
và chất lượng nội dung. Sản phẩm dưới threshold (mặc định 60) sẽ bị
AI từ chối gợi ý — "thà không gợi ý còn hơn tư vấn sai".

Weighted Scoring:
    name=20, description=25, price=15, category=10, ingredients=12,
    available_time=7, tags=5, best_paired_with=3, image_url=3

Usage::

    from src.data.scoring import ProductScorer

    scorer = ProductScorer(threshold=60)

    # Chấm điểm 1 sản phẩm
    result = scorer.score_product({"name": "Phở bò", "price": 50000, ...})
    print(result.score, result.passed)

    # Chấm điểm batch
    report = scorer.score_products(products)
    print(report.summary())  # "12/20 món sẵn sàng để AI tư vấn."
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


# ------------------------------------------------------------------
# Data classes
# ------------------------------------------------------------------

@dataclass
class ScoredProduct:
    """Kết quả chấm điểm một sản phẩm."""

    product: dict[str, Any]
    """Raw product data."""

    score: int
    """Tổng điểm 0–100."""

    passed: bool
    """True nếu score >= threshold."""

    details: dict[str, int]
    """Điểm từng field, vd: {"name": 20, "description": 25, "price": 0, ...}."""

    missing_fields: list[str]
    """Danh sách fields bị 0 điểm."""

    @property
    def product_id(self) -> str:
        return self.product.get("id", "unknown")

    @property
    def product_name(self) -> str:
        return self.product.get("name", "N/A")

    @property
    def tenant_id(self) -> str:
        return self.product.get("tenant_id", "unknown")


@dataclass
class ScoringReport:
    """Báo cáo chấm điểm batch sản phẩm."""

    total: int
    """Tổng số products đầu vào."""

    passed: int
    """Số products đạt chuẩn (>= threshold)."""

    failed: int
    """Số products bị loại (< threshold)."""

    threshold: int
    """Ngưỡng điểm đã dùng."""

    passed_products: list[ScoredProduct] = field(default_factory=list)
    """Danh sách products đạt chuẩn."""

    failed_products: list[ScoredProduct] = field(default_factory=list)
    """Danh sách products bị loại."""

    def summary(self) -> str:
        """Summary string cho tenant dashboard."""
        return f"{self.passed}/{self.total} món sẵn sàng để AI tư vấn."


# ------------------------------------------------------------------
# Scoring weights & quality checks
# ------------------------------------------------------------------

# Mỗi tuple: (field_name, weight, quality_check_fn)
# quality_check_fn nhận giá trị field, trả về True nếu đạt chất lượng
_SCORING_RULES: list[tuple[str, int, Any]] = [
    (
        "name",
        20,
        lambda v: isinstance(v, str) and len(v.strip()) >= 2,
    ),
    (
        "description",
        25,
        lambda v: isinstance(v, str) and len(v.strip()) >= 20,
    ),
    (
        "price",
        15,
        lambda v: isinstance(v, (int, float)) and v > 0,
    ),
    (
        "category",
        10,
        lambda v: isinstance(v, str) and len(v.strip()) > 0,
    ),
    (
        "ingredients",
        12,
        lambda v: isinstance(v, list) and len(v) >= 1,
    ),
    (
        "available_time",
        7,
        lambda v: isinstance(v, str) and len(v.strip()) > 0,
    ),
    (
        "tags",
        5,
        lambda v: isinstance(v, list) and len(v) >= 1,
    ),
    (
        "best_paired_with",
        3,
        lambda v: isinstance(v, list) and len(v) >= 1,
    ),
    (
        "image_url",
        3,
        lambda v: isinstance(v, str) and v.strip().startswith("http"),
    ),
]


# ------------------------------------------------------------------
# Product Scorer
# ------------------------------------------------------------------

class ProductScorer:
    """
    Weighted scorer cho sản phẩm F&B.

    Pure logic — không phụ thuộc bất kỳ service nào.
    Input: dict sản phẩm. Output: score + chi tiết.
    """

    DEFAULT_THRESHOLD = 60

    def __init__(self, threshold: int | None = None) -> None:
        """
        Args:
            threshold: Ngưỡng tối thiểu để AI được phép gợi ý.
                       Mặc định: 60.
        """
        self.threshold = threshold if threshold is not None else self.DEFAULT_THRESHOLD

        logger.info("ProductScorer initialized — threshold=%d", self.threshold)

    def score_product(self, product: dict[str, Any]) -> ScoredProduct:
        """
        Chấm điểm một sản phẩm.

        Args:
            product: Dict sản phẩm (schema từ fnb.json).

        Returns:
            ScoredProduct với score, passed, details, missing_fields.
        """
        details: dict[str, int] = {}
        missing_fields: list[str] = []

        for field_name, weight, check_fn in _SCORING_RULES:
            value = product.get(field_name)

            if value is not None and check_fn(value):
                details[field_name] = weight
            else:
                details[field_name] = 0
                missing_fields.append(field_name)

        total_score = sum(details.values())
        passed = total_score >= self.threshold

        return ScoredProduct(
            product=product,
            score=total_score,
            passed=passed,
            details=details,
            missing_fields=missing_fields,
        )

    def score_products(self, products: list[dict[str, Any]]) -> ScoringReport:
        """
        Chấm điểm batch sản phẩm.

        Args:
            products: Danh sách dicts sản phẩm.

        Returns:
            ScoringReport với passed/failed lists và summary.
        """
        passed_list: list[ScoredProduct] = []
        failed_list: list[ScoredProduct] = []

        for product in products:
            scored = self.score_product(product)
            if scored.passed:
                passed_list.append(scored)
            else:
                failed_list.append(scored)

        report = ScoringReport(
            total=len(products),
            passed=len(passed_list),
            failed=len(failed_list),
            threshold=self.threshold,
            passed_products=passed_list,
            failed_products=failed_list,
        )

        logger.info(
            "Scoring complete — %s (threshold=%d)",
            report.summary(), self.threshold,
        )

        # Log chi tiết các sản phẩm bị loại
        for sp in failed_list:
            logger.warning(
                "Product FAILED — id=%s, name='%s', score=%d, missing=%s",
                sp.product_id, sp.product_name, sp.score, sp.missing_fields,
            )

        return report
