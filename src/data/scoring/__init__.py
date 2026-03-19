"""
Scoring Package — Data quality scoring cho sản phẩm F&B.

Usage::

    from src.data.scoring import ProductScorer, ScoredProduct, ScoringReport

    scorer = ProductScorer(threshold=60)
    report = scorer.score_products(products)
    print(report.summary())  # "12/20 món sẵn sàng để AI tư vấn."
"""

from src.data.scoring.product_scorer import ProductScorer, ScoredProduct, ScoringReport

__all__ = ["ProductScorer", "ScoredProduct", "ScoringReport"]
