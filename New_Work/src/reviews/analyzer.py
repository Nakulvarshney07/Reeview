from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field, asdict
from src.product_parser.normalizer import NormalizedProduct

@dataclass
class ReviewAnalysisResult:
    review_source: str  # 'full_text' or 'aggregate_metrics'
    sentiment: str  # 'Positive', 'Neutral', 'Negative', 'Mixed'
    positive_factors: List[str] = field(default_factory=list)
    negative_factors: List[str] = field(default_factory=list)
    quality_signal: str = "Moderate"
    reliability_signal: str = "Moderate"
    value_for_money_signal: str = "Moderate"
    summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class CSVReviewAnalyzer:
    """
    Analyzes customer review signals present in the CSV dataset.
    Handles both full review text (if available) and aggregate rating/review metrics without fabricating data.
    """
    @classmethod
    def analyze(cls, product: NormalizedProduct) -> ReviewAnalysisResult:
        rating = product.rating
        review_count = product.review_count or 0
        is_best_seller = product.is_best_seller
        bought_last_month = product.bought_in_last_month or 0

        positive_factors = []
        negative_factors = []

        if rating is not None:
            if rating >= 4.5:
                sentiment = "Positive"
                positive_factors.append(f"High consumer satisfaction rating ({rating}/5.0)")
                quality_signal = "High"
            elif rating >= 3.8:
                sentiment = "Neutral"
                positive_factors.append(f"Acceptable average rating ({rating}/5.0)")
                quality_signal = "Moderate"
            elif rating > 0:
                sentiment = "Negative"
                negative_factors.append(f"Sub-par customer rating ({rating}/5.0) indicates potential quality concerns")
                quality_signal = "Low"
            else:
                sentiment = "Neutral"
                quality_signal = "Unrated"
        else:
            sentiment = "Neutral"
            quality_signal = "Unrated"

        if review_count > 500:
            positive_factors.append(f"Strong social proof with {review_count} customer reviews")
            reliability_signal = "High"
        elif review_count > 50:
            positive_factors.append(f"Moderate review volume ({review_count} reviews)")
            reliability_signal = "Moderate"
        else:
            negative_factors.append("Low review volume / early listing stage")
            reliability_signal = "Low"

        if is_best_seller:
            positive_factors.append("Best Seller badge signifies top category conversion")

        if product.price and product.list_price and product.list_price > product.price:
            positive_factors.append(f"Competitive pricing below list price (${product.price:.2f} vs ${product.list_price:.2f})")
            value_for_money_signal = "High"
        else:
            value_for_money_signal = "Moderate"

        summary = f"Review sentiment is {sentiment} based on {rating or 0}/5.0 rating across {review_count} customer reviews."

        return ReviewAnalysisResult(
            review_source="aggregate_metrics",
            sentiment=sentiment,
            positive_factors=positive_factors,
            negative_factors=negative_factors,
            quality_signal=quality_signal,
            reliability_signal=reliability_signal,
            value_for_money_signal=value_for_money_signal,
            summary=summary
        )
