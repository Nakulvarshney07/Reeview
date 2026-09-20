from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from src.product_parser.normalizer import NormalizedProduct

@dataclass
class AmazonBenchmarkResult:
    match_type: str  # 'exact', 'product_family', 'comparable_products', 'category_benchmark'
    confidence: str  # 'high', 'medium', 'low'
    historical_monthly_sales_benchmark: Optional[int]
    price: Optional[float]
    list_price: Optional[float]
    discount_pct: Optional[float]
    rating: Optional[float]
    review_count: Optional[int]
    is_best_seller: bool
    category: str
    market_signal: str  # 'Strong Momentum', 'Moderate Demand', 'Low Activity', 'Sparse Historical Data'
    details: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class AmazonMarketBenchmark:
    """
    Analyzes historical Amazon marketplace metrics available in the dataset.
    Extracts sales momentum benchmarks and price/rating stability.
    """
    
    @classmethod
    def evaluate(cls, product: NormalizedProduct) -> AmazonBenchmarkResult:
        """
        Evaluates the product row against Amazon historical market benchmarks.
        """
        bought_last_month = product.bought_in_last_month
        price = product.price
        list_price = product.list_price
        rating = product.rating
        review_count = product.review_count
        is_best_seller = product.is_best_seller
        category = product.category
        
        # Calculate discount if list price is available and greater than selling price
        discount_pct = None
        if price and list_price and list_price > price and list_price > 0:
            discount_pct = round(((list_price - price) / list_price) * 100, 1)

        # Determine match type and confidence based on ASIN and Title availability
        if product.product_id and not product.product_id.startswith("ROW_"):
            match_type = "exact"
            confidence = "high"
        elif product.brand and product.category != "General":
            match_type = "product_family"
            confidence = "medium"
        else:
            match_type = "category_benchmark"
            confidence = "low"

        # Determine market demand signal from historical sales benchmark
        if bought_last_month is not None and bought_last_month >= 500:
            market_signal = "Strong Momentum"
        elif bought_last_month is not None and bought_last_month >= 100:
            market_signal = "Moderate Demand"
        elif bought_last_month is not None and bought_last_month > 0:
            market_signal = "Low Activity"
        elif is_best_seller:
            market_signal = "Strong Momentum"
        elif rating is not None and rating >= 4.3 and (review_count or 0) > 50:
            market_signal = "Moderate Demand"
        else:
            market_signal = "Sparse Historical Data"

        details_parts = []
        if bought_last_month is not None:
            details_parts.append(f"Historical Amazon monthly purchases: {bought_last_month} units")
        if rating is not None:
            details_parts.append(f"Rating: {rating}/5.0")
        if is_best_seller:
            details_parts.append("Awarded Best Seller status")
        if discount_pct:
            details_parts.append(f"Active discount: {discount_pct}%")

        details = "; ".join(details_parts) if details_parts else "Limited historical metrics in CSV row"

        return AmazonBenchmarkResult(
            match_type=match_type,
            confidence=confidence,
            historical_monthly_sales_benchmark=bought_last_month,
            price=price,
            list_price=list_price,
            discount_pct=discount_pct,
            rating=rating,
            review_count=review_count,
            is_best_seller=is_best_seller,
            category=category,
            market_signal=market_signal,
            details=details
        )
