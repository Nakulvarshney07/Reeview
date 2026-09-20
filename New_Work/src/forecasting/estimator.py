import math
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field, asdict
from src.forecasting.evidence import StructuredEvidence
from src.ollama.client import OllamaClient
from src.ollama.prompts import OllamaPrompts

@dataclass
class ForecastResult:
    product_id: str
    product_name: str
    brand: str
    model: str
    category: str
    price: Optional[float]
    rating: Optional[float]
    review_count: Optional[int]
    
    amazon_match_type: str
    amazon_market_signal: str
    
    youtube_videos_found: int
    youtube_videos_analyzed: int
    youtube_total_views: int
    youtube_interest_signal: str
    youtube_sentiment: str
    
    reddit_posts_found: int
    reddit_posts_analyzed: int
    reddit_interest_signal: str
    reddit_sentiment: str
    reddit_purchase_intent: str
    
    review_sentiment: str
    
    overall_demand_direction: str
    demand_signal_strength: str
    
    estimated_demand_low: int
    estimated_demand: int
    estimated_demand_high: int
    
    recommended_quantity: int
    minimum_quantity: int
    maximum_quantity: int
    
    confidence: float
    
    key_positive_factors: List[str] = field(default_factory=list)
    key_negative_factors: List[str] = field(default_factory=list)
    risk_factors: List[str] = field(default_factory=list)
    reasoning: str = ""
    
    data_sources_available: Dict[str, bool] = field(default_factory=dict)
    detailed_evidence: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class DemandEstimator:
    """
    Reasoning engine for e-commerce demand estimation and listing quantity recommendation.
    Combines Ollama structured analysis with deterministic statistical fallback.
    """
    def __init__(self, ollama_client: Optional[OllamaClient] = None):
        self.ollama = ollama_client or OllamaClient()

    def estimate_demand(self, evidence: StructuredEvidence) -> ForecastResult:
        """
        Processes multi-source structured evidence to generate a demand estimate and quantity range.
        """
        pinfo = evidence.product_information
        abench = evidence.amazon_market_benchmark
        yt = evidence.youtube_analysis
        reddit = evidence.reddit_analysis
        reviews = evidence.customer_reviews

        # Attempt Ollama reasoning
        prompt = OllamaPrompts.build_demand_forecast_prompt(evidence.to_dict())
        ollama_response = self.ollama.generate(prompt)

        if ollama_response and self._is_valid_forecast(ollama_response):
            return self._build_result_from_ollama(evidence, ollama_response)
        
        # Fallback to deterministic statistical reasoning
        return self._build_statistical_fallback(evidence)

    @staticmethod
    def _is_valid_forecast(data: Dict[str, Any]) -> bool:
        required_keys = [
            "overall_demand_direction", "estimated_demand",
            "recommended_quantity", "confidence", "reasoning"
        ]
        return all(k in data for k in required_keys)

    def _build_result_from_ollama(self, evidence: StructuredEvidence, data: Dict[str, Any]) -> ForecastResult:
        pinfo = evidence.product_information
        abench = evidence.amazon_market_benchmark
        yt = evidence.youtube_analysis
        reddit = evidence.reddit_analysis
        reviews = evidence.customer_reviews

        est_demand = int(data.get("estimated_demand", 100))
        rec_qty = int(data.get("recommended_quantity", est_demand))
        
        low = int(data.get("estimated_demand_low", max(10, int(est_demand * 0.75))))
        high = int(data.get("estimated_demand_high", int(est_demand * 1.25)))
        min_qty = int(data.get("minimum_quantity", max(5, int(rec_qty * 0.7))))
        max_qty = int(data.get("maximum_quantity", int(rec_qty * 1.3)))

        return ForecastResult(
            product_id=pinfo.get("product_id", ""),
            product_name=pinfo.get("product_name", ""),
            brand=pinfo.get("brand", ""),
            model=pinfo.get("model", ""),
            category=pinfo.get("category", ""),
            price=pinfo.get("price"),
            rating=pinfo.get("rating"),
            review_count=pinfo.get("review_count"),
            
            amazon_match_type=abench.get("match_type", "exact"),
            amazon_market_signal=abench.get("market_signal", "Moderate Demand"),
            
            youtube_videos_found=yt.get("videos_found", 0),
            youtube_videos_analyzed=yt.get("videos_analyzed", 0),
            youtube_total_views=yt.get("total_views", 0),
            youtube_interest_signal=yt.get("consumer_interest", "Unknown"),
            youtube_sentiment=yt.get("sentiment", "Neutral"),
            
            reddit_posts_found=reddit.get("posts_found", 0),
            reddit_posts_analyzed=reddit.get("posts_analyzed", 0),
            reddit_interest_signal=reddit.get("consumer_interest", "Unknown"),
            reddit_sentiment=reddit.get("sentiment", "Neutral"),
            reddit_purchase_intent=reddit.get("purchase_intent", "Unknown"),
            
            review_sentiment=reviews.get("sentiment", "Neutral"),
            
            overall_demand_direction=str(data.get("overall_demand_direction", "Moderate Demand")),
            demand_signal_strength=str(data.get("demand_signal_strength", "Medium")),
            
            estimated_demand_low=low,
            estimated_demand=est_demand,
            estimated_demand_high=high,
            
            recommended_quantity=rec_qty,
            minimum_quantity=min_qty,
            maximum_quantity=max_qty,
            
            confidence=round(float(data.get("confidence", 0.65)), 2),
            
            key_positive_factors=data.get("key_positive_factors", []),
            key_negative_factors=data.get("key_negative_factors", []),
            risk_factors=data.get("risk_factors", []),
            reasoning=str(data.get("reasoning", "")),
            
            data_sources_available=evidence.data_sources_available,
            detailed_evidence=evidence.to_dict()
        )

    def _build_statistical_fallback(self, evidence: StructuredEvidence) -> ForecastResult:
        """
        Deterministic, grounded estimation algorithm when LLM is offline.
        Uses historical Amazon sales benchmark as anchor adjusted by external signals.
        """
        pinfo = evidence.product_information
        abench = evidence.amazon_market_benchmark
        yt = evidence.youtube_analysis
        reddit = evidence.reddit_analysis
        reviews = evidence.customer_reviews

        # Anchor: Historical monthly benchmark from dataset
        hist_sales = abench.get("historical_monthly_sales_benchmark")
        if hist_sales is not None and hist_sales > 0:
            base_demand = float(hist_sales)
            confidence = 0.70
        else:
            # Estimate from rating, reviews, and price
            rating = pinfo.get("rating") or 4.0
            rev_count = pinfo.get("review_count") or 10
            base_demand = max(50.0, min(500.0, rev_count * 0.5 + (rating - 3.0) * 80.0))
            confidence = 0.50

        # Adjust for external interest signals
        multiplier = 1.0
        pos_factors = list(reviews.get("positive_factors", []))
        neg_factors = list(reviews.get("negative_factors", []))
        risk_factors = []

        # Best seller bonus
        if pinfo.get("is_best_seller"):
            multiplier *= 1.25
            pos_factors.append("Historical Amazon Best Seller status indicates strong category velocity")

        # YouTube signals
        total_views = yt.get("total_views", 0)
        if total_views > 500_000:
            multiplier *= 1.15
            pos_factors.append(f"High YouTube video engagement ({total_views:,} views) shows strong consumer interest")
        elif total_views > 50_000:
            multiplier *= 1.05
            pos_factors.append(f"Moderate YouTube awareness ({total_views:,} views)")
        
        # Include qualitative strengths from video reviews
        for s in yt.get("product_strengths", [])[:2]:
            if s not in pos_factors:
                pos_factors.append(f"YouTube Review: {s}")
        for w in yt.get("product_weaknesses", [])[:1]:
            if w not in neg_factors:
                neg_factors.append(f"YouTube Tester Note: {w}")

        if yt.get("videos_found", 0) == 0:
            confidence -= 0.05
            risk_factors.append("Limited video review presence on YouTube")

        # Reddit signals
        reddit_posts = reddit.get("posts_found", 0)
        if reddit_posts > 2:
            multiplier *= 1.05
            pos_factors.append(f"Active Reddit community discussions ({reddit_posts} threads)")
            
        # Include qualitative signals from Reddit
        for adv in reddit.get("recurring_advantages", [])[:2]:
            if adv not in pos_factors:
                pos_factors.append(f"Reddit Community: {adv}")
        for comp in reddit.get("recurring_complaints", [])[:1]:
            if comp not in risk_factors:
                risk_factors.append(f"Reddit User Alert: {comp}")

        if reddit_posts == 0:
            confidence -= 0.05
            risk_factors.append("Minimal community discussion on Reddit")

        # Rating penalty
        rating = pinfo.get("rating")
        if rating is not None and rating < 3.8 and rating > 0:
            multiplier *= 0.75
            neg_factors.append(f"Low rating ({rating}/5.0) poses return and churn risk")
            risk_factors.append("Product quality or customer dissatisfaction risk")

        # Final calculated bounds
        est_demand = int(round(base_demand * multiplier))
        low = int(round(est_demand * 0.70))
        high = int(round(est_demand * 1.30))
        
        # Recommended listing quantity (standard conservative initial listing target)
        rec_qty = int(round(est_demand * 0.85))
        min_qty = int(round(rec_qty * 0.65))
        max_qty = int(round(rec_qty * 1.35))

        # Overall direction
        if multiplier >= 1.15 and est_demand >= 300:
            direction = "Strong Growth"
            signal_strength = "High"
        elif multiplier >= 0.95 and est_demand >= 100:
            direction = "Moderate Demand"
            signal_strength = "Medium"
        elif est_demand > 0:
            direction = "Stable"
            signal_strength = "Medium"
        else:
            direction = "Low Activity"
            signal_strength = "Low"

        confidence = round(max(0.35, min(0.90, confidence)), 2)

        reasoning = (
            f"Historical Amazon benchmark indicates approximately {int(base_demand)} monthly baseline demand. "
            f"Synthesizing YouTube external awareness ({total_views:,} views) and Reddit discussions "
            f"indicates {direction.lower()} momentum. Customer reviews reflect {reviews.get('sentiment', 'Neutral').lower()} sentiment. "
            f"Consequently, estimated demand is {low}–{high} units, with a recommended initial listing quantity of {rec_qty} units."
        )

        return ForecastResult(
            product_id=pinfo.get("product_id", ""),
            product_name=pinfo.get("product_name", ""),
            brand=pinfo.get("brand", ""),
            model=pinfo.get("model", ""),
            category=pinfo.get("category", ""),
            price=pinfo.get("price"),
            rating=pinfo.get("rating"),
            review_count=pinfo.get("review_count"),
            
            amazon_match_type=abench.get("match_type", "exact"),
            amazon_market_signal=abench.get("market_signal", "Moderate Demand"),
            
            youtube_videos_found=yt.get("videos_found", 0),
            youtube_videos_analyzed=yt.get("videos_analyzed", 0),
            youtube_total_views=yt.get("total_views", 0),
            youtube_interest_signal=yt.get("consumer_interest", "Stable" if yt.get("videos_found", 0) > 0 else "Unknown"),
            youtube_sentiment=yt.get("sentiment", "Neutral"),
            
            reddit_posts_found=reddit.get("posts_found", 0),
            reddit_posts_analyzed=reddit.get("posts_analyzed", 0),
            reddit_interest_signal=reddit.get("consumer_interest", "Moderate" if reddit.get("posts_found", 0) > 0 else "Unknown"),
            reddit_sentiment=reddit.get("sentiment", "Neutral"),
            reddit_purchase_intent=reddit.get("purchase_intent", "Moderate" if reddit.get("posts_found", 0) > 0 else "Unknown"),
            
            review_sentiment=reviews.get("sentiment", "Neutral"),
            
            overall_demand_direction=direction,
            demand_signal_strength=signal_strength,
            
            estimated_demand_low=low,
            estimated_demand=est_demand,
            estimated_demand_high=high,
            
            recommended_quantity=rec_qty,
            minimum_quantity=min_qty,
            maximum_quantity=max_qty,
            
            confidence=confidence,
            
            key_positive_factors=pos_factors[:4],
            key_negative_factors=neg_factors[:3],
            risk_factors=risk_factors[:3],
            reasoning=reasoning,
            
            data_sources_available=evidence.data_sources_available,
            detailed_evidence=evidence.to_dict()
        )
