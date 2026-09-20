from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field, asdict
from src.product_parser.normalizer import NormalizedProduct
from src.amazon.benchmark import AmazonBenchmarkResult
from src.reviews.analyzer import ReviewAnalysisResult
from src.youtube.search import YouTubeVideo
from src.youtube.transcript import TranscriptResult
from src.reddit.search import RedditPost

@dataclass
class StructuredEvidence:
    product_information: Dict[str, Any]
    amazon_market_benchmark: Dict[str, Any]
    customer_reviews: Dict[str, Any]
    youtube_analysis: Dict[str, Any]
    reddit_analysis: Dict[str, Any]
    quantitative_signals: Dict[str, Any]
    qualitative_signals: Dict[str, Any]
    data_sources_available: Dict[str, bool]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class EvidenceCollector:
    """
    Synthesizes and structures evidence collected across all data channels.
    Strictly separates quantitative metrics from qualitative insights.
    """

    @classmethod
    def assemble(
        cls,
        product: NormalizedProduct,
        amazon_benchmark: AmazonBenchmarkResult,
        review_analysis: ReviewAnalysisResult,
        youtube_videos: List[YouTubeVideo],
        youtube_transcripts: List[TranscriptResult],
        youtube_ollama_analysis: Optional[Dict[str, Any]],
        reddit_posts: List[RedditPost],
        reddit_ollama_analysis: Optional[Dict[str, Any]]
    ) -> StructuredEvidence:
        
        # Determine source availability
        has_amazon = amazon_benchmark is not None
        has_reviews = review_analysis is not None
        has_youtube = len(youtube_videos) > 0
        has_reddit = len(reddit_posts) > 0
        
        data_sources_available = {
            "amazon": has_amazon,
            "reviews": has_reviews,
            "youtube": has_youtube,
            "reddit": has_reddit
        }

        # Extract qualitative signals from transcripts & Reddit (used directly or as enrichment)
        from src.reviews.aspect_extractor import SignalAspectExtractor
        yt_strengths, yt_weaknesses, yt_sent = SignalAspectExtractor.extract_from_youtube(youtube_transcripts, product.product_name)
        rd_advantages, rd_complaints, rd_sent, rd_val = SignalAspectExtractor.extract_from_reddit(reddit_posts, product.product_name)

        # Merge with Ollama if available
        final_yt_strengths = (youtube_ollama_analysis or {}).get("product_strengths") or yt_strengths
        final_yt_weaknesses = (youtube_ollama_analysis or {}).get("product_weaknesses") or yt_weaknesses
        final_yt_sentiment = (youtube_ollama_analysis or {}).get("sentiment") or yt_sent

        final_rd_complaints = (reddit_ollama_analysis or {}).get("recurring_complaints") or rd_complaints
        final_rd_advantages = (reddit_ollama_analysis or {}).get("recurring_advantages") or rd_advantages
        final_rd_sentiment = (reddit_ollama_analysis or {}).get("sentiment") or rd_sent
        final_rd_val = (reddit_ollama_analysis or {}).get("price_value_perception") or rd_val

        # Aggregate YouTube metrics
        total_yt_views = sum(v.views for v in youtube_videos)
        transcripts_found = sum(1 for t in youtube_transcripts if t.transcript_available)
        
        yt_analysis_dict = {
            "videos_found": len(youtube_videos),
            "videos_analyzed": len(youtube_videos),
            "total_views": total_yt_views,
            "transcripts_available_count": transcripts_found,
            "sentiment": final_yt_sentiment,
            "product_strengths": final_yt_strengths,
            "product_weaknesses": final_yt_weaknesses,
            "purchase_intent": (youtube_ollama_analysis or {}).get("purchase_intent", "Moderate" if has_youtube else "Unknown"),
            "consumer_interest": (youtube_ollama_analysis or {}).get("consumer_interest", "Stable" if has_youtube else "Unknown"),
            "videos": [
                {
                    "video_id": v.video_id,
                    "title": v.title,
                    "channel": v.channel,
                    "url": v.url,
                    "duration": v.duration
                }
                for v in youtube_videos
            ],
            "summary": (youtube_ollama_analysis or {}).get("summary", f"Found {len(youtube_videos)} relevant YouTube videos discussing product durability, features, and user experience.")
        }

        # Aggregate Reddit metrics
        total_reddit_score = sum(p.score for p in reddit_posts)
        total_comments = sum(len(p.comments) for p in reddit_posts)
        
        reddit_analysis_dict = {
            "posts_found": len(reddit_posts),
            "posts_analyzed": len(reddit_posts),
            "total_score": total_reddit_score,
            "total_comments_collected": total_comments,
            "sentiment": final_rd_sentiment,
            "purchase_intent": (reddit_ollama_analysis or {}).get("purchase_intent", "Moderate" if has_reddit else "Unknown"),
            "user_satisfaction": (reddit_ollama_analysis or {}).get("user_satisfaction", "Moderate" if has_reddit else "Unknown"),
            "recurring_complaints": final_rd_complaints,
            "recurring_advantages": final_rd_advantages,
            "price_value_perception": final_rd_val,
            "posts": [
                {
                    "subreddit": p.subreddit,
                    "title": p.title,
                    "url": p.url
                }
                for p in reddit_posts
            ],
            "summary": (reddit_ollama_analysis or {}).get("summary", f"Collected {len(reddit_posts)} Reddit discussions covering user satisfaction, price-to-value, and recurring feedback.")
        }

        # Quantitative Signals
        quantitative_signals = {
            "price": product.price,
            "list_price": product.list_price,
            "discount_pct": amazon_benchmark.discount_pct,
            "rating": product.rating,
            "review_count": product.review_count,
            "historical_monthly_sales_benchmark": amazon_benchmark.historical_monthly_sales_benchmark,
            "is_best_seller": product.is_best_seller,
            "youtube_total_views": total_yt_views,
            "youtube_video_count": len(youtube_videos),
            "reddit_discussion_count": len(reddit_posts),
            "reddit_comment_count": total_comments
        }

        # Qualitative Signals
        qualitative_signals = {
            "amazon_market_signal": amazon_benchmark.market_signal,
            "review_sentiment": review_analysis.sentiment,
            "review_positive_factors": review_analysis.positive_factors,
            "review_negative_factors": review_analysis.negative_factors,
            "youtube_sentiment": yt_analysis_dict["sentiment"],
            "youtube_purchase_intent": yt_analysis_dict["purchase_intent"],
            "youtube_strengths": yt_analysis_dict["product_strengths"],
            "youtube_weaknesses": yt_analysis_dict["product_weaknesses"],
            "reddit_sentiment": reddit_analysis_dict["sentiment"],
            "reddit_purchase_intent": reddit_analysis_dict["purchase_intent"],
            "reddit_complaints": reddit_analysis_dict["recurring_complaints"],
            "reddit_advantages": reddit_analysis_dict["recurring_advantages"]
        }

        return StructuredEvidence(
            product_information=product.to_dict(),
            amazon_market_benchmark=amazon_benchmark.to_dict(),
            customer_reviews=review_analysis.to_dict(),
            youtube_analysis=yt_analysis_dict,
            reddit_analysis=reddit_analysis_dict,
            quantitative_signals=quantitative_signals,
            qualitative_signals=qualitative_signals,
            data_sources_available=data_sources_available
        )
