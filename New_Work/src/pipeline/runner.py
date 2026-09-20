import csv
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from src.config import (
    OUTPUT_CSV_PATH,
    OUTPUT_JSON_PATH,
    MAX_LIMIT,
    DEFAULT_LIMIT,
    YOUTUBE_MAX_RESULTS,
    REDDIT_MAX_POSTS
)
from src.csv_loader.loader import CSVProductLoader
from src.product_parser.normalizer import ProductNormalizer, NormalizedProduct
from src.amazon.benchmark import AmazonMarketBenchmark
from src.youtube.search import YouTubeSearchEngine, YouTubeVideo
from src.youtube.transcript import YouTubeTranscriptFetcher, TranscriptResult
from src.reddit.search import RedditSearchEngine, RedditPost
from src.reviews.analyzer import CSVReviewAnalyzer
from src.ollama.client import OllamaClient
from src.ollama.prompts import OllamaPrompts
from src.forecasting.evidence import EvidenceCollector, StructuredEvidence
from src.forecasting.estimator import DemandEstimator, ForecastResult

class DemandForecastingPipeline:
    """
    End-to-End Orchestrator for CSV-Based E-Commerce Demand Forecasting.
    """
    def __init__(
        self,
        output_csv: Optional[Path] = None,
        output_json: Optional[Path] = None,
    ):
        self.output_csv = output_csv or OUTPUT_CSV_PATH
        self.output_json = output_json or OUTPUT_JSON_PATH

        self.loader = CSVProductLoader()
        self.yt_search = YouTubeSearchEngine()
        self.yt_transcript = YouTubeTranscriptFetcher()
        self.reddit_search = RedditSearchEngine()
        self.ollama = OllamaClient()
        self.estimator = DemandEstimator(self.ollama)

    def run(self, limit: int = DEFAULT_LIMIT, start: int = 0) -> List[ForecastResult]:
        """
        Executes the forecasting pipeline for `limit` products starting at `start`.
        """
        # Enforce limit capping
        if limit > MAX_LIMIT:
            print(f"[Pipeline] Requested limit ({limit}) exceeds max allowed ({MAX_LIMIT}). Capping to {MAX_LIMIT}.")
            limit = MAX_LIMIT
        elif limit < 1:
            limit = 1

        print("=" * 70)
        print(f"  E-COMMERCE DEMAND FORECASTING PIPELINE (Limit: {limit}, Start: {start})")
        print("=" * 70)

        # 1. Load raw rows from CSV
        print(f"\n[1/3] Reading Amazon CSV (offset: {start}, limit: {limit})...", flush=True)
        raw_items = self.loader.load_products(start=start, limit=limit)
        print(f"Loaded {len(raw_items)} product rows from CSV.", flush=True)

        forecast_results: List[ForecastResult] = []

        # 2. Process each product
        for idx, raw_item in enumerate(raw_items, 1):
            product = ProductNormalizer.normalize(raw_item)
            short_title = product.product_name[:60] + "..." if len(product.product_name) > 60 else product.product_name
            
            print(f"\n[{idx}/{len(raw_items)}] Processing: {short_title}", flush=True)
            
            # Step A: Amazon historical market benchmark
            print(f"  [AMAZON] Matching historical market benchmark...", flush=True)
            amazon_benchmark = AmazonMarketBenchmark.evaluate(product)
            print(f"  [AMAZON] Match: {amazon_benchmark.match_type} | Signal: {amazon_benchmark.market_signal}", flush=True)
            
            # Step B: Customer Reviews analysis
            review_analysis = CSVReviewAnalyzer.analyze(product)
            
            # Step C: YouTube Search & Transcripts
            yt_query = product.search_queries.get("review", product.product_name)
            print(f"  [YOUTUBE] Searching: '{yt_query}'...", flush=True)
            yt_videos = self.yt_search.search_videos(yt_query, brand=product.brand, max_results=YOUTUBE_MAX_RESULTS)
            print(f"  [YOUTUBE] Found {len(yt_videos)} relevant videos (Total Views: {sum(v.views for v in yt_videos):,}).", flush=True)
            
            # Step D: YouTube Transcripts
            yt_transcripts: List[TranscriptResult] = []
            if yt_videos:
                print(f"  [YOUTUBE] Retrieving video transcripts...", flush=True)
                for v in yt_videos:
                    t_res = self.yt_transcript.get_transcript(v.video_id, title=v.title, views=v.views, description=v.description)
                    yt_transcripts.append(t_res)
                available_count = sum(1 for t in yt_transcripts if t.transcript_available)
                print(f"  [YOUTUBE] Transcripts available: {available_count}/{len(yt_videos)}.", flush=True)

            # Step E: YouTube Ollama Analysis
            yt_ollama_analysis = None
            if yt_transcripts and any(t.transcript_available for t in yt_transcripts):
                transcripts_payload = [
                    {"title": t.title, "views": t.views, "transcript": t.transcript}
                    for t in yt_transcripts if t.transcript_available
                ]
                yt_prompt = OllamaPrompts.build_youtube_analysis_prompt(product.product_name, product.brand, transcripts_payload)
                yt_ollama_analysis = self.ollama.generate(yt_prompt)

            # Step F: Reddit Discussions
            reddit_queries = [
                product.search_queries.get("reddit", product.product_name),
                product.search_queries.get("reddit_problems", f"{product.product_name} problems"),
                product.search_queries.get("reddit_worth_it", f"{product.product_name} worth it")
            ]
            print(f"  [REDDIT] Searching community discussions...", flush=True)
            reddit_posts = self.reddit_search.search_discussions(reddit_queries, max_posts=REDDIT_MAX_POSTS)
            print(f"  [REDDIT] Found {len(reddit_posts)} relevant threads.", flush=True)

            # Step G: Reddit Ollama Analysis
            reddit_ollama_analysis = None
            if reddit_posts:
                reddit_payload = [
                    {"subreddit": p.subreddit, "title": p.title, "text": p.text, "comments": p.comments}
                    for p in reddit_posts
                ]
                reddit_prompt = OllamaPrompts.build_reddit_analysis_prompt(product.product_name, product.brand, reddit_payload)
                reddit_ollama_analysis = self.ollama.generate(reddit_prompt)

            # Step H: Assemble Evidence & Forecast
            print(f"  [OLLAMA] Analyzing signals & estimating demand...", flush=True)
            evidence = EvidenceCollector.assemble(
                product=product,
                amazon_benchmark=amazon_benchmark,
                review_analysis=review_analysis,
                youtube_videos=yt_videos,
                youtube_transcripts=yt_transcripts,
                youtube_ollama_analysis=yt_ollama_analysis,
                reddit_posts=reddit_posts,
                reddit_ollama_analysis=reddit_ollama_analysis
            )

            forecast = self.estimator.estimate_demand(evidence)
            print(f"  [FORECAST] Recommended Listing Quantity: {forecast.recommended_quantity} units (Range: {forecast.estimated_demand_low}-{forecast.estimated_demand_high}, Conf: {forecast.confidence})", flush=True)
            print(f"  [DONE]", flush=True)
            
            forecast_results.append(forecast)

        # 3. Export to CSV and JSON
        print(f"\n[3/3] Saving results...", flush=True)
        self._export_csv(forecast_results)
        self._export_json(forecast_results)

        print("=" * 70)
        print(f"  COMPLETED: {len(forecast_results)} products processed.")
        print(f"  Output CSV:  {self.output_csv}")
        print(f"  Output JSON: {self.output_json}")
        print("=" * 70)

        return forecast_results

    def _export_csv(self, results: List[ForecastResult]):
        """Exports tabular forecast results to CSV."""
        if not results:
            return
            
        columns = [
            "product_id", "product_name", "brand", "model", "category",
            "price", "rating", "review_count",
            "amazon_match_type", "amazon_market_signal",
            "youtube_videos_found", "youtube_videos_analyzed", "youtube_total_views",
            "youtube_video_urls", "youtube_interest_signal", "youtube_sentiment",
            "reddit_posts_found", "reddit_posts_analyzed", "reddit_post_urls",
            "reddit_interest_signal", "reddit_sentiment", "reddit_purchase_intent",
            "review_sentiment",
            "overall_demand_direction", "demand_signal_strength",
            "estimated_demand_low", "estimated_demand", "estimated_demand_high",
            "recommended_quantity", "minimum_quantity", "maximum_quantity",
            "confidence",
            "key_positive_factors", "key_negative_factors", "risk_factors",
            "reasoning", "data_sources_available"
        ]

        with open(self.output_csv, mode="w", encoding="utf-8", newline="", errors="replace") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            for r in results:
                row_dict = r.to_dict()
                # Format lists/dicts as semicolon or JSON strings for CSV readability
                row_dict["key_positive_factors"] = " | ".join(r.key_positive_factors)
                row_dict["key_negative_factors"] = " | ".join(r.key_negative_factors)
                row_dict["risk_factors"] = " | ".join(r.risk_factors)
                row_dict["data_sources_available"] = json.dumps(r.data_sources_available)
                
                # Extract clickable URLs from detailed evidence
                yt_list = r.detailed_evidence.get("youtube_analysis", {}).get("videos", [])
                row_dict["youtube_video_urls"] = " | ".join([v.get("url", "") for v in yt_list if v.get("url")])
                
                rd_list = r.detailed_evidence.get("reddit_analysis", {}).get("posts", [])
                row_dict["reddit_post_urls"] = " | ".join([p.get("url", "") for p in rd_list if p.get("url")])

                # Exclude internal detailed_evidence dict from flat CSV
                row_dict.pop("detailed_evidence", None)
                writer.writerow(row_dict)

    def _export_json(self, results: List[ForecastResult]):
        """Exports complete structured evidence and forecasts to JSON."""
        with open(self.output_json, mode="w", encoding="utf-8") as f:
            json.dump([r.to_dict() for r in results], f, indent=2)
