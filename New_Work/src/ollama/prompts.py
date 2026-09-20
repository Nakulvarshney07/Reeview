import json
from typing import Dict, Any, List

class OllamaPrompts:
    """
    Centralized, specialized prompt templates for YouTube, Reddit, and Demand Forecasting.
    """

    @staticmethod
    def build_youtube_analysis_prompt(product_name: str, brand: str, videos_data: List[Dict[str, Any]]) -> str:
        """
        Prompt for analyzing YouTube transcripts and video metadata.
        """
        prompt = f"""You are an expert e-commerce demand analyst.
Analyze the following YouTube video transcripts and metadata for product: "{brand} - {product_name}".

YOUTUBE DATA:
{json.dumps(videos_data, indent=2)}

TASK:
Extract the following signals in JSON format only:
- "sentiment": "Positive" | "Neutral" | "Negative" | "Mixed"
- "product_strengths": list of repeatedly mentioned positive features/aspects
- "product_weaknesses": list of repeatedly mentioned negative features/complaints
- "purchase_intent": "High" | "Moderate" | "Low" | "Negative"
- "consumer_interest": "Increasing" | "Stable" | "Decreasing" | "Uncertain"
- "repeated_themes": list of recurring key themes across videos
- "summary": concise 2-sentence summary of YouTube discussion

IMPORTANT: Return ONLY valid JSON with no markdown wrapping or extra commentary."""
        return prompt

    @staticmethod
    def build_reddit_analysis_prompt(product_name: str, brand: str, reddit_data: List[Dict[str, Any]]) -> str:
        """
        Prompt for analyzing Reddit discussions and comments.
        """
        prompt = f"""You are an expert e-commerce consumer sentiment analyst.
Analyze the following Reddit discussion posts and comments for product: "{brand} - {product_name}".

REDDIT DISCUSSIONS:
{json.dumps(reddit_data, indent=2)}

TASK:
Extract the following qualitative signals in JSON format only:
- "sentiment": "Positive" | "Neutral" | "Negative" | "Mixed"
- "purchase_intent": "High" | "Moderate" | "Low" | "Negative"
- "user_satisfaction": "High" | "Moderate" | "Low" | "Mixed"
- "recurring_complaints": list of recurring issues or reliability concerns
- "recurring_advantages": list of recurring benefits and praised aspects
- "price_value_perception": "Good Value" | "Fair" | "Overpriced" | "Unknown"
- "consumer_interest": "Strong" | "Moderate" | "Niche" | "Low"
- "summary": concise 2-sentence summary of Reddit community opinions

IMPORTANT: Return ONLY valid JSON with no markdown wrapping or extra commentary."""
        return prompt

    @staticmethod
    def build_demand_forecast_prompt(evidence: Dict[str, Any]) -> str:
        """
        Prompt for multi-signal demand estimation and listing quantity recommendation.
        """
        prompt = f"""You are a senior e-commerce inventory and demand forecasting strategist for Amazon and Flipkart sellers.

Analyze the structured evidence collected across Historical Amazon Data, Customer Reviews, YouTube, and Reddit for this product:

EVIDENCE:
{json.dumps(evidence, indent=2)}

TASK:
Based on all available quantitative and qualitative signals, generate a demand forecast and recommended listing quantity range for an e-commerce seller.

Guidelines:
1. Historical Amazon data is market context (e.g. boughtInLastMonth benchmark).
2. YouTube and Reddit provide current external consumer interest and sentiment.
3. If evidence is weak or sources are missing, increase the range, lower confidence, and note limitations.
4. Output realistic unit quantities (e.g., for a standard monthly listing cycle).

Return ONLY valid JSON matching this exact schema:
{{
  "overall_demand_direction": "Strong Growth" | "Moderate Demand" | "Stable" | "Declining" | "Low Activity",
  "demand_signal_strength": "High" | "Medium" | "Low",
  "estimated_demand_low": integer,
  "estimated_demand": integer,
  "estimated_demand_high": integer,
  "recommended_quantity": integer,
  "minimum_quantity": integer,
  "maximum_quantity": integer,
  "confidence": float between 0.0 and 1.0,
  "key_positive_factors": ["factor 1", "factor 2"],
  "key_negative_factors": ["factor 1", "factor 2"],
  "risk_factors": ["risk 1", "risk 2"],
  "reasoning": "Clear, grounded 3-4 sentence explanation referencing the specific evidence collected."
}}

IMPORTANT: Return ONLY valid JSON with no markdown formatting or extra text."""
        return prompt
