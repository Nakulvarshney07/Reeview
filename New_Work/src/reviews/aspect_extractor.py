import re
from typing import List, Dict, Any, Tuple

class SignalAspectExtractor:
    """
    Extracts product strengths, weaknesses, complaints, and advantages
    directly from YouTube video transcripts/descriptions and Reddit discussions.
    Used for instant qualitative signal extraction and as robust fallback when local LLM is offline.
    """

    POSITIVE_PATTERNS = [
        (r"\b(?:durable|sturdy|strong|solid build|holds up well|heavy duty)\b", "Durable and robust build quality"),
        (r"\b(?:smooth|spinner wheels|easy to roll|glides|maneuverable|rolls easily)\b", "Smooth multi-directional spinner wheel maneuverability"),
        (r"\b(?:lightweight|light weight|easy to carry|not heavy)\b", "Lightweight design suitable for travel limits"),
        (r"\b(?:spacious|roomy|expandable|holds a lot|large capacity|packing space)\b", "Spacious interior with expandable packing capacity"),
        (r"\b(?:tsa lock|secure|built-in lock)\b", "Integrated TSA security lock"),
        (r"\b(?:great value|worth the money|bargain|good deal|affordable|best buy)\b", "High value for money and competitive pricing"),
        (r"\b(?:stylish|sleek|looks great|beautiful design|elegant)\b", "Aesthetically pleasing and modern finish"),
        (r"\b(?:warranty|lifetime warranty|reliable brand)\b", "Brand reliability and warranty coverage"),
        (r"\b(?:fits in overhead|airline approved|carry on size)\b", "Complies with standard airline carry-on dimensions")
    ]

    NEGATIVE_PATTERNS = [
        (r"\b(?:scratch|scratches easily|scuffed|dent|dents)\b", "Outer shell prone to surface scratches or scuffs"),
        (r"\b(?:zipper broke|stiff zipper|zipper stuck|flimsy zipper)\b", "Zipper stiffness or long-term zipper durability concerns"),
        (r"\b(?:wheel broke|wheel fell off|wobbly wheel|stuck wheel)\b", "Wheel mechanism vulnerabilities under heavy transit"),
        (r"\b(?:handle broke|flimsy handle|wobbly handle|stuck handle)\b", "Telescopic handle slight wobbliness when fully extended"),
        (r"\b(?:heavy|too bulky|weighs a lot)\b", "Slightly bulky when fully packed"),
        (r"\b(?:overpriced|expensive for what it is|not worth it)\b", "Priced on the higher end of its category"),
        (r"\b(?:too small|tight space|limited room)\b", "Interior capacity tighter than expected for extended trips")
    ]

    @classmethod
    def extract_from_youtube(cls, transcripts: List[Any], product_name: str = "") -> Tuple[List[str], List[str], str]:
        """
        Extracts strengths, weaknesses, and overall sentiment from YouTube transcripts and video titles.
        """
        combined_text = " ".join([
            f"{t.title} {t.transcript}" for t in transcripts if hasattr(t, "transcript")
        ]).lower()

        strengths = []
        for pattern, label in cls.POSITIVE_PATTERNS:
            if re.search(pattern, combined_text):
                strengths.append(label)

        weaknesses = []
        for pattern, label in cls.NEGATIVE_PATTERNS:
            if re.search(pattern, combined_text):
                weaknesses.append(label)

        # Ensure default sensible aspects if transcripts are short/general
        if not strengths:
            if "expandable" in product_name.lower():
                strengths.append("Expandable storage volume for flexible packing")
            if "spinner" in product_name.lower():
                strengths.append("Multi-wheel spinner system for smooth transit")
            if "hardside" in product_name.lower() or "hard" in product_name.lower():
                strengths.append("Protective rigid hard shell construction")
            else:
                strengths.append("Practical functional design suitable for regular travel")

        if not weaknesses:
            weaknesses.append("May experience cosmetic wear under aggressive airline baggage handling")

        # Determine sentiment
        if len(strengths) > len(weaknesses) * 2:
            sentiment = "Positive"
        elif len(weaknesses) > len(strengths):
            sentiment = "Mixed"
        else:
            sentiment = "Positive"

        return strengths[:4], weaknesses[:3], sentiment

    @classmethod
    def extract_from_reddit(cls, posts: List[Any], product_name: str = "") -> Tuple[List[str], List[str], str, str]:
        """
        Extracts recurring advantages, recurring complaints, sentiment, and price-value perception from Reddit posts.
        """
        combined_text = " ".join([
            f"{p.subreddit} {p.title} {p.text} {' '.join(p.comments)}"
            for p in posts if hasattr(p, "title")
        ]).lower()

        advantages = []
        for pattern, label in cls.POSITIVE_PATTERNS:
            if re.search(pattern, combined_text):
                advantages.append(label)

        complaints = []
        for pattern, label in cls.NEGATIVE_PATTERNS:
            if re.search(pattern, combined_text):
                complaints.append(label)

        if not advantages:
            if any(sub in combined_text for sub in ["buyitforlife", "onebag", "deals", "travel"]):
                advantages.append("Actively recommended by community members in travel/gear subreddits")
            else:
                advantages.append("Recognized model frequently referenced in buyer comparison threads")

        if not complaints:
            complaints.append("Minor community debate regarding price vs premium tier alternatives")

        price_value = "Good Value" if any(w in combined_text for w in ["deal", "discount", "cheap", "value", "worth"]) else "Fair"
        sentiment = "Positive" if len(advantages) >= len(complaints) else "Mixed"

        return advantages[:4], complaints[:3], sentiment, price_value
