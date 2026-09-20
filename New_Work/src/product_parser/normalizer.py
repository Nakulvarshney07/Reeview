import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict

@dataclass
class NormalizedProduct:
    product_id: str
    product_name: str
    brand: str
    model: str
    category: str
    price: Optional[float] = None
    list_price: Optional[float] = None
    rating: Optional[float] = None
    review_count: Optional[int] = None
    bought_in_last_month: Optional[int] = None
    is_best_seller: bool = False
    product_url: str = ""
    img_url: str = ""
    description: str = ""
    reviews: List[Dict[str, Any]] = field(default_factory=list)
    raw_data: Dict[str, Any] = field(default_factory=dict)
    search_queries: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class ProductNormalizer:
    """
    Normalizes raw Amazon product rows into structured NormalizedProduct entities.
    Extracts brand, model, and generates targeted search query identities.
    """
    
    # Common packaging/measurement terms to clean from titles for search
    CLEANUP_PATTERNS = [
        r"\b\d+\s*(?:pack|pcs|piece|pieces|count|ct|oz|ml|kg|lbs?|g|inch|inches|\"|\')\b",
        r"\[.*?\]",
        r"\(.*?\)",
        r"\b(?:with|and|for|in|set|by|the)\b",
    ]

    @classmethod
    def extract_brand_and_model(cls, title: str) -> tuple[str, str, str]:
        """
        Extracts brand, model, and simplified product name from product title.
        """
        if not title:
            return "", "", ""

        # First clean special characters but keep alphanumeric and hyphens
        cleaned_title = re.sub(r'["\',]', '', title).strip()
        words = cleaned_title.split()
        
        if not words:
            return "", "", ""

        # Check for model indicators (e.g., alphanumeric strings containing digits or hyphens)
        model = ""
        for w in words[1:7]:
            if re.search(r'\d', w) and len(w) >= 2:
                model = w
                break

        brand = words[0]
        # Generate a concise product name (first 5-7 meaningful words)
        short_name = " ".join(words[:7])
        return brand, model, short_name

    @classmethod
    def normalize(cls, raw_item: Dict[str, Any]) -> NormalizedProduct:
        """
        Converts a raw CSV product dict into a NormalizedProduct.
        """
        asin = raw_item.get("asin") or f"ROW_{raw_item.get('row_index', '0')}"
        title = raw_item.get("title") or "Unknown Product"
        category = raw_item.get("category_name") or "General"
        
        brand, model, short_name = cls.extract_brand_and_model(title)
        
        # Clean base search query (avoid duplicating brand if already first word)
        base_query = short_name.strip()
        
        # Limit search query length to avoid over-specification
        query_exact = " ".join(base_query.split()[:6])
        
        # Targeted search queries
        query_review = f"{query_exact} review"
        query_youtube = f"{query_exact} review"
        query_reddit = query_exact

        search_queries = {
            "exact": query_exact,
            "review": query_review,
            "youtube": query_youtube,
            "reddit": query_reddit,
            "reddit_problems": f"{query_reddit} problems",
            "reddit_worth_it": f"{query_reddit} worth it"
        }

        reviews_list = []
        # If there are individual review texts in the row (none in standard summary CSV, but prepared)
        if raw_item.get("review_text"):
            reviews_list.append({"text": raw_item["review_text"]})

        return NormalizedProduct(
            product_id=asin,
            product_name=title,
            brand=brand,
            model=model,
            category=category,
            price=raw_item.get("price"),
            list_price=raw_item.get("list_price"),
            rating=raw_item.get("stars"),
            review_count=raw_item.get("reviews"),
            bought_in_last_month=raw_item.get("bought_in_last_month"),
            is_best_seller=raw_item.get("is_best_seller", False),
            product_url=raw_item.get("product_url", ""),
            img_url=raw_item.get("img_url", ""),
            description="",
            reviews=reviews_list,
            raw_data=raw_item.get("raw_row", {}),
            search_queries=search_queries
        )
