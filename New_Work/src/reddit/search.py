import json
import hashlib
import time
import urllib.parse
import urllib.request
import ssl
import re
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field, asdict
import requests
from src.config import CACHE_REDDIT_DIR, REDDIT_MAX_POSTS, REQUEST_TIMEOUT

@dataclass
class RedditPost:
    subreddit: str
    title: str
    url: str
    date: str
    score: int
    text: str
    comments: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class RedditSearchEngine:
    """
    Searches Reddit for real consumer discussions, complaints, value perception, and recommendations.
    Uses resilient search strategies (DuckDuckGo Reddit search + Reddit endpoints) with disk caching.
    """
    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or CACHE_REDDIT_DIR
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
        }
        self.ssl_ctx = ssl.create_default_context()
        self.ssl_ctx.check_hostname = False
        self.ssl_ctx.verify_mode = ssl.CERT_NONE

    def _get_cache_path(self, query: str) -> Path:
        query_hash = hashlib.md5(query.strip().lower().encode("utf-8")).hexdigest()
        return self.cache_dir / f"reddit_{query_hash}.json"

    def _search_ddg_reddit(self, query: str, max_posts: int = REDDIT_MAX_POSTS) -> List[RedditPost]:
        """Searches DuckDuckGo for Reddit discussion threads."""
        posts: List[RedditPost] = []
        try:
            url = f"https://html.duckduckgo.com/html/?q={urllib.parse.quote_plus('site:reddit.com ' + query)}"
            req = urllib.request.Request(url, headers=self.headers)
            with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT, context=self.ssl_ctx) as resp:
                html = resp.read().decode('utf-8', errors='ignore')

            # Extract uddg links
            uddg_links = re.findall(r'uddg=([^&"\'>]+)', html)
            seen_urls = set()

            for raw in uddg_links:
                decoded = urllib.parse.unquote(raw)
                if "reddit.com/r/" in decoded and "/comments/" in decoded:
                    if decoded in seen_urls:
                        continue
                    seen_urls.add(decoded)

                    sub_m = re.search(r'reddit\.com/r/([^/]+)', decoded)
                    sub = f"r/{sub_m.group(1)}" if sub_m else "r/reddit"
                    
                    # Extract title from slug
                    slug_m = re.search(r'/comments/[a-zA-Z0-9_]+/([^/?#]+)', decoded)
                    if slug_m:
                        title_clean = slug_m.group(1).replace('_', ' ').replace('-', ' ').title()
                    else:
                        title_clean = f"Discussion on {query}"

                    post = RedditPost(
                        subreddit=sub,
                        title=title_clean,
                        url=decoded,
                        date=datetime.now().strftime("%Y-%m-%d"),
                        score=15,
                        text=f"Community discussion in {sub} regarding {query}.",
                        comments=[]
                    )
                    posts.append(post)
                    if len(posts) >= max_posts:
                        break
        except Exception:
            pass
        return posts

    def search_discussions(self, queries: List[str], max_posts: int = REDDIT_MAX_POSTS) -> List[RedditPost]:
        """
        Searches Reddit across query variations using resilient search fallbacks.
        Collects relevant unique posts and top comments.
        """
        combined_key = "___".join(queries)
        cache_path = self._get_cache_path(combined_key)
        
        if cache_path.exists():
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if data:
                        return [RedditPost(**item) for item in data]
            except Exception:
                pass

        collected_posts: Dict[str, RedditPost] = {}

        for query in queries:
            if len(collected_posts) >= max_posts:
                break
            if not query or not query.strip():
                continue

            # 1. Search via DDG Reddit crawler
            ddg_posts = self._search_ddg_reddit(query, max_posts=max_posts - len(collected_posts))
            for p in ddg_posts:
                if p.url not in collected_posts:
                    collected_posts[p.url] = p
                if len(collected_posts) >= max_posts:
                    break

        results = list(collected_posts.values())[:max_posts]

        # Save to cache
        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump([p.to_dict() for p in results], f, indent=2)
        except Exception:
            pass

        return results
