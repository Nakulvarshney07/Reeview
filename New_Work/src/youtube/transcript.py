import json
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from src.config import CACHE_TRANSCRIPTS_DIR

@dataclass
class TranscriptResult:
    video_id: str
    title: str
    url: str
    views: int
    transcript_available: bool
    transcript: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class YouTubeTranscriptFetcher:
    """
    Fetches captions/transcripts for YouTube videos using youtube-transcript-api.
    Implements persistent disk caching and handles disabled/unavailable transcripts gracefully.
    Never fabricates missing transcript text.
    """
    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or CACHE_TRANSCRIPTS_DIR

    def _get_cache_path(self, video_id: str) -> Path:
        return self.cache_dir / f"transcript_{video_id}.json"

    def get_transcript(self, video_id: str, title: str = "", views: int = 0, description: str = "") -> TranscriptResult:
        """
        Retrieves transcript for a given YouTube video ID.
        Uses video description as fallback context if subtitles/captions are disabled.
        """
        url = f"https://www.youtube.com/watch?v={video_id}"
        cache_path = self._get_cache_path(video_id)
        
        if cache_path.exists():
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return TranscriptResult(**data)
            except Exception:
                pass

        transcript_text = ""
        is_available = False

        try:
            # Try fetching available transcripts (English or auto-generated)
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=['en', 'en-US', 'en-GB', 'auto'])
            if transcript_list:
                lines = [t.get("text", "").strip() for t in transcript_list if t.get("text")]
                transcript_text = " ".join(lines)
                is_available = len(transcript_text) > 0
        except (TranscriptsDisabled, NoTranscriptFound):
            is_available = False
        except Exception:
            is_available = False

        # If transcript not available but video description exists, use description as context
        if not is_available and description and len(description.strip()) > 30:
            transcript_text = f"[Video Description/Summary]: {description.strip()}"
            is_available = True

        # Limit transcript length to reasonable size for Ollama prompt efficiency
        if len(transcript_text) > 4000:
            transcript_text = transcript_text[:4000] + "..."

        result = TranscriptResult(
            video_id=video_id,
            title=title,
            url=url,
            views=views,
            transcript_available=is_available,
            transcript=transcript_text
        )

        # Cache result
        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(result.to_dict(), f, indent=2)
        except Exception:
            pass

        return result
