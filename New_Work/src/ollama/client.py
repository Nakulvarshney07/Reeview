import json
import re
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional
import requests
from src.config import OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_TIMEOUT, CACHE_OLLAMA_DIR

class OllamaClient:
    """
    Centralized HTTP client for local Ollama inference.
    Features persistent disk caching, JSON extraction, and graceful failure handling.
    """
    def __init__(self, base_url: Optional[str] = None, model: Optional[str] = None, cache_dir: Optional[Path] = None):
        self.base_url = (base_url or OLLAMA_BASE_URL).rstrip("/")
        self.model = model or OLLAMA_MODEL
        self.cache_dir = cache_dir or CACHE_OLLAMA_DIR
        self.timeout = OLLAMA_TIMEOUT

    def _get_cache_path(self, prompt: str) -> Path:
        prompt_hash = hashlib.md5(f"{self.model}__{prompt.strip()}".encode("utf-8")).hexdigest()
        return self.cache_dir / f"ollama_{prompt_hash}.json"

    def is_available(self) -> bool:
        """Checks if local Ollama server is reachable."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=2)
            return resp.status_code == 200
        except Exception:
            return False

    @staticmethod
    def _extract_json(text: str) -> Optional[Dict[str, Any]]:
        """Extracts JSON object from text, handling markdown code fences if present."""
        if not text:
            return None
        text = text.strip()
        
        # Try direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try regex for ```json ... ```
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        # Try finding outermost braces { ... }
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start:end+1])
            except json.JSONDecodeError:
                pass

        return None

    def generate(self, prompt: str, system: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Sends prompt to Ollama, parses JSON output, and caches result.
        Returns parsed JSON dict or None on failure.
        """
        cache_path = self._get_cache_path(prompt)
        if cache_path.exists():
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass

        endpoint = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.2,
                "top_p": 0.9
            }
        }
        if system:
            payload["system"] = system

        try:
            resp = requests.post(endpoint, json=payload, timeout=self.timeout)
            if resp.status_code == 200:
                data = resp.json()
                raw_response = data.get("response", "")
                parsed_json = self._extract_json(raw_response)
                
                if parsed_json:
                    # Save to cache
                    try:
                        with open(cache_path, "w", encoding="utf-8") as f:
                            json.dump(parsed_json, f, indent=2)
                    except Exception:
                        pass
                    return parsed_json
        except Exception as e:
            # Network timeout or connection refused
            pass

        return None
