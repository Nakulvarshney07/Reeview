import os
from pathlib import Path
from dotenv import load_dotenv

# Base directory for New_work
BASE_DIR = Path(__file__).resolve().parent.parent

# Load .env if present
load_dotenv(BASE_DIR / ".env")

# Ollama Configuration
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "60"))

# CSV File Paths
PRODUCTS_CSV_PATH = BASE_DIR / "amazon_products.csv"
CATEGORIES_CSV_PATH = BASE_DIR / "amazon_categories.csv"

# Cache Directories
CACHE_DIR = BASE_DIR / "cache"
CACHE_YOUTUBE_DIR = CACHE_DIR / "youtube"
CACHE_TRANSCRIPTS_DIR = CACHE_DIR / "transcripts"
CACHE_REDDIT_DIR = CACHE_DIR / "reddit"
CACHE_OLLAMA_DIR = CACHE_DIR / "ollama"

# Output Directory
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_CSV_PATH = OUTPUT_DIR / "forecast_results.csv"
OUTPUT_JSON_PATH = OUTPUT_DIR / "forecast_results.json"

# Ensure runtime directories exist
for directory in [
    CACHE_DIR,
    CACHE_YOUTUBE_DIR,
    CACHE_TRANSCRIPTS_DIR,
    CACHE_REDDIT_DIR,
    CACHE_OLLAMA_DIR,
    OUTPUT_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)

# Rate limits and limits
MAX_LIMIT = 100
DEFAULT_LIMIT = 10
YOUTUBE_MAX_RESULTS = 5
REDDIT_MAX_POSTS = 5
REQUEST_TIMEOUT = 10
