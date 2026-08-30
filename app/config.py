"""
Application-wide configuration values loaded from environment variables,
plus the shared logger used across the app.
"""
import logging
import os

# ------------------------------------------------------------------
# Logging setup (console log format)
# ------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("DanAI")

# ------------------------------------------------------------------
# MongoDB
# ------------------------------------------------------------------
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")

# ------------------------------------------------------------------
# Ollama / LLM models
# ------------------------------------------------------------------
OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL", "http://localhost:11434/api/generate"
)
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "qwen3:14B")
NEWS_MODEL = os.getenv("DEFAULT_MODEL", "qwen3:14B")

# Embedding / RAG settings
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
OLLAMA_EMBED_URL = OLLAMA_BASE_URL.replace("/api/generate", "/api/embeddings")

# ------------------------------------------------------------------
# Obsidian vault (source of vocabulary / word-memory sentences)
# ------------------------------------------------------------------
OBSIDIAN_VAULT_PATH = os.getenv(
    "OBSIDIAN_VAULT_PATH",
    "/Users/danielahn/Documents/Note/Obsidian/Dan_Dev",
)

# ------------------------------------------------------------------
# Canada news RSS feeds
# ------------------------------------------------------------------
CANADA_NEWS_RSS_URLS = [
    "https://vancouver.citynews.ca/feed/",                    # CityNews Vancouver
    # "https://www.cbc.ca/cbc-stats/rss/rss-topstories.xml",  # CBC Top Stories (updated)
    # "https://globalnews.ca/canada/feed/",                   # Global News Canada (alternative)
    "https://www.yna.co.kr/rss/international.xml",
]
