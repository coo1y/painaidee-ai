"""Central configuration, loaded from environment variables (.env supported).

All settings have sensible defaults so the project runs locally and in Docker
without extra configuration beyond the API keys.
"""
from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover - dotenv is optional at runtime
    pass


# --- Paths --------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.getenv("DATA_DIR", PROJECT_ROOT / "data"))
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", PROJECT_ROOT / "chroma_db"))
FEEDBACK_DB = Path(os.getenv("FEEDBACK_DB", PROJECT_ROOT / "feedback.db"))

ATTRACTION_FILE = DATA_DIR / "attraction-sub.json"
ACTIVITY_FILE = DATA_DIR / "activity-sub.json"

# --- Collections --------------------------------------------------------
ATTRACTIONS_COLLECTION = "attractions"
EVENTS_COLLECTION = "events"

# --- OpenAI -------------------------------------------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

# --- Web search ---------------------------------------------------------
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

# --- Embedding backend --------------------------------------------------
# "openai"  -> real semantic embeddings (needs OPENAI_API_KEY)
# "local"   -> deterministic hash embeddings for OFFLINE SMOKE TESTS ONLY
EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "openai").lower()
LOCAL_EMBEDDING_DIM = 512

# --- Retrieval tuning ---------------------------------------------------
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "5"))
CANDIDATE_K = int(os.getenv("CANDIDATE_K", "20"))  # fetched before re-ranking
RRF_K = 60  # reciprocal-rank-fusion constant


def has_openai() -> bool:
    return bool(OPENAI_API_KEY) and OPENAI_API_KEY != "XXXXXX"


def has_tavily() -> bool:
    return bool(TAVILY_API_KEY) and TAVILY_API_KEY != "XXXXXX"


def use_openai_embeddings() -> bool:
    return EMBEDDING_BACKEND == "openai" and has_openai()
