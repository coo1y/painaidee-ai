"""Central configuration.

Path / model / retrieval settings may come from the environment. **API keys do
not** — the Streamlit UI collects them per browser session so they are never
kept on the server. CLI tools (ingestion, eval) can opt in via
``load_cli_keys_from_env()``.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover - dotenv is optional at runtime
    pass

# Disable ChromaDB telemetry and silence its noisy (harmless) posthog logger,
# which spams "capture() takes 1 positional argument but 3 were given".
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)


# --- Paths --------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.getenv("DATA_DIR", PROJECT_ROOT / "data"))
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", PROJECT_ROOT / "chroma_db"))
FEEDBACK_DB = Path(os.getenv("FEEDBACK_DB", PROJECT_ROOT / "feedback.db"))

ATTRACTION_FILE = DATA_DIR / "attraction.json"
ACTIVITY_FILE = DATA_DIR / "activity.json"

# --- Collections --------------------------------------------------------
ATTRACTIONS_COLLECTION = "attractions"
EVENTS_COLLECTION = "events"

# --- OpenAI (models only; keys are session-supplied) --------------------
# API keys intentionally start empty and are NOT read from os.environ /
# Streamlit secrets. The chat UI sets them on ``config`` for the current
# session only. See ``load_cli_keys_from_env`` for local CLI / Docker ingest.
OPENAI_API_KEY = ""
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-5.4-mini")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

# --- Web search (key is session-supplied; same rules as OpenAI) ---------
TAVILY_API_KEY = ""

# --- Embedding backend --------------------------------------------------
# "openai"  -> real semantic embeddings (needs OPENAI_API_KEY)
# "local"   -> deterministic hash embeddings for OFFLINE SMOKE TESTS ONLY
EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "openai").lower()
LOCAL_EMBEDDING_DIM = 512

# --- Retrieval tuning ---------------------------------------------------
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "5"))
CANDIDATE_K = int(os.getenv("CANDIDATE_K", "20"))  # fetched before re-ranking
RRF_K = 60  # reciprocal-rank-fusion constant


def scrub_server_api_keys() -> None:
    """Remove API keys from the process environment (Streamlit app only).

    ``load_dotenv()`` may have imported keys from a local ``.env``, and Streamlit
    Cloud Secrets may inject them as env vars. The chat UI must not use those —
    each user supplies keys in the sidebar for their session only.
    """
    global OPENAI_API_KEY, TAVILY_API_KEY
    os.environ.pop("OPENAI_API_KEY", None)
    os.environ.pop("TAVILY_API_KEY", None)
    OPENAI_API_KEY = ""
    TAVILY_API_KEY = ""


def load_cli_keys_from_env() -> None:
    """Load API keys from the environment for CLI tools only.

    Used by ingestion / eval scripts and Docker ``ingest``. Never call this from
    the Streamlit app — users must enter keys in the sidebar so they are not
    stored on the server.
    """
    global OPENAI_API_KEY, TAVILY_API_KEY
    openai = (os.getenv("OPENAI_API_KEY") or "").strip()
    tavily = (os.getenv("TAVILY_API_KEY") or "").strip()
    if openai and openai != "XXXXXX":
        OPENAI_API_KEY = openai
    if tavily and tavily != "XXXXXX":
        TAVILY_API_KEY = tavily


def has_openai() -> bool:
    return bool(OPENAI_API_KEY) and OPENAI_API_KEY != "XXXXXX"


def has_tavily() -> bool:
    return bool(TAVILY_API_KEY) and TAVILY_API_KEY != "XXXXXX"


def use_openai_embeddings() -> bool:
    return EMBEDDING_BACKEND == "openai" and has_openai()
