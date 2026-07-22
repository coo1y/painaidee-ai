"""PaiNaiDee-AI — 🧭 Chat page (Streamlit multipage app).

An agentic RAG assistant for planning trips in Thailand. Answers are grounded in
a TAT knowledge base (attractions + events) with optional live web search, and
every response can be rated 👍/👎 with an optional comment (stored in SQLite).

Called via ``chat.render()`` from ``streamlit_app.py``.
"""
from __future__ import annotations

import importlib
import sys
import uuid
from pathlib import Path

# Make the project root importable so ``from src import ...`` works on Cloud.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st

from src import config
from src.agent import answer as agent_answer
from src.db import init_db, log_interaction, update_feedback


# --------------------------------------------------------------------------
# One-time setup: ensure the knowledge base exists (auto-ingest on cold start).
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Preparing knowledge base…")
def bootstrap() -> dict:
    init_db()
    import chromadb
    from chromadb.config import Settings

    client = chromadb.PersistentClient(
        path=str(config.CHROMA_DIR), settings=Settings(anonymized_telemetry=False)
    )
    existing = {c.name for c in client.list_collections()}
    needed = {config.ATTRACTIONS_COLLECTION, config.EVENTS_COLLECTION}
    counts = {}
    if not needed.issubset(existing):
        # Plain (Prefect-free) ingestion keeps cold-start fast on the web app.
        from ingestion.ingest_pipeline import run_ingestion

        counts = run_ingestion(reset=True)
    else:
        for name in needed:
            counts[name] = client.get_collection(name).count()
    return counts


def apply_api_keys() -> None:
    """Apply user-entered API keys (from the sidebar) for this session.

    Keys typed into the sidebar are stored only in ``st.session_state`` (never on
    disk). We push them into ``config`` and reset the cached OpenAI/Tavily
    clients so the new keys take effect immediately.
    """
    openai_key = (st.session_state.get("openai_api_key") or "").strip()
    tavily_key = (st.session_state.get("tavily_api_key") or "").strip()

    # The OpenAI/Tavily clients read these config globals and pass api_key
    # explicitly, so we only need to update config (no os.environ needed).
    if openai_key:
        config.OPENAI_API_KEY = openai_key
    if tavily_key:
        config.TAVILY_API_KEY = tavily_key

    # The API clients are @lru_cache'd against the previous key — clear them so
    # subsequent calls rebuild with the newly supplied key.
    for module_name, factory in (
        ("src.llm", "_client"),
        ("src.embeddings", "_openai_client"),
        ("src.agent", "_tavily"),
    ):
        try:
            getattr(importlib.import_module(module_name), factory).cache_clear()
        except Exception:  # pragma: no cover - defensive; never block the UI
            pass


def save_rating(msg_idx: int) -> None:
    """Callback: persist the 👍/👎 selection for a message."""
    key = f"fb_{msg_idx}"
    value = st.session_state.get(key)
    interaction_id = st.session_state.messages[msg_idx].get("interaction_id")
    if value is not None and interaction_id:
        update_feedback(interaction_id, rating=int(value))
        st.session_state.messages[msg_idx]["rating"] = int(value)


def save_comment(msg_idx: int) -> None:
    key = f"cm_{msg_idx}"
    comment = st.session_state.get(key, "").strip()
    interaction_id = st.session_state.messages[msg_idx].get("interaction_id")
    if comment and interaction_id:
        update_feedback(interaction_id, comment=comment)
        st.session_state.messages[msg_idx]["comment_saved"] = True


def render() -> None:
    """Render the chat page (called by streamlit_app.py)."""
    # Apply any keys the user already entered this session BEFORE bootstrapping.
    apply_api_keys()

    counts = bootstrap()

    with st.sidebar:
        st.header("🗺️ PaiNaiDee AI")
        st.caption("Agentic AI travel assistant for Thailand")

        st.subheader("🔑 API keys")
        st.caption(
            "Add your own keys to use the assistant. They stay in your browser "
            "session only — never saved to disk or shared."
        )
        st.text_input(
            "OpenAI API key",
            key="openai_api_key",
            type="password",
            placeholder="sk-…",
            on_change=apply_api_keys,
            help="Required for full-quality answers and semantic (vector) search. "
            "Get one at https://platform.openai.com/api-keys",
        )
        st.text_input(
            "Tavily API key (optional)",
            key="tavily_api_key",
            type="password",
            placeholder="tvly-…",
            on_change=apply_api_keys,
            help="Enables live web search for real-time info (prices, hours, weather). "
            "Get one at https://app.tavily.com",
        )

    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Hi! I can recommend attractions and events across Thailand. "
                "Try: *“cultural riverside towns in eastern Thailand”* or "
                "*“festivals in Trat around New Year”*.",
            }
        ]

    st.title("🗺️ PaiNaiDee AI")
    st.caption("Plan your trip in Thailand — grounded in TAT attractions & events data.")

    for idx, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

            if msg.get("sources"):
                with st.expander("📚 Sources"):
                    for s in msg["sources"]:
                        label = f"**[{s['tag']}]** {s['name']}"
                        if s.get("province"):
                            label += f" — {s['province']}"
                        if s.get("url"):
                            label += f" · [link]({s['url']})"
                        st.markdown(label)

            if msg.get("route"):
                r = msg["route"]
                st.caption(
                    f"Route: {'KB + Web' if msg.get('used_web') else 'Knowledge base'} · "
                    f"{r.get('reason', '')}"
                )

            if msg["role"] == "assistant" and msg.get("interaction_id"):
                st.feedback("thumbs", key=f"fb_{idx}", on_change=save_rating, args=(idx,))
                with st.expander("💬 Tell us why (optional)"):
                    st.text_input(
                        "Why did you like / dislike this answer?",
                        key=f"cm_{idx}",
                        on_change=save_comment,
                        args=(idx,),
                        label_visibility="collapsed",
                        placeholder="Your comment…",
                    )
                    if msg.get("comment_saved"):
                        st.success("Thanks for the feedback!")

    if prompt := st.chat_input("Ask about places, events, or trip ideas in Thailand…"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                result = agent_answer(prompt)
            st.markdown(result["answer"])

        interaction_id = log_interaction(
            query=prompt,
            response=result["answer"],
            session_id=st.session_state.session_id,
            used_web=result["used_web"],
            provinces=result["provinces"],
            latency_ms=result["latency_ms"],
        )
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": result["answer"],
                "sources": result["sources"],
                "route": result["route"],
                "used_web": result["used_web"],
                "interaction_id": interaction_id,
            }
        )
        st.rerun()
