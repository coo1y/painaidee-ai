"""PaiNaiDee-AI — Streamlit chat UI.

An agentic RAG assistant for planning trips in Thailand. Answers are grounded in
a TAT knowledge base (attractions + events) with optional live web search, and
every response can be rated 👍/👎 with an optional comment (stored in SQLite).
"""
from __future__ import annotations

import uuid

import streamlit as st

from src import config
from src.agent import answer as agent_answer
from src.db import init_db, log_interaction, update_feedback

st.set_page_config(page_title="PaiNaiDee-AI — Thailand Travel", page_icon="🧭", layout="centered")


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


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
counts = bootstrap()

with st.sidebar:
    st.header("🧭 PaiNaiDee-AI")
    st.caption("Agentic RAG travel assistant for Thailand")
    st.subheader("System status")
    st.write(f"**Attractions indexed:** {counts.get(config.ATTRACTIONS_COLLECTION, 0)}")
    st.write(f"**Events indexed:** {counts.get(config.EVENTS_COLLECTION, 0)}")
    st.write(f"**OpenAI:** {'✅' if config.has_openai() else '❌ (offline mode)'}")
    st.write(f"**Web search (Tavily):** {'✅' if config.has_tavily() else '❌ (disabled)'}")
    if not config.has_openai():
        st.warning(
            "No OPENAI_API_KEY detected — running with local test embeddings and a "
            "template answer. Set your key in `.env` for full quality."
        )
    st.divider()
    st.caption("Run the monitoring dashboard with:\n\n`streamlit run dashboard.py`")


# --------------------------------------------------------------------------
# Chat state
# --------------------------------------------------------------------------
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

st.title("🧭 PaiNaiDee-AI")
st.caption("Plan your trip in Thailand — grounded in TAT attractions & events data.")


# --------------------------------------------------------------------------
# Render history
# --------------------------------------------------------------------------
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

        # Feedback widgets (assistant messages that were generated for a query).
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


# --------------------------------------------------------------------------
# Handle new input
# --------------------------------------------------------------------------
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
