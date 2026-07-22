"""Router agent: decides between the knowledge base and live web search,
then synthesizes a grounded, cited answer.

Flow:
  1. Retrieve from the vector KB (hybrid search + rewrite + rerank).
  2. Route: decide whether live web search is also needed (real-time facts,
     or weak KB coverage).
  3. Optionally call the Tavily web-search tool.
  4. Build a grounded prompt and generate an English answer with [S#] citations.
"""
from __future__ import annotations

import re
import time
from functools import lru_cache
from typing import Any

from . import config
from .llm import chat, chat_json
from .retrieval import retrieve

# Heuristic triggers for real-time / out-of-KB information.
_WEB_TRIGGERS = re.compile(
    r"\b(today|tonight|tomorrow|now|current|currently|latest|this week|weekend|"
    r"price|prices|ticket|cost|fee|how much|weather|forecast|open now|hours today|"
    r"transport|how to get|getting there|flight|train|bus|hotel|book|booking|"
    r"exchange rate|visa)\b",
    re.IGNORECASE,
)


# --------------------------------------------------------------------------
# Web search tool (Tavily)
# --------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _tavily():
    from tavily import TavilyClient

    return TavilyClient(api_key=config.TAVILY_API_KEY)


def web_search(query: str, max_results: int = 4) -> list[dict[str, Any]]:
    """Query Tavily. Returns [] gracefully if no key or on error."""
    if not config.has_tavily():
        return []
    try:
        resp = _tavily().search(
            query=f"{query} Thailand tourism",
            max_results=max_results,
            search_depth="basic",
        )
        return [
            {
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "content": r.get("content", ""),
            }
            for r in resp.get("results", [])
        ]
    except Exception:
        return []


# --------------------------------------------------------------------------
# Routing
# --------------------------------------------------------------------------
def route(user_query: str, kb_hits: list[dict[str, Any]]) -> dict[str, Any]:
    """Decide whether to also use web search. Heuristic + optional LLM."""
    heuristic_web = bool(_WEB_TRIGGERS.search(user_query)) or len(kb_hits) == 0
    decision = {"use_web": heuristic_web, "reason": "heuristic"}

    if not config.has_openai():
        return decision

    out = chat_json(
        [
            {
                "role": "system",
                "content": (
                    "You route a Thailand-travel query. The knowledge base has "
                    "static attractions and scheduled events (names, locations, "
                    "descriptions, dates) but NOT live data like current ticket "
                    "prices, weather, or transport schedules. Given the query and "
                    "whether KB results were found, decide if a live web search is "
                    "needed. Respond ONLY as JSON: {\"use_web\": bool, \"reason\": str}."
                ),
            },
            {
                "role": "user",
                "content": f"Query: {user_query}\nKB results found: {len(kb_hits)}",
            },
        ]
    )
    if isinstance(out, dict) and "use_web" in out:
        decision = {"use_web": bool(out["use_web"]), "reason": out.get("reason", "llm")}
    return decision


# --------------------------------------------------------------------------
# Context building + answer synthesis
# --------------------------------------------------------------------------
def _format_kb_source(idx: int, hit: dict[str, Any]) -> str:
    m = hit.get("metadata", {})
    bits = [f"[S{idx}] ({m.get('source_type', 'kb')})"]
    name = m.get("name_en") or m.get("name") or m.get("name_th", "")
    if name:
        bits.append(f"Name: {name}")
    if m.get("province"):
        bits.append(f"Province: {m['province']}")
    if m.get("start_date"):
        bits.append(f"Dates: {m.get('start_date')} to {m.get('end_date', '')}")
    if m.get("opening_hours"):
        bits.append(f"Hours: {m['opening_hours']}")
    if m.get("website"):
        bits.append(f"Website: {m['website']}")
    if m.get("phone") or m.get("tel"):
        bits.append(f"Tel: {m.get('phone') or m.get('tel')}")
    header = " | ".join(bits)
    return f"{header}\nContent: {hit.get('document', '')[:900]}"


def _format_web_source(idx: int, res: dict[str, Any]) -> str:
    return f"[W{idx}] {res.get('title', '')} ({res.get('url', '')})\n{res.get('content', '')[:600]}"


_ANSWER_SYS = (
    "You are PaiNaiDee, a friendly Thailand travel assistant. Answer in English. "
    "Use ONLY the provided context (knowledge base [S#] and web [W#]). Cite every "
    "claim with its tag, e.g. [S1] or [W2]. If the context lacks specific details "
    "(exact price, current hours, availability), say so honestly instead of "
    "inventing them. Recommend concrete places/events and briefly explain why. "
    "Be concise and practical."
)


def _fallback_answer(kb_hits: list[dict[str, Any]]) -> str:
    """Deterministic answer when no OpenAI key is available (offline mode)."""
    if not kb_hits:
        return "No matching places or events were found in the knowledge base."
    lines = ["Here are the closest matches from the knowledge base:"]
    for i, hit in enumerate(kb_hits, 1):
        m = hit.get("metadata", {})
        name = m.get("name_en") or m.get("name") or m.get("name_th", "")
        prov = m.get("province", "")
        lines.append(f"{i}. {name} — {prov} [S{i}]")
    return "\n".join(lines)


def answer(user_query: str, top_k: int = config.DEFAULT_TOP_K) -> dict[str, Any]:
    """Full agent flow. Returns answer text, sources, route info, and timings."""
    t0 = time.time()

    retrieval = retrieve(user_query, top_k=top_k)
    kb_hits = retrieval["results"]

    routing = route(user_query, kb_hits)
    web_results = web_search(user_query) if routing["use_web"] else []

    # Build the grounded context.
    context_parts = [_format_kb_source(i + 1, h) for i, h in enumerate(kb_hits)]
    context_parts += [_format_web_source(i + 1, r) for i, r in enumerate(web_results)]
    context = "\n\n".join(context_parts) if context_parts else "(no context found)"

    if config.has_openai():
        answer_text = chat(
            [
                {"role": "system", "content": _ANSWER_SYS},
                {
                    "role": "user",
                    "content": f"User question: {user_query}\n\nContext:\n{context}",
                },
            ],
            temperature=0.3,
        )
    else:
        answer_text = _fallback_answer(kb_hits)

    # Structured sources for the UI.
    sources = []
    for i, h in enumerate(kb_hits, 1):
        m = h.get("metadata", {})
        sources.append(
            {
                "tag": f"S{i}",
                "kind": m.get("source_type", "kb"),
                "name": m.get("name_en") or m.get("name") or m.get("name_th", ""),
                "province": m.get("province", ""),
                "url": m.get("website") or m.get("facebook", ""),
            }
        )
    for i, r in enumerate(web_results, 1):
        sources.append(
            {"tag": f"W{i}", "kind": "web", "name": r.get("title", ""),
             "province": "", "url": r.get("url", "")}
        )

    return {
        "query": user_query,
        "answer": answer_text,
        "sources": sources,
        "route": routing,
        "used_web": bool(web_results),
        "rewritten": retrieval.get("rewritten", {}),
        "provinces": sorted(
            {h.get("metadata", {}).get("province", "") for h in kb_hits} - {""}
        ),
        "latency_ms": int((time.time() - t0) * 1000),
    }
