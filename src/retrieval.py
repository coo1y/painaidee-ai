"""Retrieval layer: hybrid search + query rewriting + metadata filtering + re-ranking.

Public entry point is :func:`retrieve`. The individual pieces
(:func:`dense_search`, :func:`hybrid_search`, :func:`rerank`) are exposed so the
evaluation scripts can compare approaches head-to-head.
"""
from __future__ import annotations

import re
from functools import lru_cache
from typing import Any, Optional

import chromadb
from chromadb.config import Settings
from rank_bm25 import BM25Okapi

from . import config
from .embeddings import embed_query
from .llm import chat_json
from .utils import to_epoch_day

_TOKEN_RE = re.compile(r"[0-9A-Za-z\u0E00-\u0E7F]+")


# --------------------------------------------------------------------------
# Chroma access
# --------------------------------------------------------------------------
@lru_cache(maxsize=1)
def get_client():
    return chromadb.PersistentClient(
        path=str(config.CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False),
    )


@lru_cache(maxsize=4)
def get_collection(name: str):
    return get_client().get_collection(name)


@lru_cache(maxsize=4)
def _all_docs(name: str) -> tuple[list[str], list[str], tuple[dict, ...]]:
    """Return (ids, documents, metadatas) for the whole collection (cached)."""
    coll = get_collection(name)
    data = coll.get(include=["documents", "metadatas"])
    return data["ids"], data["documents"], tuple(data["metadatas"])


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall((text or "").lower())


@lru_cache(maxsize=4)
def _bm25_index(name: str):
    """Build and cache the BM25 index for a collection (expensive at scale)."""
    ids, docs, metas = _all_docs(name)
    corpus = [_tokenize(d) for d in docs]
    bm25 = BM25Okapi(corpus) if corpus else None
    return ids, docs, metas, bm25


def clear_caches() -> None:
    _all_docs.cache_clear()
    get_collection.cache_clear()
    _bm25_index.cache_clear()


# --------------------------------------------------------------------------
# Query rewriting
# --------------------------------------------------------------------------
_REWRITE_SYS = (
    "You rewrite a user's Thailand-travel question into an optimized search "
    "query and extract structured filters. The knowledge base stores Thai place "
    "names, so translate any location to its THAI province name. "
    "Respond ONLY as JSON with keys: "
    "search_query (string, keywords for retrieval, may mix EN+TH), "
    "province_th (Thai province name or null), "
    "date_start (YYYY-MM-DD or null), date_end (YYYY-MM-DD or null), "
    "source ('attractions' | 'events' | 'both')."
)


def rewrite_query(user_query: str) -> dict[str, Any]:
    """Rewrite the query and extract filters. Falls back to identity offline."""
    default = {
        "search_query": user_query,
        "province_th": None,
        "date_start": None,
        "date_end": None,
        "source": "both",
    }
    if not config.has_openai():
        return default
    out = chat_json(
        [
            {"role": "system", "content": _REWRITE_SYS},
            {"role": "user", "content": user_query},
        ]
    )
    if not out:
        return default
    default.update({k: out.get(k, default[k]) for k in default})
    if not default["search_query"]:
        default["search_query"] = user_query
    if default["source"] not in ("attractions", "events", "both"):
        default["source"] = "both"
    return default


# --------------------------------------------------------------------------
# Filters
# --------------------------------------------------------------------------
def build_where(
    source_type: str,
    province_th: Optional[str] = None,
    date_start: Optional[str] = None,
    date_end: Optional[str] = None,
) -> Optional[dict]:
    """Build a Chroma ``where`` clause for metadata pre-filtering."""
    clauses: list[dict] = []
    if province_th:
        clauses.append({"province": province_th})
    if source_type == "event":
        # Event overlaps the requested window if start <= win_end and end >= win_start.
        win_start = to_epoch_day(date_start)
        win_end = to_epoch_day(date_end)
        if win_end is not None:
            clauses.append({"start_day": {"$lte": win_end}})
        if win_start is not None:
            clauses.append({"end_day": {"$gte": win_start}})
    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


# --------------------------------------------------------------------------
# Search primitives
# --------------------------------------------------------------------------
def dense_search(
    collection_name: str,
    query: str,
    k: int = config.CANDIDATE_K,
    where: Optional[dict] = None,
) -> list[dict[str, Any]]:
    """Dense vector search. Returns ranked hits with a normalized score."""
    coll = get_collection(collection_name)
    n = min(k, max(coll.count(), 1))
    res = coll.query(
        query_embeddings=[embed_query(query)],
        n_results=n,
        where=where,
        include=["documents", "metadatas", "distances"],
    )
    hits = []
    ids = res["ids"][0] if res["ids"] else []
    for i, _id in enumerate(ids):
        dist = res["distances"][0][i]
        hits.append(
            {
                "id": _id,
                "document": res["documents"][0][i],
                "metadata": res["metadatas"][0][i],
                "dense_score": 1.0 / (1.0 + dist),
                "rank_dense": i + 1,
            }
        )
    return hits


def bm25_search(
    collection_name: str,
    query: str,
    k: int = config.CANDIDATE_K,
    where_ids: Optional[set[str]] = None,
) -> list[dict[str, Any]]:
    """BM25 lexical search over the collection documents."""
    ids, docs, metas, bm25 = _bm25_index(collection_name)
    if not docs or bm25 is None:
        return []
    scores = bm25.get_scores(_tokenize(query))
    order = sorted(range(len(docs)), key=lambda i: scores[i], reverse=True)
    hits = []
    rank = 0
    for i in order:
        if where_ids is not None and ids[i] not in where_ids:
            continue
        if scores[i] <= 0:
            continue
        rank += 1
        hits.append(
            {
                "id": ids[i],
                "document": docs[i],
                "metadata": metas[i],
                "bm25_score": float(scores[i]),
                "rank_bm25": rank,
            }
        )
        if rank >= k:
            break
    return hits


def _rrf_fuse(
    dense: list[dict], sparse: list[dict], k: int
) -> list[dict[str, Any]]:
    """Reciprocal Rank Fusion of two ranked lists."""
    merged: dict[str, dict[str, Any]] = {}
    for hit in dense:
        merged.setdefault(hit["id"], dict(hit))
        merged[hit["id"]]["fused"] = merged[hit["id"]].get("fused", 0.0) + 1.0 / (
            config.RRF_K + hit["rank_dense"]
        )
    for hit in sparse:
        entry = merged.setdefault(hit["id"], dict(hit))
        entry.update({kk: vv for kk, vv in hit.items() if kk not in entry})
        entry["fused"] = entry.get("fused", 0.0) + 1.0 / (
            config.RRF_K + hit["rank_bm25"]
        )
    ordered = sorted(merged.values(), key=lambda h: h["fused"], reverse=True)
    return ordered[:k]


def hybrid_search(
    collection_name: str,
    query: str,
    k: int = config.CANDIDATE_K,
    where: Optional[dict] = None,
) -> list[dict[str, Any]]:
    """Dense + BM25 combined with Reciprocal Rank Fusion."""
    dense = dense_search(collection_name, query, k=k, where=where)
    allowed = {h["id"] for h in dense} if where else None
    # When a where-filter is active, restrict BM25 to the filtered id set.
    sparse = bm25_search(collection_name, query, k=k, where_ids=allowed)
    return _rrf_fuse(dense, sparse, k)


# --------------------------------------------------------------------------
# Re-ranking (LLM cross-scoring)
# --------------------------------------------------------------------------
_RERANK_SYS = (
    "Score how relevant each candidate document is to the user query on a scale "
    "of 0-10 (10 = perfect). Respond ONLY as JSON: {\"scores\": [ints]} with one "
    "score per candidate, in the same order."
)


def rerank(
    query: str, candidates: list[dict[str, Any]], top_k: int
) -> list[dict[str, Any]]:
    """LLM-based re-ranking. Falls back to fusion order when offline."""
    if not candidates:
        return []
    if not config.has_openai() or len(candidates) == 1:
        return candidates[:top_k]

    listing = "\n\n".join(
        f"[{i}] {c['document'][:800]}" for i, c in enumerate(candidates)
    )
    out = chat_json(
        [
            {"role": "system", "content": _RERANK_SYS},
            {"role": "user", "content": f"Query: {query}\n\nCandidates:\n{listing}"},
        ]
    )
    scores = out.get("scores") if isinstance(out, dict) else None
    if not isinstance(scores, list) or len(scores) != len(candidates):
        return candidates[:top_k]
    for c, s in zip(candidates, scores):
        try:
            c["rerank_score"] = float(s)
        except (TypeError, ValueError):
            c["rerank_score"] = 0.0
    return sorted(candidates, key=lambda h: h["rerank_score"], reverse=True)[:top_k]


# --------------------------------------------------------------------------
# Orchestrated retrieval
# --------------------------------------------------------------------------
def _search_source(
    source_type: str,
    collection_name: str,
    rewritten: dict[str, Any],
    candidate_k: int,
) -> list[dict[str, Any]]:
    where = build_where(
        source_type,
        province_th=rewritten.get("province_th"),
        date_start=rewritten.get("date_start"),
        date_end=rewritten.get("date_end"),
    )
    hits = hybrid_search(collection_name, rewritten["search_query"], k=candidate_k, where=where)
    # Graceful fallback: if a strict filter removed everything, retry unfiltered.
    if not hits and where is not None:
        hits = hybrid_search(collection_name, rewritten["search_query"], k=candidate_k)
    return hits


def retrieve(
    user_query: str,
    top_k: int = config.DEFAULT_TOP_K,
    candidate_k: int = config.CANDIDATE_K,
    do_rewrite: bool = True,
    do_rerank: bool = True,
) -> dict[str, Any]:
    """Full retrieval flow. Returns hits plus debug metadata for observability."""
    rewritten = rewrite_query(user_query) if do_rewrite else {
        "search_query": user_query,
        "province_th": None,
        "date_start": None,
        "date_end": None,
        "source": "both",
    }

    source = rewritten.get("source", "both")
    candidates: list[dict[str, Any]] = []
    if source in ("attractions", "both"):
        candidates += _search_source(
            "attraction", config.ATTRACTIONS_COLLECTION, rewritten, candidate_k
        )
    if source in ("events", "both"):
        candidates += _search_source(
            "event", config.EVENTS_COLLECTION, rewritten, candidate_k
        )

    # De-duplicate by id, keep best fused score.
    dedup: dict[str, dict[str, Any]] = {}
    for c in candidates:
        prev = dedup.get(c["id"])
        if prev is None or c.get("fused", 0) > prev.get("fused", 0):
            dedup[c["id"]] = c
    candidates = list(dedup.values())

    ranked = rerank(user_query, candidates, top_k) if do_rerank else sorted(
        candidates, key=lambda h: h.get("fused", 0), reverse=True
    )[:top_k]

    return {
        "query": user_query,
        "rewritten": rewritten,
        "results": ranked,
        "n_candidates": len(candidates),
    }
