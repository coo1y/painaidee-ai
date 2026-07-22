"""Embedding backends.

Default backend is OpenAI (``text-embedding-3-small``). A deterministic
hash-based ``local`` backend is provided ONLY for offline smoke tests (no
network / no API key). Local embeddings carry no semantic meaning; in that
mode retrieval quality comes almost entirely from the BM25 lexical channel.
"""
from __future__ import annotations

import hashlib
from functools import lru_cache
from typing import List

from . import config


# --------------------------------------------------------------------------
# Local deterministic embeddings (offline test only)
# --------------------------------------------------------------------------
def _local_embed(text: str, dim: int = config.LOCAL_EMBEDDING_DIM) -> List[float]:
    vec = [0.0] * dim
    for token in text.lower().split():
        h = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
        vec[h % dim] += 1.0
    norm = sum(v * v for v in vec) ** 0.5
    if norm > 0:
        vec = [v / norm for v in vec]
    return vec


# --------------------------------------------------------------------------
# OpenAI embeddings
# --------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _openai_client():
    from openai import OpenAI

    return OpenAI(api_key=config.OPENAI_API_KEY)


def _openai_embed(texts: List[str]) -> List[List[float]]:
    client = _openai_client()
    resp = client.embeddings.create(
        model=config.OPENAI_EMBEDDING_MODEL,
        input=texts,
    )
    return [d.embedding for d in resp.data]


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------
def embed_texts(texts: List[str]) -> List[List[float]]:
    """Embed a batch of documents."""
    if not texts:
        return []
    if config.use_openai_embeddings():
        # Batch to stay well under token limits.
        out: List[List[float]] = []
        for i in range(0, len(texts), 128):
            out.extend(_openai_embed(texts[i : i + 128]))
        return out
    return [_local_embed(t) for t in texts]


def embed_query(text: str) -> List[float]:
    """Embed a single query string."""
    if config.use_openai_embeddings():
        return _openai_embed([text])[0]
    return _local_embed(text)


def backend_name() -> str:
    return "openai" if config.use_openai_embeddings() else "local(test)"
