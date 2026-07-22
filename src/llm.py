"""Thin wrapper around the OpenAI Chat Completions API with JSON helpers."""
from __future__ import annotations

import json
from functools import lru_cache
from typing import Any, Optional

from . import config


@lru_cache(maxsize=1)
def _client():
    from openai import OpenAI

    return OpenAI(api_key=config.OPENAI_API_KEY)


def chat(
    messages: list[dict[str, str]],
    model: Optional[str] = None,
    temperature: float = 0.2,
    json_mode: bool = False,
) -> str:
    """Return the assistant message content as a string."""
    kwargs: dict[str, Any] = {
        "model": model or config.OPENAI_CHAT_MODEL,
        "messages": messages,
        "temperature": temperature,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    resp = _client().chat.completions.create(**kwargs)
    return resp.choices[0].message.content or ""


def chat_json(
    messages: list[dict[str, str]],
    model: Optional[str] = None,
    temperature: float = 0.0,
) -> dict[str, Any]:
    """Return parsed JSON, or {} on any failure."""
    try:
        raw = chat(messages, model=model, temperature=temperature, json_mode=True)
        return json.loads(raw)
    except Exception:
        return {}
