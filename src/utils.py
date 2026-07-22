"""Text cleaning and record-normalization helpers.

The raw TAT exports are wrapped as ``{"<SQL query>": [ ...records ]}`` and the
narrative fields contain HTML. These helpers unwrap, clean, and normalize the
records into flat dicts that the ingestion pipeline can embed and filter on.
"""
from __future__ import annotations

import html
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from bs4 import BeautifulSoup

_WS_RE = re.compile(r"\s+")


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def load_tat_export(path: str | Path) -> list[dict[str, Any]]:
    """Load a TAT JSON export and return the list of records.

    The file is a single-key object whose key is the SQL query and whose value
    is the list of rows. We simply return that first (and only) value.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for value in payload.values():
            if isinstance(value, list):
                return value
    raise ValueError(f"Unexpected JSON structure in {path}")


# --------------------------------------------------------------------------
# Cleaning
# --------------------------------------------------------------------------
def clean_html(raw: Optional[str]) -> str:
    """Strip HTML tags/entities and collapse whitespace. Safe on None."""
    if not raw:
        return ""
    text = BeautifulSoup(raw, "html.parser").get_text(separator=" ")
    text = html.unescape(text)
    text = text.replace("\xa0", " ").replace("\r", " ").replace("\n", " ")
    return _WS_RE.sub(" ", text).strip()


def clean_text(raw: Optional[str]) -> str:
    """Trim and collapse whitespace on a plain string. Safe on None."""
    if not raw:
        return ""
    return _WS_RE.sub(" ", str(raw)).strip()


def clean_listagg(raw: Optional[str]) -> list[str]:
    """Split an Oracle LISTAGG string like ``"A ,B ,"`` into a clean list."""
    if not raw:
        return []
    parts = [p.strip() for p in str(raw).split(",")]
    return [p for p in parts if p]


def parse_latlng(raw: Optional[str]) -> tuple[Optional[float], Optional[float]]:
    """Parse a ``"lat, lng"`` string into floats. Returns (None, None) on failure."""
    if not raw:
        return None, None
    try:
        lat_s, lng_s = str(raw).split(",")[:2]
        return float(lat_s.strip()), float(lng_s.strip())
    except (ValueError, IndexError):
        return None, None


def parse_date(raw: Optional[str]) -> Optional[str]:
    """Normalize a date to ISO ``YYYY-MM-DD`` (input is already that format)."""
    if not raw:
        return None
    raw = str(raw).strip()
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(raw, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None


def to_epoch_day(date_iso: Optional[str]) -> Optional[int]:
    """Convert ``YYYY-MM-DD`` to an integer day count (for range filtering)."""
    if not date_iso:
        return None
    try:
        dt = datetime.strptime(date_iso, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        return int(dt.timestamp() // 86400)
    except ValueError:
        return None


def first_nonempty(*values: Optional[str]) -> str:
    for v in values:
        c = clean_text(v)
        if c:
            return c
    return ""
