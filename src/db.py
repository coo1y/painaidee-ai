"""SQLite store for interaction logging and user feedback.

A single ``interactions`` table records every query/response, plus optional
👍/👎 rating and a free-text comment. The dashboard reads from this table.
"""
from __future__ import annotations

import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

import pandas as pd

from . import config

_SCHEMA = """
CREATE TABLE IF NOT EXISTS interactions (
    id           TEXT PRIMARY KEY,
    ts           TEXT NOT NULL,
    session_id   TEXT,
    query        TEXT NOT NULL,
    response     TEXT,
    route        TEXT,          -- 'kb' or 'kb+web'
    used_web     INTEGER,       -- 0/1
    provinces    TEXT,          -- comma-separated
    latency_ms   INTEGER,
    rating       INTEGER,       -- 1 = 👍, 0 = 👎, NULL = none
    comment      TEXT           -- qualitative feedback
);
"""


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(config.FEEDBACK_DB))
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _connect() as conn:
        conn.execute(_SCHEMA)
        conn.commit()


def log_interaction(
    query: str,
    response: str,
    session_id: Optional[str] = None,
    used_web: bool = False,
    provinces: Optional[list[str]] = None,
    latency_ms: int = 0,
) -> str:
    """Insert an interaction and return its id (used later to attach feedback)."""
    init_db()
    interaction_id = str(uuid.uuid4())
    with _connect() as conn:
        conn.execute(
            """INSERT INTO interactions
               (id, ts, session_id, query, response, route, used_web,
                provinces, latency_ms, rating, comment)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL)""",
            (
                interaction_id,
                datetime.now(timezone.utc).isoformat(),
                session_id,
                query,
                response,
                "kb+web" if used_web else "kb",
                int(used_web),
                ",".join(provinces or []),
                latency_ms,
            ),
        )
        conn.commit()
    return interaction_id


def update_feedback(
    interaction_id: str, rating: Optional[int] = None, comment: Optional[str] = None
) -> None:
    """Attach a 👍/👎 rating and/or a text comment to an interaction."""
    init_db()
    sets, params = [], []
    if rating is not None:
        sets.append("rating = ?")
        params.append(int(rating))
    if comment is not None:
        sets.append("comment = ?")
        params.append(comment)
    if not sets:
        return
    params.append(interaction_id)
    with _connect() as conn:
        conn.execute(
            f"UPDATE interactions SET {', '.join(sets)} WHERE id = ?", params
        )
        conn.commit()


def fetch_df() -> pd.DataFrame:
    """Return all interactions as a DataFrame (empty with correct columns if none)."""
    init_db()
    with _connect() as conn:
        df = pd.read_sql_query("SELECT * FROM interactions ORDER BY ts DESC", conn)
    if not df.empty:
        df["ts"] = pd.to_datetime(df["ts"], errors="coerce", utc=True)
    return df


def stats() -> dict[str, Any]:
    df = fetch_df()
    rated = df[df["rating"].notna()] if not df.empty else df
    return {
        "total": int(len(df)),
        "rated": int(len(rated)),
        "positive": int((rated["rating"] == 1).sum()) if not rated.empty else 0,
        "negative": int((rated["rating"] == 0).sum()) if not rated.empty else 0,
    }
