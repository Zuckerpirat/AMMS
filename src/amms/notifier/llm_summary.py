"""Optional LLM-augmented end-of-day summary.

If ``ANTHROPIC_API_KEY`` is set, call Claude to add a one-paragraph
explanation to the plain summary text. Otherwise return the plain summary
unchanged.

Claude is used as a *narrator*, not a decision-maker. The bot's trades are
already decided by the strategy + risk + filter pipeline; the LLM only
explains what happened today. No retries; if the API errors we just return
the plain text. Caches the per-day prompt+response in the DB so we don't
double-bill ourselves.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from datetime import UTC, datetime

import httpx

logger = logging.getLogger(__name__)


def _ensure_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS llm_summaries (
            date TEXT PRIMARY KEY,
            input_hash TEXT NOT NULL,
            output TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )


def _cached(conn: sqlite3.Connection, date_iso: str, input_hash: str) -> str | None:
    _ensure_table(conn)
    row = conn.execute(
        "SELECT output, input_hash FROM llm_summaries WHERE date = ?",
        (date_iso,),
    ).fetchone()
    if not row:
        return None
    if row["input_hash"] != input_hash:
        return None
    return row["output"]


def _store(
    conn: sqlite3.Connection, date_iso: str, input_hash: str, output: str
) -> None:
    _ensure_table(conn)
    conn.execute(
        """
        INSERT INTO llm_summaries(date, input_hash, output, created_at)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(date) DO UPDATE SET
            input_hash = excluded.input_hash,
            output = excluded.output,
            created_at = excluded.created_at
        """,
        (date_iso, input_hash, output, datetime.now(UTC).isoformat()),
    )


def _hash_inputs(plain_summary: str, trades_today: list[dict]) -> str:
    import hashlib

    payload = json.dumps({"s": plain_summary, "t": trades_today}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def augment_summary(
    plain_summary: str,
    *,
    trades_today: list[dict],
    conn: sqlite3.Connection,
    model: str = "claude-haiku-4-5-20251001",
    timeout: float = 20.0,
) -> str:
    """Return ``plain_summary`` with an LLM-generated paragraph appended,
    or the plain text unchanged when no API key is configured or the call
    fails. Cached per UTC date in the ``llm_summaries`` table.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        return plain_summary

    date_iso = datetime.now(UTC).date().isoformat()
    digest = _hash_inputs(plain_summary, trades_today)
    cached = _cached(conn, date_iso, digest)
    if cached is not None:
        return cached

    prompt = (
        "You narrate one paragraph (3-4 sentences) summarizing a paper-trading "
        "bot's day. Be specific about trades and equity. Do not make trading "
        "recommendations. Do not invent numbers; only use the data given.\n\n"
        f"Plain summary:\n{plain_summary}\n\n"
        f"Trades today (JSON):\n{json.dumps(trades_today, indent=2)}"
    )

    try:
        resp = httpx.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": model,
                "max_tokens": 400,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        narration = "".join(
            block.get("text", "")
            for block in data.get("content", [])
            if block.get("type") == "text"
        ).strip()
    except Exception:
        logger.warning("LLM summary failed; falling back to plain", exc_info=True)
        return plain_summary

    if not narration:
        return plain_summary

    out = f"{plain_summary}\n\n{narration}"
    try:
        _store(conn, date_iso, digest, out)
    except Exception:
        logger.warning("LLM summary cache write failed", exc_info=True)
    return out
