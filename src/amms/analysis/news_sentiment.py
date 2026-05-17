"""News sentiment analysis via Claude.

Fetches recent Alpaca news articles for a symbol and asks Claude Haiku
to produce a structured sentiment assessment: score, conclusion, and a
brief reasoning chain.

Output:
  NewsSentimentResult(
      symbol        : str,
      score         : float,   # –1.0 (very bearish) … +1.0 (very bullish)
      confidence    : float,   # 0.0 … 1.0
      conclusion    : str,     # one-line summary, e.g. "Mixed: tariff risk offset by strong earnings"
      reasoning     : list[str],
      article_count : int,
      is_bullish    : bool,
      cached        : bool,
  )

Caching:
  Results are cached in SQLite table ``news_sentiment_cache`` per
  (symbol, date). Re-uses the cached result within the same UTC day
  so we don't double-bill the API.

Fallback:
  If ANTHROPIC_API_KEY is not set, returns a neutral result with score=0
  and a note that LLM analysis is unavailable. The bot can still run.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime

import httpx

logger = logging.getLogger(__name__)

_CLAUDE_URL = "https://api.anthropic.com/v1/messages"
_MODEL = "claude-haiku-4-5-20251001"
_MAX_TOKENS = 512
_TIMEOUT = 20.0


@dataclass
class NewsSentimentResult:
    symbol: str
    score: float          # −1.0 … +1.0
    confidence: float     # 0.0 … 1.0
    conclusion: str
    reasoning: list[str]
    article_count: int
    is_bullish: bool
    cached: bool = False


def _ensure_table(conn) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS news_sentiment_cache (
            symbol      TEXT NOT NULL,
            date        TEXT NOT NULL,
            input_hash  TEXT NOT NULL,
            result_json TEXT NOT NULL,
            created_at  TEXT NOT NULL,
            PRIMARY KEY (symbol, date)
        )
        """
    )
    conn.commit()


def _input_hash(articles: list[dict]) -> str:
    payload = json.dumps(
        [{"h": a.get("headline", ""), "s": a.get("summary", "")} for a in articles],
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _load_cache(conn, symbol: str, date_iso: str, input_hash: str) -> NewsSentimentResult | None:
    try:
        _ensure_table(conn)
        row = conn.execute(
            "SELECT result_json, input_hash FROM news_sentiment_cache WHERE symbol=? AND date=?",
            (symbol.upper(), date_iso),
        ).fetchone()
        if row and row["input_hash"] == input_hash:
            data = json.loads(row["result_json"])
            data["cached"] = True
            return NewsSentimentResult(**data)
    except Exception as exc:
        logger.debug("News cache read failed: %s", exc)
    return None


def _save_cache(conn, symbol: str, date_iso: str, input_hash: str,
                result: NewsSentimentResult) -> None:
    try:
        _ensure_table(conn)
        payload = json.dumps({
            "symbol": result.symbol,
            "score": result.score,
            "confidence": result.confidence,
            "conclusion": result.conclusion,
            "reasoning": result.reasoning,
            "article_count": result.article_count,
            "is_bullish": result.is_bullish,
            "cached": False,
        })
        conn.execute(
            """
            INSERT INTO news_sentiment_cache(symbol, date, input_hash, result_json, created_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(symbol, date) DO UPDATE SET
                input_hash  = excluded.input_hash,
                result_json = excluded.result_json,
                created_at  = excluded.created_at
            """,
            (symbol.upper(), date_iso, input_hash, payload, datetime.now(UTC).isoformat()),
        )
        conn.commit()
    except Exception as exc:
        logger.debug("News cache write failed: %s", exc)


def _neutral(symbol: str, article_count: int, reason: str) -> NewsSentimentResult:
    return NewsSentimentResult(
        symbol=symbol,
        score=0.0,
        confidence=0.0,
        conclusion=reason,
        reasoning=[],
        article_count=article_count,
        is_bullish=False,
    )


def _call_claude(symbol: str, articles: list[dict], api_key: str) -> NewsSentimentResult:
    article_text = "\n\n".join(
        f"[{i+1}] {a.get('headline', '(no headline)')}\n"
        f"    {(a.get('summary') or '')[:300]}"
        for i, a in enumerate(articles[:6])
    )

    prompt = f"""You are a financial news analyst. Analyze these {len(articles)} recent news articles for {symbol} and provide a JSON response.

Articles:
{article_text}

Respond with ONLY valid JSON matching this exact schema:
{{
  "score": <float from -1.0 (very bearish) to +1.0 (very bullish)>,
  "confidence": <float 0.0 to 1.0 — how confident based on article quality>,
  "conclusion": "<one sentence: what this news means for the stock>",
  "reasoning": ["<key point 1>", "<key point 2>", "<key point 3 max>"]
}}

Rules:
- Be specific to {symbol}, not generic
- Score 0.0 if articles are irrelevant or neutral
- Low confidence if articles are old, vague, or few
- Do NOT make a buy/sell recommendation — only assess sentiment"""

    resp = httpx.post(
        _CLAUDE_URL,
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": _MODEL,
            "max_tokens": _MAX_TOKENS,
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=_TIMEOUT,
    )
    resp.raise_for_status()
    raw = resp.json()
    text = "".join(
        b.get("text", "") for b in raw.get("content", []) if b.get("type") == "text"
    ).strip()

    # Parse JSON — strip markdown fences if present
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    parsed = json.loads(text)

    score = float(parsed.get("score", 0.0))
    score = max(-1.0, min(1.0, score))
    confidence = float(parsed.get("confidence", 0.5))
    confidence = max(0.0, min(1.0, confidence))

    return NewsSentimentResult(
        symbol=symbol,
        score=score,
        confidence=confidence,
        conclusion=str(parsed.get("conclusion", ""))[:200],
        reasoning=[str(r)[:150] for r in parsed.get("reasoning", [])[:3]],
        article_count=len(articles),
        is_bullish=score > 0.15,
    )


def analyze_news(
    symbol: str,
    articles: list[dict],
    *,
    conn=None,
) -> NewsSentimentResult:
    """Analyze news articles for a symbol using Claude Haiku.

    Args:
        symbol:   Ticker (e.g. "AAPL")
        articles: List of article dicts from Alpaca news API
                  (keys: headline, summary, url, created_at, symbols)
        conn:     Optional SQLite connection for caching. If None, no caching.

    Returns:
        NewsSentimentResult with score, conclusion, reasoning.
        Falls back to neutral (score=0) if no API key or call fails.
    """
    symbol = symbol.upper()

    if not articles:
        return _neutral(symbol, 0, "No news articles available.")

    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        return _neutral(symbol, len(articles), "LLM news analysis unavailable (no API key).")

    date_iso = datetime.now(UTC).date().isoformat()
    digest = _input_hash(articles)

    # Try cache first
    if conn is not None:
        cached = _load_cache(conn, symbol, date_iso, digest)
        if cached is not None:
            logger.debug("News sentiment cache hit for %s", symbol)
            return cached

    try:
        result = _call_claude(symbol, articles, api_key)
        if conn is not None:
            _save_cache(conn, symbol, date_iso, digest, result)
        return result
    except Exception as exc:
        logger.warning("Claude news analysis failed for %s: %s", symbol, exc)
        return _neutral(symbol, len(articles), f"LLM analysis failed: {exc}")


def format_news_sentiment(result: NewsSentimentResult) -> str:
    """Format a NewsSentimentResult for Telegram display."""
    bar_len = 10
    filled = round(abs(result.score) * bar_len)
    bar = ("█" * filled).ljust(bar_len)
    direction = "▲ Bullish" if result.score > 0.15 else ("▼ Bearish" if result.score < -0.15 else "→ Neutral")
    sign = "+" if result.score >= 0 else ""

    lines = [
        f"── News Sentiment: {result.symbol} ──",
        f"  {direction}  [{bar}]  {sign}{result.score:.2f}  (conf {result.confidence:.0%})",
        f"  {result.conclusion}",
    ]
    if result.reasoning:
        lines.append("  Reasoning:")
        for r in result.reasoning:
            lines.append(f"    • {r}")
    lines.append(f"  Based on {result.article_count} article(s)")
    if result.cached:
        lines.append("  (cached today)")
    return "\n".join(lines)
