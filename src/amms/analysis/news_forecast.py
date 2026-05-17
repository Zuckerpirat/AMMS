"""News-based price forecast via Claude.

Reads recent news articles for a symbol and produces a structured
price-movement forecast: expected direction, magnitude, timeframe,
key catalysts, and risk factors.

This is explicitly NOT a financial recommendation — it is a
probabilistic, AI-generated research signal to be weighed alongside
technical and fundamental analysis. Confidence should be treated as
low; all forecasts may be wrong.

Output: NewsForecast dataclass with:
  direction   : "up" | "down" | "sideways" | "unknown"
  magnitude   : expected move in % (e.g. 5.0 = ~5% move expected)
  horizon     : "1d" | "1w" | "1m" — estimated timeframe
  confidence  : 0.0–1.0 (how confident the model is given the news)
  catalysts   : list[str] — specific drivers (e.g. "AI chip demand surge")
  risks       : list[str] — what could invalidate the forecast
  summary     : one-line forecast statement
  cached      : bool

Caching: per (symbol, UTC date) in SQLite table ``news_forecast_cache``.
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
_MAX_TOKENS = 700
_TIMEOUT = 25.0


@dataclass
class NewsForecast:
    symbol: str
    direction: str        # "up" | "down" | "sideways" | "unknown"
    magnitude: float      # expected % move (always positive; direction says which way)
    horizon: str          # "1d" | "1w" | "1m"
    confidence: float     # 0.0–1.0
    catalysts: list[str]  # specific positive/negative drivers from news
    risks: list[str]      # what could invalidate this forecast
    summary: str          # one-line forecast statement
    article_count: int
    cached: bool = False


def _ensure_table(conn) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS news_forecast_cache (
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


def _load_cache(conn, symbol: str, date_iso: str, input_hash: str) -> NewsForecast | None:
    try:
        _ensure_table(conn)
        row = conn.execute(
            "SELECT result_json, input_hash FROM news_forecast_cache WHERE symbol=? AND date=?",
            (symbol.upper(), date_iso),
        ).fetchone()
        if row and row["input_hash"] == input_hash:
            data = json.loads(row["result_json"])
            data["cached"] = True
            return NewsForecast(**data)
    except Exception as exc:
        logger.debug("Forecast cache read failed: %s", exc)
    return None


def _save_cache(conn, symbol: str, date_iso: str, input_hash: str,
                result: NewsForecast) -> None:
    try:
        _ensure_table(conn)
        payload = json.dumps({
            "symbol": result.symbol,
            "direction": result.direction,
            "magnitude": result.magnitude,
            "horizon": result.horizon,
            "confidence": result.confidence,
            "catalysts": result.catalysts,
            "risks": result.risks,
            "summary": result.summary,
            "article_count": result.article_count,
            "cached": False,
        })
        conn.execute(
            """
            INSERT INTO news_forecast_cache(symbol, date, input_hash, result_json, created_at)
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
        logger.debug("Forecast cache write failed: %s", exc)


def _unknown(symbol: str, article_count: int, reason: str) -> NewsForecast:
    return NewsForecast(
        symbol=symbol,
        direction="unknown",
        magnitude=0.0,
        horizon="1w",
        confidence=0.0,
        catalysts=[],
        risks=[],
        summary=reason,
        article_count=article_count,
    )


def _call_claude(symbol: str, articles: list[dict], api_key: str) -> NewsForecast:
    article_text = "\n\n".join(
        f"[{i+1}] {a.get('headline', '(no headline)')}\n"
        f"    {(a.get('summary') or '')[:400]}"
        for i, a in enumerate(articles[:6])
    )

    prompt = f"""You are a quantitative financial analyst. Based on these news articles about {symbol}, produce a price-movement forecast in JSON.

Articles ({len(articles)} total, newest first):
{article_text}

Today's date: {datetime.now(UTC).date().isoformat()}

Respond with ONLY valid JSON matching this exact schema:
{{
  "direction": "<'up' | 'down' | 'sideways' | 'unknown'>",
  "magnitude": <expected % price move, float 0.0-50.0, e.g. 5.0 means ~5% move>,
  "horizon": "<'1d' | '1w' | '1m'>",
  "confidence": <float 0.0-1.0>,
  "catalysts": ["<specific catalyst 1>", "<catalyst 2>", "<catalyst 3 max>"],
  "risks": ["<risk that could invalidate this forecast>", "<risk 2 max>"],
  "summary": "<one sentence: e.g. 'NVDA likely up 5-8% within 1 week due to AI chip demand surge beating estimates'>"
}}

Strict rules:
- Be specific to {symbol}'s actual business situation, not generic market commentary
- magnitude = 0 if direction is 'sideways' or 'unknown'
- confidence < 0.3 if news is vague, old (>3 days), or unrelated to {symbol}'s financials
- confidence > 0.6 only for clear catalysts: earnings beats, major contract wins, analyst upgrades with specific targets
- catalysts must be concrete facts from the articles, not speculation
- risks must be specific to {symbol}, not generic ("market could go down")
- IMPORTANT: This is a research signal, not advice. Never say "buy" or "sell"."""

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

    # Strip markdown fences
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]

    parsed = json.loads(text)

    direction = str(parsed.get("direction", "unknown")).lower()
    if direction not in {"up", "down", "sideways", "unknown"}:
        direction = "unknown"

    magnitude = float(parsed.get("magnitude", 0.0))
    magnitude = max(0.0, min(50.0, magnitude))

    horizon = str(parsed.get("horizon", "1w")).lower()
    if horizon not in {"1d", "1w", "1m"}:
        horizon = "1w"

    confidence = float(parsed.get("confidence", 0.3))
    confidence = max(0.0, min(1.0, confidence))

    catalysts = [str(c)[:200] for c in parsed.get("catalysts", [])[:3]]
    risks = [str(r)[:200] for r in parsed.get("risks", [])[:2]]
    summary = str(parsed.get("summary", ""))[:300]

    return NewsForecast(
        symbol=symbol,
        direction=direction,
        magnitude=magnitude,
        horizon=horizon,
        confidence=confidence,
        catalysts=catalysts,
        risks=risks,
        summary=summary,
        article_count=len(articles),
    )


def forecast_from_news(
    symbol: str,
    articles: list[dict],
    *,
    conn=None,
) -> NewsForecast:
    """Generate a price-movement forecast from news articles.

    Args:
        symbol:   Ticker symbol
        articles: News articles from Alpaca (headline, summary, created_at, ...)
        conn:     Optional SQLite connection for caching (one call per symbol per day)

    Returns:
        NewsForecast with direction, magnitude, horizon, confidence.
        Returns direction='unknown', confidence=0 if no API key or call fails.
    """
    symbol = symbol.upper()

    if not articles:
        return _unknown(symbol, 0, "Keine News verfügbar.")

    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        return _unknown(symbol, len(articles), "KI-Prognose nicht verfügbar (kein API-Key).")

    date_iso = datetime.now(UTC).date().isoformat()
    digest = _input_hash(articles)

    if conn is not None:
        cached = _load_cache(conn, symbol, date_iso, digest)
        if cached is not None:
            logger.debug("News forecast cache hit for %s", symbol)
            return cached

    try:
        result = _call_claude(symbol, articles, api_key)
        if conn is not None:
            _save_cache(conn, symbol, date_iso, digest, result)
        return result
    except Exception as exc:
        logger.warning("Claude forecast failed for %s: %s", symbol, exc)
        return _unknown(symbol, len(articles), f"Prognose-Fehler: {exc}")


def format_forecast(fc: NewsForecast) -> str:
    """Format a NewsForecast for Telegram display."""
    direction_icons = {
        "up": "📈",
        "down": "📉",
        "sideways": "➡️",
        "unknown": "❓",
    }
    icon = direction_icons.get(fc.direction, "❓")

    if fc.direction == "up":
        dir_text = f"▲ Anstieg ~{fc.magnitude:.1f}%"
    elif fc.direction == "down":
        dir_text = f"▼ Rückgang ~{fc.magnitude:.1f}%"
    elif fc.direction == "sideways":
        dir_text = "→ Seitwärts"
    else:
        dir_text = "Unbekannt"

    horizon_map = {"1d": "1 Tag", "1w": "1 Woche", "1m": "1 Monat"}
    horizon_text = horizon_map.get(fc.horizon, fc.horizon)

    conf_bar_len = 8
    conf_filled = round(fc.confidence * conf_bar_len)
    conf_bar = ("█" * conf_filled).ljust(conf_bar_len)

    lines = [
        f"{icon} {fc.symbol} — News-Prognose",
        f"  Richtung:   {dir_text}  ({horizon_text})",
        f"  Konfidenz:  [{conf_bar}]  {fc.confidence:.0%}",
        f"  Fazit: {fc.summary}",
    ]

    if fc.catalysts:
        lines.append("  Treiber:")
        for c in fc.catalysts:
            lines.append(f"    ✦ {c}")

    if fc.risks:
        lines.append("  Risiken:")
        for r in fc.risks:
            lines.append(f"    ⚠ {r}")

    lines.append(f"  Basis: {fc.article_count} News-Artikel")

    if fc.cached:
        lines.append("  (zwischengespeichert heute)")

    lines.append("")
    lines.append("⚠️  KI-Prognose — kein Anlageberater. Eigenes Urteil verwenden.")

    return "\n".join(lines)
