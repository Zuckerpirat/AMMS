"""Deep multi-factor price forecast via Claude.

Combines five analytical layers into one structured forecast:

  1. RECENT NEWS     — current catalysts and events
  2. PRICE HISTORY   — 30/90/252-day returns, 52-week position, volatility
  3. TECHNICAL       — RSI, trend (MA50/MA200), momentum state
  4. BEHAVIORAL      — market psychology (panic/FOMO/capitulation zones)
  5. CYCLE ANALYSIS  — bull/bear phase, historical analogues, modern deviations

Claude reasons across all five layers like a senior research analyst would:
not just "news says bullish" but "AI hardware cycle resembles 2000 CSCO
but with stronger fundamentals — expect 15–25% upside IF demand holds."

Output: DeepForecast dataclass with:
  direction        : "up" | "down" | "sideways" | "unknown"
  magnitude        : expected % move (e.g. 12.0)
  horizon          : "1d" | "1w" | "1m" | "3m"
  confidence       : 0.0–1.0
  catalysts        : list[str] — specific positive drivers
  risks            : list[str] — invalidation risks
  summary          : one-line executive summary
  behavioral_state : "panic" | "fomo" | "capitulation" | "euphoria" | "neutral"
  cycle_phase      : "early_bull" | "mid_bull" | "late_bull" | "early_bear" | "mid_bear" | "recovery" | "unknown"
  historical_note  : str — comparable past event and how it resolved
  cycle_deviation  : str — whether this cycle deviates from the analogue and why
  price_context    : dict — computed stats (RSI, returns, vol, etc.)
  cached           : bool
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_CLAUDE_URL = "https://api.anthropic.com/v1/messages"
_MODEL = "claude-haiku-4-5-20251001"
_MAX_TOKENS = 1200
_TIMEOUT = 30.0


@dataclass
class DeepForecast:
    symbol: str
    direction: str           # "up" | "down" | "sideways" | "unknown"
    magnitude: float         # expected % move
    horizon: str             # "1d" | "1w" | "1m" | "3m"
    confidence: float        # 0.0–1.0
    catalysts: list[str]
    risks: list[str]
    summary: str
    behavioral_state: str    # "panic" | "fomo" | "capitulation" | "euphoria" | "neutral"
    cycle_phase: str         # "early_bull" | "mid_bull" | "late_bull" | etc.
    historical_note: str     # comparable past event
    cycle_deviation: str     # how current cycle deviates from the analogue
    price_context: dict      # computed stats passed to Claude
    article_count: int
    cached: bool = False


# ── Price context computation ─────────────────────────────────────────────────

def _compute_price_context(bars) -> dict[str, Any]:
    """Compute statistical price context from bar data."""
    if not bars or len(bars) < 20:
        return {}

    closes = [float(b.close) for b in bars]
    n = len(closes)
    cur = closes[-1]

    def pct_change(ago: int) -> float | None:
        if n > ago:
            ref = closes[-(ago + 1)]
            return round((cur / ref - 1.0) * 100.0, 2) if ref > 0 else None
        return None

    # Returns
    ret_1d = pct_change(1)
    ret_5d = pct_change(5)
    ret_20d = pct_change(20)
    ret_60d = pct_change(min(60, n - 1))
    ret_252d = pct_change(min(252, n - 1))

    # 52-week high/low and position
    highs_52w = [float(b.high) for b in bars[-252:]] if n >= 252 else [float(b.high) for b in bars]
    lows_52w = [float(b.low) for b in bars[-252:]] if n >= 252 else [float(b.low) for b in bars]
    high_52w = max(highs_52w)
    low_52w = min(lows_52w)
    range_52w = high_52w - low_52w
    pct_from_high = round((cur / high_52w - 1.0) * 100.0, 1) if high_52w > 0 else 0.0
    pct_from_low = round((cur / low_52w - 1.0) * 100.0, 1) if low_52w > 0 else 0.0
    position_in_range = round((cur - low_52w) / range_52w * 100.0, 1) if range_52w > 0 else 50.0

    # Moving averages
    def sma(period: int) -> float | None:
        if n >= period:
            return round(sum(closes[-period:]) / period, 2)
        return None

    ma20 = sma(20)
    ma50 = sma(50)
    ma200 = sma(200)
    above_ma50 = (cur > ma50) if ma50 else None
    above_ma200 = (cur > ma200) if ma200 else None
    golden_cross = (ma50 > ma200) if (ma50 and ma200) else None

    # RSI (14)
    rsi_val = None
    try:
        gains = []
        losses = []
        for i in range(1, min(15, n)):
            diff = closes[-(i)] - closes[-(i + 1)]
            (gains if diff >= 0 else losses).append(abs(diff))
        if gains and losses:
            avg_g = sum(gains) / len(gains)
            avg_l = sum(losses) / len(losses)
            if avg_l > 0:
                rs = avg_g / avg_l
                rsi_val = round(100 - 100 / (1 + rs), 1)
    except Exception:
        pass

    # Volatility (20d annualised)
    vol_20d = None
    try:
        rets = [(closes[i] / closes[i - 1] - 1.0) for i in range(max(1, n - 20), n)]
        if len(rets) >= 5:
            mean_r = sum(rets) / len(rets)
            var = sum((r - mean_r) ** 2 for r in rets) / len(rets)
            vol_20d = round((var ** 0.5) * (252 ** 0.5) * 100.0, 1)
    except Exception:
        pass

    # Behavioral zone
    behavioral = "neutral"
    if pct_from_high < -30:
        behavioral = "panic" if (ret_20d or 0) < -10 else "capitulation"
    elif position_in_range > 90:
        behavioral = "euphoria" if (ret_20d or 0) > 15 else "fomo"

    # Trend strength
    trend = "unknown"
    if above_ma50 is not None and above_ma200 is not None:
        if above_ma50 and above_ma200 and golden_cross:
            trend = "strong_uptrend"
        elif above_ma50 and above_ma200:
            trend = "uptrend"
        elif not above_ma50 and not above_ma200:
            trend = "downtrend"
        else:
            trend = "mixed"

    return {
        "price": round(cur, 2),
        "ret_1d_pct": ret_1d,
        "ret_5d_pct": ret_5d,
        "ret_20d_pct": ret_20d,
        "ret_60d_pct": ret_60d,
        "ret_252d_pct": ret_252d,
        "high_52w": round(high_52w, 2),
        "low_52w": round(low_52w, 2),
        "pct_from_52w_high": pct_from_high,
        "pct_from_52w_low": pct_from_low,
        "position_in_52w_range_pct": position_in_range,
        "ma_50": ma50,
        "ma_200": ma200,
        "above_ma50": above_ma50,
        "above_ma200": above_ma200,
        "golden_cross": golden_cross,
        "rsi_14": rsi_val,
        "vol_20d_annualised_pct": vol_20d,
        "trend": trend,
        "behavioral_zone": behavioral,
        "bars_available": n,
    }


# ── Caching ───────────────────────────────────────────────────────────────────

def _ensure_table(conn) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS deep_forecast_cache (
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


def _input_hash(articles: list[dict], price_context: dict) -> str:
    payload = json.dumps({
        "a": [{"h": a.get("headline", "")} for a in articles],
        "p": {k: price_context.get(k) for k in ["ret_20d_pct", "ret_252d_pct", "rsi_14", "trend"]},
    }, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _load_cache(conn, symbol: str, date_iso: str, digest: str) -> DeepForecast | None:
    try:
        _ensure_table(conn)
        row = conn.execute(
            "SELECT result_json, input_hash FROM deep_forecast_cache WHERE symbol=? AND date=?",
            (symbol.upper(), date_iso),
        ).fetchone()
        if row and row["input_hash"] == digest:
            data = json.loads(row["result_json"])
            data["cached"] = True
            return DeepForecast(**data)
    except Exception as exc:
        logger.debug("Deep forecast cache read: %s", exc)
    return None


def _save_cache(conn, symbol: str, date_iso: str, digest: str, result: DeepForecast) -> None:
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
            "behavioral_state": result.behavioral_state,
            "cycle_phase": result.cycle_phase,
            "historical_note": result.historical_note,
            "cycle_deviation": result.cycle_deviation,
            "price_context": result.price_context,
            "article_count": result.article_count,
            "cached": False,
        })
        conn.execute(
            """
            INSERT INTO deep_forecast_cache(symbol, date, input_hash, result_json, created_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(symbol, date) DO UPDATE SET
                input_hash  = excluded.input_hash,
                result_json = excluded.result_json,
                created_at  = excluded.created_at
            """,
            (symbol.upper(), date_iso, digest, payload, datetime.now(UTC).isoformat()),
        )
        conn.commit()
    except Exception as exc:
        logger.debug("Deep forecast cache write: %s", exc)


def _fallback(symbol: str, article_count: int, price_context: dict, reason: str) -> DeepForecast:
    return DeepForecast(
        symbol=symbol,
        direction="unknown",
        magnitude=0.0,
        horizon="1w",
        confidence=0.0,
        catalysts=[],
        risks=[],
        summary=reason,
        behavioral_state=price_context.get("behavioral_zone", "neutral"),
        cycle_phase="unknown",
        historical_note="",
        cycle_deviation="",
        price_context=price_context,
        article_count=article_count,
    )


# ── Claude call ───────────────────────────────────────────────────────────────

def _call_claude(
    symbol: str,
    articles: list[dict],
    price_context: dict,
    macro_level: str,
    api_key: str,
) -> DeepForecast:
    # Build news section
    if articles:
        news_text = "\n\n".join(
            f"[{i+1}] {a.get('headline', '(no headline)')}\n"
            f"    {(a.get('summary') or '')[:350]}"
            for i, a in enumerate(articles[:5])
        )
    else:
        news_text = "(no recent news available)"

    # Build price context section
    ctx = price_context
    price_lines = []
    if ctx.get("price"):
        price_lines.append(f"  Current price: ${ctx['price']}")
    for k, label in [
        ("ret_1d_pct", "1d return"), ("ret_5d_pct", "5d return"),
        ("ret_20d_pct", "20d return"), ("ret_60d_pct", "60d return"),
        ("ret_252d_pct", "1y return"),
    ]:
        v = ctx.get(k)
        if v is not None:
            price_lines.append(f"  {label}: {v:+.1f}%")
    if ctx.get("pct_from_52w_high") is not None:
        price_lines.append(f"  From 52w high: {ctx['pct_from_52w_high']:+.1f}%")
    if ctx.get("position_in_52w_range_pct") is not None:
        price_lines.append(f"  52w range position: {ctx['position_in_52w_range_pct']:.0f}% (0=low, 100=high)")
    if ctx.get("rsi_14") is not None:
        price_lines.append(f"  RSI(14): {ctx['rsi_14']:.1f}")
    if ctx.get("trend"):
        price_lines.append(f"  Trend (MA50/200): {ctx['trend']}")
    if ctx.get("vol_20d_annualised_pct") is not None:
        price_lines.append(f"  Volatility (20d annualised): {ctx['vol_20d_annualised_pct']:.1f}%")
    if ctx.get("golden_cross") is not None:
        price_lines.append(f"  Golden cross (MA50>MA200): {ctx['golden_cross']}")
    if ctx.get("behavioral_zone"):
        price_lines.append(f"  Behavioral zone: {ctx['behavioral_zone']}")

    price_section = "\n".join(price_lines) if price_lines else "  (no price data)"
    today = datetime.now(UTC).date().isoformat()

    prompt = f"""You are a senior quantitative research analyst with expertise in behavioral finance, market cycles, and historical pattern recognition. Analyze {symbol} and produce a deep multi-factor price forecast.

=== TODAY: {today} | MACRO REGIME: {macro_level} ===

=== 1. PRICE & TECHNICAL DATA ===
{price_section}

=== 2. RECENT NEWS ({len(articles)} articles) ===
{news_text}

=== YOUR ANALYTICAL FRAMEWORK ===
Reason like a CFA with behavioral finance expertise. Integrate ALL layers:

A) NEWS IMPACT: What do current events actually mean for this specific company's fundamentals?

B) HISTORICAL ANALOGUES: What comparable historical situations exist?
   Examples to consider where relevant:
   - Semiconductor cycles: 2000 CSCO bubble, 2016 NVDA AI onset, 2022 chip glut
   - Rate cycle impacts: 2022 growth selloff, 1999-2000 tech bubble
   - Sector rotation patterns from similar macro regimes
   - Post-earnings drift patterns for this company type
   - Recovery patterns from comparable drawdowns (depth, time to recover)

C) BEHAVIORAL PSYCHOLOGY: What is the crowd doing and what does that imply?
   - RSI extremes: >70 = overbought/FOMO risk; <30 = oversold/panic opportunity
   - 52w range position: near high = distribution/euphoria; near low = accumulation
   - Capitulation patterns (high vol + strong decline + washout)
   - Momentum psychology: trend continuation vs. mean reversion

D) MARKET CYCLE PHASE: Where is this stock in its cycle?
   - MA alignment (price vs MA50 vs MA200, golden/death cross)
   - Duration and magnitude of current trend
   - Volume patterns if available
   - Sector cycle position

E) MODERN DEVIATIONS: How does the current environment differ from historical analogues?
   - AI/tech structural change vs. prior cycles
   - Post-COVID supply chain normalization effects
   - Rate environment differences
   - Concentration/index effects

Respond with ONLY valid JSON:
{{
  "direction": "<'up' | 'down' | 'sideways' | 'unknown'>",
  "magnitude": <expected % move over the horizon, float 0.0-50.0>,
  "horizon": "<'1d' | '1w' | '1m' | '3m'>",
  "confidence": <float 0.0-1.0>,
  "catalysts": [
    "<specific catalyst 1 — from news or technical setup>",
    "<catalyst 2>",
    "<catalyst 3 max>"
  ],
  "risks": [
    "<specific invalidation risk 1>",
    "<risk 2 max>"
  ],
  "summary": "<one-sentence executive summary integrating all layers>",
  "behavioral_state": "<'panic' | 'fomo' | 'capitulation' | 'euphoria' | 'neutral'>",
  "cycle_phase": "<'early_bull' | 'mid_bull' | 'late_bull' | 'early_bear' | 'mid_bear' | 'recovery' | 'unknown'>",
  "historical_note": "<1-2 sentences: most comparable past event and how it resolved>",
  "cycle_deviation": "<1 sentence: how the current situation differs from that historical analogue>"
}}

STRICT RULES:
- magnitude=0 if direction is 'sideways' or 'unknown'
- confidence<0.25 if only price data, no news
- confidence>0.7 only for very clear multi-factor convergence
- historical_note must be specific: name the year, company/sector, outcome
- cycle_deviation must address WHY this time is different (or the same)
- NEVER say "buy" or "sell" — this is research, not advice"""

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
    if horizon not in {"1d", "1w", "1m", "3m"}:
        horizon = "1w"

    confidence = max(0.0, min(1.0, float(parsed.get("confidence", 0.3))))

    behavioral = str(parsed.get("behavioral_state", "neutral")).lower()
    if behavioral not in {"panic", "fomo", "capitulation", "euphoria", "neutral"}:
        behavioral = "neutral"

    cycle = str(parsed.get("cycle_phase", "unknown")).lower()
    valid_cycles = {"early_bull", "mid_bull", "late_bull", "early_bear", "mid_bear", "recovery", "unknown"}
    if cycle not in valid_cycles:
        cycle = "unknown"

    return DeepForecast(
        symbol=symbol,
        direction=direction,
        magnitude=magnitude,
        horizon=horizon,
        confidence=confidence,
        catalysts=[str(c)[:250] for c in parsed.get("catalysts", [])[:3]],
        risks=[str(r)[:250] for r in parsed.get("risks", [])[:2]],
        summary=str(parsed.get("summary", ""))[:300],
        behavioral_state=behavioral,
        cycle_phase=cycle,
        historical_note=str(parsed.get("historical_note", ""))[:400],
        cycle_deviation=str(parsed.get("cycle_deviation", ""))[:300],
        price_context=price_context,
        article_count=len(articles),
    )


# ── Public API ────────────────────────────────────────────────────────────────

def deep_forecast(
    symbol: str,
    bars,
    articles: list[dict],
    *,
    macro_level: str = "calm",
    conn=None,
) -> DeepForecast:
    """Generate a deep multi-factor price forecast.

    Args:
        symbol:      Ticker symbol
        bars:        Price bars (e.g. from data.get_bars) — used for technical context
        articles:    Recent news articles from Alpaca
        macro_level: Current macro regime ("calm" | "elevated" | "stressed")
        conn:        Optional SQLite connection for caching

    Returns:
        DeepForecast combining news, technicals, behavioral, and cycle analysis.
        Falls back to direction='unknown' if no API key or call fails.
    """
    symbol = symbol.upper()
    price_context = _compute_price_context(bars) if bars else {}

    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        return _fallback(symbol, len(articles), price_context,
                         "KI-Prognose nicht verfügbar (kein API-Key — /setkey anthropic_key ...).")

    date_iso = datetime.now(UTC).date().isoformat()
    digest = _input_hash(articles, price_context)

    if conn is not None:
        cached = _load_cache(conn, symbol, date_iso, digest)
        if cached is not None:
            logger.debug("Deep forecast cache hit for %s", symbol)
            return cached

    try:
        result = _call_claude(symbol, articles, price_context, macro_level, api_key)
        if conn is not None:
            _save_cache(conn, symbol, date_iso, digest, result)
        return result
    except Exception as exc:
        logger.warning("Deep forecast failed for %s: %s", symbol, exc)
        return _fallback(symbol, len(articles), price_context, f"Analyse-Fehler: {exc}")


# ── Formatting ────────────────────────────────────────────────────────────────

def format_deep_forecast(fc: DeepForecast) -> str:
    """Format a DeepForecast for Telegram — rich multi-section output."""
    dir_icons = {"up": "📈", "down": "📉", "sideways": "➡️", "unknown": "❓"}
    icon = dir_icons.get(fc.direction, "❓")

    if fc.direction == "up":
        move = f"▲ +{fc.magnitude:.1f}%"
    elif fc.direction == "down":
        move = f"▼ -{fc.magnitude:.1f}%"
    elif fc.direction == "sideways":
        move = "→ Seitwärts"
    else:
        move = "Unbekannt"

    horizon_map = {"1d": "1 Tag", "1w": "1 Woche", "1m": "1 Monat", "3m": "3 Monate"}
    hor = horizon_map.get(fc.horizon, fc.horizon)

    conf_bar = ("█" * round(fc.confidence * 10)).ljust(10)

    behavior_icons = {
        "panic": "😱 Panik", "fomo": "🚀 FOMO", "capitulation": "🏳️ Kapitulation",
        "euphoria": "🎉 Euphorie", "neutral": "😐 Neutral",
    }
    beh = behavior_icons.get(fc.behavioral_state, fc.behavioral_state)

    cycle_labels = {
        "early_bull": "Frühphase Hausse", "mid_bull": "Mittphase Hausse",
        "late_bull": "Spätphase Hausse", "early_bear": "Frühphase Baisse",
        "mid_bear": "Mittphase Baisse", "recovery": "Erholung", "unknown": "Unbekannt",
    }
    cycle = cycle_labels.get(fc.cycle_phase, fc.cycle_phase)

    lines = [
        f"{icon} {fc.symbol} — Tiefenanalyse",
        f"  Prognose:  {move}  ({hor})",
        f"  Konfidenz: [{conf_bar}]  {fc.confidence:.0%}",
        f"",
        f"  📋 {fc.summary}",
        f"",
    ]

    if fc.catalysts:
        lines.append("  ✦ Treiber:")
        for c in fc.catalysts:
            lines.append(f"    → {c}")
        lines.append("")

    if fc.risks:
        lines.append("  ⚠ Risiken:")
        for r in fc.risks:
            lines.append(f"    → {r}")
        lines.append("")

    lines.append(f"  🧠 Psychologie:  {beh}")
    lines.append(f"  📊 Marktzyklus:  {cycle}")
    lines.append("")

    if fc.historical_note:
        lines.append(f"  📚 Historisches Vorbild:")
        lines.append(f"    {fc.historical_note}")

    if fc.cycle_deviation:
        lines.append(f"  🔄 Abweichung vom Vorbild:")
        lines.append(f"    {fc.cycle_deviation}")

    # Show key price stats
    ctx = fc.price_context
    stats = []
    if ctx.get("ret_20d_pct") is not None:
        stats.append(f"20T: {ctx['ret_20d_pct']:+.1f}%")
    if ctx.get("ret_252d_pct") is not None:
        stats.append(f"1J: {ctx['ret_252d_pct']:+.1f}%")
    if ctx.get("rsi_14") is not None:
        stats.append(f"RSI: {ctx['rsi_14']:.0f}")
    if ctx.get("pct_from_52w_high") is not None:
        stats.append(f"vs 52W-Hoch: {ctx['pct_from_52w_high']:+.1f}%")
    if stats:
        lines.append("")
        lines.append(f"  📈 Kursdaten:  {' | '.join(stats)}")

    lines.append(f"  Basis: {fc.article_count} News + {ctx.get('bars_available', 0)} Kursdaten")
    if fc.cached:
        lines.append("  (zwischengespeichert heute)")

    lines.append("")
    lines.append("⚠️  KI-Tiefenanalyse — kein Anlageberater. Eigenes Urteil verwenden.")

    return "\n".join(lines)
