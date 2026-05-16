"""Candlestick Pattern Recognizer.

Identifies common single-, two-, and three-bar candlestick patterns
from OHLC bar data and assigns a directional bias per pattern.

Patterns detected:
  Single-bar:  Hammer, Hanging Man, Inverted Hammer, Shooting Star,
               Doji, Dragonfly Doji, Gravestone Doji, Marubozu
               (Bull/Bear), Spinning Top
  Two-bar:     Bullish/Bearish Engulfing, Piercing Line, Dark Cloud
               Cover, Tweezer Top/Bottom
  Three-bar:   Morning Star, Evening Star, Three White Soldiers,
               Three Black Crows
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class CandlePattern:
    name: str
    bias: str          # "bullish", "bearish", "neutral"
    strength: str      # "strong", "moderate", "weak"
    bar_idx: int       # index of the last bar of the pattern (0-based from start)
    description: str


@dataclass(frozen=True)
class CandleReport:
    symbol: str
    patterns: list[CandlePattern]
    recent_patterns: list[CandlePattern]   # last 5 bars only
    bullish_count: int
    bearish_count: int
    dominant_bias: str   # "bullish", "bearish", "neutral"
    bias_score: float    # -1.0 (bear) to +1.0 (bull)
    last_signal: CandlePattern | None
    current_price: float
    bars_used: int
    verdict: str


# ── Candle geometry helpers ────────────────────────────────────────────

def _body(o: float, c: float) -> float:
    return abs(c - o)


def _upper_wick(o: float, h: float, c: float) -> float:
    return h - max(o, c)


def _lower_wick(o: float, l: float, c: float) -> float:
    return min(o, c) - l


def _range(h: float, l: float) -> float:
    return h - l if h > l else 1e-9


def _is_bull(o: float, c: float) -> bool:
    return c > o


def _is_bear(o: float, c: float) -> bool:
    return c < o


def _mid(o: float, c: float) -> float:
    return (o + c) / 2.0


# ── Single-bar patterns ───────────────────────────────────────────────

def _single_bar_patterns(bars: list, i: int) -> list[CandlePattern]:
    b = bars[i]
    o, h, l, c = float(b.open), float(b.high), float(b.low), float(b.close)

    rng = _range(h, l)
    body = _body(o, c)
    upper = _upper_wick(o, h, c)
    lower = _lower_wick(o, l, c)

    body_pct = body / rng
    upper_pct = upper / rng
    lower_pct = lower / rng

    found: list[CandlePattern] = []

    # Doji: tiny body
    if body_pct < 0.1:
        if lower_pct > 0.4 and upper_pct < 0.1:
            found.append(CandlePattern("Dragonfly Doji", "bullish", "moderate", i,
                                       "Long lower wick, almost no upper wick"))
        elif upper_pct > 0.4 and lower_pct < 0.1:
            found.append(CandlePattern("Gravestone Doji", "bearish", "moderate", i,
                                       "Long upper wick, almost no lower wick"))
        else:
            found.append(CandlePattern("Doji", "neutral", "weak", i,
                                       "Indecision candle, open ≈ close"))
        return found

    # Marubozu: very small wicks
    if body_pct > 0.90:
        if _is_bull(o, c):
            found.append(CandlePattern("Bullish Marubozu", "bullish", "strong", i,
                                       "Full bull body, minimal wicks — strong buying"))
        else:
            found.append(CandlePattern("Bearish Marubozu", "bearish", "strong", i,
                                       "Full bear body, minimal wicks — strong selling"))
        return found

    # Spinning Top: small body, long wicks on both sides
    if body_pct < 0.3 and upper_pct > 0.25 and lower_pct > 0.25:
        found.append(CandlePattern("Spinning Top", "neutral", "weak", i,
                                   "Small body, both wicks long — indecision"))
        return found

    # Hammer / Hanging Man: lower wick ≥ 2× body, small upper wick
    if lower_pct >= 0.5 and body_pct >= 0.1 and upper_pct <= 0.15:
        if _is_bull(o, c):
            found.append(CandlePattern("Hammer", "bullish", "moderate", i,
                                       "Long lower wick — buying rejected lows"))
        else:
            found.append(CandlePattern("Hanging Man", "bearish", "moderate", i,
                                       "Long lower wick after uptrend — reversal risk"))

    # Inverted Hammer / Shooting Star: upper wick ≥ 2× body, small lower wick
    if upper_pct >= 0.5 and body_pct >= 0.1 and lower_pct <= 0.15:
        if _is_bull(o, c):
            found.append(CandlePattern("Inverted Hammer", "bullish", "weak", i,
                                       "Long upper wick up — potential bull reversal"))
        else:
            found.append(CandlePattern("Shooting Star", "bearish", "moderate", i,
                                       "Long upper wick — selling rejected highs"))

    return found


# ── Two-bar patterns ──────────────────────────────────────────────────

def _two_bar_patterns(bars: list, i: int) -> list[CandlePattern]:
    if i < 1:
        return []
    p = bars[i - 1]
    b = bars[i]
    po, ph, pl, pc = float(p.open), float(p.high), float(p.low), float(p.close)
    co, ch, cl, cc = float(b.open), float(b.high), float(b.low), float(b.close)

    found: list[CandlePattern] = []

    p_body = _body(po, pc)
    c_body = _body(co, cc)
    p_rng = _range(ph, pl)
    c_rng = _range(ch, cl)

    if p_body < 1e-9 or c_body < 1e-9:
        return found

    # Bullish Engulfing
    if _is_bear(po, pc) and _is_bull(co, cc) and co <= pc and cc >= po:
        found.append(CandlePattern("Bullish Engulfing", "bullish", "strong", i,
                                   "Bull bar fully engulfs prior bear bar"))

    # Bearish Engulfing
    if _is_bull(po, pc) and _is_bear(co, cc) and co >= pc and cc <= po:
        found.append(CandlePattern("Bearish Engulfing", "bearish", "strong", i,
                                   "Bear bar fully engulfs prior bull bar"))

    # Piercing Line: prev bear, current bull opens below prev low, closes above midpoint
    if _is_bear(po, pc) and _is_bull(co, cc):
        if co < pl and cc > _mid(po, pc) and cc < po:
            found.append(CandlePattern("Piercing Line", "bullish", "moderate", i,
                                       "Bull bar closes above mid of prior bear bar"))

    # Dark Cloud Cover: prev bull, current bear opens above prev high, closes below midpoint
    if _is_bull(po, pc) and _is_bear(co, cc):
        if co > ph and cc < _mid(po, pc) and cc > po:
            found.append(CandlePattern("Dark Cloud Cover", "bearish", "moderate", i,
                                       "Bear bar closes below mid of prior bull bar"))

    # Tweezer Bottom: similar lows, prior bear then bull
    if _is_bear(po, pc) and _is_bull(co, cc) and abs(pl - cl) / max(p_rng, c_rng) < 0.05:
        found.append(CandlePattern("Tweezer Bottom", "bullish", "moderate", i,
                                   "Two candles share same low — support confirmed"))

    # Tweezer Top: similar highs, prior bull then bear
    if _is_bull(po, pc) and _is_bear(co, cc) and abs(ph - ch) / max(p_rng, c_rng) < 0.05:
        found.append(CandlePattern("Tweezer Top", "bearish", "moderate", i,
                                   "Two candles share same high — resistance confirmed"))

    return found


# ── Three-bar patterns ────────────────────────────────────────────────

def _three_bar_patterns(bars: list, i: int) -> list[CandlePattern]:
    if i < 2:
        return []
    b0 = bars[i - 2]
    b1 = bars[i - 1]
    b2 = bars[i]
    o0, h0, l0, c0 = float(b0.open), float(b0.high), float(b0.low), float(b0.close)
    o1, h1, l1, c1 = float(b1.open), float(b1.high), float(b1.low), float(b1.close)
    o2, h2, l2, c2 = float(b2.open), float(b2.high), float(b2.low), float(b2.close)

    found: list[CandlePattern] = []

    body0 = _body(o0, c0)
    body1 = _body(o1, c1)
    body2 = _body(o2, c2)

    if body0 < 1e-9 or body2 < 1e-9:
        return found

    # Morning Star: bear, small middle, bull — middle below both
    if (_is_bear(o0, c0) and body1 < body0 * 0.5
            and _is_bull(o2, c2) and c2 > _mid(o0, c0)):
        found.append(CandlePattern("Morning Star", "bullish", "strong", i,
                                   "Three-bar bull reversal: bear → star → bull"))

    # Evening Star: bull, small middle, bear — middle above both
    if (_is_bull(o0, c0) and body1 < body0 * 0.5
            and _is_bear(o2, c2) and c2 < _mid(o0, c0)):
        found.append(CandlePattern("Evening Star", "bearish", "strong", i,
                                   "Three-bar bear reversal: bull → star → bear"))

    # Three White Soldiers: three consecutive bull bars, each closing higher
    if (_is_bull(o0, c0) and _is_bull(o1, c1) and _is_bull(o2, c2)
            and c1 > c0 and c2 > c1
            and o1 > o0 and o2 > o1):
        found.append(CandlePattern("Three White Soldiers", "bullish", "strong", i,
                                   "Three consecutive rising bull bars — strong uptrend"))

    # Three Black Crows: three consecutive bear bars, each closing lower
    if (_is_bear(o0, c0) and _is_bear(o1, c1) and _is_bear(o2, c2)
            and c1 < c0 and c2 < c1
            and o1 < o0 and o2 < o1):
        found.append(CandlePattern("Three Black Crows", "bearish", "strong", i,
                                   "Three consecutive falling bear bars — strong downtrend"))

    return found


# ── Main ──────────────────────────────────────────────────────────────

def analyze(bars: list, *, symbol: str = "", lookback: int = 20) -> CandleReport | None:
    """Scan bars for candlestick patterns.

    bars: bar objects with .open, .high, .low, .close attributes.
    lookback: how many recent bars to scan for patterns (capped at len(bars)).
    """
    if not bars or len(bars) < 3:
        return None

    # Validate required attributes
    try:
        _ = float(bars[0].open), float(bars[0].high), float(bars[0].low), float(bars[0].close)
    except (AttributeError, TypeError, ValueError):
        return None

    scan_start = max(0, len(bars) - lookback)
    all_patterns: list[CandlePattern] = []

    for i in range(scan_start, len(bars)):
        all_patterns.extend(_single_bar_patterns(bars, i))
        all_patterns.extend(_two_bar_patterns(bars, i))
        all_patterns.extend(_three_bar_patterns(bars, i))

    # Sort by bar_idx so most recent last
    all_patterns.sort(key=lambda p: p.bar_idx)

    recent_cutoff = len(bars) - 5
    recent = [p for p in all_patterns if p.bar_idx >= recent_cutoff]

    bull = sum(1 for p in all_patterns if p.bias == "bullish")
    bear = sum(1 for p in all_patterns if p.bias == "bearish")
    total = bull + bear
    if total > 0:
        bias_score = (bull - bear) / total
    else:
        bias_score = 0.0

    if bias_score > 0.2:
        dominant_bias = "bullish"
    elif bias_score < -0.2:
        dominant_bias = "bearish"
    else:
        dominant_bias = "neutral"

    last_signal = all_patterns[-1] if all_patterns else None

    current_price = float(bars[-1].close)

    # Verdict
    if not all_patterns:
        verdict = "No significant candlestick patterns detected in the lookback window."
    else:
        parts = [f"{len(all_patterns)} pattern(s) found: {bull} bullish, {bear} bearish"]
        if last_signal:
            parts.append(f"last: {last_signal.name} ({last_signal.bias})")
        parts.append(f"bias {dominant_bias}")
        verdict = "Candlestick analysis: " + "; ".join(parts) + "."

    return CandleReport(
        symbol=symbol,
        patterns=all_patterns,
        recent_patterns=recent,
        bullish_count=bull,
        bearish_count=bear,
        dominant_bias=dominant_bias,
        bias_score=round(bias_score, 3),
        last_signal=last_signal,
        current_price=current_price,
        bars_used=len(bars),
        verdict=verdict,
    )
