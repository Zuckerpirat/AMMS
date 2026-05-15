"""Candlestick pattern detector.

Detects common single-candle and two-candle patterns:

Single-candle:
  - Doji: open ≈ close (body < 10% of total range)
  - Hammer: small body at top, long lower shadow (bullish)
  - Shooting star: small body at bottom, long upper shadow (bearish)
  - Marubozu: no shadows, pure momentum candle
  - Spinning top: small body with equal shadows (indecision)

Two-candle:
  - Bullish engulfing: bearish candle followed by larger bullish candle
  - Bearish engulfing: bullish candle followed by larger bearish candle
  - Bullish harami: large bearish candle, small bullish candle inside
  - Bearish harami: large bullish candle, small bearish candle inside

Each pattern returns a direction (bullish/bearish/neutral) and confidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from amms.data.bars import Bar


@dataclass(frozen=True)
class CandlePattern:
    name: str
    direction: str    # "bullish" | "bearish" | "neutral"
    confidence: float  # 0..1
    description: str


def detect_patterns(bars: list[Bar], lookback: int = 5) -> list[CandlePattern]:
    """Detect candlestick patterns from the most recent bars.

    Returns list of detected patterns (may be empty).
    lookback: how many bars to analyze (most recent)
    """
    if not bars:
        return []

    recent = bars[-lookback:] if len(bars) >= lookback else bars
    patterns: list[CandlePattern] = []

    # Single-candle patterns on the last bar
    last = recent[-1]
    p = _single_candle(last)
    if p:
        patterns.append(p)

    # Two-candle patterns
    if len(recent) >= 2:
        prev = recent[-2]
        tp = _two_candle(prev, last)
        if tp:
            patterns.append(tp)

    return patterns


def _single_candle(bar: Bar) -> CandlePattern | None:
    total_range = bar.high - bar.low
    if total_range < 0.001:
        return None

    body = abs(bar.close - bar.open)
    body_pct = body / total_range
    upper_shadow = bar.high - max(bar.open, bar.close)
    lower_shadow = min(bar.open, bar.close) - bar.low
    upper_pct = upper_shadow / total_range
    lower_pct = lower_shadow / total_range

    # Doji
    if body_pct < 0.1:
        return CandlePattern("Doji", "neutral", 0.6 + (0.1 - body_pct) * 3,
                             "Open ≈ Close: market indecision, potential reversal.")

    # Hammer (bullish): small body at top, long lower shadow
    if body_pct < 0.3 and lower_pct > 0.6 and upper_pct < 0.1:
        conf = min(0.5 + lower_pct * 0.5, 0.95)
        return CandlePattern("Hammer", "bullish", round(conf, 2),
                             "Long lower shadow: buyers rejected lower prices.")

    # Shooting star (bearish): small body at bottom, long upper shadow
    if body_pct < 0.3 and upper_pct > 0.6 and lower_pct < 0.1:
        conf = min(0.5 + upper_pct * 0.5, 0.95)
        return CandlePattern("Shooting Star", "bearish", round(conf, 2),
                             "Long upper shadow: sellers rejected higher prices.")

    # Marubozu (strong momentum)
    if body_pct > 0.9 and upper_pct < 0.05 and lower_pct < 0.05:
        direction = "bullish" if bar.close > bar.open else "bearish"
        label = "Bullish Marubozu" if direction == "bullish" else "Bearish Marubozu"
        return CandlePattern(label, direction, 0.85,
                             "Full-body candle: strong one-sided momentum.")

    # Spinning top (indecision)
    if body_pct < 0.3 and upper_pct > 0.25 and lower_pct > 0.25:
        return CandlePattern("Spinning Top", "neutral", 0.55,
                             "Small body with equal shadows: buyer/seller balance.")

    return None


def _two_candle(prev: Bar, curr: Bar) -> CandlePattern | None:
    prev_body = abs(prev.close - prev.open)
    curr_body = abs(curr.close - curr.open)
    prev_bullish = prev.close > prev.open
    curr_bullish = curr.close > curr.open

    if prev_body < 0.001 or curr_body < 0.001:
        return None

    # Bullish engulfing
    if not prev_bullish and curr_bullish:
        if curr.open <= prev.close and curr.close >= prev.open and curr_body > prev_body * 1.1:
            conf = min(0.6 + (curr_body / prev_body - 1) * 0.2, 0.95)
            return CandlePattern("Bullish Engulfing", "bullish", round(conf, 2),
                                 "Bullish candle fully engulfs prior bearish: reversal signal.")

    # Bearish engulfing
    if prev_bullish and not curr_bullish:
        if curr.open >= prev.close and curr.close <= prev.open and curr_body > prev_body * 1.1:
            conf = min(0.6 + (curr_body / prev_body - 1) * 0.2, 0.95)
            return CandlePattern("Bearish Engulfing", "bearish", round(conf, 2),
                                 "Bearish candle fully engulfs prior bullish: reversal signal.")

    # Bullish harami
    if not prev_bullish and curr_bullish:
        if curr.open > prev.close and curr.close < prev.open and curr_body < prev_body * 0.5:
            return CandlePattern("Bullish Harami", "bullish", 0.60,
                                 "Small bullish inside large bearish: potential bottom.")

    # Bearish harami
    if prev_bullish and not curr_bullish:
        if curr.open < prev.close and curr.close > prev.open and curr_body < prev_body * 0.5:
            return CandlePattern("Bearish Harami", "bearish", 0.60,
                                 "Small bearish inside large bullish: potential top.")

    return None
