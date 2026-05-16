"""Wyckoff Phase Detector.

Attempts to identify the current Wyckoff market phase using price action,
volume, and range analysis. Wyckoff defined four phases:

  Phase A (Stopping):    End of previous trend; high volume climax bars
  Phase B (Building):    Sideways range; testing supply/demand
  Phase C (Testing):     Spring (accumulation) or UTAD (distribution)
  Phase D (Markup/down): Trend begins; price leaves the range
  Phase E (Trend):       Sustained trend phase

This implementation uses heuristics based on:
  - Price range width relative to history (tight = building/testing)
  - Volume trend (climactic vs dry)
  - Trend direction
  - Position within recent high/low range
  - Volatility patterns
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class WyckoffReport:
    symbol: str

    phase: str           # "accumulation_a", "accumulation_b", "accumulation_c",
                         # "markup", "distribution_a", "distribution_b",
                         # "distribution_c", "markdown", "trend_up", "trend_down", "unknown"
    phase_label: str     # human-readable
    confidence: float    # 0-1

    # Supporting evidence
    is_sideways: bool
    range_pct: float         # (high - low) / mid_price over lookback
    vol_climax: bool         # recent volume spike
    vol_drying: bool         # volume declining
    trend_direction: str     # "up", "down", "flat"
    price_position: float    # 0=at low of range, 1=at high of range

    # Spring/UTAD detection
    spring_detected: bool    # price below range then quickly recovered
    utad_detected: bool      # price above range then quickly failed

    current_price: float
    range_high: float
    range_low: float
    bars_used: int
    verdict: str


def _wilder_atr(highs: list[float], lows: list[float], closes: list[float], period: int) -> list[float]:
    if len(closes) < period + 1:
        return []
    trs = []
    for i in range(1, len(closes)):
        tr = max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1]))
        trs.append(tr)
    if len(trs) < period:
        return []
    atr = [sum(trs[:period]) / period]
    for tr in trs[period:]:
        atr.append((atr[-1] * (period - 1) + tr) / period)
    return atr


def analyze(
    bars: list,
    *,
    symbol: str = "",
    lookback: int = 60,
) -> WyckoffReport | None:
    """Detect Wyckoff phase from price/volume history.

    bars: bar objects with .high, .low, .close and optionally .volume.
    lookback: bars to analyse for range and phase detection.
    """
    if not bars or len(bars) < 30:
        return None

    try:
        highs = [float(b.high) for b in bars]
        lows = [float(b.low) for b in bars]
        closes = [float(b.close) for b in bars]
    except (AttributeError, TypeError, ValueError):
        return None

    try:
        volumes = [float(b.volume) for b in bars]
        has_vol = True
    except AttributeError:
        volumes = [1.0] * len(bars)
        has_vol = False

    current = closes[-1]
    if current <= 0:
        return None

    n = len(bars)
    window = min(lookback, n)
    w_closes = closes[-window:]
    w_highs = highs[-window:]
    w_lows = lows[-window:]
    w_vols = volumes[-window:]

    # Range analysis
    range_high = max(w_highs)
    range_low = min(w_lows)
    mid_price = (range_high + range_low) / 2.0
    range_pct = (range_high - range_low) / mid_price * 100.0 if mid_price > 0 else 0.0

    # Is it sideways? Compare recent range to longer-term ATR
    atr_vals = _wilder_atr(highs[-min(n, 100):], lows[-min(n, 100):], closes[-min(n, 100):], 14)
    avg_atr = sum(atr_vals) / len(atr_vals) if atr_vals else 1.0

    # Normalised range vs ATR
    range_span = range_high - range_low
    is_sideways = range_span < avg_atr * 6.0 if avg_atr > 0 else False

    # Trend direction: SMA comparison
    if len(w_closes) >= 20:
        sma20 = sum(w_closes[-20:]) / 20
        sma_start = sum(w_closes[:20]) / 20
        if current > sma20 * 1.01 and sma20 > sma_start * 1.005:
            trend_dir = "up"
        elif current < sma20 * 0.99 and sma20 < sma_start * 0.995:
            trend_dir = "down"
        else:
            trend_dir = "flat"
    else:
        trend_dir = "flat"

    # Volume analysis
    avg_vol = sum(w_vols) / len(w_vols) if w_vols else 1.0
    recent_vol = sum(w_vols[-5:]) / 5 if len(w_vols) >= 5 else avg_vol
    vol_climax = recent_vol > avg_vol * 1.8 if has_vol else False
    vol_drying = recent_vol < avg_vol * 0.5 if has_vol else False

    # Price position within range
    if range_high > range_low:
        pos = (current - range_low) / (range_high - range_low)
    else:
        pos = 0.5
    pos = max(0.0, min(1.0, pos))

    # Spring detection: price briefly went below range_low then recovered
    spring = False
    utad = False
    recent_lows = lows[-10:] if len(lows) >= 10 else lows
    recent_highs = highs[-10:] if len(highs) >= 10 else highs
    early_low = min(lows[-window:-10]) if window > 10 else range_low
    early_high = max(highs[-window:-10]) if window > 10 else range_high

    if any(l < early_low for l in recent_lows) and current > early_low:
        spring = True
    if any(h > early_high for h in recent_highs) and current < early_high:
        utad = True

    # Phase determination heuristics
    if not is_sideways:
        if trend_dir == "up":
            phase = "trend_up"
            label = "Trend Up (Phase E Markup)"
            conf = 0.6
        elif trend_dir == "down":
            phase = "trend_down"
            label = "Trend Down (Phase E Markdown)"
            conf = 0.6
        else:
            phase = "unknown"
            label = "Transitional / Unknown"
            conf = 0.3
    else:
        # Sideways — determine accumulation vs distribution
        if pos < 0.35:
            # Trading in lower part of range
            if vol_climax and not vol_drying:
                phase = "accumulation_a"
                label = "Accumulation Phase A (Stopping — climactic vol)"
                conf = 0.55
            elif spring:
                phase = "accumulation_c"
                label = "Accumulation Phase C (Spring detected)"
                conf = 0.70
            else:
                phase = "accumulation_b"
                label = "Accumulation Phase B (Building cause)"
                conf = 0.50
        elif pos > 0.65:
            # Trading in upper part of range
            if vol_climax and not vol_drying:
                phase = "distribution_a"
                label = "Distribution Phase A (Stopping — climactic vol)"
                conf = 0.55
            elif utad:
                phase = "distribution_c"
                label = "Distribution Phase C (UTAD detected)"
                conf = 0.70
            else:
                phase = "distribution_b"
                label = "Distribution Phase B (Building cause)"
                conf = 0.50
        else:
            # Middle of range
            if vol_drying:
                # Testing — could be Phase C testing
                phase = "accumulation_c" if trend_dir != "down" else "distribution_c"
                label = f"{'Accumulation' if trend_dir != 'down' else 'Distribution'} Phase C (Vol dry-up, testing)"
                conf = 0.45
            else:
                phase = "accumulation_b" if pos >= 0.5 else "distribution_b"
                label = "Sideways Range — Phase B (cause building)"
                conf = 0.40

    # Verdict
    verdict = (
        f"Wyckoff: {label}. "
        f"Confidence: {conf:.0%}. "
        f"Range {range_pct:.1f}%, price at {pos * 100:.0f}th percentile of range."
    )
    if spring:
        verdict += " Spring detected."
    if utad:
        verdict += " UTAD detected."

    return WyckoffReport(
        symbol=symbol,
        phase=phase,
        phase_label=label,
        confidence=round(conf, 2),
        is_sideways=is_sideways,
        range_pct=round(range_pct, 2),
        vol_climax=vol_climax,
        vol_drying=vol_drying,
        trend_direction=trend_dir,
        price_position=round(pos, 3),
        spring_detected=spring,
        utad_detected=utad,
        current_price=round(current, 4),
        range_high=round(range_high, 4),
        range_low=round(range_low, 4),
        bars_used=n,
        verdict=verdict,
    )
