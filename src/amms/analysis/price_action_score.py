"""Price Action Composite Score.

Combines multiple price-action factors into a single 0-100 score without
relying on volume or external data. Useful for instruments with unreliable
volume or for a pure price-based assessment.

Factors (each scored 0-100, then weighted):
  1. Trend alignment   (25%) — SMA 20/50/100 alignment
  2. Momentum          (20%) — ROC across multiple windows
  3. Mean reversion    (15%) — z-score distance from mean
  4. Volatility regime (15%) — current ATR vs historical ATR
  5. Bar quality       (15%) — avg body/range ratio (directional conviction)
  6. Range expansion   (10%) — current range vs historical range (breakout energy)
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class PAFactor:
    name: str
    raw: float         # raw value before scoring
    score: float       # 0-100 (higher = more bullish except for mean reversion)
    weight: float      # contribution weight
    weighted: float    # score × weight
    description: str


@dataclass(frozen=True)
class PAScoreReport:
    symbol: str
    composite: float      # 0-100 final score
    grade: str            # "strong_bull", "bull", "neutral", "bear", "strong_bear"
    factors: list[PAFactor]

    # Key raw values
    current_price: float
    sma20: float | None
    sma50: float | None
    sma100: float | None
    roc5: float
    roc20: float
    z_score: float
    atr_pct: float
    avg_body_ratio: float
    bars_used: int
    verdict: str


def _sma(closes: list[float], period: int) -> float | None:
    if len(closes) < period:
        return None
    return sum(closes[-period:]) / period


def _atr_simple(closes: list[float], period: int) -> float:
    """Simplified ATR using |close[i] - close[i-1]| as proxy."""
    if len(closes) < period + 1:
        return 0.0
    diffs = [abs(closes[i] - closes[i - 1]) for i in range(len(closes) - period, len(closes))]
    return sum(diffs) / period


def analyze(bars: list, *, symbol: str = "") -> PAScoreReport | None:
    """Compute Price Action Composite Score.

    bars: bar objects with .open, .high, .low, .close attributes.
    Minimum 105 bars required (for SMA-100 + lookback).
    """
    if not bars or len(bars) < 105:
        return None

    try:
        opens = [float(b.open) for b in bars]
        highs = [float(b.high) for b in bars]
        lows = [float(b.low) for b in bars]
        closes = [float(b.close) for b in bars]
    except (AttributeError, TypeError, ValueError):
        return None

    current = closes[-1]
    if current <= 0:
        return None

    n = len(closes)

    # ── Factor 1: Trend Alignment ─────────────────────────────────────
    sma20 = _sma(closes, 20)
    sma50 = _sma(closes, 50)
    sma100 = _sma(closes, 100)

    trend_points = 0
    max_trend = 6
    if sma20 is not None and current > sma20:
        trend_points += 2
    if sma50 is not None and current > sma50:
        trend_points += 2
    if sma100 is not None and current > sma100:
        trend_points += 2
    if sma20 is not None and sma50 is not None and sma20 > sma50:
        trend_points += 0  # already counted via price position
    trend_score = trend_points / max_trend * 100.0

    # ── Factor 2: Momentum ────────────────────────────────────────────
    roc5 = (current / closes[-6] - 1.0) * 100.0 if closes[-6] > 0 else 0.0
    roc20 = (current / closes[-21] - 1.0) * 100.0 if n > 20 and closes[-21] > 0 else 0.0
    roc50 = (current / closes[-51] - 1.0) * 100.0 if n > 50 and closes[-51] > 0 else 0.0

    # Map ROCs: neutral=0, ±5% = ±50 points
    def _roc_score(roc: float, scale: float = 5.0) -> float:
        return max(0.0, min(100.0, 50.0 + roc / scale * 50.0))

    mom_score = (_roc_score(roc5, 2.0) + _roc_score(roc20, 5.0) + _roc_score(roc50, 10.0)) / 3.0

    # ── Factor 3: Mean Reversion ──────────────────────────────────────
    period = 20
    subset = closes[-period:]
    mean = sum(subset) / period
    variance = sum((v - mean) ** 2 for v in subset) / period
    std = math.sqrt(variance) if variance > 0 else 1e-9
    z = (current - mean) / std

    # Invert for bullish: oversold (z<-1) = high score, overbought = low score
    # But we want it as a "reversion opportunity" metric: extreme = opportunity
    # Actually for composite: near mean = 50, oversold = 80 (buying opp), overbought = 20
    mr_score = max(0.0, min(100.0, 50.0 - z * 15.0))

    # ── Factor 4: Volatility Regime ───────────────────────────────────
    recent_atr = _atr_simple(closes, 14)
    hist_atr = _atr_simple(closes[:-14], 50) if n > 64 else recent_atr
    atr_pct = recent_atr / current * 100.0 if current > 0 else 0.0

    # Moderate volatility = good (not too high, not too low)
    # vol_ratio: recent/hist; ~1.0 = normal, <0.5 = squeeze, >2.0 = panic
    vol_ratio = recent_atr / hist_atr if hist_atr > 0 else 1.0
    # Optimal around 0.8-1.2; score peaks at 1.0, falls off
    vol_score = max(0.0, min(100.0, 100.0 - abs(vol_ratio - 1.0) * 40.0))

    # ── Factor 5: Bar Quality ─────────────────────────────────────────
    body_ratios = []
    for i in range(max(0, n - 20), n):
        rng = highs[i] - lows[i]
        if rng > 0:
            body_ratios.append(abs(closes[i] - opens[i]) / rng)
    avg_body = sum(body_ratios) / len(body_ratios) if body_ratios else 0.5

    # Higher body ratio = more directional conviction
    # Bull body ratio: also check direction of bodies
    bull_bodies = sum(1 for i in range(max(0, n - 20), n) if closes[i] > opens[i])
    direction_pct = bull_bodies / min(20, n) * 100.0

    # Quality score: body ratio × direction bias
    body_score = avg_body * 100.0 * 0.5 + direction_pct * 0.5

    # ── Factor 6: Range Expansion ─────────────────────────────────────
    current_range = highs[-1] - lows[-1]
    hist_ranges = [highs[i] - lows[i] for i in range(max(0, n - 20), n - 1)]
    avg_hist_range = sum(hist_ranges) / len(hist_ranges) if hist_ranges else current_range
    range_ratio = current_range / avg_hist_range if avg_hist_range > 0 else 1.0

    # Range expansion in direction of trend = bullish
    price_up = closes[-1] > closes[-2]
    if range_ratio > 1.2 and price_up:
        range_score = min(100.0, 50.0 + (range_ratio - 1.0) * 30.0)
    elif range_ratio > 1.2 and not price_up:
        range_score = max(0.0, 50.0 - (range_ratio - 1.0) * 30.0)
    else:
        range_score = 50.0

    # ── Composite ─────────────────────────────────────────────────────
    factors = [
        PAFactor("Trend Alignment", trend_points / max_trend, round(trend_score, 1), 0.25,
                 round(trend_score * 0.25, 2), f"Price vs SMA20/50/100"),
        PAFactor("Momentum", (roc5 + roc20) / 2, round(mom_score, 1), 0.20,
                 round(mom_score * 0.20, 2), f"ROC5={roc5:+.1f}%, ROC20={roc20:+.1f}%"),
        PAFactor("Mean Reversion", z, round(mr_score, 1), 0.15,
                 round(mr_score * 0.15, 2), f"z={z:+.2f} from 20-bar mean"),
        PAFactor("Volatility", vol_ratio, round(vol_score, 1), 0.15,
                 round(vol_score * 0.15, 2), f"ATR ratio {vol_ratio:.2f} vs history"),
        PAFactor("Bar Quality", avg_body, round(body_score, 1), 0.15,
                 round(body_score * 0.15, 2), f"Avg body/range {avg_body:.2f}, {direction_pct:.0f}% bull"),
        PAFactor("Range Expansion", range_ratio, round(range_score, 1), 0.10,
                 round(range_score * 0.10, 2), f"Range ratio {range_ratio:.2f}"),
    ]

    composite = sum(f.weighted for f in factors)
    composite = max(0.0, min(100.0, composite))

    if composite >= 70:
        grade = "strong_bull"
    elif composite >= 60:
        grade = "bull"
    elif composite <= 30:
        grade = "strong_bear"
    elif composite <= 40:
        grade = "bear"
    else:
        grade = "neutral"

    # Verdict
    top_factor = max(factors, key=lambda f: abs(f.weighted - 50 * f.weight))
    verdict = (
        f"Price Action Score: {composite:.0f}/100 ({grade.replace('_', ' ')}). "
        f"Key driver: {top_factor.name} ({top_factor.score:.0f}/100). "
        f"Price {roc20:+.1f}% over 20 bars."
    )

    return PAScoreReport(
        symbol=symbol,
        composite=round(composite, 1),
        grade=grade,
        factors=factors,
        current_price=round(current, 4),
        sma20=round(sma20, 4) if sma20 else None,
        sma50=round(sma50, 4) if sma50 else None,
        sma100=round(sma100, 4) if sma100 else None,
        roc5=round(roc5, 3),
        roc20=round(roc20, 3),
        z_score=round(z, 4),
        atr_pct=round(atr_pct, 2),
        avg_body_ratio=round(avg_body, 3),
        bars_used=len(bars),
        verdict=verdict,
    )
