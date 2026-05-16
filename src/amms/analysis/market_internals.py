"""Market Internals Score.

Computes a synthetic market internals score from a basket of symbols,
aggregating multiple technical factors to assess overall market health.

For each symbol, scores:
  1. Price vs SMA-20/50/200 (trend alignment)
  2. RSI momentum (above/below 50)
  3. Rate of change (positive/negative)
  4. New highs vs new lows (52-bar window)
  5. Volume trend (increasing/decreasing on up vs down days)

Then aggregates to produce:
  - Advance/decline ratio proxy
  - New highs / new lows ratio
  - % of symbols above SMA-50
  - Overall internals score (0-100)
  - Market breadth health label
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class SymbolInternals:
    symbol: str
    score: float          # 0-100
    above_sma20: bool
    above_sma50: bool
    above_sma200: bool
    rsi: float
    roc20: float
    new_high_52: bool
    new_low_52: bool
    trend: str            # "bull", "neutral", "bear"


@dataclass(frozen=True)
class MarketInternalsReport:
    symbols_scored: int
    symbols_bull: int
    symbols_bear: int
    symbols_neutral: int

    pct_above_sma20: float
    pct_above_sma50: float
    pct_above_sma200: float
    pct_new_highs: float
    pct_new_lows: float

    advance_decline_ratio: float   # bull / (bull + bear)
    nh_nl_ratio: float             # new highs / (new highs + new lows)

    composite_score: float         # 0-100
    health_label: str              # "strong_bull", "bull", "neutral", "bear", "strong_bear"

    by_symbol: list[SymbolInternals]
    verdict: str


def _sma(vals: list[float], period: int) -> float | None:
    if len(vals) < period:
        return None
    return sum(vals[-period:]) / period


def _rsi(closes: list[float], period: int = 14) -> float:
    if len(closes) < period + 1:
        return 50.0
    gains = []
    losses = []
    for i in range(len(closes) - period, len(closes)):
        change = closes[i] - closes[i - 1]
        gains.append(max(change, 0.0))
        losses.append(abs(min(change, 0.0)))
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    if avg_loss < 1e-9:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - 100.0 / (1.0 + rs)


def _score_symbol(closes: list[float], highs: list[float], lows: list[float]) -> SymbolInternals | None:
    if not closes or len(closes) < 55:
        return None
    current = closes[-1]
    if current <= 0:
        return None

    sma20 = _sma(closes, 20)
    sma50 = _sma(closes, 50)
    sma200 = _sma(closes, 200)

    above20 = current > sma20 if sma20 else False
    above50 = current > sma50 if sma50 else False
    above200 = current > sma200 if sma200 else False

    rsi_val = _rsi(closes, 14)

    roc20 = (current / closes[-21] - 1.0) * 100.0 if len(closes) > 20 and closes[-21] > 0 else 0.0

    # 52-bar new high/low
    window = min(52, len(highs))
    high52 = max(highs[-window:])
    low52 = min(lows[-window:])
    new_high = highs[-1] >= high52 * 0.995
    new_low = lows[-1] <= low52 * 1.005

    # Score components
    trend_pts = sum([above20, above50, above200])
    trend_score = trend_pts / 3 * 100.0
    rsi_score = min(100.0, max(0.0, (rsi_val - 30) / 40 * 100.0))
    roc_score = max(0.0, min(100.0, 50.0 + roc20 * 5.0))
    nh_score = 80.0 if new_high else (20.0 if new_low else 50.0)

    score = trend_score * 0.35 + rsi_score * 0.25 + roc_score * 0.25 + nh_score * 0.15
    score = max(0.0, min(100.0, score))

    if score >= 65:
        trend = "bull"
    elif score <= 35:
        trend = "bear"
    else:
        trend = "neutral"

    return SymbolInternals(
        symbol="",  # filled by caller
        score=round(score, 1),
        above_sma20=above20,
        above_sma50=above50,
        above_sma200=above200,
        rsi=round(rsi_val, 1),
        roc20=round(roc20, 2),
        new_high_52=new_high,
        new_low_52=new_low,
        trend=trend,
    )


def analyze(bars_by_symbol: dict) -> MarketInternalsReport | None:
    """Compute market internals from a basket of symbol bar data.

    bars_by_symbol: dict mapping symbol str to list of bar objects
                    with .close, .high, .low attributes.
    """
    if not bars_by_symbol:
        return None

    results: list[SymbolInternals] = []

    for sym, bars in bars_by_symbol.items():
        if not bars or len(bars) < 55:
            continue
        try:
            closes = [float(b.close) for b in bars]
            highs = [float(b.high) for b in bars]
            lows = [float(b.low) for b in bars]
        except (AttributeError, TypeError, ValueError):
            continue

        si = _score_symbol(closes, highs, lows)
        if si is not None:
            # Attach symbol name
            results.append(SymbolInternals(
                symbol=sym,
                score=si.score,
                above_sma20=si.above_sma20,
                above_sma50=si.above_sma50,
                above_sma200=si.above_sma200,
                rsi=si.rsi,
                roc20=si.roc20,
                new_high_52=si.new_high_52,
                new_low_52=si.new_low_52,
                trend=si.trend,
            ))

    if not results:
        return None

    n = len(results)

    pct_sma20 = sum(1 for s in results if s.above_sma20) / n * 100.0
    pct_sma50 = sum(1 for s in results if s.above_sma50) / n * 100.0
    pct_sma200 = sum(1 for s in results if s.above_sma200) / n * 100.0
    pct_nh = sum(1 for s in results if s.new_high_52) / n * 100.0
    pct_nl = sum(1 for s in results if s.new_low_52) / n * 100.0

    bull_n = sum(1 for s in results if s.trend == "bull")
    bear_n = sum(1 for s in results if s.trend == "bear")
    neutral_n = n - bull_n - bear_n

    ad_ratio = bull_n / (bull_n + bear_n) if (bull_n + bear_n) > 0 else 0.5
    nh_total = sum(1 for s in results if s.new_high_52) + sum(1 for s in results if s.new_low_52)
    nh_nl_ratio = sum(1 for s in results if s.new_high_52) / nh_total if nh_total > 0 else 0.5

    composite = (
        pct_sma50 * 0.25
        + pct_sma20 * 0.20
        + ad_ratio * 100 * 0.25
        + nh_nl_ratio * 100 * 0.15
        + pct_sma200 * 0.15
    )
    composite = max(0.0, min(100.0, composite))

    if composite >= 70:
        health = "strong_bull"
    elif composite >= 60:
        health = "bull"
    elif composite <= 30:
        health = "strong_bear"
    elif composite <= 40:
        health = "bear"
    else:
        health = "neutral"

    verdict = (
        f"Market Internals ({n} symbols): {composite:.0f}/100 ({health.replace('_', ' ')}). "
        f"{pct_sma50:.0f}% above SMA-50, A/D ratio {ad_ratio:.2f}. "
        f"New highs: {pct_nh:.0f}%, new lows: {pct_nl:.0f}%."
    )

    return MarketInternalsReport(
        symbols_scored=n,
        symbols_bull=bull_n,
        symbols_bear=bear_n,
        symbols_neutral=neutral_n,
        pct_above_sma20=round(pct_sma20, 1),
        pct_above_sma50=round(pct_sma50, 1),
        pct_above_sma200=round(pct_sma200, 1),
        pct_new_highs=round(pct_nh, 1),
        pct_new_lows=round(pct_nl, 1),
        advance_decline_ratio=round(ad_ratio, 3),
        nh_nl_ratio=round(nh_nl_ratio, 3),
        composite_score=round(composite, 1),
        health_label=health,
        by_symbol=results,
        verdict=verdict,
    )
