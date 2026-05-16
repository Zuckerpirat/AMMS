"""Multi-Timeframe Momentum Analyser.

Compares momentum across multiple lookback windows (short/medium/long term)
to classify the overall momentum regime and detect alignment or divergence.

Momentum windows:
  - Short:  5 bars  (very recent)
  - Medium: 20 bars (standard)
  - Long:   50 bars
  - Trend:  100 bars (structural)

Each window produces a Rate-of-Change (ROC), an RSI, and an EMA slope sign.
Alignment = all windows agree on direction.
Divergence = short and long are in opposite directions.

Score:
  +1 per window where ROC > 0
  -1 per window where ROC < 0
  Total: -4 to +4 → maps to regime
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class MomentumWindow:
    bars: int           # lookback window
    roc: float          # rate of change %
    rsi: float          # RSI for this window
    ema_slope_pct: float  # slope of last EMA value as % of EMA
    direction: str      # "bullish", "bearish", "neutral"
    score: int          # +1 / 0 / -1


@dataclass(frozen=True)
class MTFMomentumReport:
    symbol: str
    windows: list[MomentumWindow]   # short → trend
    total_score: int                # -4 to +4
    regime: str                     # "strong_bull", "bull", "neutral", "bear", "strong_bear"
    aligned: bool                   # all windows same direction
    divergence: bool                # short vs long in opposite directions
    current_price: float
    short_roc: float                # 5-bar ROC
    medium_roc: float               # 20-bar ROC
    long_roc: float                 # 50-bar ROC
    trend_roc: float                # 100-bar ROC
    bars_used: int
    verdict: str


_WINDOW_SIZES = [5, 20, 50, 100]


def _roc(closes: list[float], period: int) -> float:
    if len(closes) < period + 1 or closes[-(period + 1)] <= 0:
        return 0.0
    return (closes[-1] / closes[-(period + 1)] - 1.0) * 100.0


def _rsi(closes: list[float], period: int = 14) -> float:
    if len(closes) < period + 1:
        return 50.0
    changes = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    last = changes[-period:]
    gains = [max(0.0, c) for c in last]
    losses = [abs(min(0.0, c)) for c in last]
    avg_g = sum(gains) / period
    avg_l = sum(losses) / period
    if avg_l == 0:
        return 100.0
    return 100.0 - 100.0 / (1.0 + avg_g / avg_l)


def _ema_slope(closes: list[float], period: int) -> float:
    """EMA slope as % of last EMA value."""
    if len(closes) < period + 2:
        return 0.0
    k = 2.0 / (period + 1)
    ema = sum(closes[:period]) / period
    for v in closes[period:]:
        ema = v * k + ema * (1 - k)
    # Compute previous EMA
    k2 = k
    ema_prev = sum(closes[:period]) / period
    for v in closes[period:-1]:
        ema_prev = v * k2 + ema_prev * (1 - k2)
    if ema_prev <= 0:
        return 0.0
    return (ema - ema_prev) / ema_prev * 100.0


def _classify(roc: float, rsi: float) -> tuple[str, int]:
    """Return (direction, score) from ROC and RSI."""
    if roc > 1.0 and rsi > 55:
        return "bullish", 1
    elif roc < -1.0 and rsi < 45:
        return "bearish", -1
    elif roc > 0.5:
        return "bullish", 1
    elif roc < -0.5:
        return "bearish", -1
    return "neutral", 0


def analyze(bars: list, *, symbol: str = "") -> MTFMomentumReport | None:
    """Analyse multi-timeframe momentum from bar history.

    bars: list of bar objects with .close attribute.
    symbol: optional label.
    """
    if not bars or len(bars) < _WINDOW_SIZES[-1] + 2:
        return None

    try:
        closes = [float(b.close) for b in bars]
    except Exception:
        return None

    current = closes[-1]
    if current <= 0:
        return None

    windows: list[MomentumWindow] = []
    for w in _WINDOW_SIZES:
        if len(closes) < w + 1:
            roc_val = 0.0
            rsi_val = 50.0
            slope = 0.0
        else:
            roc_val = _roc(closes, w)
            rsi_val = _rsi(closes[-w - 15:], period=min(14, w))
            slope = _ema_slope(closes, min(w, len(closes) - 1))

        direction, score = _classify(roc_val, rsi_val)
        windows.append(MomentumWindow(
            bars=w,
            roc=round(roc_val, 3),
            rsi=round(rsi_val, 1),
            ema_slope_pct=round(slope, 4),
            direction=direction,
            score=score,
        ))

    total = sum(w.score for w in windows)
    directions = [w.direction for w in windows]

    # Regime
    if total >= 3:
        regime = "strong_bull"
    elif total >= 1:
        regime = "bull"
    elif total <= -3:
        regime = "strong_bear"
    elif total <= -1:
        regime = "bear"
    else:
        regime = "neutral"

    aligned = len(set(directions)) == 1
    divergence = (windows[0].direction == "bullish" and windows[-1].direction == "bearish") or \
                 (windows[0].direction == "bearish" and windows[-1].direction == "bullish")

    # Verdict
    parts = [f"regime: {regime} (score {total:+d}/4)"]
    if aligned:
        parts.append(f"all {len(windows)} timeframes aligned {windows[0].direction}")
    elif divergence:
        parts.append(f"divergence: short {windows[0].direction}, long {windows[-1].direction}")
    rocs = [w.roc for w in windows]
    parts.append(f"ROC 5/20/50/100: {rocs[0]:+.1f}% / {rocs[1]:+.1f}% / {rocs[2]:+.1f}% / {rocs[3]:+.1f}%")

    verdict = "MTF momentum: " + "; ".join(parts) + "."

    return MTFMomentumReport(
        symbol=symbol,
        windows=windows,
        total_score=total,
        regime=regime,
        aligned=aligned,
        divergence=divergence,
        current_price=round(current, 2),
        short_roc=windows[0].roc,
        medium_roc=windows[1].roc,
        long_roc=windows[2].roc,
        trend_roc=windows[3].roc,
        bars_used=len(bars),
        verdict=verdict,
    )
