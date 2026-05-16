"""Watchlist Opportunity Scorer.

Ranks watchlist symbols by a composite opportunity score using
technical signals derived from bar data. Each criterion contributes
to a 0-100 score. The score helps prioritise which symbols to research
or add as positions.

Scoring criteria (equal weight unless noted):
  1. Momentum (20%): 20-day rate of change — positive = bullish
  2. RSI positioning (20%): RSI < 35 = oversold opportunity, > 65 = extended
  3. Volume trend (15%): 5-day avg volume vs 20-day avg — rising volume confirms
  4. Price vs SMA-50 (15%): how close to / above SMA-50
  5. Volatility score (15%): lower ATR% preferred for risk-adjusted entries
  6. Trend quality (15%): EMA slope consistency, less reversal = higher score

Each criterion produces a 0-100 sub-score. Final = weighted average.
Grade: A (≥80), B (≥65), C (≥50), D (≥35), F (<35).
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class WatchlistScore:
    symbol: str
    total_score: float      # 0-100 composite
    grade: str              # A/B/C/D/F
    momentum_score: float   # 0-100
    rsi_score: float        # 0-100
    volume_score: float     # 0-100
    sma_score: float        # 0-100
    vol_score: float        # 0-100 (lower volatility = higher score here)
    trend_score: float      # 0-100
    current_price: float
    rsi: float
    roc_20: float           # 20-day rate of change %
    bars_used: int
    summary: str            # one-line verdict


@dataclass(frozen=True)
class WatchlistScorerReport:
    scores: list[WatchlistScore]   # sorted best first
    top_pick: WatchlistScore | None
    n_symbols: int
    n_graded: int


def _sma(closes: list[float], period: int) -> float | None:
    if len(closes) < period:
        return None
    return sum(closes[-period:]) / period


def _ema_series(closes: list[float], period: int) -> list[float]:
    if len(closes) < period:
        return []
    k = 2.0 / (period + 1)
    result = [sum(closes[:period]) / period]
    for v in closes[period:]:
        result.append(v * k + result[-1] * (1 - k))
    return result


def _rsi(closes: list[float], period: int = 14) -> float | None:
    if len(closes) < period + 1:
        return None
    changes = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    gains = [max(0.0, c) for c in changes[-period:]]
    losses = [abs(min(0.0, c)) for c in changes[-period:]]
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def _atr_pct(bars: list, period: int = 14) -> float:
    if len(bars) < 2:
        return 0.0
    trs = []
    for i in range(1, min(period + 1, len(bars))):
        h = float(bars[-i].high)
        l = float(bars[-i].low)
        pc = float(bars[-i - 1].close)
        trs.append(max(h - l, abs(h - pc), abs(l - pc)))
    atr = sum(trs) / len(trs) if trs else 0.0
    last_close = float(bars[-1].close)
    return atr / last_close * 100.0 if last_close > 0 else 0.0


def score_symbol(bars: list, *, symbol: str = "") -> WatchlistScore | None:
    """Score a single symbol from its bar history."""
    if not bars or len(bars) < 25:
        return None

    try:
        closes = [float(b.close) for b in bars]
        vols = [float(b.volume) if hasattr(b, "volume") else 1.0 for b in bars]
        current = closes[-1]
    except Exception:
        return None

    if current <= 0:
        return None

    # 1. Momentum: 20-day ROC
    roc_20 = (closes[-1] / closes[-21] - 1.0) * 100.0 if len(closes) >= 21 and closes[-21] > 0 else 0.0
    mom_score = min(100.0, max(0.0, 50.0 + roc_20 * 5))  # +10% ROC → score 100

    # 2. RSI score
    rsi_val = _rsi(closes) or 50.0
    if rsi_val < 30:
        rsi_score = 90.0  # oversold = opportunity
    elif rsi_val < 40:
        rsi_score = 70.0
    elif rsi_val < 50:
        rsi_score = 55.0
    elif rsi_val < 60:
        rsi_score = 45.0
    elif rsi_val < 70:
        rsi_score = 30.0
    else:
        rsi_score = 15.0  # overbought

    # 3. Volume trend: 5-day vs 20-day avg
    vol_5 = sum(vols[-5:]) / 5.0 if len(vols) >= 5 else 1.0
    vol_20 = sum(vols[-20:]) / 20.0 if len(vols) >= 20 else vol_5
    vol_ratio = vol_5 / vol_20 if vol_20 > 0 else 1.0
    volume_score = min(100.0, max(0.0, 50.0 + (vol_ratio - 1.0) * 50.0))

    # 4. Price vs SMA-50
    sma50 = _sma(closes, 50)
    if sma50:
        pct_from_sma = (current - sma50) / sma50 * 100.0
        if pct_from_sma > 5:
            sma_score = 80.0
        elif pct_from_sma > 0:
            sma_score = 65.0
        elif pct_from_sma > -3:
            sma_score = 50.0  # near SMA = potential bounce
        elif pct_from_sma > -8:
            sma_score = 35.0
        else:
            sma_score = 20.0
    else:
        sma_score = 50.0

    # 5. Volatility score (lower ATR% = less risky entry)
    atr = _atr_pct(bars)
    if atr < 1.0:
        vol_score = 80.0
    elif atr < 2.0:
        vol_score = 65.0
    elif atr < 3.0:
        vol_score = 50.0
    elif atr < 5.0:
        vol_score = 35.0
    else:
        vol_score = 20.0

    # 6. Trend quality: EMA-20 slope consistency
    ema20 = _ema_series(closes, 20)
    if len(ema20) >= 5:
        recent_slopes = [(ema20[i] - ema20[i - 1]) / ema20[i - 1] * 100.0
                         for i in range(-5, 0) if ema20[i - 1] > 0]
        if recent_slopes:
            all_positive = all(s > 0 for s in recent_slopes)
            all_negative = all(s < 0 for s in recent_slopes)
            if all_positive:
                trend_score = 80.0
            elif all_negative:
                trend_score = 30.0
            else:
                trend_score = 50.0
        else:
            trend_score = 50.0
    else:
        trend_score = 50.0

    # Weighted composite
    total = (
        mom_score * 0.20 +
        rsi_score * 0.20 +
        volume_score * 0.15 +
        sma_score * 0.15 +
        vol_score * 0.15 +
        trend_score * 0.15
    )

    grade = "A" if total >= 80 else "B" if total >= 65 else "C" if total >= 50 else "D" if total >= 35 else "F"

    summary = (
        f"Score {total:.0f}/100 ({grade}) — "
        f"ROC {roc_20:+.1f}%, RSI {rsi_val:.0f}, "
        f"ATR {atr:.1f}%, vol ratio {vol_ratio:.2f}×"
    )

    return WatchlistScore(
        symbol=symbol,
        total_score=round(total, 1),
        grade=grade,
        momentum_score=round(mom_score, 1),
        rsi_score=round(rsi_score, 1),
        volume_score=round(volume_score, 1),
        sma_score=round(sma_score, 1),
        vol_score=round(vol_score, 1),
        trend_score=round(trend_score, 1),
        current_price=round(current, 2),
        rsi=round(rsi_val, 1),
        roc_20=round(roc_20, 2),
        bars_used=len(bars),
        summary=summary,
    )


def score_many(bars_by_symbol: dict[str, list]) -> WatchlistScorerReport:
    """Score all symbols and return ranked report."""
    scores: list[WatchlistScore] = []
    for sym, bars in bars_by_symbol.items():
        s = score_symbol(bars, symbol=sym)
        if s is not None:
            scores.append(s)

    scores.sort(key=lambda s: -s.total_score)
    return WatchlistScorerReport(
        scores=scores,
        top_pick=scores[0] if scores else None,
        n_symbols=len(bars_by_symbol),
        n_graded=len(scores),
    )
