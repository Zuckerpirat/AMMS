"""Multi-criteria symbol screener.

Screens a list of symbols against configurable filters and returns
a ranked list sorted by composite score.

Screening criteria (all optional, configurable):
  - RSI range (e.g. 40-60 for non-extreme, or >60 for momentum)
  - ROC minimum (e.g. >5% over 20 bars)
  - ATR expansion (current ATR > avg ATR × threshold)
  - Price above / below SMA
  - Volume above average

Each passing criterion adds to a composite score (0-100).
Results sorted by score descending.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ScreenResult:
    symbol: str
    score: float          # 0-100 composite
    price: float
    rsi: float | None
    roc_20: float | None  # 20-bar Rate of Change %
    atr_ratio: float | None  # current ATR / avg ATR
    above_sma20: bool | None
    above_sma50: bool | None
    volume_ratio: float | None  # current vol / avg vol
    passed_filters: int   # how many filters passed
    total_filters: int
    verdict: str


@dataclass(frozen=True)
class ScreenReport:
    results: list[ScreenResult]   # sorted by score desc
    n_screened: int
    n_passed: int
    filters_applied: list[str]


def _rsi(closes: list[float], period: int = 14) -> float | None:
    if len(closes) < period + 1:
        return None
    deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    gains = [max(d, 0) for d in deltas[-period:]]
    losses = [abs(min(d, 0)) for d in deltas[-period:]]
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - 100 / (1 + rs)


def _sma(closes: list[float], period: int) -> float | None:
    if len(closes) < period:
        return None
    return sum(closes[-period:]) / period


def _atr(bars: list, period: int = 14) -> tuple[float | None, float | None]:
    """Returns (current_atr, avg_atr_over_2×period)."""
    if len(bars) < period + 1:
        return None, None
    trs = []
    for i in range(1, len(bars)):
        high = float(bars[i].high)
        low = float(bars[i].low)
        prev_close = float(bars[i - 1].close)
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        trs.append(tr)
    if len(trs) < period:
        return None, None
    current_atr = sum(trs[-period:]) / period
    long_period = min(period * 2, len(trs))
    avg_atr = sum(trs[-long_period:]) / long_period
    return current_atr, avg_atr


def screen(
    symbols_bars: dict[str, list],
    *,
    rsi_min: float | None = None,
    rsi_max: float | None = None,
    roc_min: float | None = None,
    roc_max: float | None = None,
    require_above_sma20: bool | None = None,
    require_above_sma50: bool | None = None,
    atr_expansion_min: float | None = None,
    volume_ratio_min: float | None = None,
    min_score: float = 0.0,
) -> ScreenReport | None:
    """Screen symbols against multiple criteria.

    symbols_bars: {symbol: list_of_bars}  (bars need .high .low .close .volume)
    rsi_min/max: RSI filter range (None = no filter)
    roc_min/max: 20-bar ROC % filter
    require_above_sma20/50: True=must be above, False=must be below, None=no filter
    atr_expansion_min: current ATR / avg ATR must be >= this (e.g. 1.2 for 20% expansion)
    volume_ratio_min: current vol / avg vol must be >= this
    min_score: minimum composite score to include in results

    Returns None if no symbols provided.
    """
    if not symbols_bars:
        return None

    filters_applied: list[str] = []
    if rsi_min is not None:
        filters_applied.append(f"RSI>={rsi_min:.0f}")
    if rsi_max is not None:
        filters_applied.append(f"RSI<={rsi_max:.0f}")
    if roc_min is not None:
        filters_applied.append(f"ROC20>={roc_min:.1f}%%")
    if roc_max is not None:
        filters_applied.append(f"ROC20<={roc_max:.1f}%%")
    if require_above_sma20 is True:
        filters_applied.append("above SMA20")
    elif require_above_sma20 is False:
        filters_applied.append("below SMA20")
    if require_above_sma50 is True:
        filters_applied.append("above SMA50")
    if atr_expansion_min is not None:
        filters_applied.append(f"ATR ratio>={atr_expansion_min:.2f}")
    if volume_ratio_min is not None:
        filters_applied.append(f"vol ratio>={volume_ratio_min:.2f}")

    total_filters = len(filters_applied)
    results: list[ScreenResult] = []

    for sym, bars in symbols_bars.items():
        if not bars or len(bars) < 5:
            continue

        try:
            closes = [float(b.close) for b in bars]
            volumes = [float(b.volume) for b in bars if hasattr(b, "volume")]
        except Exception:
            continue

        price = closes[-1]
        n = len(closes)

        # RSI
        rsi_val = _rsi(closes)

        # ROC-20
        roc_20 = None
        if n >= 21:
            roc_20 = (closes[-1] / closes[-21] - 1) * 100

        # SMA
        sma20_val = _sma(closes, 20)
        sma50_val = _sma(closes, 50)
        above_sma20 = (price > sma20_val) if sma20_val is not None else None
        above_sma50 = (price > sma50_val) if sma50_val is not None else None

        # ATR
        cur_atr, avg_atr = _atr(bars)
        atr_ratio = (cur_atr / avg_atr) if (cur_atr and avg_atr and avg_atr > 0) else None

        # Volume ratio
        vol_ratio = None
        if volumes and len(volumes) >= 5:
            avg_vol = sum(volumes[-20:]) / len(volumes[-20:])
            vol_ratio = volumes[-1] / avg_vol if avg_vol > 0 else None

        # Check filters and score
        passed = 0
        score = 0.0

        def check(condition: bool, weight: float = 1.0):
            nonlocal passed, score
            if condition:
                passed += 1
                score += weight

        if rsi_val is not None and rsi_min is not None:
            check(rsi_val >= rsi_min)
        if rsi_val is not None and rsi_max is not None:
            check(rsi_val <= rsi_max)
        if roc_20 is not None and roc_min is not None:
            check(roc_20 >= roc_min)
        if roc_20 is not None and roc_max is not None:
            check(roc_20 <= roc_max)
        if require_above_sma20 is not None and above_sma20 is not None:
            check(above_sma20 == require_above_sma20)
        if require_above_sma50 is not None and above_sma50 is not None:
            check(above_sma50 == require_above_sma50)
        if atr_ratio is not None and atr_expansion_min is not None:
            check(atr_ratio >= atr_expansion_min)
        if vol_ratio is not None and volume_ratio_min is not None:
            check(vol_ratio >= volume_ratio_min)

        composite = (score / max(total_filters, 1)) * 100 if total_filters > 0 else 50.0

        if composite < min_score:
            continue

        if composite >= 80:
            verdict = "Strong match"
        elif composite >= 60:
            verdict = "Good match"
        elif composite >= 40:
            verdict = "Partial match"
        else:
            verdict = "Weak match"

        results.append(ScreenResult(
            symbol=sym,
            score=round(composite, 1),
            price=round(price, 2),
            rsi=round(rsi_val, 1) if rsi_val is not None else None,
            roc_20=round(roc_20, 2) if roc_20 is not None else None,
            atr_ratio=round(atr_ratio, 3) if atr_ratio is not None else None,
            above_sma20=above_sma20,
            above_sma50=above_sma50,
            volume_ratio=round(vol_ratio, 2) if vol_ratio is not None else None,
            passed_filters=passed,
            total_filters=total_filters,
            verdict=verdict,
        ))

    results.sort(key=lambda r: -r.score)
    n_passed = sum(1 for r in results if r.passed_filters == total_filters)

    return ScreenReport(
        results=results,
        n_screened=len(symbols_bars),
        n_passed=n_passed,
        filters_applied=filters_applied,
    )
