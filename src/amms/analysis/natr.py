"""Normalized ATR (NATR) Analyser.

ATR expressed as a percentage of the closing price, making volatility
comparable across symbols regardless of price level.

  TR(t)   = max(High - Low, |High - Close[t-1]|, |Low - Close[t-1]|)
  ATR(t)  = Wilder SMMA of TR over `period` bars
  NATR(t) = ATR(t) / Close(t) × 100

Also computes:
  - Historical NATR percentile (where is current vol vs recent history)
  - Volatility regime: low / normal / elevated / high
  - NATR slope: rising or falling
  - Multi-period NATR (fast / slow) for regime comparison
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NATRReport:
    symbol: str

    natr: float             # current NATR (%)
    atr: float              # raw ATR (price units)
    natr_fast: float        # NATR with fast_period
    natr_slow: float        # NATR with slow_period

    natr_percentile: float  # vs recent history (0-100)
    avg_natr: float         # mean NATR over history
    natr_std: float         # std dev over history

    regime: str             # "low", "normal", "elevated", "high"
    slope_up: bool          # NATR rising vs 3 bars ago
    compression: bool       # natr_fast < natr_slow (recent compression)

    score: float            # 0-100 (0=calm, 100=extreme volatility)
    signal: str             # "calm", "normal", "active", "volatile", "extreme"

    history: list[float]    # recent NATR values
    bars_used: int
    verdict: str


def _smma(values: list[float], period: int) -> list[float]:
    """Wilder Smoothed Moving Average (SMMA) series."""
    if len(values) < period:
        return []
    seed = sum(values[:period]) / period
    out = [seed]
    for v in values[period:]:
        out.append((out[-1] * (period - 1) + v) / period)
    return out


def _tr_series(highs: list[float], lows: list[float], closes: list[float]) -> list[float]:
    """True Range series (length = len(closes) - 1 using previous close)."""
    trs = []
    for i in range(1, len(closes)):
        hl  = highs[i] - lows[i]
        hpc = abs(highs[i] - closes[i - 1])
        lpc = abs(lows[i] - closes[i - 1])
        trs.append(max(hl, hpc, lpc))
    return trs


def _mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def _std(vals: list[float]) -> float:
    if len(vals) < 2:
        return 0.0
    m = _mean(vals)
    return (sum((v - m) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5


def _natr_at(atr_val: float, close: float) -> float:
    return atr_val / close * 100.0 if close > 1e-9 else 0.0


def analyze(
    bars: list,
    *,
    symbol: str = "",
    period: int = 14,
    fast_period: int = 7,
    slow_period: int = 28,
    history: int = 30,
) -> NATRReport | None:
    """Compute Normalized ATR.

    bars: bar objects with .high, .low, .close attributes.
    period: main ATR period (Wilder SMMA).
    fast_period / slow_period: secondary periods for regime comparison.
    """
    min_bars = slow_period + history + 5
    if not bars or len(bars) < min_bars:
        return None

    try:
        highs  = [float(b.high)  for b in bars]
        lows   = [float(b.low)   for b in bars]
        closes = [float(b.close) for b in bars]
    except (AttributeError, TypeError, ValueError):
        return None

    if closes[-1] <= 0:
        return None

    trs = _tr_series(highs, lows, closes)
    if not trs:
        return None

    # Compute ATR series using SMMA
    atr_series  = _smma(trs, period)
    fast_series = _smma(trs, fast_period)
    slow_series = _smma(trs, slow_period)

    if not atr_series or not fast_series or not slow_series:
        return None

    # Align series to same length (slow is shortest)
    # Each SMMA of length k started from trs index (k-1),
    # so slow_series[-1] aligns with atr_series at offset (slow_period - period)
    if len(atr_series) < len(slow_series):
        return None

    # Current values (last bar)
    # Close index aligns: trs has len n-1, atr_series has len n-1-period+1 = n-period
    # Last atr_series element corresponds to closes[-1]
    n_closes = len(closes)
    atr_val  = atr_series[-1]
    fast_val = fast_series[-1]
    slow_val = slow_series[-1]
    cur_close = closes[-1]

    natr_val  = _natr_at(atr_val, cur_close)
    natr_fast = _natr_at(fast_val, cur_close)
    natr_slow = _natr_at(slow_val, cur_close)

    # NATR history: use closes aligned to atr_series
    # atr_series[i] corresponds to closes[period + i] (0-indexed)
    offset = period  # first atr value corresponds to closes[period]
    natr_hist_raw: list[float] = []
    for i in range(len(atr_series)):
        close_idx = offset + i
        if close_idx < len(closes) and closes[close_idx] > 1e-9:
            natr_hist_raw.append(atr_series[i] / closes[close_idx] * 100.0)

    if len(natr_hist_raw) < history + 3:
        return None

    recent_natr = natr_hist_raw[-history:]
    avg = _mean(recent_natr)
    std = _std(recent_natr)
    pctile = sum(1 for v in recent_natr[:-1] if v < natr_val) / max(len(recent_natr) - 1, 1) * 100.0

    # Slope: compare to 3 bars ago
    natr_3ago = natr_hist_raw[-4] if len(natr_hist_raw) >= 4 else natr_hist_raw[0]
    slope_up = natr_val > natr_3ago

    compression = natr_fast < natr_slow

    # Regime based on percentile
    if pctile >= 85:
        regime = "high"
    elif pctile >= 65:
        regime = "elevated"
    elif pctile <= 20:
        regime = "low"
    else:
        regime = "normal"

    # Score: percentile rank as score
    score = pctile

    if score >= 85:
        signal = "extreme"
    elif score >= 65:
        signal = "volatile"
    elif score >= 40:
        signal = "active"
    elif score >= 20:
        signal = "normal"
    else:
        signal = "calm"

    compress_str = " Compression (fast<slow)." if compression else ""
    verdict = (
        f"NATR ({symbol}): {natr_val:.2f}% (ATR={atr_val:.4f}). "
        f"Regime: {regime} (pctile={pctile:.0f}%). "
        f"Signal: {signal}.{compress_str}"
    )

    return NATRReport(
        symbol=symbol,
        natr=round(natr_val, 4),
        atr=round(atr_val, 6),
        natr_fast=round(natr_fast, 4),
        natr_slow=round(natr_slow, 4),
        natr_percentile=round(pctile, 1),
        avg_natr=round(avg, 4),
        natr_std=round(std, 4),
        regime=regime,
        slope_up=slope_up,
        compression=compression,
        score=round(score, 1),
        signal=signal,
        history=[round(v, 4) for v in recent_natr],
        bars_used=n_closes,
        verdict=verdict,
    )
