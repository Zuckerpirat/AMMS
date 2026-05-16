"""TRIX and Know Sure Thing (KST) Oscillators.

Both indicators use smoothed Rate-of-Change to measure momentum.

TRIX:
  Triple-smoothed EMA; TRIX = 1-bar ROC of triple EMA (%)
  Signal line = EMA(TRIX, signal_period)
  Above zero = positive momentum; below zero = negative momentum
  Histogram = TRIX - Signal

KST (Know Sure Thing, Martin Pring):
  Weighted sum of four smoothed ROC values at different periods
  KST = (RCMA1×1) + (RCMA2×2) + (RCMA3×3) + (RCMA4×4)
    where RCMAi = SMA(ROC(roc_period_i), sma_period_i)
  Signal = SMA(KST, 9)
  Above zero = bullish; below zero = bearish
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class TrixKSTReport:
    symbol: str

    # TRIX
    trix: float             # current TRIX value (%)
    trix_signal: float      # signal line (EMA of TRIX)
    trix_histogram: float   # trix - signal
    trix_bullish: bool      # trix > 0
    trix_cross_up: bool     # trix crossed above signal
    trix_cross_down: bool

    # KST
    kst: float              # current KST value
    kst_signal: float       # 9-bar SMA of KST
    kst_histogram: float    # kst - signal
    kst_bullish: bool       # kst > 0
    kst_cross_up: bool      # kst crossed above signal
    kst_cross_down: bool

    # Composite
    score: float            # -100 to +100
    signal: str             # "strong_bull", "bull", "neutral", "bear", "strong_bear"

    current_price: float
    bars_used: int
    verdict: str


def _ema(series: list[float], period: int) -> list[float]:
    if len(series) < period:
        return []
    k = 2.0 / (period + 1)
    val = sum(series[:period]) / period
    result = [val]
    for x in series[period:]:
        val = x * k + val * (1 - k)
        result.append(val)
    return result


def _sma(series: list[float], period: int) -> list[float]:
    if len(series) < period:
        return []
    return [
        sum(series[i:i + period]) / period
        for i in range(len(series) - period + 1)
    ]


def _roc(series: list[float], period: int) -> list[float]:
    """Rate of change as percentage."""
    result = []
    for i in range(period, len(series)):
        prev = series[i - period]
        if prev > 1e-9:
            result.append((series[i] / prev - 1.0) * 100.0)
        else:
            result.append(0.0)
    return result


def _trix_series(closes: list[float], period: int) -> list[float]:
    """Triple EMA, then 1-period ROC."""
    e1 = _ema(closes, period)
    e2 = _ema(e1, period)
    e3 = _ema(e2, period)
    return _roc(e3, 1)


def _kst_rcma(closes: list[float], roc_period: int, sma_period: int) -> list[float]:
    roc_vals = _roc(closes, roc_period)
    return _sma(roc_vals, sma_period)


def analyze(
    bars: list,
    *,
    symbol: str = "",
    trix_period: int = 18,
    trix_signal_period: int = 9,
    # KST default periods (Pring's short-term daily)
    kst_roc_periods: tuple[int, int, int, int] = (10, 13, 14, 15),
    kst_sma_periods: tuple[int, int, int, int] = (10, 13, 14, 15),
    kst_signal_period: int = 9,
) -> TrixKSTReport | None:
    """Compute TRIX and KST momentum oscillators.

    bars: bar objects with .close attribute.
    """
    # Min bars = 3 * trix_period (for triple EMA) + trix_signal_period + 5
    min_bars = 3 * trix_period + trix_signal_period + max(kst_roc_periods) + max(kst_sma_periods) + 5
    if not bars or len(bars) < min_bars:
        return None

    try:
        closes = [float(b.close) for b in bars]
    except (AttributeError, TypeError, ValueError):
        return None

    current = closes[-1]
    if current <= 0:
        return None

    # TRIX
    trix_vals = _trix_series(closes, trix_period)
    if len(trix_vals) < trix_signal_period + 2:
        return None

    trix_signal_vals = _ema(trix_vals, trix_signal_period)
    if not trix_signal_vals:
        return None

    cur_trix = trix_vals[-1]
    cur_trix_sig = trix_signal_vals[-1]
    trix_hist = cur_trix - cur_trix_sig

    # TRIX cross: compare last 2 values
    if len(trix_vals) >= 2 and len(trix_signal_vals) >= 2:
        prev_trix = trix_vals[-2]
        prev_sig  = trix_signal_vals[-2]
        trix_cross_up   = cur_trix >= cur_trix_sig and prev_trix < prev_sig
        trix_cross_down = cur_trix <= cur_trix_sig and prev_trix > prev_sig
    else:
        trix_cross_up = trix_cross_down = False

    # KST
    rcma_series = []
    for i, (rp, sp) in enumerate(zip(kst_roc_periods, kst_sma_periods)):
        rcma = _kst_rcma(closes, rp, sp)
        rcma_series.append(rcma)

    # Align lengths — take from end
    min_len = min(len(s) for s in rcma_series)
    if min_len < kst_signal_period + 2:
        return None

    rcma_aligned = [s[-min_len:] for s in rcma_series]
    weights = [1, 2, 3, 4]
    kst_vals = [
        sum(w * rcma_aligned[i][j] for i, w in enumerate(weights))
        for j in range(min_len)
    ]

    kst_signal_vals = _sma(kst_vals, kst_signal_period)
    if not kst_signal_vals:
        return None

    cur_kst = kst_vals[-1]
    cur_kst_sig = kst_signal_vals[-1]
    kst_hist = cur_kst - cur_kst_sig

    # KST cross
    if len(kst_vals) >= 2 and len(kst_signal_vals) >= 2:
        prev_kst  = kst_vals[-2]
        prev_ksig = kst_signal_vals[-2]
        kst_cross_up   = cur_kst >= cur_kst_sig and prev_kst < prev_ksig
        kst_cross_down = cur_kst <= cur_kst_sig and prev_kst > prev_ksig
    else:
        kst_cross_up = kst_cross_down = False

    # Composite score
    # TRIX: normalise by max abs trix in recent window
    trix_window = trix_vals[-20:] if len(trix_vals) >= 20 else trix_vals
    trix_max = max(abs(v) for v in trix_window) if trix_window else 1.0
    trix_norm = max(-100.0, min(100.0, cur_trix / trix_max * 100.0)) if trix_max > 1e-9 else 0.0

    kst_window = kst_vals[-20:] if len(kst_vals) >= 20 else kst_vals
    kst_max = max(abs(v) for v in kst_window) if kst_window else 1.0
    kst_norm = max(-100.0, min(100.0, cur_kst / kst_max * 100.0)) if kst_max > 1e-9 else 0.0

    score = trix_norm * 0.5 + kst_norm * 0.5
    score = max(-100.0, min(100.0, score))

    if score >= 60:
        signal = "strong_bull"
    elif score >= 20:
        signal = "bull"
    elif score <= -60:
        signal = "strong_bear"
    elif score <= -20:
        signal = "bear"
    else:
        signal = "neutral"

    verdict = (
        f"TRIX/KST ({symbol}): {signal.replace('_', ' ')}. "
        f"TRIX={cur_trix:+.4f}% (hist {trix_hist:+.4f}), "
        f"KST={cur_kst:+.2f} (hist {kst_hist:+.2f})."
    )
    if trix_cross_up:
        verdict += " TRIX bullish cross."
    if trix_cross_down:
        verdict += " TRIX bearish cross."
    if kst_cross_up:
        verdict += " KST bullish cross."
    if kst_cross_down:
        verdict += " KST bearish cross."

    return TrixKSTReport(
        symbol=symbol,
        trix=round(cur_trix, 5),
        trix_signal=round(cur_trix_sig, 5),
        trix_histogram=round(trix_hist, 5),
        trix_bullish=cur_trix > 0,
        trix_cross_up=trix_cross_up,
        trix_cross_down=trix_cross_down,
        kst=round(cur_kst, 4),
        kst_signal=round(cur_kst_sig, 4),
        kst_histogram=round(kst_hist, 4),
        kst_bullish=cur_kst > 0,
        kst_cross_up=kst_cross_up,
        kst_cross_down=kst_cross_down,
        score=round(score, 2),
        signal=signal,
        current_price=round(current, 4),
        bars_used=len(bars),
        verdict=verdict,
    )
