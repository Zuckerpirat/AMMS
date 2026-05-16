"""Chaikin Money Flow (CMF) Analyser.

CMF measures the volume-weighted buying/selling pressure over a rolling
window. Positive CMF suggests accumulation (buyers in control), negative
suggests distribution (sellers in control).

Formula:
  Money Flow Multiplier (MFM) = ((close - low) - (high - close)) / (high - low)
  Money Flow Volume (MFV) = MFM × volume
  CMF = sum(MFV, n) / sum(volume, n)

CMF ranges from -1 to +1. Above 0.05 = buying pressure, below -0.05 = selling.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CMFSnapshot:
    bar_idx: int
    cmf: float
    mfm: float     # money flow multiplier for this bar
    volume: float


@dataclass(frozen=True)
class CMFReport:
    symbol: str
    period: int

    cmf: float              # current Chaikin Money Flow (-1 to +1)
    signal: str             # "strong_buy", "buy", "neutral", "sell", "strong_sell"
    signal_strength: float  # abs(cmf) capped at 1

    cmf_trend: str          # "improving", "worsening", "stable"
    trend_bars: int         # how many bars CMF has been in same direction

    avg_mfm: float          # average money flow multiplier
    avg_volume: float       # average volume in period

    above_zero_pct: float   # % of recent bars with CMF > 0
    zero_crossings: int     # number of sign changes in history

    history: list[CMFSnapshot]
    current_price: float
    bars_used: int
    verdict: str


def _mfm(high: float, low: float, close: float) -> float:
    """Money Flow Multiplier: -1 to +1."""
    rng = high - low
    if rng < 1e-9:
        return 0.0
    return ((close - low) - (high - close)) / rng


def analyze(
    bars: list,
    *,
    symbol: str = "",
    period: int = 20,
    history_bars: int = 40,
) -> CMFReport | None:
    """Compute Chaikin Money Flow from bar history.

    bars: bar objects with .high, .low, .close, .volume attributes.
    period: rolling window for CMF calculation.
    history_bars: number of historical CMF values to return.
    """
    if not bars or len(bars) < period + 2:
        return None

    try:
        highs = [float(b.high) for b in bars]
        lows = [float(b.low) for b in bars]
        closes = [float(b.close) for b in bars]
        volumes = [float(b.volume) for b in bars]
    except (AttributeError, TypeError, ValueError):
        return None

    current = closes[-1]
    if current <= 0:
        return None

    n = len(bars)

    # Per-bar MFM and MFV
    mfms = [_mfm(highs[i], lows[i], closes[i]) for i in range(n)]
    mfvs = [mfms[i] * volumes[i] for i in range(n)]

    # Rolling CMF
    cmf_series: list[float] = []
    vol_series: list[float] = []
    idx_series: list[int] = []

    for i in range(period - 1, n):
        sum_mfv = sum(mfvs[i - period + 1: i + 1])
        sum_vol = sum(volumes[i - period + 1: i + 1])
        cmf_val = sum_mfv / sum_vol if sum_vol > 0 else 0.0
        cmf_series.append(cmf_val)
        vol_series.append(sum_vol / period)
        idx_series.append(i)

    if not cmf_series:
        return None

    cur_cmf = cmf_series[-1]

    # Signal
    if cur_cmf >= 0.15:
        signal = "strong_buy"
    elif cur_cmf >= 0.05:
        signal = "buy"
    elif cur_cmf <= -0.15:
        signal = "strong_sell"
    elif cur_cmf <= -0.05:
        signal = "sell"
    else:
        signal = "neutral"

    strength = min(1.0, abs(cur_cmf))

    # CMF trend: compare last quarter vs prior quarter
    qlen = max(1, len(cmf_series) // 4)
    recent_avg = sum(cmf_series[-qlen:]) / qlen
    earlier_avg = sum(cmf_series[-2 * qlen:-qlen]) / qlen if len(cmf_series) >= 2 * qlen else recent_avg

    if recent_avg > earlier_avg + 0.02:
        cmf_trend = "improving"
    elif recent_avg < earlier_avg - 0.02:
        cmf_trend = "worsening"
    else:
        cmf_trend = "stable"

    # Trend bars: consecutive bars in same direction
    trend_bars = 0
    rising = cur_cmf >= 0
    for k in range(len(cmf_series) - 2, -1, -1):
        was_rising = cmf_series[k] >= 0
        if was_rising != rising:
            break
        trend_bars += 1

    # Above zero percentage
    above_zero_pct = sum(1 for c in cmf_series if c > 0) / len(cmf_series) * 100.0

    # Zero crossings
    crossings = sum(
        1 for k in range(1, len(cmf_series))
        if (cmf_series[k] >= 0) != (cmf_series[k - 1] >= 0)
    )

    # Avg MFM and volume in current period
    window_mfms = mfms[-period:]
    avg_mfm_val = sum(window_mfms) / len(window_mfms)
    avg_vol = vol_series[-1] if vol_series else 0.0

    # History snapshots
    history: list[CMFSnapshot] = []
    n_hist = min(history_bars, len(cmf_series))
    for k in range(len(cmf_series) - n_hist, len(cmf_series)):
        i = idx_series[k]
        history.append(CMFSnapshot(
            bar_idx=i,
            cmf=round(cmf_series[k], 4),
            mfm=round(mfms[i], 4),
            volume=round(volumes[i], 0),
        ))

    # Verdict
    if signal in {"strong_buy", "buy"}:
        action = f"Accumulation ({signal.replace('_', ' ')}) — buyers in control"
    elif signal in {"strong_sell", "sell"}:
        action = f"Distribution ({signal.replace('_', ' ')}) — sellers in control"
    else:
        action = "Neutral money flow — no clear direction"

    verdict = (
        f"CMF({period}): {cur_cmf:+.3f} — {action}. "
        f"Trend: {cmf_trend}. {above_zero_pct:.0f}% of bars positive."
    )

    return CMFReport(
        symbol=symbol,
        period=period,
        cmf=round(cur_cmf, 4),
        signal=signal,
        signal_strength=round(strength, 3),
        cmf_trend=cmf_trend,
        trend_bars=trend_bars,
        avg_mfm=round(avg_mfm_val, 4),
        avg_volume=round(avg_vol, 0),
        above_zero_pct=round(above_zero_pct, 1),
        zero_crossings=crossings,
        history=history,
        current_price=round(current, 4),
        bars_used=len(bars),
        verdict=verdict,
    )
