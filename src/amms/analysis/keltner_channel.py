"""Keltner Channel Analyser.

Keltner Channels use an EMA as the middle band with ATR-based upper/lower
bands. They adapt to volatility changes more smoothly than Bollinger Bands.

Key uses:
  - Trend confirmation: price sustained above/below channel midline
  - Breakout detection: price closing outside the outer bands
  - Squeeze setup: when Bollinger Bands contract inside Keltner (volatility compression)
  - Channel riding: price bouncing between mid and outer band in a trend

Formulas:
  Middle: EMA(close, period)
  Upper:  EMA + mult × ATR(atr_period)
  Lower:  EMA - mult × ATR(atr_period)
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class KCSnapshot:
    bar_idx: int
    price: float
    upper: float
    middle: float
    lower: float
    position: float    # 0=lower band, 0.5=middle, 1=upper band


@dataclass(frozen=True)
class KeltnerReport:
    symbol: str
    period: int
    atr_period: int
    mult: float

    current_price: float
    upper: float
    middle: float
    lower: float
    channel_width: float        # upper - lower
    channel_width_pct: float    # channel_width / middle * 100

    price_position: float       # 0=at lower, 0.5=at middle, 1=at upper
    position_label: str         # "above_upper", "upper_half", "middle", "lower_half", "below_lower"

    breakout_up: bool           # price > upper band
    breakout_down: bool         # price < lower band
    trend_direction: str        # "up", "down", "sideways"
    trend_bars: int             # bars since last crossing of midline

    atr: float
    atr_pct: float              # ATR / price * 100

    history: list[KCSnapshot]
    bars_used: int
    verdict: str


def _ema(values: list[float], period: int) -> list[float]:
    if len(values) < period:
        return []
    k = 2.0 / (period + 1)
    result = [sum(values[:period]) / period]
    for v in values[period:]:
        result.append(v * k + result[-1] * (1 - k))
    return result


def _atr_series(highs: list[float], lows: list[float], closes: list[float], period: int) -> list[float]:
    """True Range then smoothed ATR (RMA / Wilder smoothing)."""
    if len(closes) < period + 1:
        return []
    trs = []
    for i in range(1, len(closes)):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )
        trs.append(tr)
    if len(trs) < period:
        return []
    # Wilder smoothing
    atr = [sum(trs[:period]) / period]
    for tr in trs[period:]:
        atr.append((atr[-1] * (period - 1) + tr) / period)
    return atr


def analyze(
    bars: list,
    *,
    symbol: str = "",
    period: int = 20,
    atr_period: int = 10,
    mult: float = 2.0,
    history_bars: int = 30,
) -> KeltnerReport | None:
    """Compute Keltner Channels from bar history.

    bars: bar objects with .high, .low, .close attributes.
    period: EMA period for midline.
    atr_period: ATR smoothing period.
    mult: ATR multiplier for band width.
    history_bars: number of historical snapshots to return.
    """
    min_bars = max(period, atr_period) + 5
    if not bars or len(bars) < min_bars:
        return None

    try:
        highs = [float(b.high) for b in bars]
        lows = [float(b.low) for b in bars]
        closes = [float(b.close) for b in bars]
    except (AttributeError, TypeError, ValueError):
        return None

    current = closes[-1]
    if current <= 0:
        return None

    # EMA of close
    ema_vals = _ema(closes, period)
    if not ema_vals:
        return None

    # ATR series
    atr_vals = _atr_series(highs, lows, closes, atr_period)
    if not atr_vals:
        return None

    # Align: ema_vals[0] corresponds to closes[period-1], atr_vals[0] to closes[atr_period]
    # We need overlapping indices
    # ema starts at index (period-1) in closes → ema_vals[k] ↔ closes[period-1+k]
    # atr starts at index (atr_period) in closes → atr_vals[k] ↔ closes[atr_period+k]
    # Overlap: start from max(period-1, atr_period)
    ema_offset = period - 1
    atr_offset = atr_period  # atr_vals[0] corresponds to closes[atr_period]

    overlap_start = max(ema_offset, atr_offset)
    ema_start_k = overlap_start - ema_offset
    atr_start_k = overlap_start - atr_offset

    n_overlap = min(len(ema_vals) - ema_start_k, len(atr_vals) - atr_start_k)
    if n_overlap <= 0:
        return None

    # Build channel for each overlap bar
    upper_vals = []
    middle_vals = []
    lower_vals = []
    prices_at = []

    for k in range(n_overlap):
        m = ema_vals[ema_start_k + k]
        atr = atr_vals[atr_start_k + k]
        upper_vals.append(m + mult * atr)
        middle_vals.append(m)
        lower_vals.append(m - mult * atr)
        prices_at.append(closes[overlap_start + k])

    if not upper_vals:
        return None

    cur_upper = upper_vals[-1]
    cur_mid = middle_vals[-1]
    cur_lower = lower_vals[-1]
    cur_atr = atr_vals[-1]

    channel_w = cur_upper - cur_lower
    channel_w_pct = channel_w / cur_mid * 100.0 if cur_mid > 0 else 0.0

    # Price position: 0=lower, 0.5=mid, 1=upper
    if channel_w > 0:
        pos = (current - cur_lower) / channel_w
    else:
        pos = 0.5
    pos = max(0.0, min(1.5, pos))

    if current > cur_upper:
        pos_label = "above_upper"
    elif current > cur_mid:
        pos_label = "upper_half"
    elif current > cur_lower:
        pos_label = "lower_half"
    else:
        pos_label = "below_lower"
    # Fix: if exactly near mid
    if abs(pos - 0.5) < 0.08:
        pos_label = "middle"

    breakout_up = current > cur_upper
    breakout_down = current < cur_lower

    # Trend direction: compare EMA now vs EMA `period` bars ago
    if len(middle_vals) >= period:
        ema_then = middle_vals[-period]
        if cur_mid > ema_then * 1.001:
            trend = "up"
        elif cur_mid < ema_then * 0.999:
            trend = "down"
        else:
            trend = "sideways"
    else:
        trend = "sideways"

    # How many bars since last midline cross
    trend_bars = 0
    above_mid = current >= cur_mid
    for k in range(len(prices_at) - 2, -1, -1):
        was_above = prices_at[k] >= middle_vals[k]
        if was_above != above_mid:
            break
        trend_bars += 1

    # History snapshots
    history: list[KCSnapshot] = []
    n_hist = min(history_bars, len(prices_at))
    for k in range(len(prices_at) - n_hist, len(prices_at)):
        w = upper_vals[k] - lower_vals[k]
        p = (prices_at[k] - lower_vals[k]) / w if w > 0 else 0.5
        history.append(KCSnapshot(
            bar_idx=overlap_start + k,
            price=round(prices_at[k], 4),
            upper=round(upper_vals[k], 4),
            middle=round(middle_vals[k], 4),
            lower=round(lower_vals[k], 4),
            position=round(max(0.0, min(1.5, p)), 3),
        ))

    atr_pct = cur_atr / current * 100.0 if current > 0 else 0.0

    # Verdict
    if breakout_up:
        action = "Breakout ABOVE upper band"
    elif breakout_down:
        action = "Breakout BELOW lower band"
    elif pos_label == "upper_half":
        action = "Price in upper half — trending up"
    elif pos_label == "lower_half":
        action = "Price in lower half — trending down"
    else:
        action = f"Price near middle ({pos_label})"

    verdict = (
        f"Keltner Channel ({period}/{atr_period}, ×{mult}): {action}. "
        f"Channel width {channel_w_pct:.1f}% of price. ATR {atr_pct:.1f}%."
    )

    return KeltnerReport(
        symbol=symbol,
        period=period,
        atr_period=atr_period,
        mult=mult,
        current_price=round(current, 4),
        upper=round(cur_upper, 4),
        middle=round(cur_mid, 4),
        lower=round(cur_lower, 4),
        channel_width=round(channel_w, 4),
        channel_width_pct=round(channel_w_pct, 2),
        price_position=round(pos, 3),
        position_label=pos_label,
        breakout_up=breakout_up,
        breakout_down=breakout_down,
        trend_direction=trend,
        trend_bars=trend_bars,
        atr=round(cur_atr, 4),
        atr_pct=round(atr_pct, 2),
        history=history,
        bars_used=len(bars),
        verdict=verdict,
    )
