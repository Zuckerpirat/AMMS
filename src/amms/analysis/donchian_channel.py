"""Donchian Channel Analyser.

Donchian Channels use the rolling highest high and lowest low over a
period to define upper, lower, and middle bands. Used in the Turtle
Trading system for breakout signals.

Key metrics:
  - Upper band: highest high over period
  - Lower band: lowest low over period
  - Middle band: (upper + lower) / 2
  - Price position: where price sits within the channel
  - Breakout detection: new N-period high/low
  - Channel contraction/expansion over time
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DCSnapshot:
    bar_idx: int
    price: float
    upper: float
    middle: float
    lower: float
    channel_width: float
    position: float     # 0=lower, 0.5=middle, 1=upper


@dataclass(frozen=True)
class DonchianReport:
    symbol: str
    period: int

    current_price: float
    upper: float           # highest high over period
    middle: float          # (upper + lower) / 2
    lower: float           # lowest low over period
    channel_width: float
    channel_width_pct: float

    price_position: float   # 0=at lower, 0.5=at middle, 1=at upper
    position_label: str     # "above_upper", "at_upper", "upper_half", "middle", "lower_half", "at_lower", "below_lower"

    breakout_up: bool       # price >= upper (new period high)
    breakout_down: bool     # price <= lower (new period low)
    near_upper: bool        # within 1% of upper band
    near_lower: bool        # within 1% of lower band

    bars_since_upper: int   # bars since price was at/above upper
    bars_since_lower: int   # bars since price was at/above lower

    channel_trend: str      # "expanding", "contracting", "stable"
    width_change_pct: float # recent vs earlier channel width change

    history: list[DCSnapshot]
    bars_used: int
    verdict: str


def analyze(
    bars: list,
    *,
    symbol: str = "",
    period: int = 20,
    history_bars: int = 40,
    near_threshold_pct: float = 1.0,
) -> DonchianReport | None:
    """Compute Donchian Channels from bar history.

    bars: bar objects with .high, .low, .close attributes.
    period: rolling window for highest high / lowest low.
    history_bars: number of history snapshots to return.
    near_threshold_pct: % distance to consider "near" a band.
    """
    if not bars or len(bars) < period + 2:
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

    n = len(bars)

    # Build channel for each bar from period onwards
    upper_vals: list[float] = []
    lower_vals: list[float] = []
    prices_at: list[float] = []

    for i in range(period - 1, n):
        window_h = highs[i - period + 1: i + 1]
        window_l = lows[i - period + 1: i + 1]
        upper_vals.append(max(window_h))
        lower_vals.append(min(window_l))
        prices_at.append(closes[i])

    if not upper_vals:
        return None

    cur_upper = upper_vals[-1]
    cur_lower = lower_vals[-1]
    cur_mid = (cur_upper + cur_lower) / 2.0
    channel_w = cur_upper - cur_lower
    channel_w_pct = channel_w / cur_mid * 100.0 if cur_mid > 0 else 0.0

    # Position
    pos = (current - cur_lower) / channel_w if channel_w > 0 else 0.5
    pos = max(0.0, min(1.0, pos))

    near_up = (cur_upper - current) / current * 100.0 <= near_threshold_pct if current > 0 else False
    near_lo = (current - cur_lower) / current * 100.0 <= near_threshold_pct if current > 0 else False
    breakout_up = current >= cur_upper
    breakout_down = current <= cur_lower

    if breakout_up:
        pos_label = "at_upper" if current == cur_upper else "above_upper"
    elif breakout_down:
        pos_label = "at_lower" if current == cur_lower else "below_lower"
    elif pos > 0.65:
        pos_label = "upper_half"
    elif pos < 0.35:
        pos_label = "lower_half"
    else:
        pos_label = "middle"

    # Bars since last at upper/lower
    bars_since_upper = 0
    for k in range(len(prices_at) - 2, -1, -1):
        if prices_at[k] >= upper_vals[k]:
            break
        bars_since_upper += 1

    bars_since_lower = 0
    for k in range(len(prices_at) - 2, -1, -1):
        if prices_at[k] <= lower_vals[k]:
            break
        bars_since_lower += 1

    # Channel trend: compare recent width vs earlier width
    half = max(1, len(upper_vals) // 2)
    recent_widths = [upper_vals[k] - lower_vals[k] for k in range(len(upper_vals) - half, len(upper_vals))]
    earlier_widths = [upper_vals[k] - lower_vals[k] for k in range(half)]
    recent_avg = sum(recent_widths) / len(recent_widths) if recent_widths else channel_w
    earlier_avg = sum(earlier_widths) / len(earlier_widths) if earlier_widths else channel_w

    if earlier_avg > 0:
        width_change = (recent_avg - earlier_avg) / earlier_avg * 100.0
    else:
        width_change = 0.0

    if width_change > 5.0:
        ch_trend = "expanding"
    elif width_change < -5.0:
        ch_trend = "contracting"
    else:
        ch_trend = "stable"

    # History
    history: list[DCSnapshot] = []
    n_hist = min(history_bars, len(upper_vals))
    for k in range(len(upper_vals) - n_hist, len(upper_vals)):
        bar_idx = (period - 1) + k
        w = upper_vals[k] - lower_vals[k]
        p = (prices_at[k] - lower_vals[k]) / w if w > 0 else 0.5
        history.append(DCSnapshot(
            bar_idx=bar_idx,
            price=round(prices_at[k], 4),
            upper=round(upper_vals[k], 4),
            middle=round((upper_vals[k] + lower_vals[k]) / 2, 4),
            lower=round(lower_vals[k], 4),
            channel_width=round(w, 4),
            position=round(max(0.0, min(1.0, p)), 3),
        ))

    # Verdict
    if breakout_up:
        action = f"NEW {period}-bar HIGH breakout at {current:.2f}"
    elif breakout_down:
        action = f"NEW {period}-bar LOW breakdown at {current:.2f}"
    elif near_up:
        action = f"Near upper band ({cur_upper:.2f}) — potential breakout setup"
    elif near_lo:
        action = f"Near lower band ({cur_lower:.2f}) — potential breakdown risk"
    else:
        action = f"Price at {pos_label} of channel ({pos * 100:.0f}th percentile)"

    verdict = (
        f"Donchian({period}): {action}. "
        f"Channel width {channel_w_pct:.1f}% ({ch_trend})."
    )

    return DonchianReport(
        symbol=symbol,
        period=period,
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
        near_upper=near_up,
        near_lower=near_lo,
        bars_since_upper=bars_since_upper,
        bars_since_lower=bars_since_lower,
        channel_trend=ch_trend,
        width_change_pct=round(width_change, 2),
        history=history,
        bars_used=len(bars),
        verdict=verdict,
    )
