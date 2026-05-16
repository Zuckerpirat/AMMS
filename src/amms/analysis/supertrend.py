"""Supertrend Indicator.

Supertrend uses ATR-based bands around the (high+low)/2 midpoint.
It flips between bullish and bearish mode when price crosses the bands,
providing a single clear trend signal with dynamic stop levels.

Formula:
  Basic Upper Band = HL2 + mult × ATR
  Basic Lower Band = HL2 - mult × ATR
  Final Upper Band = min(current, prev upper) when in uptrend
  Final Lower Band = max(current, prev lower) when in downtrend
  Trend = bullish when close > Final Upper Band (prev), bearish otherwise
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class STSnapshot:
    bar_idx: int
    price: float
    supertrend: float    # the active band level (support in bull, resistance in bear)
    direction: str       # "bull" or "bear"
    flipped: bool        # True when direction changed on this bar


@dataclass(frozen=True)
class SupertrendReport:
    symbol: str
    period: int
    mult: float

    direction: str           # "bull" or "bear"
    supertrend_level: float  # current support (bull) or resistance (bear) line
    current_price: float
    distance_pct: float      # (price - level) / price * 100

    trend_age: int           # bars since last flip
    last_flip_bar: int | None
    flip_count: int          # number of flips in history

    atr: float
    history: list[STSnapshot]
    bars_used: int
    verdict: str


def _wilder_atr(highs: list[float], lows: list[float], closes: list[float], period: int) -> list[float]:
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
    atr = [sum(trs[:period]) / period]
    for tr in trs[period:]:
        atr.append((atr[-1] * (period - 1) + tr) / period)
    return atr


def analyze(
    bars: list,
    *,
    symbol: str = "",
    period: int = 10,
    mult: float = 3.0,
    history_bars: int = 40,
) -> SupertrendReport | None:
    """Compute Supertrend from bar history.

    bars: bar objects with .high, .low, .close attributes.
    period: ATR smoothing period.
    mult: ATR multiplier for band distance.
    history_bars: number of historical snapshots to return.
    """
    min_bars = period + 5
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

    atr_vals = _wilder_atr(highs, lows, closes, period)
    if not atr_vals:
        return None

    # atr_vals[0] ↔ closes[period]
    n = len(atr_vals)
    atr_offset = period  # closes[atr_offset + k] ↔ atr_vals[k]

    # Supertrend computation
    # basic bands
    final_upper: list[float] = []
    final_lower: list[float] = []
    directions: list[str] = []

    for k in range(n):
        idx = atr_offset + k
        hl2 = (highs[idx] + lows[idx]) / 2.0
        atr = atr_vals[k]
        basic_upper = hl2 + mult * atr
        basic_lower = hl2 - mult * atr

        if k == 0:
            final_upper.append(basic_upper)
            final_lower.append(basic_lower)
            directions.append("bull" if closes[idx] > basic_upper else "bear")
            continue

        prev_upper = final_upper[-1]
        prev_lower = final_lower[-1]
        prev_close = closes[idx - 1]

        # Final upper: tighten if possible, but only when prev was bull
        fu = basic_upper if basic_upper < prev_upper or prev_close > prev_upper else prev_upper
        # Final lower: lift if possible, but only when prev was bear
        fl = basic_lower if basic_lower > prev_lower or prev_close < prev_lower else prev_lower

        final_upper.append(fu)
        final_lower.append(fl)

        # Direction
        prev_dir = directions[-1]
        if prev_dir == "bull":
            new_dir = "bear" if closes[idx] < fl else "bull"
        else:
            new_dir = "bull" if closes[idx] > fu else "bear"
        directions.append(new_dir)

    if not directions:
        return None

    cur_dir = directions[-1]
    cur_st = final_lower[-1] if cur_dir == "bull" else final_upper[-1]
    cur_atr = atr_vals[-1]

    distance_pct = (current - cur_st) / current * 100.0 if current > 0 else 0.0

    # Flip analysis
    flip_bars: list[int] = []
    for k in range(1, len(directions)):
        if directions[k] != directions[k - 1]:
            flip_bars.append(atr_offset + k)

    trend_age = 0
    if flip_bars:
        last_flip_abs = flip_bars[-1]
        trend_age = len(bars) - 1 - last_flip_abs
        last_flip_bar: int | None = last_flip_abs
    else:
        trend_age = len(atr_vals)
        last_flip_bar = None

    # History snapshots
    history: list[STSnapshot] = []
    n_hist = min(history_bars, n)
    for k in range(n - n_hist, n):
        idx = atr_offset + k
        d = directions[k]
        st_level = final_lower[k] if d == "bull" else final_upper[k]
        flipped = k > 0 and directions[k] != directions[k - 1]
        history.append(STSnapshot(
            bar_idx=idx,
            price=round(closes[idx], 4),
            supertrend=round(st_level, 4),
            direction=d,
            flipped=flipped,
        ))

    # Verdict
    age_str = f"{trend_age} bars"
    dist_str = f"{distance_pct:+.1f}%"
    if cur_dir == "bull":
        action = f"BULLISH — support at {cur_st:.2f} ({dist_str} below price)"
    else:
        action = f"BEARISH — resistance at {cur_st:.2f} ({dist_str} above price)"

    verdict = (
        f"Supertrend ({period}, ×{mult}): {action}. "
        f"Trend age: {age_str}. Flips: {len(flip_bars)}."
    )

    return SupertrendReport(
        symbol=symbol,
        period=period,
        mult=mult,
        direction=cur_dir,
        supertrend_level=round(cur_st, 4),
        current_price=round(current, 4),
        distance_pct=round(distance_pct, 2),
        trend_age=trend_age,
        last_flip_bar=last_flip_bar,
        flip_count=len(flip_bars),
        atr=round(cur_atr, 4),
        history=history,
        bars_used=len(bars),
        verdict=verdict,
    )
