"""Parabolic SAR Analyser.

The Parabolic SAR (Stop and Reverse) is a trend-following indicator that
places stop levels that accelerate as the trend develops. When price crosses
the SAR, the indicator flips to the opposite side.

Formula:
  Rising SAR:  SAR_t = SAR_{t-1} + AF × (EP - SAR_{t-1})
  Falling SAR: SAR_t = SAR_{t-1} - AF × (SAR_{t-1} - EP)
  EP (Extreme Point): highest high in uptrend / lowest low in downtrend
  AF (Acceleration Factor): starts at step, increases by step each new EP, max at max_af
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SARSnapshot:
    bar_idx: int
    price: float
    sar: float
    direction: str    # "bull" or "bear"
    af: float
    ep: float
    flipped: bool


@dataclass(frozen=True)
class PSARReport:
    symbol: str
    af_step: float
    af_max: float

    direction: str          # "bull" or "bear"
    sar: float              # current SAR level (stop price)
    current_price: float
    distance_pct: float     # |price - sar| / price * 100

    trend_age: int          # bars since last flip
    flip_count: int         # number of flips in history
    last_flip_bar: int | None

    current_af: float       # current acceleration factor
    current_ep: float       # current extreme point

    avg_trend_duration: float    # avg bars between flips
    bull_pct: float              # % of bars that were bullish

    history: list[SARSnapshot]
    bars_used: int
    verdict: str


def analyze(
    bars: list,
    *,
    symbol: str = "",
    af_step: float = 0.02,
    af_max: float = 0.2,
    history_bars: int = 40,
) -> PSARReport | None:
    """Compute Parabolic SAR from bar history.

    bars: bar objects with .high, .low, .close attributes.
    af_step: acceleration factor increment.
    af_max: maximum acceleration factor.
    history_bars: number of historical snapshots to return.
    """
    if not bars or len(bars) < 5:
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

    # Initialize: determine initial direction from first 2 bars
    if closes[1] >= closes[0]:
        direction = "bull"
        sar = min(lows[0], lows[1])
        ep = max(highs[0], highs[1])
    else:
        direction = "bear"
        sar = max(highs[0], highs[1])
        ep = min(lows[0], lows[1])

    af = af_step

    sar_vals: list[float] = [sar, sar]
    dir_vals: list[str] = [direction, direction]
    af_vals: list[float] = [af, af]
    ep_vals: list[float] = [ep, ep]
    flip_vals: list[bool] = [False, False]

    for i in range(2, n):
        h = highs[i]
        l = lows[i]
        prev_sar = sar_vals[-1]

        flipped = False

        if direction == "bull":
            # New SAR
            new_sar = prev_sar + af * (ep - prev_sar)
            # SAR cannot be above prior two lows
            new_sar = min(new_sar, lows[i - 1], lows[i - 2] if i >= 2 else lows[i - 1])

            if l < new_sar:
                # Flip to bear
                direction = "bear"
                new_sar = ep  # SAR starts at highest high
                ep = l
                af = af_step
                flipped = True
            else:
                if h > ep:
                    ep = h
                    af = min(af + af_step, af_max)
        else:  # bear
            new_sar = prev_sar - af * (prev_sar - ep)
            # SAR cannot be below prior two highs
            new_sar = max(new_sar, highs[i - 1], highs[i - 2] if i >= 2 else highs[i - 1])

            if h > new_sar:
                # Flip to bull
                direction = "bull"
                new_sar = ep  # SAR starts at lowest low
                ep = h
                af = af_step
                flipped = True
            else:
                if l < ep:
                    ep = l
                    af = min(af + af_step, af_max)

        sar = new_sar
        sar_vals.append(sar)
        dir_vals.append(direction)
        af_vals.append(af)
        ep_vals.append(ep)
        flip_vals.append(flipped)

    cur_sar = sar_vals[-1]
    cur_dir = dir_vals[-1]
    cur_af = af_vals[-1]
    cur_ep = ep_vals[-1]

    distance_pct = abs(current - cur_sar) / current * 100.0 if current > 0 else 0.0

    # Flips analysis
    flip_indices = [i for i in range(n) if flip_vals[i]]
    flip_count = len(flip_indices)
    last_flip_bar = flip_indices[-1] if flip_indices else None

    # Trend age
    trend_age = n - 1 - last_flip_bar if last_flip_bar is not None else n

    # Average trend duration
    if flip_count >= 2:
        durations = [flip_indices[k + 1] - flip_indices[k] for k in range(len(flip_indices) - 1)]
        avg_dur = sum(durations) / len(durations)
    elif flip_count == 1:
        avg_dur = float(flip_indices[0])
    else:
        avg_dur = float(n)

    # Bull percentage
    bull_pct = sum(1 for d in dir_vals if d == "bull") / len(dir_vals) * 100.0

    # History
    history: list[SARSnapshot] = []
    n_hist = min(history_bars, n)
    for i in range(n - n_hist, n):
        history.append(SARSnapshot(
            bar_idx=i,
            price=round(closes[i], 4),
            sar=round(sar_vals[i], 4),
            direction=dir_vals[i],
            af=round(af_vals[i], 4),
            ep=round(ep_vals[i], 4),
            flipped=flip_vals[i],
        ))

    # Verdict
    if cur_dir == "bull":
        action = f"BULLISH — SAR support at {cur_sar:.2f} ({distance_pct:.1f}% below price)"
    else:
        action = f"BEARISH — SAR resistance at {cur_sar:.2f} ({distance_pct:.1f}% above price)"

    verdict = (
        f"Parabolic SAR (step={af_step}, max={af_max}): {action}. "
        f"Trend age: {trend_age} bars, AF={cur_af:.3f}. "
        f"{flip_count} flip(s) in history."
    )

    return PSARReport(
        symbol=symbol,
        af_step=af_step,
        af_max=af_max,
        direction=cur_dir,
        sar=round(cur_sar, 4),
        current_price=round(current, 4),
        distance_pct=round(distance_pct, 2),
        trend_age=trend_age,
        flip_count=flip_count,
        last_flip_bar=last_flip_bar,
        current_af=round(cur_af, 4),
        current_ep=round(cur_ep, 4),
        avg_trend_duration=round(avg_dur, 1),
        bull_pct=round(bull_pct, 1),
        history=history,
        bars_used=n,
        verdict=verdict,
    )
