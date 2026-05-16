"""Anchored VWAP analysis.

Computes VWAP anchored to a specific starting bar — either an explicit
index (bars[-n]) or auto-detected swing high/low. Unlike rolling VWAP,
anchored VWAP accumulates from one meaningful price event and gives a
stable reference that traders use as dynamic support/resistance.

Anchors supported:
  - "auto_high": most significant swing high within the window
  - "auto_low":  most significant swing low within the window
  - n (int):     anchor n bars back from the most recent bar

Also computes upper/lower bands at ±1 and ±2 standard deviations of
price × volume, giving envelope bands similar to Bollinger-band VWAP.

Interpretation:
  - Price above AVWAP: bullish — buyers in control since anchor
  - Price below AVWAP: bearish — sellers in control since anchor
  - Price crossing AVWAP: potential reversal or trend change
  - Upper band (1σ/2σ): dynamic resistance
  - Lower band (1σ/2σ): dynamic support
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AVWAPReport:
    symbol: str
    anchor_bar_idx: int         # index in the original bar list
    anchor_label: str           # human-readable anchor description
    avwap: float                # anchored VWAP price
    upper_1: float              # AVWAP + 1σ
    upper_2: float              # AVWAP + 2σ
    lower_1: float              # AVWAP - 1σ
    lower_2: float              # AVWAP - 2σ
    current_price: float
    pct_from_avwap: float       # (price - avwap) / avwap * 100
    price_position: str         # "above" / "below" / "at"
    bars_in_window: int         # bars since anchor
    total_bars: int
    verdict: str


def _swing_high_idx(bars: list, window: int) -> int:
    """Return index of the bar with the highest high in bars[-window:]."""
    n = len(bars)
    start = max(0, n - window)
    best_idx = start
    best_val = float(bars[start].high)
    for i in range(start + 1, n):
        v = float(bars[i].high)
        if v > best_val:
            best_val = v
            best_idx = i
    return best_idx


def _swing_low_idx(bars: list, window: int) -> int:
    """Return index of the bar with the lowest low in bars[-window:]."""
    n = len(bars)
    start = max(0, n - window)
    best_idx = start
    best_val = float(bars[start].low)
    for i in range(start + 1, n):
        v = float(bars[i].low)
        if v < best_val:
            best_val = v
            best_idx = i
    return best_idx


def _compute_avwap(bars: list, anchor_idx: int) -> tuple[float, float] | None:
    """Return (avwap, vol_weighted_variance) from anchor_idx onward."""
    cum_pv = 0.0
    cum_v = 0.0
    cum_pv2 = 0.0   # for variance: Σ vol * typical_price²

    for b in bars[anchor_idx:]:
        try:
            high = float(b.high)
            low = float(b.low)
            close = float(b.close)
            vol = float(b.volume) if hasattr(b, "volume") else 1.0
        except Exception:
            continue
        tp = (high + low + close) / 3.0
        cum_pv += tp * vol
        cum_v += vol
        cum_pv2 += tp * tp * vol

    if cum_v <= 0:
        return None

    avwap = cum_pv / cum_v
    # Population variance: E[x²] - E[x]²
    variance = max(0.0, cum_pv2 / cum_v - avwap ** 2)
    return avwap, variance ** 0.5  # return (avwap, std_dev)


def analyze(
    bars: list,
    *,
    symbol: str = "",
    anchor: str | int = "auto_low",
    swing_window: int = 50,
) -> AVWAPReport | None:
    """Compute Anchored VWAP from bars.

    bars: list[Bar] with .high .low .close and optionally .volume.
    symbol: ticker for display.
    anchor: "auto_high" | "auto_low" | int (bars back from end).
    swing_window: how many bars back to search for the swing (default 50).
    """
    if not bars or len(bars) < 5:
        return None

    n = len(bars)

    try:
        if anchor == "auto_high":
            anchor_idx = _swing_high_idx(bars, swing_window)
            anchor_label = f"swing high ({swing_window}b window)"
        elif anchor == "auto_low":
            anchor_idx = _swing_low_idx(bars, swing_window)
            anchor_label = f"swing low ({swing_window}b window)"
        elif isinstance(anchor, int):
            offset = max(1, min(anchor, n - 1))
            anchor_idx = n - offset
            anchor_label = f"{offset} bars back"
        else:
            return None
    except Exception:
        return None

    result = _compute_avwap(bars, anchor_idx)
    if result is None:
        return None

    avwap, sigma = result

    try:
        current_price = float(bars[-1].close)
    except Exception:
        return None

    pct = (current_price - avwap) / avwap * 100.0 if avwap > 0 else 0.0
    at_threshold = 0.15  # within 0.15% = "at" AVWAP
    if abs(pct) <= at_threshold:
        price_pos = "at"
    elif current_price > avwap:
        price_pos = "above"
    else:
        price_pos = "below"

    pos_desc = {
        "above": f"above AVWAP ({pct:+.2f}%) — bullish: buyers in control since anchor",
        "below": f"below AVWAP ({pct:+.2f}%) — bearish: sellers in control since anchor",
        "at": "at AVWAP — equilibrium, watch for directional break",
    }.get(price_pos, "")

    verdict = (
        f"AVWAP ({anchor_label}): {avwap:.2f}. "
        f"Price ({current_price:.2f}) is {pos_desc}. "
        f"Bands: {avwap - 2*sigma:.2f} / {avwap - sigma:.2f} | "
        f"{avwap + sigma:.2f} / {avwap + 2*sigma:.2f}."
    )

    return AVWAPReport(
        symbol=symbol,
        anchor_bar_idx=anchor_idx,
        anchor_label=anchor_label,
        avwap=round(avwap, 4),
        upper_1=round(avwap + sigma, 4),
        upper_2=round(avwap + 2 * sigma, 4),
        lower_1=round(avwap - sigma, 4),
        lower_2=round(avwap - 2 * sigma, 4),
        current_price=round(current_price, 2),
        pct_from_avwap=round(pct, 3),
        price_position=price_pos,
        bars_in_window=n - anchor_idx,
        total_bars=n,
        verdict=verdict,
    )
