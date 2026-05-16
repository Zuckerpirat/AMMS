"""Order Flow Imbalance (OFI) analysis.

Estimates intra-bar buy vs. sell pressure from OHLCV data.
Without tick data, we use two proxies:

1. **Close Position Ratio (CPR)**:
   CPR = (close - low) / (high - low)  ∈ [0, 1]
   Close near high → buyers dominated (CPR → 1)
   Close near low  → sellers dominated (CPR → 0)

2. **Volume-Weighted OFI**:
   buy_vol  = volume × CPR
   sell_vol = volume × (1 - CPR)
   OFI      = (buy_vol - sell_vol) / (buy_vol + sell_vol)  ∈ [-1, 1]

Cumulative OFI (COFI) sums OFI across a window — rising COFI signals
sustained buying pressure; falling COFI signals selling pressure.

Interpretation:
  - COFI trending up + price up: confirmed uptrend (strong hands buying)
  - COFI trending up + price flat: accumulation (hidden buying)
  - COFI trending down + price down: confirmed downtrend
  - COFI trending down + price flat: distribution (hidden selling)
  - Divergence (COFI vs price): potential reversal signal
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OFIBar:
    cpr: float          # close position ratio [0, 1]
    ofi: float          # per-bar OFI [-1, 1]
    buy_vol: float
    sell_vol: float


@dataclass(frozen=True)
class OFIReport:
    symbol: str
    bars: list[OFIBar]          # recent OFI bars (last lookback)
    cofi: float                 # cumulative OFI over window
    cofi_slope: float           # linear trend of COFI series (per bar)
    cofi_direction: str         # "rising" / "falling" / "flat"
    avg_ofi: float              # mean OFI over window
    buy_pressure_pct: float     # % bars where OFI > 0 (buying dominant)
    price_trend: str            # "up" / "down" / "flat"
    divergence: bool            # OFI direction contradicts price trend
    current_price: float
    bars_used: int
    verdict: str


def _ofi_bar(b) -> OFIBar | None:
    try:
        high = float(b.high)
        low = float(b.low)
        close = float(b.close)
        vol = float(b.volume) if hasattr(b, "volume") else 1.0
    except Exception:
        return None

    spread = high - low
    if spread <= 0:
        cpr = 0.5
    else:
        cpr = (close - low) / spread

    buy_vol = vol * cpr
    sell_vol = vol * (1.0 - cpr)
    total = buy_vol + sell_vol
    ofi = (buy_vol - sell_vol) / total if total > 0 else 0.0

    return OFIBar(
        cpr=round(cpr, 4),
        ofi=round(ofi, 4),
        buy_vol=round(buy_vol, 2),
        sell_vol=round(sell_vol, 2),
    )


def _slope(series: list[float]) -> float:
    """OLS slope of series (y = slope * x + intercept)."""
    n = len(series)
    if n < 2:
        return 0.0
    xs = list(range(n))
    mx = (n - 1) / 2.0
    my = sum(series) / n
    ss_xx = sum((x - mx) ** 2 for x in xs)
    ss_xy = sum((xs[i] - mx) * (series[i] - my) for i in range(n))
    return ss_xy / ss_xx if ss_xx > 0 else 0.0


def analyze(
    bars: list,
    *,
    symbol: str = "",
    lookback: int = 20,
    flat_threshold: float = 0.01,
) -> OFIReport | None:
    """Compute Order Flow Imbalance from bars.

    bars: list[Bar] with .high .low .close and optionally .volume.
    symbol: ticker for display.
    lookback: number of recent bars to analyse (default 20).
    flat_threshold: OFI/slope magnitude below which trend is "flat".
    """
    if not bars or len(bars) < 5:
        return None

    try:
        current_price = float(bars[-1].close)
    except Exception:
        return None

    recent = bars[-lookback:]
    ofi_bars: list[OFIBar] = []
    for b in recent:
        ob = _ofi_bar(b)
        if ob is not None:
            ofi_bars.append(ob)

    if len(ofi_bars) < 3:
        return None

    # Cumulative OFI series
    cofi_series: list[float] = []
    running = 0.0
    for ob in ofi_bars:
        running += ob.ofi
        cofi_series.append(running)

    cofi = cofi_series[-1]
    avg_ofi = sum(ob.ofi for ob in ofi_bars) / len(ofi_bars)
    buy_pressure_pct = sum(1 for ob in ofi_bars if ob.ofi > 0) / len(ofi_bars) * 100.0

    # COFI trend (slope of cumulative OFI)
    cofi_slope = _slope(cofi_series)
    if cofi_slope > flat_threshold:
        cofi_direction = "rising"
    elif cofi_slope < -flat_threshold:
        cofi_direction = "falling"
    else:
        cofi_direction = "flat"

    # Price trend: compare first vs last third
    chunk = max(1, len(recent) // 3)
    try:
        first_avg = sum(float(b.close) for b in recent[:chunk]) / chunk
        last_avg = sum(float(b.close) for b in recent[-chunk:]) / chunk
        pct_change = (last_avg - first_avg) / first_avg * 100.0 if first_avg > 0 else 0.0
    except Exception:
        pct_change = 0.0

    price_flat = 0.3  # % threshold for "flat"
    if pct_change > price_flat:
        price_trend = "up"
    elif pct_change < -price_flat:
        price_trend = "down"
    else:
        price_trend = "flat"

    # Divergence: OFI says one thing, price says another
    divergence = (
        (cofi_direction == "rising" and price_trend == "down") or
        (cofi_direction == "falling" and price_trend == "up")
    )

    # Build verdict
    flow_desc = {
        "rising": "buying pressure building",
        "falling": "selling pressure building",
        "flat": "balanced order flow",
    }.get(cofi_direction, cofi_direction)

    context_desc = {
        ("rising", "up"): "Confirmed uptrend — buyers in control.",
        ("rising", "flat"): "Accumulation signal — hidden buying despite flat price.",
        ("rising", "down"): "Divergence — buyers absorbing selling; potential reversal.",
        ("falling", "down"): "Confirmed downtrend — sellers in control.",
        ("falling", "up"): "Divergence — distribution signal; selling into rally.",
        ("falling", "flat"): "Distribution — hidden selling despite flat price.",
        ("flat", "up"): "Price rising on balanced flow — momentum may fade.",
        ("flat", "down"): "Price falling on balanced flow — drift lower.",
        ("flat", "flat"): "Equilibrium — no directional edge.",
    }.get((cofi_direction, price_trend), "Mixed signals.")

    verdict = (
        f"OFI: {flow_desc} (COFI {cofi:+.2f}, avg {avg_ofi:+.3f}/bar). "
        f"Buy pressure: {buy_pressure_pct:.0f}% of bars. "
        f"{context_desc}"
    )
    if divergence:
        verdict += " ⚠️ Divergence detected."

    return OFIReport(
        symbol=symbol,
        bars=ofi_bars[-10:],
        cofi=round(cofi, 4),
        cofi_slope=round(cofi_slope, 6),
        cofi_direction=cofi_direction,
        avg_ofi=round(avg_ofi, 4),
        buy_pressure_pct=round(buy_pressure_pct, 1),
        price_trend=price_trend,
        divergence=divergence,
        current_price=round(current_price, 2),
        bars_used=len(bars),
        verdict=verdict,
    )
