"""Ichimoku Cloud Analyser.

Computes all five Ichimoku Kinko Hyo components and derives a directional
signal from their relative positions.

Components:
  Tenkan-sen (Conversion Line): (highest high + lowest low) / 2 over 9 bars
  Kijun-sen  (Base Line):       (highest high + lowest low) / 2 over 26 bars
  Senkou Span A (Lead A):       (Tenkan + Kijun) / 2, plotted 26 bars ahead
  Senkou Span B (Lead B):       (highest high + lowest low) / 2 over 52 bars,
                                 plotted 26 bars ahead
  Chikou Span  (Lagging):       Current close plotted 26 bars behind

Cloud analysis at the current bar:
  - Price above both Spans  → bullish cloud context
  - Price below both Spans  → bearish cloud context
  - Price inside cloud      → neutral / transitional
  - Span A > Span B         → bullish cloud (green cloud)
  - Span A < Span B         → bearish cloud (red cloud)

Signal rules (all six checks):
  1. Price above Kijun
  2. Tenkan above Kijun (TK cross)
  3. Price above cloud
  4. Cloud is bullish (Span A > Span B)
  5. Chikou above price 26 bars ago
  6. Tenkan slope is positive

Score: 0-6 bullish signals → maps to signal label.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class IchimokuReport:
    symbol: str

    # Lines at current bar (index -1)
    tenkan: float | None
    kijun: float  | None
    senkou_a: float | None   # value for current bar (from 26-bar offset)
    senkou_b: float | None   # value for current bar
    chikou: float            # current close (displayed 26 bars back, compare to price 26 ago)

    # Cloud context
    cloud_top: float | None
    cloud_bottom: float | None
    cloud_bullish: bool | None   # None if no cloud yet

    # Price vs cloud
    price_above_cloud: bool | None
    price_below_cloud: bool | None
    price_in_cloud: bool | None

    # Signals
    bullish_signals: int    # 0-6
    bearish_signals: int    # 0-6
    signal: str             # "strong_bull", "bull", "neutral", "bear", "strong_bear"
    tk_cross_bullish: bool  # tenkan just crossed above kijun
    tk_cross_bearish: bool

    current_price: float
    bars_used: int
    verdict: str


def _donchian_mid(highs: list[float], lows: list[float], period: int) -> float | None:
    if len(highs) < period or len(lows) < period:
        return None
    return (max(highs[-period:]) + min(lows[-period:])) / 2.0


def analyze(
    bars: list,
    *,
    symbol: str = "",
    tenkan_period: int = 9,
    kijun_period: int = 26,
    senkou_b_period: int = 52,
    displacement: int = 26,
) -> IchimokuReport | None:
    """Compute Ichimoku Cloud components and derive a signal.

    bars: bar objects with .high, .low, .close attributes.
    Requires at least senkou_b_period + displacement bars (default 78).
    """
    min_bars = senkou_b_period + displacement
    if not bars or len(bars) < min_bars:
        return None

    try:
        highs  = [float(b.high)  for b in bars]
        lows   = [float(b.low)   for b in bars]
        closes = [float(b.close) for b in bars]
    except (AttributeError, TypeError, ValueError):
        return None

    current = closes[-1]
    if current <= 0:
        return None

    n = len(bars)

    # Tenkan-sen at current bar
    tenkan = _donchian_mid(highs, lows, tenkan_period)

    # Kijun-sen at current bar
    kijun = _donchian_mid(highs, lows, kijun_period)

    # Senkou Span A = (Tenkan + Kijun) / 2, plotted `displacement` bars ahead
    # So the value visible at current bar was computed `displacement` bars ago
    if n >= kijun_period + displacement:
        h_disp = highs[: n - displacement]
        l_disp = lows[: n - displacement]
        c_disp = closes[: n - displacement]
        tenkan_d = _donchian_mid(h_disp, l_disp, tenkan_period)
        kijun_d  = _donchian_mid(h_disp, l_disp, kijun_period)
        senkou_a = (tenkan_d + kijun_d) / 2.0 if (tenkan_d is not None and kijun_d is not None) else None
    else:
        senkou_a = None

    # Senkou Span B = donchian mid of 52 bars, plotted displacement bars ahead
    if n >= senkou_b_period + displacement:
        h_disp = highs[: n - displacement]
        l_disp = lows[: n - displacement]
        senkou_b = _donchian_mid(h_disp, l_disp, senkou_b_period)
    else:
        senkou_b = None

    # Cloud top/bottom
    if senkou_a is not None and senkou_b is not None:
        cloud_top    = max(senkou_a, senkou_b)
        cloud_bottom = min(senkou_a, senkou_b)
        cloud_bullish = senkou_a >= senkou_b
        price_above_cloud = current > cloud_top
        price_below_cloud = current < cloud_bottom
        price_in_cloud    = not price_above_cloud and not price_below_cloud
    else:
        cloud_top = cloud_bottom = None
        cloud_bullish = price_above_cloud = price_below_cloud = price_in_cloud = None

    # Chikou span: current close, compared to close 26 bars ago
    chikou = current
    chikou_above_past = None
    if n > displacement:
        past_close = closes[-displacement - 1]
        chikou_above_past = current > past_close

    # TK cross: check if tenkan crossed kijun recently (last 3 bars)
    tk_bull_cross = False
    tk_bear_cross = False
    if n >= kijun_period + 3:
        prev_h = highs[:-3]
        prev_l = lows[:-3]
        prev_tenkan = _donchian_mid(prev_h, prev_l, tenkan_period)
        prev_kijun  = _donchian_mid(prev_h, prev_l, kijun_period)
        if tenkan is not None and kijun is not None and prev_tenkan is not None and prev_kijun is not None:
            if tenkan >= kijun and prev_tenkan < prev_kijun:
                tk_bull_cross = True
            elif tenkan <= kijun and prev_tenkan > prev_kijun:
                tk_bear_cross = True

    # Tenkan slope (compare to 3 bars ago)
    tenkan_slope_up = False
    if n >= kijun_period + 4:
        h_prev3 = highs[:-3]
        l_prev3 = lows[:-3]
        tenkan_prev3 = _donchian_mid(h_prev3, l_prev3, tenkan_period)
        if tenkan is not None and tenkan_prev3 is not None:
            tenkan_slope_up = tenkan > tenkan_prev3

    # Bullish signal count (0-6)
    bull = 0
    bear = 0

    if tenkan is not None and kijun is not None:
        if current > kijun:
            bull += 1
        else:
            bear += 1
        if tenkan > kijun:
            bull += 1
        else:
            bear += 1

    if price_above_cloud is True:
        bull += 1
    elif price_below_cloud is True:
        bear += 1

    if cloud_bullish is True:
        bull += 1
    elif cloud_bullish is False:
        bear += 1

    if chikou_above_past is True:
        bull += 1
    elif chikou_above_past is False:
        bear += 1

    if tenkan_slope_up:
        bull += 1
    else:
        bear += 1

    # Net signal
    net = bull - bear
    if net >= 4:
        signal = "strong_bull"
    elif net >= 2:
        signal = "bull"
    elif net <= -4:
        signal = "strong_bear"
    elif net <= -2:
        signal = "bear"
    else:
        signal = "neutral"

    # Verdict
    cloud_str = "above" if price_above_cloud else ("below" if price_below_cloud else "inside")
    verdict = (
        f"Ichimoku ({symbol}): {signal.replace('_', ' ')} "
        f"({bull} bull / {bear} bear signals). "
        f"Price {cloud_str} cloud."
    )
    if tenkan is not None and kijun is not None:
        verdict += f" Tenkan {tenkan:.2f}, Kijun {kijun:.2f}."
    if tk_bull_cross:
        verdict += " TK bullish cross."
    if tk_bear_cross:
        verdict += " TK bearish cross."

    return IchimokuReport(
        symbol=symbol,
        tenkan=round(tenkan, 4) if tenkan is not None else None,
        kijun=round(kijun, 4) if kijun is not None else None,
        senkou_a=round(senkou_a, 4) if senkou_a is not None else None,
        senkou_b=round(senkou_b, 4) if senkou_b is not None else None,
        chikou=round(chikou, 4),
        cloud_top=round(cloud_top, 4) if cloud_top is not None else None,
        cloud_bottom=round(cloud_bottom, 4) if cloud_bottom is not None else None,
        cloud_bullish=cloud_bullish,
        price_above_cloud=price_above_cloud,
        price_below_cloud=price_below_cloud,
        price_in_cloud=price_in_cloud,
        bullish_signals=bull,
        bearish_signals=bear,
        signal=signal,
        tk_cross_bullish=tk_bull_cross,
        tk_cross_bearish=tk_bear_cross,
        current_price=round(current, 4),
        bars_used=n,
        verdict=verdict,
    )
