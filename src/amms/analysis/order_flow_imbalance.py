"""Order Flow Imbalance (OFI) Analyser.

Estimates buy vs sell pressure bar-by-bar using price direction and volume.
Without tick-level data, approximates order flow using:

  - Up bars (close > open): volume classified as buy-dominant
  - Down bars (close < open): volume classified as sell-dominant
  - Bar body fraction used to weight the classification

Per bar:
  buy_vol  = volume * body_pct  (if up bar)
  sell_vol = volume * body_pct  (if down bar)
  neutral  = remainder

Cumulative OFI = running (buy_vol - sell_vol), normalised to a -100..+100
score over the lookback window.

Also computes:
  - Rolling OFI (recent N bars) vs longer baseline
  - OFI divergence: price direction vs OFI direction
  - Signal: strong_buy, buy, neutral, sell, strong_sell
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class OFIBar:
    index: int
    buy_vol: float
    sell_vol: float
    net: float      # buy_vol - sell_vol


@dataclass(frozen=True)
class OFIReport:
    symbol: str

    # Cumulative
    cumulative_ofi: float       # running sum over lookback
    cumulative_ofi_norm: float  # normalised -100..+100

    # Rolling vs baseline
    recent_ofi: float           # sum of last `short_window` bars
    baseline_ofi: float         # sum of lookback bars (excluding recent)
    ofi_ratio: float            # recent / |baseline| clamped -3..+3

    # Score and signal
    score: float                # -100..+100
    signal: str                 # "strong_buy", "buy", "neutral", "sell", "strong_sell"

    # Divergence
    price_direction: str        # "up", "down", "flat"
    ofi_direction: str          # "up", "down", "flat"
    divergence: bool            # price up but OFI down, or vice versa

    # Volume stats
    total_buy_vol: float
    total_sell_vol: float
    buy_pct: float              # buy_vol / total_vol %

    bars_used: int
    by_bar: list[OFIBar]
    verdict: str


def _body_pct(o: float, h: float, l: float, c: float) -> float:
    bar_range = h - l
    if bar_range < 1e-9:
        return 0.0
    return abs(c - o) / bar_range


def analyze(
    bars: list,
    *,
    symbol: str = "",
    lookback: int = 50,
    short_window: int = 10,
) -> OFIReport | None:
    """Estimate order flow imbalance from OHLCV bars.

    bars: bar objects with .open, .high, .low, .close, .volume attributes.
    lookback: bars to analyse.
    short_window: bars for the recent OFI window vs baseline comparison.
    """
    if not bars or len(bars) < max(lookback, 20):
        return None

    try:
        opens  = [float(b.open)   for b in bars]
        highs  = [float(b.high)   for b in bars]
        lows   = [float(b.low)    for b in bars]
        closes = [float(b.close)  for b in bars]
    except (AttributeError, TypeError, ValueError):
        return None

    try:
        volumes = [float(b.volume) for b in bars]
    except (AttributeError, TypeError, ValueError):
        volumes = [1.0] * len(bars)

    current = closes[-1]
    if current <= 0:
        return None

    # Work on the lookback window
    n = min(lookback, len(bars))
    w_opens  = opens[-n:]
    w_highs  = highs[-n:]
    w_lows   = lows[-n:]
    w_closes = closes[-n:]
    w_vols   = volumes[-n:]

    ofi_bars: list[OFIBar] = []
    cumulative = 0.0

    for i in range(n):
        o, h, l, c, v = w_opens[i], w_highs[i], w_lows[i], w_closes[i], w_vols[i]
        bp = _body_pct(o, h, l, c)
        if c >= o:
            buy_v  = v * bp
            sell_v = 0.0
        else:
            buy_v  = 0.0
            sell_v = v * bp
        net = buy_v - sell_v
        cumulative += net
        ofi_bars.append(OFIBar(index=i, buy_vol=round(buy_v, 2), sell_vol=round(sell_v, 2), net=round(net, 2)))

    # Normalise cumulative OFI to -100..+100
    max_possible = sum(abs(b.net) for b in ofi_bars)
    cum_norm = (cumulative / max_possible * 100.0) if max_possible > 1e-9 else 0.0
    cum_norm = max(-100.0, min(100.0, cum_norm))

    # Recent vs baseline
    sw = min(short_window, n)
    recent_nets = [b.net for b in ofi_bars[-sw:]]
    baseline_bars = ofi_bars[:-sw] if len(ofi_bars) > sw else ofi_bars

    recent_ofi = sum(recent_nets)
    baseline_ofi = sum(b.net for b in baseline_bars)

    denom = abs(baseline_ofi) if abs(baseline_ofi) > 1e-9 else 1.0
    ofi_ratio = max(-3.0, min(3.0, recent_ofi / denom))

    # Overall score: blend cumulative norm and ratio signal
    ratio_score = ofi_ratio / 3.0 * 100.0  # -100..+100
    score = cum_norm * 0.6 + ratio_score * 0.4
    score = max(-100.0, min(100.0, score))

    # Signal
    if score >= 50:
        signal = "strong_buy"
    elif score >= 20:
        signal = "buy"
    elif score <= -50:
        signal = "strong_sell"
    elif score <= -20:
        signal = "sell"
    else:
        signal = "neutral"

    # Price direction over window
    start_price = w_closes[0]
    if current > start_price * 1.005:
        price_dir = "up"
    elif current < start_price * 0.995:
        price_dir = "down"
    else:
        price_dir = "flat"

    # OFI direction
    if cumulative > max_possible * 0.05:
        ofi_dir = "up"
    elif cumulative < -max_possible * 0.05:
        ofi_dir = "down"
    else:
        ofi_dir = "flat"

    # Divergence
    divergence = (
        (price_dir == "up"   and ofi_dir == "down") or
        (price_dir == "down" and ofi_dir == "up")
    )

    # Volume totals
    total_buy  = sum(b.buy_vol  for b in ofi_bars)
    total_sell = sum(b.sell_vol for b in ofi_bars)
    total_vol  = total_buy + total_sell
    buy_pct = total_buy / total_vol * 100.0 if total_vol > 0 else 50.0

    verdict = (
        f"OFI ({symbol}, {n} bars): score {score:+.1f}/100 ({signal.replace('_', ' ')}). "
        f"Buy vol {buy_pct:.1f}%, cumulative OFI {cum_norm:+.1f}."
    )
    if divergence:
        verdict += f" Divergence: price {price_dir}, OFI {ofi_dir}."

    return OFIReport(
        symbol=symbol,
        cumulative_ofi=round(cumulative, 2),
        cumulative_ofi_norm=round(cum_norm, 2),
        recent_ofi=round(recent_ofi, 2),
        baseline_ofi=round(baseline_ofi, 2),
        ofi_ratio=round(ofi_ratio, 3),
        score=round(score, 2),
        signal=signal,
        price_direction=price_dir,
        ofi_direction=ofi_dir,
        divergence=divergence,
        total_buy_vol=round(total_buy, 2),
        total_sell_vol=round(total_sell, 2),
        buy_pct=round(buy_pct, 2),
        bars_used=len(bars),
        by_bar=ofi_bars,
        verdict=verdict,
    )
