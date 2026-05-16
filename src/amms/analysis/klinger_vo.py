"""Klinger Volume Oscillator (KVO).

Developed by Stephen Klinger. Measures long-term money flow trends using
volume, price direction, and a volume force concept.

Volume Force (VF):
  Typical Price (TP) = (H + L + C) / 3
  Direction: +1 if TP > prev_TP, else -1
  DM = H - L  (daily range / daily movement)
  CM = CM_prev + DM  (cumulative movement, resets on direction change)
  VF = Volume × (2 × (DM / CM) - 1) × direction × 100

KVO:
  KVO = EMA(VF, 34) - EMA(VF, 55)
  Signal = EMA(KVO, 13)
  Histogram = KVO - Signal

Signals:
  KVO above zero + trending up → bullish
  KVO crosses signal → buy/sell trigger
  Divergence from price → potential reversal
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class KVOReport:
    symbol: str

    kvo: float              # current KVO value
    kvo_signal: float       # signal line (EMA-13 of KVO)
    kvo_histogram: float    # KVO - Signal

    kvo_bullish: bool       # KVO > 0
    above_signal: bool      # KVO > signal

    cross_up: bool
    cross_down: bool

    # Normalised score
    score: float            # -100 to +100
    signal: str             # "strong_bull", "bull", "neutral", "bear", "strong_bear"

    # Divergence from price
    price_direction: str    # "up", "down", "flat"
    kvo_direction: str      # "up", "down", "flat"
    divergence: bool

    # History (last 15 values)
    kvo_series: list[float]
    signal_series: list[float]

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


def analyze(
    bars: list,
    *,
    symbol: str = "",
    fast: int = 34,
    slow: int = 55,
    signal_period: int = 13,
    history: int = 15,
) -> KVOReport | None:
    """Compute the Klinger Volume Oscillator.

    bars: bar objects with .high, .low, .close and optionally .volume.
    """
    min_bars = slow + signal_period + history + 5
    if not bars or len(bars) < min_bars:
        return None

    try:
        highs  = [float(b.high)  for b in bars]
        lows   = [float(b.low)   for b in bars]
        closes = [float(b.close) for b in bars]
    except (AttributeError, TypeError, ValueError):
        return None

    try:
        volumes = [float(b.volume) for b in bars]
    except (AttributeError, TypeError, ValueError):
        volumes = [1.0] * len(bars)

    current = closes[-1]
    if current <= 0:
        return None

    n = len(bars)

    # Compute Volume Force series
    tps = [(highs[i] + lows[i] + closes[i]) / 3.0 for i in range(n)]

    vf_series = []
    cm = highs[0] - lows[0]  # initial CM
    direction = 1

    for i in range(1, n):
        dm = highs[i] - lows[i]
        new_dir = 1 if tps[i] >= tps[i - 1] else -1
        if new_dir != direction:
            cm = highs[i - 1] - lows[i - 1] + dm  # reset
        else:
            cm = cm + dm
        direction = new_dir

        if cm > 1e-9:
            vf = volumes[i] * (2.0 * (dm / cm) - 1.0) * direction * 100.0
        else:
            vf = 0.0
        vf_series.append(vf)

    if len(vf_series) < slow + signal_period + 2:
        return None

    # KVO = EMA(VF, fast) - EMA(VF, slow)
    ema_fast = _ema(vf_series, fast)
    ema_slow = _ema(vf_series, slow)

    if not ema_fast or not ema_slow:
        return None

    # Align: ema_fast and ema_slow have different lengths; align from end
    min_len = min(len(ema_fast), len(ema_slow))
    kvo_vals = [ema_fast[-(min_len - i)] - ema_slow[-(min_len - i)] for i in range(min_len)]

    sig_vals = _ema(kvo_vals, signal_period)
    if not sig_vals:
        return None

    cur_kvo = kvo_vals[-1]
    cur_sig = sig_vals[-1]
    kvo_hist = cur_kvo - cur_sig

    # Cross detection
    if len(kvo_vals) >= 2 and len(sig_vals) >= 2:
        prev_k = kvo_vals[-2]
        prev_s = sig_vals[-2]
        cross_up   = cur_kvo >= cur_sig and prev_k < prev_s
        cross_down = cur_kvo <= cur_sig and prev_k > prev_s
    else:
        cross_up = cross_down = False

    # Score: normalise KVO by recent max-abs
    kvo_window = kvo_vals[-50:] if len(kvo_vals) >= 50 else kvo_vals
    kvo_max = max(abs(v) for v in kvo_window) if kvo_window else 1.0
    kvo_norm = max(-100.0, min(100.0, cur_kvo / kvo_max * 100.0)) if kvo_max > 1e-9 else 0.0

    hist_max = max(abs(kvo_vals[-i] - sig_vals[-i]) for i in range(1, min(len(sig_vals), 20) + 1))
    hist_norm = max(-100.0, min(100.0, kvo_hist / hist_max * 100.0)) if hist_max > 1e-9 else 0.0

    score = kvo_norm * 0.6 + hist_norm * 0.4
    score = max(-100.0, min(100.0, score))

    if score >= 55:
        signal = "strong_bull"
    elif score >= 15:
        signal = "bull"
    elif score <= -55:
        signal = "strong_bear"
    elif score <= -15:
        signal = "bear"
    else:
        signal = "neutral"

    # Price vs KVO direction (over last 10 bars)
    price_window = closes[-11:]
    if len(price_window) > 1 and price_window[0] > 0:
        price_chg = (price_window[-1] - price_window[0]) / price_window[0]
        price_dir = "up" if price_chg > 0.005 else ("down" if price_chg < -0.005 else "flat")
    else:
        price_dir = "flat"

    kvo_window10 = kvo_vals[-10:]
    kvo_dir = "up" if kvo_window10[-1] > kvo_window10[0] else ("down" if kvo_window10[-1] < kvo_window10[0] else "flat")

    divergence = (price_dir == "up" and kvo_dir == "down") or (price_dir == "down" and kvo_dir == "up")

    # History
    hist_kvo = kvo_vals[-history:]
    hist_sig = sig_vals[-history:]

    verdict = (
        f"KVO ({symbol}): {signal.replace('_', ' ')}. "
        f"KVO={cur_kvo:+.1f}, Signal={cur_sig:+.1f}, Hist={kvo_hist:+.1f}."
    )
    if cross_up:
        verdict += " Bullish cross."
    if cross_down:
        verdict += " Bearish cross."
    if divergence:
        verdict += f" Divergence: price {price_dir}, KVO {kvo_dir}."

    return KVOReport(
        symbol=symbol,
        kvo=round(cur_kvo, 2),
        kvo_signal=round(cur_sig, 2),
        kvo_histogram=round(kvo_hist, 2),
        kvo_bullish=cur_kvo > 0,
        above_signal=cur_kvo > cur_sig,
        cross_up=cross_up,
        cross_down=cross_down,
        score=round(score, 2),
        signal=signal,
        price_direction=price_dir,
        kvo_direction=kvo_dir,
        divergence=divergence,
        kvo_series=[round(v, 2) for v in hist_kvo],
        signal_series=[round(v, 2) for v in hist_sig],
        current_price=round(current, 4),
        bars_used=n,
        verdict=verdict,
    )
