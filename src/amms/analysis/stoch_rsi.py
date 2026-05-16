"""Stochastic RSI (StochRSI) Analyser.

Applies the Stochastic oscillator formula to RSI, producing a faster,
more sensitive version bounded 0–100. Typically smoothed to generate
%K and %D lines.

Algorithm:
  RSI = Wilder RSI(rsi_period)
  StochRSI = (RSI - min(RSI, stoch_period)) /
              (max(RSI, stoch_period) - min(RSI, stoch_period))   → 0..1
  %K  = SMA(StochRSI, k_smooth) × 100    → 0..100
  %D  = SMA(%K, d_smooth)

Signal zones:
  %K > 80 → overbought
  %K < 20 → oversold
  %K crossing %D → momentum shift
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StochRSIReport:
    symbol: str

    stoch_rsi: float    # raw StochRSI × 100 (before smoothing)
    k: float            # %K (smoothed StochRSI × 100)
    d: float            # %D (signal line)
    rsi: float          # underlying RSI value

    overbought: bool    # k > 80
    oversold: bool      # k < 20
    k_above_d: bool     # bullish alignment
    cross_up: bool      # %K just crossed above %D
    cross_down: bool    # %K just crossed below %D

    score: float        # -100..+100
    signal: str         # "strong_buy", "buy", "neutral", "sell", "strong_sell"

    history_k: list[float]
    history_d: list[float]
    bars_used: int
    verdict: str


def _wilder_rsi(closes: list[float], period: int) -> list[float]:
    """Wilder RSI series (length = len(closes) - period)."""
    if len(closes) < period + 1:
        return []
    gains, losses = 0.0, 0.0
    for i in range(1, period + 1):
        diff = closes[i] - closes[i - 1]
        if diff > 0:
            gains += diff
        else:
            losses -= diff
    avg_g = gains / period
    avg_l = losses / period

    def rsi_from_avg(ag: float, al: float) -> float:
        if al < 1e-9:
            return 100.0
        return 100.0 - 100.0 / (1.0 + ag / al)

    out = [rsi_from_avg(avg_g, avg_l)]
    for i in range(period + 1, len(closes)):
        diff = closes[i] - closes[i - 1]
        g = max(diff, 0.0)
        l = abs(min(diff, 0.0))
        avg_g = (avg_g * (period - 1) + g) / period
        avg_l = (avg_l * (period - 1) + l) / period
        out.append(rsi_from_avg(avg_g, avg_l))
    return out


def _sma_series(values: list[float], period: int) -> list[float]:
    if len(values) < period:
        return []
    out = []
    window_sum = sum(values[:period])
    out.append(window_sum / period)
    for i in range(period, len(values)):
        window_sum += values[i] - values[i - period]
        out.append(window_sum / period)
    return out


def analyze(
    bars: list,
    *,
    symbol: str = "",
    rsi_period: int = 14,
    stoch_period: int = 14,
    k_smooth: int = 3,
    d_smooth: int = 3,
    history: int = 20,
) -> StochRSIReport | None:
    """Compute Stochastic RSI.

    bars: bar objects with .close attribute.
    rsi_period: RSI lookback.
    stoch_period: Stochastic window applied to RSI.
    k_smooth / d_smooth: SMA smoothing periods.
    """
    min_bars = rsi_period + stoch_period + k_smooth + d_smooth + history + 5
    if not bars or len(bars) < min_bars:
        return None

    try:
        closes = [float(b.close) for b in bars]
    except (AttributeError, TypeError, ValueError):
        return None

    if closes[-1] <= 0:
        return None

    # Compute RSI series
    rsi_vals = _wilder_rsi(closes, rsi_period)
    if len(rsi_vals) < stoch_period + k_smooth + d_smooth + history:
        return None

    # Stochastic of RSI
    raw_stoch: list[float] = []
    for i in range(len(rsi_vals)):
        if i < stoch_period - 1:
            raw_stoch.append(50.0)
            continue
        window = rsi_vals[i - stoch_period + 1 : i + 1]
        lo = min(window)
        hi = max(window)
        rng = hi - lo
        if rng < 1e-9:
            raw_stoch.append(raw_stoch[-1] if raw_stoch else 50.0)
        else:
            raw_stoch.append((rsi_vals[i] - lo) / rng * 100.0)

    # %K = SMA(raw_stoch, k_smooth)
    k_series = _sma_series(raw_stoch, k_smooth)
    if not k_series:
        return None

    # %D = SMA(%K, d_smooth)
    d_series = _sma_series(k_series, d_smooth)
    if len(d_series) < history + 2:
        return None

    k_val = k_series[-1]
    d_val = d_series[-1]
    k_prev = k_series[-2] if len(k_series) >= 2 else k_val
    d_prev = d_series[-2] if len(d_series) >= 2 else d_val
    raw_val = raw_stoch[-1]
    rsi_cur = rsi_vals[-1]

    overbought = k_val > 80.0
    oversold   = k_val < 20.0
    k_above_d  = k_val > d_val
    cross_up   = k_prev <= d_prev and k_val > d_val
    cross_down = k_prev >= d_prev and k_val < d_val

    # Score: position + cross bonus (mean-reversion aware)
    pos_score = (k_val - 50.0) * 2.0
    cross_bonus = 15.0 if cross_up else (-15.0 if cross_down else 0.0)
    score = max(-100.0, min(100.0, pos_score + cross_bonus))

    if oversold and cross_up:
        signal = "strong_buy"
    elif oversold or (score <= -50):
        signal = "buy"
    elif overbought and cross_down:
        signal = "strong_sell"
    elif overbought or (score >= 50):
        signal = "sell"
    else:
        signal = "neutral"

    cross_str = ""
    if cross_up:
        cross_str = " %K crossed above %D."
    elif cross_down:
        cross_str = " %K crossed below %D."

    ob_str = " [OVERBOUGHT]" if overbought else (" [OVERSOLD]" if oversold else "")
    verdict = (
        f"StochRSI ({symbol}): %K={k_val:.1f}, %D={d_val:.1f}{ob_str}. "
        f"RSI={rsi_cur:.1f}. Signal: {signal.replace('_', ' ')}.{cross_str}"
    )

    hist_k = k_series[-history:] if len(k_series) >= history else k_series
    hist_d = d_series[-history:] if len(d_series) >= history else d_series

    return StochRSIReport(
        symbol=symbol,
        stoch_rsi=round(raw_val, 2),
        k=round(k_val, 2),
        d=round(d_val, 2),
        rsi=round(rsi_cur, 2),
        overbought=overbought,
        oversold=oversold,
        k_above_d=k_above_d,
        cross_up=cross_up,
        cross_down=cross_down,
        score=round(score, 2),
        signal=signal,
        history_k=[round(v, 2) for v in hist_k],
        history_d=[round(v, 2) for v in hist_d],
        bars_used=len(bars),
        verdict=verdict,
    )
