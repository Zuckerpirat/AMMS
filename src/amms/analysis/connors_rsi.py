"""Connors RSI (CRSI) Analyser.

Composite RSI developed by Larry Connors combining three momentum measures:

  1. Short RSI (3-period) of price — fast momentum
  2. Streak RSI (2-period) of consecutive up/down day count — mean reversion bias
  3. Percentile rank — today's % change vs last N bars (pct_lookback)

  CRSI = (RSI_price + RSI_streak + Percentile) / 3

Signal interpretation (mean reversion):
  CRSI < 10   → strong oversold, mean reversion buy opportunity
  CRSI < 20   → oversold, buy bias
  CRSI > 90   → strong overbought, mean reversion sell opportunity
  CRSI > 80   → overbought, sell bias
  20-80       → neutral
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CRSIReport:
    symbol: str

    crsi: float             # 0-100 composite
    rsi_price: float        # RSI(rsi_period) of close
    rsi_streak: float       # RSI(streak_period) of streak series
    percentile: float       # 0-100 percentile rank of today's % move

    streak: int             # current consecutive up(+) / down(-) days count
    pct_change: float       # today's % change

    overbought: bool        # crsi > 90
    oversold: bool          # crsi < 10
    ob_soft: bool           # crsi > 80
    os_soft: bool           # crsi < 20

    score: float            # -100..+100 (inverted: oversold = bullish)
    signal: str             # "strong_buy", "buy", "neutral", "sell", "strong_sell"

    history: list[float]    # recent CRSI values
    bars_used: int
    verdict: str


def _rsi(series: list[float], period: int) -> float:
    """Simple RSI of an arbitrary float series."""
    if len(series) < period + 1:
        return 50.0
    gains, losses = 0.0, 0.0
    for i in range(len(series) - period, len(series)):
        diff = series[i] - series[i - 1]
        if diff > 0:
            gains += diff
        else:
            losses -= diff
    avg_g = gains / period
    avg_l = losses / period
    if avg_l < 1e-9:
        return 100.0
    rs = avg_g / avg_l
    return 100.0 - 100.0 / (1.0 + rs)


def _streak_series(closes: list[float]) -> list[float]:
    """Compute consecutive up/down streak at each bar (signed int as float)."""
    if not closes:
        return []
    streaks: list[float] = [0.0]
    for i in range(1, len(closes)):
        if closes[i] > closes[i - 1]:
            prev = streaks[-1]
            streaks.append((prev + 1) if prev > 0 else 1.0)
        elif closes[i] < closes[i - 1]:
            prev = streaks[-1]
            streaks.append((prev - 1) if prev < 0 else -1.0)
        else:
            streaks.append(0.0)
    return streaks


def _percentile_rank(series: list[float], value: float) -> float:
    """Percentile rank of `value` within `series` (0-100)."""
    if not series:
        return 50.0
    below = sum(1 for x in series if x < value)
    return below / len(series) * 100.0


def analyze(
    bars: list,
    *,
    symbol: str = "",
    rsi_period: int = 3,
    streak_period: int = 2,
    pct_lookback: int = 100,
    history: int = 20,
) -> CRSIReport | None:
    """Compute Connors RSI from OHLCV bars.

    bars: bar objects with .close attribute.
    rsi_period: RSI period for price (default 3).
    streak_period: RSI period for streak (default 2).
    pct_lookback: bars for percentile rank of % change (default 100).
    """
    min_bars = pct_lookback + max(rsi_period, streak_period) + history + 5
    if not bars or len(bars) < min_bars:
        return None

    try:
        closes = [float(b.close) for b in bars]
    except (AttributeError, TypeError, ValueError):
        return None

    current = closes[-1]
    if current <= 0:
        return None

    n = len(closes)

    # Component 1: Price RSI(rsi_period)
    rsi_p = _rsi(closes, rsi_period)

    # Component 2: Streak RSI
    streaks = _streak_series(closes)
    rsi_s = _rsi(streaks, streak_period)

    # Component 3: Percentile rank of today's % change
    pct_today = (closes[-1] / closes[-2] - 1.0) * 100.0 if closes[-2] > 0 else 0.0
    pct_series = [
        (closes[i] / closes[i - 1] - 1.0) * 100.0
        for i in range(max(1, n - pct_lookback), n)
        if closes[i - 1] > 0
    ]
    pct_rank = _percentile_rank(pct_series[:-1], pct_today)  # exclude today itself

    crsi = (rsi_p + rsi_s + pct_rank) / 3.0

    # Streak at current bar
    streak_val = int(streaks[-1])

    overbought = crsi > 90.0
    oversold   = crsi < 10.0
    ob_soft    = crsi > 80.0
    os_soft    = crsi < 20.0

    # Score: mean-reversion interpretation — oversold → bullish score
    # Invert so that CRSI < 50 → positive score (buy), > 50 → negative (sell)
    score = (50.0 - crsi) * 2.0
    score = max(-100.0, min(100.0, score))

    if oversold:
        signal = "strong_buy"
    elif os_soft:
        signal = "buy"
    elif overbought:
        signal = "strong_sell"
    elif ob_soft:
        signal = "sell"
    else:
        signal = "neutral"

    # History: compute CRSI at each of the last `history` bars
    crsi_history: list[float] = []
    for t in range(n - history, n):
        c_slice = closes[: t + 1]
        s_slice = streaks[: t + 1]
        rp = _rsi(c_slice, rsi_period)
        rs = _rsi(s_slice, streak_period)
        if t > 0 and closes[t - 1] > 0:
            pct_t = (closes[t] / closes[t - 1] - 1.0) * 100.0
        else:
            pct_t = 0.0
        pct_win = [
            (closes[i] / closes[i - 1] - 1.0) * 100.0
            for i in range(max(1, t - pct_lookback), t)
            if closes[i - 1] > 0
        ]
        pr = _percentile_rank(pct_win, pct_t)
        crsi_history.append(round((rp + rs + pr) / 3.0, 2))

    streak_str = f"+{streak_val}" if streak_val > 0 else str(streak_val)
    ob_os = " [OVERBOUGHT]" if overbought else (" [OVERSOLD]" if oversold else "")
    verdict = (
        f"CRSI ({symbol}): {crsi:.1f}{ob_os}. "
        f"RSI(price)={rsi_p:.1f}, RSI(streak)={rsi_s:.1f}, Pctile={pct_rank:.1f}. "
        f"Streak: {streak_str} bars. Signal: {signal.replace('_', ' ')}."
    )

    return CRSIReport(
        symbol=symbol,
        crsi=round(crsi, 2),
        rsi_price=round(rsi_p, 2),
        rsi_streak=round(rsi_s, 2),
        percentile=round(pct_rank, 2),
        streak=streak_val,
        pct_change=round(pct_today, 4),
        overbought=overbought,
        oversold=oversold,
        ob_soft=ob_soft,
        os_soft=os_soft,
        score=round(score, 2),
        signal=signal,
        history=crsi_history,
        bars_used=n,
        verdict=verdict,
    )
