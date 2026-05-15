from __future__ import annotations

from amms.data.bars import Bar


def n_day_return(bars: list[Bar], n: int = 20) -> float | None:
    """Simple return over the last ``n`` bars: close[-1] / close[-(n+1)] - 1.

    Returns None if there isn't enough history or the prior close is zero.
    """
    if n <= 0:
        raise ValueError(f"n must be > 0, got {n}")
    if len(bars) < n + 1:
        return None
    last = bars[-1].close
    prior = bars[-(n + 1)].close
    if prior <= 0:
        return None
    return last / prior - 1.0


def ema(bars: list[Bar], n: int = 20) -> float | None:
    """Exponential moving average of closing prices over ``n`` bars.

    Uses the standard multiplier: k = 2 / (n + 1).
    Returns None when fewer than n bars are available.
    """
    if n <= 0:
        raise ValueError(f"n must be > 0, got {n}")
    if len(bars) < n:
        return None
    closes = [b.close for b in bars[-n - n:]]  # use up to 2× window for warm-up
    k = 2.0 / (n + 1)
    ema_val = closes[0]
    for c in closes[1:]:
        ema_val = c * k + ema_val * (1 - k)
    return ema_val


def sma(bars: list[Bar], n: int = 20) -> float | None:
    """Simple moving average of the last ``n`` closing prices."""
    if n <= 0:
        raise ValueError(f"n must be > 0, got {n}")
    if len(bars) < n:
        return None
    return sum(b.close for b in bars[-n:]) / n


def rsi(bars: list[Bar], n: int = 14) -> float | None:
    """Wilder-style RSI computed over the last ``n`` close-to-close changes.

    Uses simple averages of gains/losses (no exponential smoothing) for
    transparency. Returns 100.0 when there are no losses, or None when
    history is too short.
    """
    if n <= 0:
        raise ValueError(f"n must be > 0, got {n}")
    if len(bars) < n + 1:
        return None
    gains: list[float] = []
    losses: list[float] = []
    for i in range(len(bars) - n, len(bars)):
        diff = bars[i].close - bars[i - 1].close
        if diff >= 0:
            gains.append(diff)
            losses.append(0.0)
        else:
            gains.append(0.0)
            losses.append(-diff)
    avg_gain = sum(gains) / n
    avg_loss = sum(losses) / n
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))
