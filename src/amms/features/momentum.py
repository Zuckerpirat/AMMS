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


def macd(
    bars: list[Bar],
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[float, float, float] | None:
    """MACD line, signal line, and histogram.

    Returns (macd_line, signal_line, histogram) or None if not enough history.
    MACD line = EMA(fast) - EMA(slow)
    Signal = EMA(MACD, signal)
    Histogram = MACD - Signal
    """
    if len(bars) < slow + signal:
        return None
    k_fast = 2.0 / (fast + 1)
    k_slow = 2.0 / (slow + 1)
    k_sig = 2.0 / (signal + 1)

    closes = [b.close for b in bars]
    # warm-up: compute EMA for entire series
    ema_fast = closes[0]
    ema_slow = closes[0]
    for c in closes[1:]:
        ema_fast = c * k_fast + ema_fast * (1 - k_fast)
        ema_slow = c * k_slow + ema_slow * (1 - k_slow)

    macd_line = ema_fast - ema_slow

    # Compute MACD line over last (slow + signal) bars to get signal EMA
    macd_values: list[float] = []
    ema_f = closes[-(slow + signal)]; ema_s = closes[-(slow + signal)]
    for c in closes[-(slow + signal):]:
        ema_f = c * k_fast + ema_f * (1 - k_fast)
        ema_s = c * k_slow + ema_s * (1 - k_slow)
        macd_values.append(ema_f - ema_s)

    sig_val = macd_values[0]
    for mv in macd_values[1:]:
        sig_val = mv * k_sig + sig_val * (1 - k_sig)

    histogram = macd_line - sig_val
    return macd_line, sig_val, histogram


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
