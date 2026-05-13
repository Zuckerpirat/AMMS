from __future__ import annotations

from dataclasses import dataclass

from amms.data.bars import Bar
from amms.strategy.base import Signal


@dataclass(frozen=True)
class SmaCross:
    """Long-only fast/slow simple-moving-average crossover.

    Emits 'buy' when fast SMA crosses above slow SMA on the latest bar, 'sell'
    when it crosses below, and 'hold' otherwise. Crossover detection requires
    one extra bar of history.
    """

    fast: int = 10
    slow: int = 30
    name: str = "sma_cross"

    def __post_init__(self) -> None:
        if self.fast <= 0 or self.slow <= 0:
            raise ValueError("fast and slow must be positive")
        if self.fast >= self.slow:
            raise ValueError(f"fast ({self.fast}) must be < slow ({self.slow})")

    @property
    def lookback(self) -> int:
        return self.slow + 1

    def evaluate(self, symbol: str, bars: list[Bar]) -> Signal:
        price = bars[-1].close if bars else 0.0
        if len(bars) < self.lookback:
            return Signal(
                symbol,
                "hold",
                f"need {self.lookback} bars, have {len(bars)}",
                price,
            )
        closes = [b.close for b in bars]

        def sma(window: int, offset: int) -> float:
            window_slice = (
                closes[-(window + offset) : -offset] if offset else closes[-window:]
            )
            return sum(window_slice) / window

        fast_now = sma(self.fast, 0)
        slow_now = sma(self.slow, 0)
        fast_prev = sma(self.fast, 1)
        slow_prev = sma(self.slow, 1)

        if fast_prev <= slow_prev and fast_now > slow_now:
            return Signal(
                symbol,
                "buy",
                f"SMA{self.fast} crossed above SMA{self.slow}",
                price,
            )
        if fast_prev >= slow_prev and fast_now < slow_now:
            return Signal(
                symbol,
                "sell",
                f"SMA{self.fast} crossed below SMA{self.slow}",
                price,
            )
        return Signal(symbol, "hold", "no crossover", price)
