"""Bollinger-style mean reversion: buy when price drops below the lower band,
sell when it returns above the mean. Long-only.

This is intentionally simple and orthogonal to the trend-following
CompositeStrategy — useful as a diversifier or to A/B against in
``amms compare-strategies``.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass

from amms.data.bars import Bar
from amms.strategy.base import Signal


@dataclass(frozen=True)
class MeanReversion:
    window: int = 20
    z_buy: float = -2.0   # buy when (close - mean) / std <= z_buy
    z_exit: float = 0.0   # sell when z >= z_exit (returns to mean)
    name: str = "mean_reversion"

    def __post_init__(self) -> None:
        if self.window <= 1:
            raise ValueError(f"window must be > 1, got {self.window}")
        if self.z_buy >= self.z_exit:
            raise ValueError(
                f"z_buy ({self.z_buy}) must be < z_exit ({self.z_exit})"
            )

    @property
    def lookback(self) -> int:
        return self.window + 1

    def evaluate(self, symbol: str, bars: list[Bar]) -> Signal:
        price = bars[-1].close if bars else 0.0
        if len(bars) < self.lookback:
            return Signal(symbol, "hold", "insufficient history", price)
        closes = [b.close for b in bars[-self.window :]]
        mean = statistics.fmean(closes)
        std = statistics.stdev(closes) if len(closes) > 1 else 0.0
        if std == 0:
            return Signal(symbol, "hold", "zero stdev", price)
        z = (price - mean) / std
        if z <= self.z_buy:
            return Signal(
                symbol,
                "buy",
                f"z={z:.2f} <= {self.z_buy:.2f}",
                price,
                score=-z,  # more oversold → higher score
            )
        if z >= self.z_exit:
            return Signal(
                symbol,
                "sell",
                f"z={z:.2f} >= exit {self.z_exit:.2f}",
                price,
            )
        return Signal(symbol, "hold", f"z={z:.2f}", price)
