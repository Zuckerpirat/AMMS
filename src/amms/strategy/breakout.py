"""Donchian-channel breakout: buy on a new N-day high, exit on a new N-day low.

A canonical trend-following baseline. Pairs well with the mean-reversion
strategy in ``amms compare-strategies`` for sanity-checking the composite.
"""

from __future__ import annotations

from dataclasses import dataclass

from amms.data.bars import Bar
from amms.strategy.base import Signal


@dataclass(frozen=True)
class Breakout:
    entry_window: int = 20
    exit_window: int = 10
    name: str = "breakout"

    def __post_init__(self) -> None:
        if self.entry_window <= 1 or self.exit_window <= 1:
            raise ValueError("entry_window and exit_window must be > 1")

    @property
    def lookback(self) -> int:
        return max(self.entry_window, self.exit_window) + 1

    def evaluate(self, symbol: str, bars: list[Bar]) -> Signal:
        price = bars[-1].close if bars else 0.0
        if len(bars) < self.lookback:
            return Signal(symbol, "hold", "insufficient history", price)

        # Reference highs/lows exclude the latest bar so a fresh high "breaks" it.
        entry_window = bars[-(self.entry_window + 1) : -1]
        exit_window = bars[-(self.exit_window + 1) : -1]
        prior_high = max(b.high for b in entry_window)
        prior_low = min(b.low for b in exit_window)

        if price > prior_high:
            return Signal(
                symbol,
                "buy",
                f"close {price:.2f} > {self.entry_window}d high {prior_high:.2f}",
                price,
                score=(price - prior_high) / max(prior_high, 0.01),
            )
        if price < prior_low:
            return Signal(
                symbol,
                "sell",
                f"close {price:.2f} < {self.exit_window}d low {prior_low:.2f}",
                price,
            )
        return Signal(symbol, "hold", "no breakout", price)
