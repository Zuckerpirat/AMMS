"""VWAP-deviation strategy.

Generates signals based on price deviation from the Volume-Weighted Average Price:
  - BUY: price is significantly below VWAP (discount / mean-reversion opportunity)
  - SELL: price is significantly above VWAP (extended / take profit)
  - HOLD: price is near VWAP

Parameters:
  window: int = 20           — number of bars to compute VWAP over
  buy_deviation: float = -1.5 — buy when price < VWAP * (1 + buy_deviation/100)
  sell_deviation: float = 1.5 — sell when price > VWAP * (1 + sell_deviation/100)
"""

from __future__ import annotations

from dataclasses import dataclass

from amms.data.bars import Bar
from amms.features.vwap import vwap_deviation_pct
from amms.strategy.base import Signal


@dataclass(frozen=True)
class VwapStrategy:
    window: int = 20
    buy_deviation: float = -1.5
    sell_deviation: float = 1.5
    name: str = "vwap"

    def __post_init__(self) -> None:
        if self.buy_deviation >= 0:
            raise ValueError(f"buy_deviation must be negative, got {self.buy_deviation}")
        if self.sell_deviation <= 0:
            raise ValueError(f"sell_deviation must be positive, got {self.sell_deviation}")
        if self.window < 2:
            raise ValueError(f"window must be >= 2, got {self.window}")

    @property
    def lookback(self) -> int:
        return self.window

    def evaluate(self, symbol: str, bars: list[Bar]) -> Signal:
        price = bars[-1].close if bars else 0.0
        if len(bars) < self.lookback:
            return Signal(symbol, "hold", "insufficient history", price)

        dev = vwap_deviation_pct(price, bars, n=self.window)
        if dev is None:
            return Signal(symbol, "hold", "VWAP undefined (zero volume)", price)

        if dev <= self.buy_deviation:
            score = abs(dev - self.buy_deviation) * 10
            return Signal(
                symbol,
                "buy",
                f"price {dev:+.1f}% vs VWAP (threshold {self.buy_deviation:+.1f}%)",
                price,
                score=min(score, 100.0),
            )

        if dev >= self.sell_deviation:
            return Signal(
                symbol,
                "sell",
                f"price {dev:+.1f}% vs VWAP (threshold {self.sell_deviation:+.1f}%)",
                price,
            )

        return Signal(symbol, "hold", f"price {dev:+.1f}% vs VWAP", price)
