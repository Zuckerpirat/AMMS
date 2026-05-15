"""RSI-based reversal strategy.

Generates BUY signals when RSI drops below the oversold threshold,
and SELL signals when RSI rises above the overbought threshold.

Differs from MeanReversion (which uses Bollinger z-scores) by operating
directly on the RSI momentum oscillator rather than price distance from mean.
This makes it more responsive to short-term momentum exhaustion.

Parameters:
  rsi_period: int = 14         — RSI lookback period
  oversold: float = 30.0       — RSI below this → buy
  overbought: float = 70.0     — RSI above this → sell
  strong_oversold: float = 20  — boosts score for extreme readings
"""

from __future__ import annotations

from dataclasses import dataclass, field

from amms.data.bars import Bar
from amms.features.momentum import rsi as compute_rsi
from amms.strategy.base import Signal


@dataclass(frozen=True)
class RsiReversal:
    rsi_period: int = 14
    oversold: float = 30.0
    overbought: float = 70.0
    strong_oversold: float = 20.0
    name: str = "rsi_reversal"

    def __post_init__(self) -> None:
        if not (0 < self.oversold < self.overbought < 100):
            raise ValueError(
                f"Must have 0 < oversold ({self.oversold}) < "
                f"overbought ({self.overbought}) < 100"
            )
        if self.rsi_period <= 1:
            raise ValueError(f"rsi_period must be > 1, got {self.rsi_period}")

    @property
    def lookback(self) -> int:
        return self.rsi_period + 1

    def evaluate(self, symbol: str, bars: list[Bar]) -> Signal:
        price = bars[-1].close if bars else 0.0
        if len(bars) < self.lookback:
            return Signal(symbol, "hold", "insufficient history", price)

        r = compute_rsi(bars, self.rsi_period)
        if r is None:
            return Signal(symbol, "hold", "RSI undefined", price)

        if r <= self.oversold:
            score = (self.oversold - r) / self.oversold * 100
            if r <= self.strong_oversold:
                score = min(score * 1.5, 100.0)
            return Signal(
                symbol,
                "buy",
                f"RSI {r:.1f} ≤ oversold {self.oversold:.0f}",
                price,
                score=score,
            )

        if r >= self.overbought:
            return Signal(
                symbol,
                "sell",
                f"RSI {r:.1f} ≥ overbought {self.overbought:.0f}",
                price,
            )

        return Signal(symbol, "hold", f"RSI {r:.1f}", price)
