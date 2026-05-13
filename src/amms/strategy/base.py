from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol

from amms.data.bars import Bar

SignalKind = Literal["buy", "sell", "hold"]


@dataclass(frozen=True)
class Signal:
    symbol: str
    kind: SignalKind
    reason: str
    price: float
    score: float = 0.0


class Strategy(Protocol):
    """A strategy maps recent bars to a single signal for a symbol.

    Pure: no I/O, no DB, no broker calls. Easy to unit-test and to swap into
    the backtester unchanged.
    """

    name: str

    @property
    def lookback(self) -> int:
        """Minimum number of bars the strategy needs to emit a real signal."""
        ...

    def evaluate(self, symbol: str, bars: list[Bar]) -> Signal:
        ...


def build_strategy(name: str, params: dict[str, Any]) -> Strategy:
    """Construct a strategy by name."""
    if name == "sma_cross":
        from amms.strategy.sma_cross import SmaCross

        return SmaCross(**params)
    if name == "composite":
        from amms.strategy.composite import CompositeStrategy

        return CompositeStrategy(**params)
    raise ValueError(f"Unknown strategy: {name!r}")
