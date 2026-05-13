from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol

SignalKind = Literal["buy", "sell", "hold"]


@dataclass(frozen=True)
class Signal:
    symbol: str
    kind: SignalKind
    reason: str
    price: float


class Strategy(Protocol):
    """A strategy maps recent closes to a single signal for a symbol.

    Pure: no I/O, no DB, no broker calls. Easy to unit-test and to swap into
    the backtester unchanged.
    """

    name: str

    @property
    def lookback(self) -> int:
        """Minimum number of bars the strategy needs to emit a real signal."""
        ...

    def evaluate(self, symbol: str, closes: list[float]) -> Signal:
        ...


def build_strategy(name: str, params: dict[str, Any]) -> Strategy:
    """Construct a strategy by name. Kept small until we have more than one."""
    from amms.strategy.sma_cross import SmaCross

    if name == "sma_cross":
        return SmaCross(**params)
    raise ValueError(f"Unknown strategy: {name!r}")
