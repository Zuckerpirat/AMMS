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


_STRATEGY_REGISTRY: dict[str, type] = {}


def register_strategy(name: str, cls: type) -> None:
    """Register a Strategy class under ``name`` so it can be built via
    ``build_strategy(name, params)`` and named in config.yaml."""
    _STRATEGY_REGISTRY[name] = cls


def registered_strategies() -> dict[str, type]:
    return dict(_STRATEGY_REGISTRY)


def build_strategy(name: str, params: dict[str, Any]) -> Strategy:
    """Construct a strategy by name."""
    if name not in _STRATEGY_REGISTRY:
        # Lazy-import the built-ins so they show up after first import.
        from amms.strategy.composite import CompositeStrategy
        from amms.strategy.sma_cross import SmaCross

        _STRATEGY_REGISTRY.setdefault("sma_cross", SmaCross)
        _STRATEGY_REGISTRY.setdefault("composite", CompositeStrategy)
    cls = _STRATEGY_REGISTRY.get(name)
    if cls is None:
        known = ", ".join(sorted(_STRATEGY_REGISTRY)) or "(none)"
        raise ValueError(f"Unknown strategy: {name!r}. Known: {known}")
    return cls(**params)
