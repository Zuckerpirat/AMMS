from amms.strategy.base import (
    Signal,
    SignalKind,
    Strategy,
    build_strategy,
    register_strategy,
)
from amms.strategy.breakout import Breakout
from amms.strategy.composite import CompositeStrategy
from amms.strategy.mean_reversion import MeanReversion
from amms.strategy.sma_cross import SmaCross

register_strategy("sma_cross", SmaCross)
register_strategy("composite", CompositeStrategy)
register_strategy("mean_reversion", MeanReversion)
register_strategy("breakout", Breakout)

__all__ = [
    "Breakout",
    "CompositeStrategy",
    "MeanReversion",
    "Signal",
    "SignalKind",
    "SmaCross",
    "Strategy",
    "build_strategy",
    "register_strategy",
]
