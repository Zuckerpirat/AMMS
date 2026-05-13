from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class Signal:
    symbol: str
    side: str        # 'buy' | 'sell'
    reason: str
    confidence: float  # 0.0 – 1.0


class BaseStrategy(ABC):
    @abstractmethod
    def generate_signals(self, symbols: list[str]) -> list[Signal]:
        ...
