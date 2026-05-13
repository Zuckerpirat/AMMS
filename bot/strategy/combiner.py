import logging
from collections import defaultdict
from bot.strategy.base import BaseStrategy, Signal

logger = logging.getLogger(__name__)


class StrategySignalCombiner(BaseStrategy):
    """
    Runs multiple strategies and merges their signals.

    min_agreement=1  → any single strategy can trigger a trade
    min_agreement=2  → at least two strategies must agree on the same symbol
    """

    def __init__(self, strategies: list[BaseStrategy], min_agreement: int = 1):
        self.strategies = strategies
        self.min_agreement = min_agreement

    def generate_signals(self, symbols: list[str]) -> list[Signal]:
        by_symbol: dict[str, list[Signal]] = defaultdict(list)

        for strategy in self.strategies:
            for signal in strategy.generate_signals(symbols):
                by_symbol[signal.symbol].append(signal)

        combined: list[Signal] = []
        for symbol, signals in by_symbol.items():
            buy_signals = [s for s in signals if s.side == "buy"]
            if len(buy_signals) < self.min_agreement:
                logger.debug(
                    "%s: %d/%d strategies agree — need %d",
                    symbol, len(buy_signals), len(self.strategies), self.min_agreement,
                )
                continue

            reasons = " | ".join(s.reason for s in buy_signals)
            avg_confidence = sum(s.confidence for s in buy_signals) / len(buy_signals)
            agreement_tag = f"[{len(buy_signals)}/{len(self.strategies)} strategies]"

            combined.append(Signal(
                symbol=symbol,
                side="buy",
                reason=f"{agreement_tag} {reasons}",
                confidence=avg_confidence,
            ))
            logger.info(
                "Combined BUY: %s  agreement=%d/%d  confidence=%.2f",
                symbol, len(buy_signals), len(self.strategies), avg_confidence,
            )

        return combined
