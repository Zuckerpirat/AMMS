import logging
from bot.strategy.base import BaseStrategy, Signal
from bot.data import market
from config import settings

logger = logging.getLogger(__name__)


class MomentumStrategy(BaseStrategy):
    """
    Buy when: price breaks above the N-day high AND volume is >= VOLUME_MULTIPLIER
    times the N-day average. Both conditions must be true simultaneously.
    """

    def generate_signals(self, symbols: list[str]) -> list[Signal]:
        signals: list[Signal] = []
        for symbol in symbols:
            try:
                signal = self._evaluate(symbol)
                if signal:
                    signals.append(signal)
            except Exception as exc:
                logger.warning("Error evaluating %s: %s", symbol, exc)
        return signals

    def _evaluate(self, symbol: str) -> Signal | None:
        bars = market.get_bars(symbol, days=settings.MOMENTUM_LOOKBACK + 14)
        if len(bars) < settings.MOMENTUM_LOOKBACK:
            logger.debug("%s: only %d bars, need %d — skipping", symbol, len(bars), settings.MOMENTUM_LOOKBACK)
            return None

        lookback = bars[-settings.MOMENTUM_LOOKBACK :]
        prior = lookback[:-1]   # all bars except today
        today = lookback[-1]

        prior_high = max(b["h"] for b in prior)
        avg_volume = sum(b["v"] for b in prior) / len(prior)

        price = today["c"]
        volume = today["v"]
        vol_ratio = volume / avg_volume if avg_volume else 0.0

        breakout = price > prior_high
        volume_surge = vol_ratio >= settings.VOLUME_MULTIPLIER

        if breakout and volume_surge:
            logger.info(
                "BUY signal: %s  price=%.2f  prior_high=%.2f  vol_ratio=%.1fx",
                symbol, price, prior_high, vol_ratio,
            )
            return Signal(
                symbol=symbol,
                side="buy",
                reason=f"breakout above {prior_high:.2f} with {vol_ratio:.1f}x volume",
                confidence=min(1.0, (vol_ratio - 1.0) * 0.4),
            )

        return None
