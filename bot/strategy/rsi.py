import logging
from bot.strategy.base import BaseStrategy, Signal
from bot.data import market
from config import settings

logger = logging.getLogger(__name__)


def _calculate_rsi(closes: list[float], period: int) -> list[float]:
    """
    Wilder's RSI. Returns one RSI value per bar starting after the first `period`
    bars used to seed the initial average. Returns [] when not enough data.
    """
    if len(closes) < period + 1:
        return []

    deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    gains = [max(d, 0.0) for d in deltas]
    losses = [abs(min(d, 0.0)) for d in deltas]

    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    rsi: list[float] = []
    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss == 0:
            rsi.append(100.0)
        else:
            rs = avg_gain / avg_loss
            rsi.append(100.0 - (100.0 / (1.0 + rs)))

    return rsi


class RSIStrategy(BaseStrategy):
    """
    Buy when RSI(14) crosses upward through the oversold threshold (default 30).
    Requires the previous bar's RSI to be below the threshold and today's above it,
    confirming a bounce rather than just touching the level.
    """

    def generate_signals(self, symbols: list[str]) -> list[Signal]:
        signals: list[Signal] = []
        for symbol in symbols:
            try:
                signal = self._evaluate(symbol)
                if signal:
                    signals.append(signal)
            except Exception as exc:
                logger.warning("RSI error on %s: %s", symbol, exc)
        return signals

    def _evaluate(self, symbol: str) -> Signal | None:
        # Need period*2 + a few bars so Wilder's smoothing is stable
        min_bars = settings.RSI_PERIOD * 2 + 5
        bars = market.get_bars(symbol, days=min_bars + 14)
        if len(bars) < min_bars:
            logger.debug("%s: only %d bars for RSI, need %d", symbol, len(bars), min_bars)
            return None

        closes = [b["c"] for b in bars]
        rsi_values = _calculate_rsi(closes, settings.RSI_PERIOD)

        if len(rsi_values) < 2:
            return None

        prev_rsi = rsi_values[-2]
        curr_rsi = rsi_values[-1]
        threshold = settings.RSI_OVERSOLD

        if prev_rsi < threshold <= curr_rsi:
            logger.info(
                "RSI BUY signal: %s  rsi=%.1f (was %.1f, threshold=%.1f)",
                symbol, curr_rsi, prev_rsi, threshold,
            )
            return Signal(
                symbol=symbol,
                side="buy",
                reason=f"RSI crossed above {threshold:.0f} (was {prev_rsi:.1f}, now {curr_rsi:.1f})",
                confidence=min(1.0, (threshold - prev_rsi) / threshold * 2),
            )

        return None
