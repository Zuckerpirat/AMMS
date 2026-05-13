import logging
from bot.data import market
from config import settings

logger = logging.getLogger(__name__)


def _true_ranges(bars: list[dict]) -> list[float]:
    trs: list[float] = []
    for i in range(1, len(bars)):
        high, low, prev_close = bars[i]["h"], bars[i]["l"], bars[i - 1]["c"]
        trs.append(max(high - low, abs(high - prev_close), abs(low - prev_close)))
    return trs


def _atr(bars: list[dict], period: int) -> float | None:
    trs = _true_ranges(bars)
    if len(trs) < period:
        return None
    return sum(trs[-period:]) / period


def atr_filter(symbols: list[str]) -> list[str]:
    """
    Return the subset of symbols whose ATR(14) as a percentage of price falls
    within [ATR_MIN_PCT, ATR_MAX_PCT]. Symbols with too little movement (boring)
    or too much (dangerous) are dropped before strategies run.
    """
    passing: list[str] = []
    needed = settings.ATR_PERIOD + 14  # buffer for weekends

    for symbol in symbols:
        try:
            bars = market.get_bars(symbol, days=needed)
            if len(bars) < settings.ATR_PERIOD + 1:
                logger.debug("%s: not enough bars for ATR filter", symbol)
                passing.append(symbol)  # give benefit of the doubt
                continue

            atr_val = _atr(bars, settings.ATR_PERIOD)
            if atr_val is None:
                passing.append(symbol)
                continue

            price = bars[-1]["c"]
            atr_pct = atr_val / price

            if settings.ATR_MIN_PCT <= atr_pct <= settings.ATR_MAX_PCT:
                passing.append(symbol)
            else:
                logger.debug(
                    "FILTER %s: ATR%% = %.3f (allowed %.3f–%.3f)",
                    symbol, atr_pct, settings.ATR_MIN_PCT, settings.ATR_MAX_PCT,
                )
        except Exception as exc:
            logger.warning("ATR filter error on %s: %s — letting it through", symbol, exc)
            passing.append(symbol)

    return passing
