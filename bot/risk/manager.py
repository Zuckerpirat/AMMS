import logging
from config import settings

logger = logging.getLogger(__name__)


def can_buy(symbol: str, price: float, portfolio_value: float, open_positions: int) -> bool:
    if price < settings.MIN_STOCK_PRICE:
        logger.debug("SKIP %s: price %.2f below min %.2f", symbol, price, settings.MIN_STOCK_PRICE)
        return False
    if price > settings.MAX_STOCK_PRICE:
        logger.debug("SKIP %s: price %.2f above max %.2f", symbol, price, settings.MAX_STOCK_PRICE)
        return False
    if open_positions >= settings.MAX_OPEN_POSITIONS:
        logger.debug("SKIP %s: at max open positions (%d)", symbol, settings.MAX_OPEN_POSITIONS)
        return False
    return True


def calculate_qty(price: float, portfolio_value: float) -> int:
    dollars = portfolio_value * settings.MAX_POSITION_PCT
    qty = int(dollars / price)
    return qty if qty >= 1 else 0


def should_stop_loss(entry_price: float, current_price: float) -> bool:
    return current_price <= entry_price * (1.0 - settings.STOP_LOSS_PCT)


def should_take_profit(entry_price: float, current_price: float) -> bool:
    return current_price >= entry_price * (1.0 + settings.TAKE_PROFIT_PCT)
