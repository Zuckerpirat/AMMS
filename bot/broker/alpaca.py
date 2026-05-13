import logging
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from config import settings

logger = logging.getLogger(__name__)

_client: TradingClient | None = None


def get_client() -> TradingClient:
    global _client
    if _client is None:
        _client = TradingClient(
            settings.ALPACA_API_KEY,
            settings.ALPACA_API_SECRET,
            paper=True,
        )
    return _client


def get_account():
    return get_client().get_account()


def get_positions() -> list:
    return get_client().get_all_positions()


def get_clock():
    return get_client().get_clock()


def place_market_order(symbol: str, qty: int, side: str):
    order_side = OrderSide.BUY if side == "buy" else OrderSide.SELL
    request = MarketOrderRequest(
        symbol=symbol,
        qty=qty,
        side=order_side,
        time_in_force=TimeInForce.DAY,
    )
    order = get_client().submit_order(request)
    logger.info("Order placed: %s %s x%d", side.upper(), symbol, qty)
    return order


def cancel_all_orders() -> None:
    get_client().cancel_orders()
