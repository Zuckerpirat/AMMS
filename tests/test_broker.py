from unittest.mock import MagicMock, patch
import pytest


@pytest.fixture(autouse=True)
def mock_env(monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY", "test_key")
    monkeypatch.setenv("ALPACA_API_SECRET", "test_secret")


@pytest.fixture(autouse=True)
def reset_client():
    # Each test gets a fresh _client so mocks don't bleed between tests
    import bot.broker.alpaca as alpaca
    alpaca._client = None
    yield
    alpaca._client = None


def _make_mock_order(order_id: str):
    order = MagicMock()
    order.id = order_id
    return order


def test_place_buy_order():
    import bot.broker.alpaca as alpaca

    mock_client = MagicMock()
    mock_client.submit_order.return_value = _make_mock_order("buy-123")

    with patch("bot.broker.alpaca.TradingClient", return_value=mock_client):
        order = alpaca.place_market_order("AAPL", 10, "buy")

    mock_client.submit_order.assert_called_once()
    call_args = mock_client.submit_order.call_args[0][0]
    assert call_args.symbol == "AAPL"
    assert call_args.qty == 10
    assert order.id == "buy-123"


def test_place_sell_order():
    import bot.broker.alpaca as alpaca
    from alpaca.trading.enums import OrderSide

    mock_client = MagicMock()
    mock_client.submit_order.return_value = _make_mock_order("sell-456")

    with patch("bot.broker.alpaca.TradingClient", return_value=mock_client):
        order = alpaca.place_market_order("MSFT", 5, "sell")

    call_args = mock_client.submit_order.call_args[0][0]
    assert call_args.side == OrderSide.SELL
    assert order.id == "sell-456"


def test_get_positions_returns_list():
    import bot.broker.alpaca as alpaca

    mock_client = MagicMock()
    mock_client.get_all_positions.return_value = []

    with patch("bot.broker.alpaca.TradingClient", return_value=mock_client):
        result = alpaca.get_positions()

    assert result == []
    mock_client.get_all_positions.assert_called_once()


def test_cancel_all_orders():
    import bot.broker.alpaca as alpaca

    mock_client = MagicMock()

    with patch("bot.broker.alpaca.TradingClient", return_value=mock_client):
        alpaca.cancel_all_orders()

    mock_client.cancel_orders.assert_called_once()
