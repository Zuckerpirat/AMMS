from unittest.mock import patch
import pytest


# Patch env vars before importing settings
@pytest.fixture(autouse=True)
def mock_env(monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY", "test_key")
    monkeypatch.setenv("ALPACA_API_SECRET", "test_secret")


def test_can_buy_normal():
    from bot.risk import manager
    assert manager.can_buy("AAPL", 150.0, 100_000.0, 5) is True


def test_can_buy_price_below_min():
    from bot.risk import manager
    assert manager.can_buy("AAPL", 0.50, 100_000.0, 5) is False


def test_can_buy_price_above_max():
    from bot.risk import manager
    assert manager.can_buy("AAPL", 600.0, 100_000.0, 5) is False


def test_can_buy_at_max_positions():
    from bot.risk import manager
    assert manager.can_buy("AAPL", 150.0, 100_000.0, 10) is False


def test_calculate_qty_normal():
    from bot.risk import manager
    # 5% of $100k = $5000, $5000 / $150 = 33
    assert manager.calculate_qty(150.0, 100_000.0) == 33


def test_calculate_qty_zero_when_price_exceeds_budget():
    from bot.risk import manager
    # 5% of $100 = $5, $5 / $10 = 0 (floor)
    assert manager.calculate_qty(10.0, 100.0) == 0


def test_stop_loss_triggered():
    from bot.risk import manager
    # 5% below entry
    assert manager.should_stop_loss(100.0, 94.0) is True


def test_stop_loss_not_triggered():
    from bot.risk import manager
    assert manager.should_stop_loss(100.0, 96.0) is False


def test_take_profit_triggered():
    from bot.risk import manager
    # 15% above entry
    assert manager.should_take_profit(100.0, 116.0) is True


def test_take_profit_not_triggered():
    from bot.risk import manager
    assert manager.should_take_profit(100.0, 114.0) is False
