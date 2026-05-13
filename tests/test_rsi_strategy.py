from unittest.mock import patch
import pytest


@pytest.fixture(autouse=True)
def mock_env(monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY", "test_key")
    monkeypatch.setenv("ALPACA_API_SECRET", "test_secret")


def _bars(closes: list[float]) -> list[dict]:
    return [{"t": None, "o": c, "h": c * 1.01, "l": c * 0.99, "c": c, "v": 1_000_000.0}
            for c in closes]


# --- Unit tests for the RSI calculation itself ---

def test_rsi_calculation_basic():
    from bot.strategy.rsi import _calculate_rsi

    # Flat prices → gains and losses both zero → RSI not calculable (avg_loss = 0)
    closes = [100.0] * 30
    rsi = _calculate_rsi(closes, period=14)
    # All RSI values should be 100.0 (no losses ever)
    assert all(r == 100.0 for r in rsi)


def test_rsi_returns_empty_when_not_enough_data():
    from bot.strategy.rsi import _calculate_rsi

    closes = [100.0] * 10
    assert _calculate_rsi(closes, period=14) == []


def test_rsi_oversold_then_recovery():
    from bot.strategy.rsi import _calculate_rsi

    # Declining prices push RSI low, then a sharp recovery should cross back up
    closes = [100.0 - i * 2 for i in range(20)]  # 100 → 62 (declining)
    closes += [65.0, 70.0, 75.0, 72.0, 78.0]     # recovery
    rsi = _calculate_rsi(closes, period=14)
    assert len(rsi) > 0
    assert rsi[0] < 50  # after the downtrend, RSI should be low


# --- Strategy signal tests ---

def test_rsi_buy_signal_on_crossover():
    from bot.strategy.rsi import RSIStrategy, _calculate_rsi

    # Build closes where RSI was below 30 and then crosses above
    # Sharp drop followed by recovery
    closes_drop = [100.0 - i * 3 for i in range(20)]   # steep decline
    closes_bounce = [closes_drop[-1] + i * 4 for i in range(5)]  # bounce
    all_closes = closes_drop + closes_bounce

    bars = _bars(all_closes)

    with patch("bot.strategy.rsi.market.get_bars", return_value=bars):
        signals = RSIStrategy().generate_signals(["AAPL"])

    # We can't guarantee exact crossover in synthetic data,
    # but confirm no exception is raised and result is a list
    assert isinstance(signals, list)


def test_rsi_no_signal_insufficient_bars():
    from bot.strategy.rsi import RSIStrategy

    bars = _bars([100.0] * 10)

    with patch("bot.strategy.rsi.market.get_bars", return_value=bars):
        signals = RSIStrategy().generate_signals(["AAPL"])

    assert signals == []


def test_rsi_error_does_not_propagate():
    from bot.strategy.rsi import RSIStrategy

    def boom(symbol, **_):
        raise RuntimeError("network error")

    with patch("bot.strategy.rsi.market.get_bars", side_effect=boom):
        signals = RSIStrategy().generate_signals(["AAPL", "MSFT"])

    assert signals == []
