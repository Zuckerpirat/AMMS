from unittest.mock import patch
import pytest


@pytest.fixture(autouse=True)
def mock_env(monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY", "test_key")
    monkeypatch.setenv("ALPACA_API_SECRET", "test_secret")


def _bars(prices: list[float], volumes: list[float]) -> list[dict]:
    return [
        {"t": None, "o": p, "h": p * 1.01, "l": p * 0.99, "c": p, "v": v}
        for p, v in zip(prices, volumes)
    ]


def test_buy_signal_on_breakout_with_volume():
    from bot.strategy.momentum import MomentumStrategy

    # 20 bars: first 19 at $100, last at $105 (breakout); last volume is 2x avg
    prices = [100.0] * 19 + [105.0]
    volumes = [1_000_000.0] * 19 + [2_000_000.0]
    bars = _bars(prices, volumes)

    with patch("bot.strategy.momentum.market.get_bars", return_value=bars):
        signals = MomentumStrategy().generate_signals(["AAPL"])

    assert len(signals) == 1
    assert signals[0].symbol == "AAPL"
    assert signals[0].side == "buy"


def test_no_signal_without_volume_surge():
    from bot.strategy.momentum import MomentumStrategy

    prices = [100.0] * 19 + [105.0]
    volumes = [1_000_000.0] * 20  # flat volume, no surge
    bars = _bars(prices, volumes)

    with patch("bot.strategy.momentum.market.get_bars", return_value=bars):
        signals = MomentumStrategy().generate_signals(["AAPL"])

    assert signals == []


def test_no_signal_without_price_breakout():
    from bot.strategy.momentum import MomentumStrategy

    prices = [100.0] * 20  # price stays flat, never breaks prior high
    volumes = [1_000_000.0] * 19 + [2_000_000.0]
    bars = _bars(prices, volumes)

    with patch("bot.strategy.momentum.market.get_bars", return_value=bars):
        signals = MomentumStrategy().generate_signals(["AAPL"])

    assert signals == []


def test_no_signal_with_insufficient_bars():
    from bot.strategy.momentum import MomentumStrategy

    bars = _bars([100.0] * 5, [1_000_000.0] * 5)

    with patch("bot.strategy.momentum.market.get_bars", return_value=bars):
        signals = MomentumStrategy().generate_signals(["AAPL"])

    assert signals == []


def test_error_in_one_symbol_does_not_stop_others():
    from bot.strategy.momentum import MomentumStrategy

    good_prices = [100.0] * 19 + [105.0]
    good_volumes = [1_000_000.0] * 19 + [2_000_000.0]
    good_bars = _bars(good_prices, good_volumes)

    def fake_get_bars(symbol, **kwargs):
        if symbol == "BAD":
            raise RuntimeError("simulated data error")
        return good_bars

    with patch("bot.strategy.momentum.market.get_bars", side_effect=fake_get_bars):
        signals = MomentumStrategy().generate_signals(["BAD", "AAPL"])

    assert len(signals) == 1
    assert signals[0].symbol == "AAPL"
