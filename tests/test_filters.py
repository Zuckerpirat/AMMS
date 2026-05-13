from unittest.mock import patch
import pytest


@pytest.fixture(autouse=True)
def mock_env(monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY", "test_key")
    monkeypatch.setenv("ALPACA_API_SECRET", "test_secret")


def _bars(price: float, atr_pct: float, n: int = 20) -> list[dict]:
    """Generate bars where the true range = atr_pct * price every day."""
    half = price * atr_pct / 2
    return [
        {"t": None, "o": price, "h": price + half, "l": price - half, "c": price, "v": 1_000_000.0}
        for _ in range(n)
    ]


def test_symbol_passes_when_atr_in_range():
    from bot.strategy.filters import atr_filter

    # ATR per bar = 2% of price → within default [0.5%, 8%]
    bars = _bars(price=100.0, atr_pct=0.02, n=20)

    with patch("bot.strategy.filters.market.get_bars", return_value=bars):
        result = atr_filter(["AAPL"])

    assert "AAPL" in result


def test_symbol_filtered_when_atr_too_low():
    from bot.strategy.filters import atr_filter

    # ATR per bar = 0.1% → below ATR_MIN_PCT (0.5%)
    bars = _bars(price=100.0, atr_pct=0.001, n=20)

    with patch("bot.strategy.filters.market.get_bars", return_value=bars):
        result = atr_filter(["BORING"])

    assert "BORING" not in result


def test_symbol_filtered_when_atr_too_high():
    from bot.strategy.filters import atr_filter

    # ATR per bar = 15% → above ATR_MAX_PCT (8%)
    bars = _bars(price=100.0, atr_pct=0.15, n=20)

    with patch("bot.strategy.filters.market.get_bars", return_value=bars):
        result = atr_filter(["WILD"])

    assert "WILD" not in result


def test_symbol_passes_when_not_enough_bars():
    from bot.strategy.filters import atr_filter

    # Too few bars → benefit of the doubt, symbol passes
    bars = _bars(price=100.0, atr_pct=0.02, n=5)

    with patch("bot.strategy.filters.market.get_bars", return_value=bars):
        result = atr_filter(["SHORT"])

    assert "SHORT" in result


def test_error_passes_symbol_through():
    from bot.strategy.filters import atr_filter

    with patch("bot.strategy.filters.market.get_bars", side_effect=RuntimeError("oops")):
        result = atr_filter(["ERR"])

    assert "ERR" in result


def test_multiple_symbols_mixed():
    from bot.strategy.filters import atr_filter

    good_bars = _bars(price=100.0, atr_pct=0.02, n=20)
    bad_bars = _bars(price=100.0, atr_pct=0.20, n=20)

    def fake_get_bars(symbol, **_):
        return good_bars if symbol == "GOOD" else bad_bars

    with patch("bot.strategy.filters.market.get_bars", side_effect=fake_get_bars):
        result = atr_filter(["GOOD", "BAD"])

    assert "GOOD" in result
    assert "BAD" not in result
