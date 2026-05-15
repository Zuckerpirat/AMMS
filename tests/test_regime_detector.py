from __future__ import annotations

import pytest

from amms.analysis.regime import MarketRegime, detect_regime, _FALLBACK
from amms.data.bars import Bar


def _bar(symbol: str, close: float, i: int = 0) -> Bar:
    return Bar(symbol, "1D", f"2026-01-{1 + i:02d}", close, close + 0.5, close - 0.5, close, 1000)


class _FakeData:
    """Configurable fake data client for regime tests."""

    def __init__(self, spy_prices: list[float], vixy_prices: list[float] | None = None) -> None:
        self._spy = spy_prices
        self._vixy = vixy_prices or [20.0, 20.5]  # calm default

    def get_bars(self, symbol: str, *, limit: int = 210) -> list[Bar]:
        if symbol == "SPY":
            prices = self._spy[-limit:]
        elif symbol == "VIXY":
            prices = (self._vixy or [])[-limit:]
        else:
            return []
        return [_bar(symbol, p, i) for i, p in enumerate(prices)]


def _spy_prices(n: int, start: float = 400.0, trend: float = 0.5) -> list[float]:
    """Generate n SPY prices with a given daily drift."""
    prices = [start]
    for _ in range(n - 1):
        prices.append(prices[-1] + trend)
    return prices


def test_fallback_when_no_data() -> None:
    class _Empty:
        def get_bars(self, symbol, *, limit=210):
            return []

    regime = detect_regime(_Empty())
    assert regime.label == "neutral"
    assert regime.confidence == 0.0


def test_bull_regime_spy_above_both_smas() -> None:
    # 210 bars trending up → price above SMA-50 and SMA-200
    spy = _spy_prices(210, start=350.0, trend=0.5)
    data = _FakeData(spy_prices=spy)
    regime = detect_regime(data)
    assert regime.label == "bull"
    assert regime.spy_vs_sma50 is not None
    assert regime.spy_vs_sma50 > 0


def test_bear_regime_vixy_spike() -> None:
    spy = _spy_prices(210, start=400.0, trend=0.0)
    # VIXY spikes 10%
    data = _FakeData(spy_prices=spy, vixy_prices=[20.0, 22.0])
    regime = detect_regime(data)
    assert regime.label == "bear"
    assert "stressed" in regime.reason


def test_bear_regime_spy_below_sma200() -> None:
    # Declining market: 210 bars trending down strongly
    spy = _spy_prices(210, start=500.0, trend=-1.0)
    data = _FakeData(spy_prices=spy)
    regime = detect_regime(data)
    assert regime.label == "bear"
    assert regime.spy_vs_sma200 is not None
    assert regime.spy_vs_sma200 < 0


def test_risk_multiplier_values() -> None:
    assert MarketRegime("bull", 1.0, "").risk_multiplier == 1.0
    assert MarketRegime("neutral", 0.5, "").risk_multiplier == 0.75
    assert MarketRegime("bear", 1.0, "").risk_multiplier == 0.5


def test_is_bull_is_bear_properties() -> None:
    bull = MarketRegime("bull", 0.8, "")
    bear = MarketRegime("bear", 0.9, "")
    neutral = MarketRegime("neutral", 0.5, "")
    assert bull.is_bull and not bull.is_bear
    assert bear.is_bear and not bear.is_bull
    assert not neutral.is_bull and not neutral.is_bear


def test_confidence_range() -> None:
    spy = _spy_prices(210)
    data = _FakeData(spy_prices=spy)
    regime = detect_regime(data)
    assert 0.0 <= regime.confidence <= 1.0


def test_error_in_data_client_returns_fallback() -> None:
    class _Broken:
        def get_bars(self, symbol, *, limit=210):
            raise RuntimeError("network error")

    regime = detect_regime(_Broken())
    assert regime.label == "neutral"
    assert regime.confidence == 0.0
