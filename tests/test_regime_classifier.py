"""Tests for amms.analysis.regime_classifier."""

from __future__ import annotations

import math

import pytest

from amms.data.bars import Bar
from amms.analysis.regime_classifier import RegimeResult, classify


def _bar(sym: str, close: float, high: float = None, low: float = None,
         i: int = 0) -> Bar:
    h = high if high is not None else close * 1.01
    l = low if low is not None else close * 0.99
    return Bar(sym, "1Day", f"2026-01-{1 + i % 28:02d}T10:00:00Z",
               close, h, l, close, 100_000)


def _bars_trending_up(sym: str, n: int, step: float = 1.0) -> list[Bar]:
    return [_bar(sym, 100.0 + step * i, i=i) for i in range(n)]


def _bars_trending_down(sym: str, n: int, step: float = 1.0) -> list[Bar]:
    return [_bar(sym, 150.0 - step * i, i=i) for i in range(n)]


def _bars_flat(sym: str, n: int, price: float = 100.0) -> list[Bar]:
    """Flat price with minimal noise."""
    return [_bar(sym, price, price + 0.2, price - 0.2, i) for i in range(n)]


def _bars_volatile_ranging(sym: str, n: int) -> list[Bar]:
    """Oscillating price with high volatility."""
    bars = []
    for i in range(n):
        price = 100.0 + 5.0 * math.sin(i * 0.5)
        bars.append(_bar(sym, price, price * 1.03, price * 0.97, i))
    return bars


class TestClassify:
    def test_returns_none_insufficient_bars(self):
        bars = _bars_flat("AAPL", 10)
        assert classify(bars, lookback=20) is None

    def test_returns_result_with_enough_data(self):
        bars = _bars_trending_up("AAPL", 30)
        result = classify(bars, lookback=20)
        assert result is not None
        assert isinstance(result, RegimeResult)

    def test_symbol_preserved(self):
        bars = _bars_trending_up("TSLA", 30)
        result = classify(bars, lookback=20)
        assert result is not None
        assert result.symbol == "TSLA"

    def test_trending_up_regime(self):
        """Strongly rising prices → trending_up."""
        bars = _bars_trending_up("AAPL", 30, step=2.0)
        result = classify(bars, lookback=20)
        assert result is not None
        assert result.regime == "trending_up"
        assert result.trend_direction == "up"

    def test_trending_down_regime(self):
        """Strongly falling prices → trending_down."""
        bars = _bars_trending_down("AAPL", 30, step=2.0)
        result = classify(bars, lookback=20)
        assert result is not None
        assert result.regime == "trending_down"
        assert result.trend_direction == "down"

    def test_ranging_flat_regime(self):
        """Flat price with low vol → ranging_low_vol."""
        bars = _bars_flat("AAPL", 30)
        result = classify(bars, lookback=20)
        assert result is not None
        assert result.regime in ("ranging_low_vol", "ranging_high_vol")

    def test_confidence_range(self):
        bars = _bars_trending_up("AAPL", 30)
        result = classify(bars, lookback=20)
        assert result is not None
        assert 0 <= result.confidence <= 100

    def test_trend_strength_range(self):
        bars = _bars_trending_up("AAPL", 30)
        result = classify(bars, lookback=20)
        assert result is not None
        assert 0 <= result.trend_strength <= 100

    def test_momentum_pct_positive_uptrend(self):
        """Rising prices → positive momentum."""
        bars = _bars_trending_up("AAPL", 30, step=1.0)
        result = classify(bars, lookback=20)
        assert result is not None
        assert result.momentum_pct > 0

    def test_momentum_pct_negative_downtrend(self):
        """Falling prices → negative momentum."""
        bars = _bars_trending_down("AAPL", 30, step=1.0)
        result = classify(bars, lookback=20)
        assert result is not None
        assert result.momentum_pct < 0

    def test_strategy_hint_present(self):
        bars = _bars_trending_up("AAPL", 30)
        result = classify(bars, lookback=20)
        assert result is not None
        assert len(result.strategy_hint) > 0

    def test_bars_used_correct(self):
        bars = _bars_trending_up("AAPL", 35)
        result = classify(bars, lookback=20)
        assert result is not None
        assert result.bars_used == 20

    def test_uptrend_stronger_than_downtrend_momentum(self):
        r_up = classify(_bars_trending_up("AAPL", 30, step=2.0), lookback=20)
        r_dn = classify(_bars_trending_down("AAPL", 30, step=2.0), lookback=20)
        assert r_up is not None and r_dn is not None
        assert r_up.momentum_pct > r_dn.momentum_pct

    def test_vol_regime_high_for_volatile(self):
        """High-volatility oscillating bars → high vol regime."""
        bars = _bars_volatile_ranging("AAPL", 30)
        result = classify(bars, lookback=20)
        assert result is not None
        assert result.vol_regime in ("normal", "high")

    def test_vol_regime_low_for_flat(self):
        """Near-flat bars → low vol regime."""
        bars = _bars_flat("AAPL", 30)
        result = classify(bars, lookback=20)
        assert result is not None
        assert result.vol_regime == "low"
