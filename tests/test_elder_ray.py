"""Tests for amms.analysis.elder_ray."""

from __future__ import annotations

import pytest

from amms.analysis.elder_ray import ElderRayReport, analyze


class _Bar:
    def __init__(self, high: float, low: float, close: float):
        self.high  = high
        self.low   = low
        self.close = close


MIN_BARS = 32  # 13 + 5 + 10 + 3 + 1


def _flat(n: int = MIN_BARS, price: float = 100.0) -> list[_Bar]:
    return [_Bar(price + 1.0, price - 1.0, price) for _ in range(n)]


def _uptrend(n: int = 80, start: float = 50.0, step: float = 1.0) -> list[_Bar]:
    bars = []
    price = start
    for _ in range(n):
        bars.append(_Bar(price + 1.0, price - 0.5, price))
        price += step
    return bars


def _downtrend(n: int = 80, start: float = 200.0, step: float = 1.0) -> list[_Bar]:
    bars = []
    price = start
    for _ in range(n):
        bars.append(_Bar(price + 0.5, price - 1.0, price))
        price = max(price - step, 1.0)
    return bars


class TestEdgeCases:
    def test_returns_none_empty(self):
        assert analyze([]) is None

    def test_returns_none_too_few_bars(self):
        assert analyze(_flat(10)) is None

    def test_returns_result_min_bars(self):
        result = analyze(_flat(MIN_BARS))
        assert result is not None
        assert isinstance(result, ElderRayReport)

    def test_bad_attr_returns_none(self):
        class _Bad:
            high = 1.0
            low  = 1.0
        assert analyze([_Bad()] * MIN_BARS) is None


class TestComponents:
    def test_ema_positive(self):
        result = analyze(_flat(MIN_BARS))
        assert result is not None
        assert result.ema > 0

    def test_ema_period_stored(self):
        result = analyze(_flat(MIN_BARS), ema_period=13)
        assert result is not None
        assert result.ema_period == 13

    def test_bull_power_type(self):
        result = analyze(_flat(MIN_BARS))
        assert result is not None
        assert isinstance(result.bull_power, float)

    def test_bear_power_type(self):
        result = analyze(_flat(MIN_BARS))
        assert result is not None
        assert isinstance(result.bear_power, float)

    def test_uptrend_bull_power_positive(self):
        result = analyze(_uptrend())
        assert result is not None
        # In uptrend, highs should exceed EMA
        assert result.bull_power > 0

    def test_downtrend_bear_power_negative(self):
        result = analyze(_downtrend())
        assert result is not None
        # In downtrend, lows should be below EMA
        assert result.bear_power < 0


class TestSignal:
    def test_signal_valid_values(self):
        valid = {"strong_bull", "bull", "neutral", "bear", "strong_bear"}
        for bars in [_flat(MIN_BARS), _uptrend(), _downtrend()]:
            result = analyze(bars)
            if result:
                assert result.signal in valid

    def test_score_in_range(self):
        for bars in [_flat(MIN_BARS), _uptrend(), _downtrend()]:
            result = analyze(bars)
            if result:
                assert -100.0 <= result.score <= 100.0

    def test_rising_bools(self):
        result = analyze(_flat(MIN_BARS))
        assert result is not None
        assert isinstance(result.bull_rising, bool)
        assert isinstance(result.bear_rising, bool)


class TestSeries:
    def test_bull_series_non_empty(self):
        result = analyze(_flat(MIN_BARS))
        assert result is not None
        assert len(result.bull_series) > 0

    def test_bear_series_non_empty(self):
        result = analyze(_flat(MIN_BARS))
        assert result is not None
        assert len(result.bear_series) > 0

    def test_bull_series_last_matches_current(self):
        result = analyze(_flat(MIN_BARS))
        assert result is not None
        assert abs(result.bull_series[-1] - result.bull_power) < 0.01

    def test_bear_series_last_matches_current(self):
        result = analyze(_flat(MIN_BARS))
        assert result is not None
        assert abs(result.bear_series[-1] - result.bear_power) < 0.01


class TestMetadata:
    def test_symbol_stored(self):
        result = analyze(_flat(MIN_BARS), symbol="META")
        assert result is not None
        assert result.symbol == "META"

    def test_bars_used_correct(self):
        bars = _flat(50)
        result = analyze(bars)
        assert result is not None
        assert result.bars_used == 50

    def test_current_price_correct(self):
        result = analyze(_flat(MIN_BARS, price=500.0))
        assert result is not None
        assert abs(result.current_price - 500.0) < 1.0


class TestVerdict:
    def test_verdict_present(self):
        result = analyze(_flat(MIN_BARS))
        assert result is not None
        assert len(result.verdict) > 10

    def test_verdict_mentions_elder(self):
        result = analyze(_flat(MIN_BARS))
        assert result is not None
        assert "Elder" in result.verdict or "Bull Power" in result.verdict
