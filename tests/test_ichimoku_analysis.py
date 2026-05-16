"""Tests for amms.analysis.ichimoku (extended Ichimoku Cloud analyser)."""

from __future__ import annotations

import pytest

from amms.analysis.ichimoku import IchimokuReport, analyze


class _Bar:
    def __init__(self, high: float, low: float, close: float):
        self.high  = high
        self.low   = low
        self.close = close


def _flat(n: int = 90, price: float = 100.0) -> list[_Bar]:
    return [_Bar(price + 0.5, price - 0.5, price) for _ in range(n)]


def _uptrend(n: int = 120, start: float = 50.0, step: float = 1.0) -> list[_Bar]:
    bars = []
    price = start
    for _ in range(n):
        bars.append(_Bar(price + 0.5, price - 0.5, price))
        price += step
    return bars


def _downtrend(n: int = 120, start: float = 200.0, step: float = 1.0) -> list[_Bar]:
    bars = []
    price = start
    for _ in range(n):
        bars.append(_Bar(price + 0.5, price - 0.5, price))
        price = max(price - step, 1.0)
    return bars


MIN_BARS = 78  # senkou_b_period(52) + displacement(26)


class TestEdgeCases:
    def test_returns_none_empty(self):
        assert analyze([]) is None

    def test_returns_none_too_few_bars(self):
        assert analyze(_flat(50)) is None

    def test_returns_result_min_bars(self):
        result = analyze(_flat(MIN_BARS))
        assert result is not None
        assert isinstance(result, IchimokuReport)

    def test_bad_attr_returns_none(self):
        class _Bad:
            high = low = 1.0
        assert analyze([_Bad()] * MIN_BARS) is None


class TestComponents:
    def test_tenkan_not_none(self):
        result = analyze(_flat(MIN_BARS))
        assert result is not None
        assert result.tenkan is not None

    def test_kijun_not_none(self):
        result = analyze(_flat(MIN_BARS))
        assert result is not None
        assert result.kijun is not None

    def test_senkou_a_not_none(self):
        result = analyze(_flat(MIN_BARS))
        assert result is not None
        assert result.senkou_a is not None

    def test_senkou_b_not_none(self):
        result = analyze(_flat(MIN_BARS))
        assert result is not None
        assert result.senkou_b is not None

    def test_flat_tenkan_equals_kijun(self):
        result = analyze(_flat(MIN_BARS))
        assert result is not None
        assert abs(result.tenkan - result.kijun) < 0.01

    def test_cloud_top_above_bottom(self):
        result = analyze(_flat(MIN_BARS))
        assert result is not None
        if result.cloud_top is not None and result.cloud_bottom is not None:
            assert result.cloud_top >= result.cloud_bottom


class TestCloudContext:
    def test_uptrend_price_in_or_above_cloud(self):
        result = analyze(_uptrend())
        assert result is not None
        assert result.price_above_cloud is True or result.price_in_cloud is True

    def test_downtrend_price_in_or_below_cloud(self):
        result = analyze(_downtrend())
        assert result is not None
        assert result.price_below_cloud is True or result.price_in_cloud is True

    def test_price_context_mutually_exclusive(self):
        result = analyze(_flat(MIN_BARS))
        assert result is not None
        if result.price_above_cloud is not None:
            true_count = sum([
                bool(result.price_above_cloud),
                bool(result.price_below_cloud),
                bool(result.price_in_cloud),
            ])
            assert true_count <= 1


class TestSignal:
    def test_signal_valid_values(self):
        valid = {"strong_bull", "bull", "neutral", "bear", "strong_bear"}
        for bars in [_flat(MIN_BARS), _uptrend(), _downtrend()]:
            result = analyze(bars)
            if result:
                assert result.signal in valid

    def test_uptrend_bullish_or_neutral(self):
        result = analyze(_uptrend())
        assert result is not None
        assert result.signal in {"bull", "strong_bull", "neutral"}

    def test_downtrend_bearish_or_neutral(self):
        result = analyze(_downtrend())
        assert result is not None
        assert result.signal in {"bear", "strong_bear", "neutral"}

    def test_bullish_signals_in_range(self):
        result = analyze(_flat(MIN_BARS))
        assert result is not None
        assert 0 <= result.bullish_signals <= 6

    def test_bearish_signals_in_range(self):
        result = analyze(_flat(MIN_BARS))
        assert result is not None
        assert 0 <= result.bearish_signals <= 6


class TestTKCross:
    def test_tk_cross_bools(self):
        result = analyze(_flat(MIN_BARS))
        assert result is not None
        assert isinstance(result.tk_cross_bullish, bool)
        assert isinstance(result.tk_cross_bearish, bool)

    def test_tk_not_both_cross(self):
        result = analyze(_flat(MIN_BARS))
        assert result is not None
        assert not (result.tk_cross_bullish and result.tk_cross_bearish)


class TestMetadata:
    def test_symbol_stored(self):
        result = analyze(_flat(MIN_BARS), symbol="NVDA")
        assert result is not None
        assert result.symbol == "NVDA"

    def test_bars_used_correct(self):
        bars = _flat(90)
        result = analyze(bars)
        assert result is not None
        assert result.bars_used == 90

    def test_current_price_correct(self):
        result = analyze(_flat(MIN_BARS, price=200.0))
        assert result is not None
        assert abs(result.current_price - 200.0) < 1.0

    def test_chikou_equals_current_price(self):
        result = analyze(_flat(MIN_BARS, price=150.0))
        assert result is not None
        assert abs(result.chikou - result.current_price) < 0.01


class TestVerdict:
    def test_verdict_present(self):
        result = analyze(_flat(MIN_BARS))
        assert result is not None
        assert len(result.verdict) > 10

    def test_verdict_contains_ichimoku(self):
        result = analyze(_flat(MIN_BARS))
        assert result is not None
        assert "Ichimoku" in result.verdict

    def test_verdict_mentions_cloud(self):
        result = analyze(_flat(MIN_BARS))
        assert result is not None
        text = result.verdict.lower()
        assert "cloud" in text or "above" in text or "below" in text
