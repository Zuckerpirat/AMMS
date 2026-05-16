"""Tests for amms.analysis.williams_aroon."""

from __future__ import annotations

import pytest

from amms.analysis.williams_aroon import WilliamsAroonReport, analyze


class _Bar:
    def __init__(self, high: float, low: float, close: float):
        self.high  = high
        self.low   = low
        self.close = close


MIN_BARS = 45  # max(14, 25) + 10 + 5


def _flat(n: int = MIN_BARS, price: float = 100.0) -> list[_Bar]:
    return [_Bar(price + 1.0, price - 1.0, price) for _ in range(n)]


def _uptrend(n: int = 80, start: float = 50.0, step: float = 1.0) -> list[_Bar]:
    bars = []
    price = start
    for _ in range(n):
        bars.append(_Bar(price + 0.5, price - 0.5, price))
        price += step
    return bars


def _downtrend(n: int = 80, start: float = 200.0, step: float = 1.0) -> list[_Bar]:
    bars = []
    price = start
    for _ in range(n):
        bars.append(_Bar(price + 0.5, price - 0.5, price))
        price = max(price - step, 1.0)
    return bars


def _overbought(n: int = MIN_BARS, price: float = 100.0) -> list[_Bar]:
    """Price consistently near the high of the range — overbought."""
    bars = []
    for i in range(n):
        h = price + 5.0
        l = price - 0.1
        c = price + 4.9  # close near high
        bars.append(_Bar(h, l, c))
    return bars


def _oversold(n: int = MIN_BARS, price: float = 100.0) -> list[_Bar]:
    """Price consistently near the low of the range — oversold."""
    bars = []
    for _ in range(n):
        h = price + 5.0
        l = price - 5.0
        c = price - 4.9  # close near low
        bars.append(_Bar(h, l, c))
    return bars


class TestEdgeCases:
    def test_returns_none_empty(self):
        assert analyze([]) is None

    def test_returns_none_too_few_bars(self):
        assert analyze(_flat(10)) is None

    def test_returns_result_enough_bars(self):
        result = analyze(_flat(MIN_BARS))
        assert result is not None
        assert isinstance(result, WilliamsAroonReport)

    def test_bad_attr_returns_none(self):
        class _Bad:
            high = 1.0
            low  = 1.0
        assert analyze([_Bad()] * MIN_BARS) is None


class TestWilliamsR:
    def test_williams_r_in_range(self):
        for bars in [_flat(MIN_BARS), _uptrend(), _downtrend()]:
            result = analyze(bars)
            if result:
                assert -100.0 <= result.williams_r <= 0.0

    def test_overbought_detected(self):
        result = analyze(_overbought())
        assert result is not None
        assert result.williams_overbought is True
        assert result.williams_signal == "overbought"

    def test_oversold_detected(self):
        result = analyze(_oversold())
        assert result is not None
        assert result.williams_oversold is True
        assert result.williams_signal == "oversold"

    def test_williams_signal_valid(self):
        for bars in [_flat(MIN_BARS), _overbought(), _oversold()]:
            result = analyze(bars)
            if result:
                assert result.williams_signal in {"overbought", "neutral", "oversold"}

    def test_not_both_overbought_and_oversold(self):
        result = analyze(_flat(MIN_BARS))
        assert result is not None
        assert not (result.williams_overbought and result.williams_oversold)


class TestAroon:
    def test_aroon_up_in_range(self):
        for bars in [_flat(MIN_BARS), _uptrend(), _downtrend()]:
            result = analyze(bars)
            if result:
                assert 0.0 <= result.aroon_up <= 100.0

    def test_aroon_down_in_range(self):
        for bars in [_flat(MIN_BARS), _uptrend(), _downtrend()]:
            result = analyze(bars)
            if result:
                assert 0.0 <= result.aroon_down <= 100.0

    def test_aroon_osc_in_range(self):
        for bars in [_flat(MIN_BARS), _uptrend(), _downtrend()]:
            result = analyze(bars)
            if result:
                assert -100.0 <= result.aroon_osc <= 100.0

    def test_aroon_signal_valid(self):
        valid = {"strong_up", "up", "neutral", "down", "strong_down"}
        for bars in [_flat(MIN_BARS), _uptrend(), _downtrend()]:
            result = analyze(bars)
            if result:
                assert result.aroon_signal in valid

    def test_uptrend_high_aroon_up(self):
        result = analyze(_uptrend())
        assert result is not None
        # In a sustained uptrend, Aroon Up should be high
        assert result.aroon_up >= 50.0

    def test_downtrend_high_aroon_down(self):
        result = analyze(_downtrend())
        assert result is not None
        assert result.aroon_down >= 50.0


class TestComposite:
    def test_composite_signal_valid(self):
        valid = {"strong_bull", "bull", "neutral", "bear", "strong_bear"}
        for bars in [_flat(MIN_BARS), _uptrend(), _downtrend()]:
            result = analyze(bars)
            if result:
                assert result.composite_signal in valid

    def test_composite_score_in_range(self):
        for bars in [_flat(MIN_BARS), _uptrend(), _downtrend()]:
            result = analyze(bars)
            if result:
                assert -100.0 <= result.composite_score <= 100.0

    def test_uptrend_bullish_composite(self):
        result = analyze(_uptrend())
        assert result is not None
        assert result.composite_signal in {"bull", "strong_bull", "neutral"}

    def test_downtrend_bearish_composite(self):
        result = analyze(_downtrend())
        assert result is not None
        assert result.composite_signal in {"bear", "strong_bear", "neutral"}


class TestHistory:
    def test_williams_r_series_non_empty(self):
        result = analyze(_flat(MIN_BARS))
        assert result is not None
        assert len(result.williams_r_series) > 0

    def test_williams_r_series_last_is_current(self):
        result = analyze(_flat(MIN_BARS))
        assert result is not None
        assert abs(result.williams_r_series[-1] - result.williams_r) < 0.01


class TestMetadata:
    def test_symbol_stored(self):
        result = analyze(_flat(MIN_BARS), symbol="AAPL")
        assert result is not None
        assert result.symbol == "AAPL"

    def test_bars_used_correct(self):
        bars = _flat(60)
        result = analyze(bars)
        assert result is not None
        assert result.bars_used == 60

    def test_current_price_correct(self):
        result = analyze(_flat(MIN_BARS, price=300.0))
        assert result is not None
        assert abs(result.current_price - 300.0) < 1.0


class TestVerdict:
    def test_verdict_present(self):
        result = analyze(_flat(MIN_BARS))
        assert result is not None
        assert len(result.verdict) > 10

    def test_verdict_mentions_williams(self):
        result = analyze(_flat(MIN_BARS))
        assert result is not None
        assert "Williams" in result.verdict or "%R" in result.verdict

    def test_verdict_mentions_aroon(self):
        result = analyze(_flat(MIN_BARS))
        assert result is not None
        assert "Aroon" in result.verdict
