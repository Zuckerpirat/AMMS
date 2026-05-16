"""Tests for amms.analysis.supertrend."""

from __future__ import annotations

import pytest

from amms.analysis.supertrend import STSnapshot, SupertrendReport, analyze


class _Bar:
    def __init__(self, close: float, spread: float = 1.0):
        self.close = close
        self.high = close + spread
        self.low = close - spread


def _flat(n: int = 40, price: float = 100.0) -> list[_Bar]:
    return [_Bar(price)] * n


def _uptrend(n: int = 60, start: float = 50.0, step: float = 1.0) -> list[_Bar]:
    bars = []
    price = start
    for _ in range(n):
        bars.append(_Bar(price))
        price += step
    return bars


def _downtrend(n: int = 60, start: float = 200.0, step: float = 1.0) -> list[_Bar]:
    bars = []
    price = start
    for _ in range(n):
        bars.append(_Bar(price))
        price -= step
    return bars


def _flip_bars(n: int = 60) -> list[_Bar]:
    """Uptrend then sharp reversal."""
    bars = []
    price = 100.0
    for i in range(n):
        if i < n // 2:
            price += 1.0
        else:
            price -= 3.0
        bars.append(_Bar(max(price, 1.0)))
    return bars


class TestEdgeCases:
    def test_returns_none_empty(self):
        assert analyze([]) is None

    def test_returns_none_too_few_bars(self):
        assert analyze(_flat(5)) is None

    def test_returns_result_enough_bars(self):
        result = analyze(_flat(20))
        assert result is not None
        assert isinstance(result, SupertrendReport)

    def test_returns_none_no_high_low(self):
        class _BadBar:
            close = 100.0
        assert analyze([_BadBar()] * 20) is None


class TestDirection:
    def test_direction_valid(self):
        for bars in [_flat(40), _uptrend(60), _downtrend(60)]:
            result = analyze(bars)
            if result:
                assert result.direction in {"bull", "bear"}

    def test_uptrend_bull(self):
        result = analyze(_uptrend(80, start=50.0, step=1.5))
        assert result is not None
        assert result.direction == "bull"

    def test_downtrend_bear(self):
        result = analyze(_downtrend(80, start=300.0, step=1.5))
        assert result is not None
        assert result.direction == "bear"


class TestSupertrendLevel:
    def test_level_positive(self):
        result = analyze(_flat(40))
        assert result is not None
        assert result.supertrend_level > 0

    def test_bull_level_below_price(self):
        result = analyze(_uptrend(80, start=50.0, step=1.0))
        assert result is not None
        if result.direction == "bull":
            assert result.supertrend_level <= result.current_price

    def test_bear_level_above_price(self):
        result = analyze(_downtrend(80, start=300.0, step=1.0))
        assert result is not None
        if result.direction == "bear":
            assert result.supertrend_level >= result.current_price


class TestTrendAge:
    def test_trend_age_non_negative(self):
        result = analyze(_flat(40))
        assert result is not None
        assert result.trend_age >= 0

    def test_stable_trend_high_age(self):
        result = analyze(_uptrend(80, step=1.0))
        assert result is not None
        assert result.trend_age > 0


class TestFlips:
    def test_flip_count_non_negative(self):
        result = analyze(_flat(40))
        assert result is not None
        assert result.flip_count >= 0

    def test_flip_count_increases_for_reversal(self):
        result = analyze(_flip_bars(80))
        assert result is not None
        assert result.flip_count >= 1

    def test_last_flip_bar_valid(self):
        result = analyze(_flip_bars(80))
        assert result is not None
        if result.last_flip_bar is not None:
            assert 0 <= result.last_flip_bar < result.bars_used


class TestATR:
    def test_atr_positive(self):
        result = analyze(_flat(40))
        assert result is not None
        assert result.atr > 0


class TestHistory:
    def test_history_not_empty(self):
        result = analyze(_flat(40))
        assert result is not None
        assert len(result.history) > 0

    def test_history_items_are_snapshots(self):
        result = analyze(_flat(40))
        assert result is not None
        for s in result.history:
            assert isinstance(s, STSnapshot)

    def test_history_directions_valid(self):
        result = analyze(_flat(40))
        assert result is not None
        for s in result.history:
            assert s.direction in {"bull", "bear"}

    def test_history_flipped_is_bool(self):
        result = analyze(_flat(40))
        assert result is not None
        for s in result.history:
            assert isinstance(s.flipped, bool)

    def test_flip_in_history_for_reversal(self):
        result = analyze(_flip_bars(80))
        assert result is not None
        assert any(s.flipped for s in result.history)


class TestMetadata:
    def test_bars_used_correct(self):
        bars = _flat(40)
        result = analyze(bars)
        assert result is not None
        assert result.bars_used == 40

    def test_current_price_correct(self):
        result = analyze(_flat(40, price=150.0))
        assert result is not None
        assert abs(result.current_price - 150.0) < 1.0

    def test_symbol_stored(self):
        result = analyze(_flat(40), symbol="AMZN")
        assert result is not None
        assert result.symbol == "AMZN"

    def test_period_stored(self):
        result = analyze(_flat(40), period=7)
        assert result is not None
        assert result.period == 7

    def test_mult_stored(self):
        result = analyze(_flat(40), mult=2.5)
        assert result is not None
        assert result.mult == 2.5


class TestVerdict:
    def test_verdict_present(self):
        result = analyze(_flat(40))
        assert result is not None
        assert len(result.verdict) > 10

    def test_verdict_mentions_supertrend(self):
        result = analyze(_flat(40))
        assert result is not None
        assert "supertrend" in result.verdict.lower()
