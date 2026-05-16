"""Tests for amms.analysis.keltner_channel."""

from __future__ import annotations

import pytest

from amms.analysis.keltner_channel import KCSnapshot, KeltnerReport, analyze


class _Bar:
    def __init__(self, close: float, spread: float = 1.0):
        self.close = close
        self.high = close + spread
        self.low = close - spread


def _flat(n: int = 50, price: float = 100.0) -> list[_Bar]:
    return [_Bar(price)] * n


def _uptrend(n: int = 60, start: float = 100.0, step: float = 0.5) -> list[_Bar]:
    bars = []
    price = start
    for _ in range(n):
        bars.append(_Bar(price))
        price += step
    return bars


def _downtrend(n: int = 60, start: float = 200.0, step: float = 0.5) -> list[_Bar]:
    bars = []
    price = start
    for _ in range(n):
        bars.append(_Bar(price))
        price -= step
    return bars


def _breakout_up(n: int = 50, base: float = 100.0) -> list[_Bar]:
    bars = _flat(n - 3, price=base)
    bars.append(_Bar(base + 15.0, spread=0.5))
    bars.append(_Bar(base + 20.0, spread=0.5))
    bars.append(_Bar(base + 25.0, spread=0.5))
    return bars


def _breakout_down(n: int = 50, base: float = 100.0) -> list[_Bar]:
    bars = _flat(n - 3, price=base)
    bars.append(_Bar(base - 15.0, spread=0.5))
    bars.append(_Bar(base - 20.0, spread=0.5))
    bars.append(_Bar(base - 25.0, spread=0.5))
    return bars


class TestEdgeCases:
    def test_returns_none_empty(self):
        assert analyze([]) is None

    def test_returns_none_too_few_bars(self):
        assert analyze(_flat(10)) is None

    def test_returns_result_enough_bars(self):
        result = analyze(_flat(40))
        assert result is not None
        assert isinstance(result, KeltnerReport)

    def test_returns_none_no_high_low(self):
        class _BadBar:
            close = 100.0
        assert analyze([_BadBar()] * 30) is None


class TestBands:
    def test_upper_above_middle(self):
        result = analyze(_flat(50))
        assert result is not None
        assert result.upper > result.middle

    def test_lower_below_middle(self):
        result = analyze(_flat(50))
        assert result is not None
        assert result.lower < result.middle

    def test_channel_width_positive(self):
        result = analyze(_flat(50))
        assert result is not None
        assert result.channel_width > 0

    def test_channel_width_pct_positive(self):
        result = analyze(_flat(50))
        assert result is not None
        assert result.channel_width_pct > 0

    def test_middle_near_price_for_flat(self):
        result = analyze(_flat(50, price=100.0))
        assert result is not None
        assert abs(result.middle - 100.0) < 2.0


class TestPricePosition:
    def test_price_position_in_range(self):
        result = analyze(_flat(50))
        assert result is not None
        assert 0.0 <= result.price_position <= 1.5

    def test_position_label_valid(self):
        for bars in [_flat(50), _uptrend(60), _downtrend(60)]:
            result = analyze(bars)
            if result:
                assert result.position_label in {
                    "above_upper", "upper_half", "middle", "lower_half", "below_lower"
                }

    def test_breakout_up_detected(self):
        result = analyze(_breakout_up(50))
        assert result is not None
        assert result.breakout_up

    def test_breakout_down_detected(self):
        result = analyze(_breakout_down(50))
        assert result is not None
        assert result.breakout_down

    def test_no_breakout_for_flat(self):
        result = analyze(_flat(50))
        assert result is not None
        assert not result.breakout_up
        assert not result.breakout_down


class TestTrend:
    def test_trend_valid(self):
        for bars in [_flat(50), _uptrend(60), _downtrend(60)]:
            result = analyze(bars)
            if result:
                assert result.trend_direction in {"up", "down", "sideways"}

    def test_uptrend_detected(self):
        result = analyze(_uptrend(80, step=1.0))
        assert result is not None
        assert result.trend_direction in {"up", "sideways"}

    def test_downtrend_detected(self):
        result = analyze(_downtrend(80, step=1.0))
        assert result is not None
        assert result.trend_direction in {"down", "sideways"}

    def test_trend_bars_non_negative(self):
        result = analyze(_flat(50))
        assert result is not None
        assert result.trend_bars >= 0


class TestATR:
    def test_atr_positive(self):
        result = analyze(_flat(50))
        assert result is not None
        assert result.atr > 0

    def test_atr_pct_positive(self):
        result = analyze(_flat(50))
        assert result is not None
        assert result.atr_pct > 0


class TestHistory:
    def test_history_not_empty(self):
        result = analyze(_flat(50))
        assert result is not None
        assert len(result.history) > 0

    def test_history_items_are_snapshots(self):
        result = analyze(_flat(50))
        assert result is not None
        for s in result.history:
            assert isinstance(s, KCSnapshot)

    def test_history_positions_in_range(self):
        result = analyze(_flat(50))
        assert result is not None
        for s in result.history:
            assert 0.0 <= s.position <= 1.5

    def test_history_upper_above_lower(self):
        result = analyze(_flat(50))
        assert result is not None
        for s in result.history:
            assert s.upper > s.lower


class TestMetadata:
    def test_bars_used_correct(self):
        bars = _flat(50)
        result = analyze(bars)
        assert result is not None
        assert result.bars_used == 50

    def test_current_price_correct(self):
        result = analyze(_flat(50, price=150.0))
        assert result is not None
        assert abs(result.current_price - 150.0) < 1.0

    def test_symbol_stored(self):
        result = analyze(_flat(50), symbol="SPY")
        assert result is not None
        assert result.symbol == "SPY"

    def test_period_stored(self):
        result = analyze(_flat(50), period=14)
        assert result is not None
        assert result.period == 14


class TestVerdict:
    def test_verdict_present(self):
        result = analyze(_flat(50))
        assert result is not None
        assert len(result.verdict) > 10

    def test_verdict_mentions_keltner(self):
        result = analyze(_flat(50))
        assert result is not None
        assert "keltner" in result.verdict.lower() or "channel" in result.verdict.lower()
