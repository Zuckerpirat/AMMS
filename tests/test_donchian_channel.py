"""Tests for amms.analysis.donchian_channel."""

from __future__ import annotations

import pytest

from amms.analysis.donchian_channel import DCSnapshot, DonchianReport, analyze


class _Bar:
    def __init__(self, close: float, spread: float = 1.0):
        self.close = close
        self.high = close + spread
        self.low = close - spread


def _flat(n: int = 40, price: float = 100.0) -> list[_Bar]:
    return [_Bar(price)] * n


def _uptrend(n: int = 50, start: float = 50.0, step: float = 1.0) -> list[_Bar]:
    bars = []
    price = start
    for _ in range(n):
        bars.append(_Bar(price))
        price += step
    return bars


def _downtrend(n: int = 50, start: float = 200.0, step: float = 1.0) -> list[_Bar]:
    bars = []
    price = start
    for _ in range(n):
        bars.append(_Bar(price))
        price -= step
    return bars


def _breakout_up_bars(n: int = 30, base: float = 100.0) -> list[_Bar]:
    """Flat then new high breakout (spread=0 so close==high==upper)."""
    bars = _flat(n - 1, price=base)
    bars.append(_Bar(base + 20.0, spread=0.0))
    return bars


def _breakout_down_bars(n: int = 30, base: float = 100.0) -> list[_Bar]:
    """Flat then new low breakdown (spread=0 so close==low==lower)."""
    bars = _flat(n - 1, price=base)
    bars.append(_Bar(base - 20.0, spread=0.0))
    return bars


class TestEdgeCases:
    def test_returns_none_empty(self):
        assert analyze([]) is None

    def test_returns_none_too_few_bars(self):
        assert analyze(_flat(15)) is None

    def test_returns_result_enough_bars(self):
        result = analyze(_flat(25))
        assert result is not None
        assert isinstance(result, DonchianReport)

    def test_returns_none_no_high_low(self):
        class _BadBar:
            close = 100.0
        assert analyze([_BadBar()] * 25) is None


class TestBands:
    def test_upper_above_lower(self):
        result = analyze(_uptrend(50))
        assert result is not None
        assert result.upper >= result.lower

    def test_middle_between_bands(self):
        result = analyze(_flat(40))
        assert result is not None
        assert result.lower <= result.middle <= result.upper

    def test_channel_width_positive(self):
        result = analyze(_uptrend(50))
        assert result is not None
        assert result.channel_width >= 0

    def test_flat_tight_channel(self):
        result = analyze(_flat(40, price=100.0))
        assert result is not None
        assert result.channel_width_pct < 5.0

    def test_upper_is_period_high(self):
        bars = _flat(30, 100.0)
        result = analyze(bars, period=20)
        assert result is not None
        # All closes same, high = close + 1 = 101
        assert abs(result.upper - 101.0) < 0.1


class TestPricePosition:
    def test_price_position_in_range(self):
        for bars in [_flat(40), _uptrend(50), _downtrend(50)]:
            result = analyze(bars)
            if result:
                assert 0.0 <= result.price_position <= 1.0

    def test_position_label_valid(self):
        for bars in [_flat(40), _uptrend(50), _downtrend(50)]:
            result = analyze(bars)
            if result:
                assert result.position_label in {
                    "above_upper", "at_upper", "upper_half",
                    "middle", "lower_half", "at_lower", "below_lower"
                }

    def test_breakout_up_detected(self):
        result = analyze(_breakout_up_bars(30))
        assert result is not None
        assert result.breakout_up

    def test_breakout_down_detected(self):
        result = analyze(_breakout_down_bars(30))
        assert result is not None
        assert result.breakout_down

    def test_no_breakout_for_flat(self):
        result = analyze(_flat(40))
        assert result is not None
        assert not result.breakout_up
        assert not result.breakout_down


class TestBreakoutDetails:
    def test_bars_since_upper_non_negative(self):
        result = analyze(_flat(40))
        assert result is not None
        assert result.bars_since_upper >= 0

    def test_bars_since_lower_non_negative(self):
        result = analyze(_flat(40))
        assert result is not None
        assert result.bars_since_lower >= 0

    def test_breakout_up_bars_since_non_negative(self):
        result = analyze(_breakout_up_bars(30))
        assert result is not None
        assert result.bars_since_upper >= 0


class TestChannelTrend:
    def test_channel_trend_valid(self):
        for bars in [_flat(40), _uptrend(50), _downtrend(50)]:
            result = analyze(bars)
            if result:
                assert result.channel_trend in {"expanding", "contracting", "stable"}


class TestHistory:
    def test_history_not_empty(self):
        result = analyze(_flat(40))
        assert result is not None
        assert len(result.history) > 0

    def test_history_items_are_snapshots(self):
        result = analyze(_flat(40))
        assert result is not None
        for s in result.history:
            assert isinstance(s, DCSnapshot)

    def test_history_positions_in_range(self):
        result = analyze(_flat(40))
        assert result is not None
        for s in result.history:
            assert 0.0 <= s.position <= 1.0

    def test_history_upper_ge_lower(self):
        result = analyze(_flat(40))
        assert result is not None
        for s in result.history:
            assert s.upper >= s.lower


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
        result = analyze(_flat(40), symbol="GLD")
        assert result is not None
        assert result.symbol == "GLD"


class TestVerdict:
    def test_verdict_present(self):
        result = analyze(_flat(40))
        assert result is not None
        assert len(result.verdict) > 10

    def test_verdict_mentions_donchian(self):
        result = analyze(_flat(40))
        assert result is not None
        assert "donchian" in result.verdict.lower()
