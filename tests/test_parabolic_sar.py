"""Tests for amms.analysis.parabolic_sar."""

from __future__ import annotations

import pytest

from amms.analysis.parabolic_sar import PSARReport, SARSnapshot, analyze


class _Bar:
    def __init__(self, close: float, spread: float = 1.0):
        self.close = close
        self.high = close + spread
        self.low = close - spread


def _flat(n: int = 30, price: float = 100.0) -> list[_Bar]:
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


def _reversal_bars(n: int = 60) -> list[_Bar]:
    """Up then sharp reversal."""
    bars = []
    price = 100.0
    for i in range(n):
        if i < n // 2:
            price += 1.5
        else:
            price -= 3.0
        bars.append(_Bar(max(price, 1.0)))
    return bars


class TestEdgeCases:
    def test_returns_none_empty(self):
        assert analyze([]) is None

    def test_returns_none_too_few_bars(self):
        assert analyze(_flat(3)) is None

    def test_returns_result_enough_bars(self):
        result = analyze(_flat(10))
        assert result is not None
        assert isinstance(result, PSARReport)

    def test_returns_none_no_high_low(self):
        class _BadBar:
            close = 100.0
        assert analyze([_BadBar()] * 10) is None


class TestDirection:
    def test_direction_valid(self):
        for bars in [_flat(30), _uptrend(50), _downtrend(50)]:
            result = analyze(bars)
            if result:
                assert result.direction in {"bull", "bear"}

    def test_uptrend_bull(self):
        result = analyze(_uptrend(60, step=2.0))
        assert result is not None
        assert result.direction == "bull"

    def test_downtrend_bear(self):
        result = analyze(_downtrend(60, step=2.0))
        assert result is not None
        assert result.direction == "bear"


class TestSARLevel:
    def test_sar_positive(self):
        result = analyze(_flat(20))
        assert result is not None
        assert result.sar > 0

    def test_bull_sar_below_price(self):
        result = analyze(_uptrend(60, step=2.0))
        assert result is not None
        if result.direction == "bull":
            assert result.sar < result.current_price

    def test_bear_sar_above_price(self):
        result = analyze(_downtrend(60, step=2.0))
        assert result is not None
        if result.direction == "bear":
            assert result.sar > result.current_price

    def test_distance_pct_non_negative(self):
        result = analyze(_flat(20))
        assert result is not None
        assert result.distance_pct >= 0


class TestAccelerationFactor:
    def test_af_in_range(self):
        result = analyze(_flat(20))
        assert result is not None
        assert result.af_step <= result.current_af <= result.af_max

    def test_custom_af_step_stored(self):
        result = analyze(_flat(20), af_step=0.01)
        assert result is not None
        assert result.af_step == 0.01

    def test_custom_af_max_stored(self):
        result = analyze(_flat(20), af_max=0.1)
        assert result is not None
        assert result.af_max == 0.1


class TestFlips:
    def test_flip_count_non_negative(self):
        result = analyze(_flat(30))
        assert result is not None
        assert result.flip_count >= 0

    def test_reversal_causes_flip(self):
        result = analyze(_reversal_bars(80))
        assert result is not None
        assert result.flip_count >= 1

    def test_last_flip_bar_valid(self):
        result = analyze(_reversal_bars(80))
        assert result is not None
        if result.last_flip_bar is not None:
            assert 0 <= result.last_flip_bar < result.bars_used

    def test_trend_age_non_negative(self):
        result = analyze(_flat(30))
        assert result is not None
        assert result.trend_age >= 0


class TestStats:
    def test_bull_pct_in_range(self):
        result = analyze(_flat(30))
        assert result is not None
        assert 0.0 <= result.bull_pct <= 100.0

    def test_uptrend_high_bull_pct(self):
        result = analyze(_uptrend(60, step=2.0))
        assert result is not None
        assert result.bull_pct > 50.0

    def test_avg_trend_duration_positive(self):
        result = analyze(_flat(30))
        assert result is not None
        assert result.avg_trend_duration > 0


class TestHistory:
    def test_history_not_empty(self):
        result = analyze(_flat(20))
        assert result is not None
        assert len(result.history) > 0

    def test_history_items_are_snapshots(self):
        result = analyze(_flat(20))
        assert result is not None
        for s in result.history:
            assert isinstance(s, SARSnapshot)

    def test_history_directions_valid(self):
        result = analyze(_flat(20))
        assert result is not None
        for s in result.history:
            assert s.direction in {"bull", "bear"}

    def test_history_af_in_range(self):
        result = analyze(_flat(20))
        assert result is not None
        for s in result.history:
            assert result.af_step <= s.af <= result.af_max

    def test_flip_in_history_for_reversal(self):
        result = analyze(_reversal_bars(80))
        assert result is not None
        assert any(s.flipped for s in result.history)


class TestMetadata:
    def test_bars_used_correct(self):
        bars = _flat(30)
        result = analyze(bars)
        assert result is not None
        assert result.bars_used == 30

    def test_current_price_correct(self):
        result = analyze(_flat(30, price=150.0))
        assert result is not None
        assert abs(result.current_price - 150.0) < 1.0

    def test_symbol_stored(self):
        result = analyze(_flat(30), symbol="NFLX")
        assert result is not None
        assert result.symbol == "NFLX"


class TestVerdict:
    def test_verdict_present(self):
        result = analyze(_flat(20))
        assert result is not None
        assert len(result.verdict) > 10

    def test_verdict_mentions_parabolic(self):
        result = analyze(_flat(20))
        assert result is not None
        assert "parabolic" in result.verdict.lower() or "sar" in result.verdict.lower()
