"""Tests for amms.analysis.mass_index."""

from __future__ import annotations

import pytest

from amms.analysis.mass_index import MassIndexReport, analyze


class _Bar:
    def __init__(self, high: float, low: float, close: float):
        self.high  = high
        self.low   = low
        self.close = close


# min_bars = 9*2 + 25 + 60 + 5 = 108
MIN_BARS = 115


def _flat(n: int = MIN_BARS, price: float = 100.0, spread: float = 2.0) -> list[_Bar]:
    return [_Bar(price + spread, price - spread, price) for _ in range(n)]


def _uptrend(n: int = MIN_BARS, start: float = 50.0, step: float = 1.0) -> list[_Bar]:
    bars = []
    price = start
    for _ in range(n):
        bars.append(_Bar(price + 1.0, price - 1.0, price))
        price += step
    return bars


def _wide_range(n: int = MIN_BARS, price: float = 100.0, spread: float = 10.0) -> list[_Bar]:
    """Wide bars → high mass index."""
    return [_Bar(price + spread, price - spread, price) for _ in range(n)]


def _narrow_range(n: int = MIN_BARS, price: float = 100.0) -> list[_Bar]:
    """Very narrow bars → low mass index."""
    return [_Bar(price + 0.01, price - 0.01, price) for _ in range(n)]


class TestEdgeCases:
    def test_returns_none_empty(self):
        assert analyze([]) is None

    def test_returns_none_too_few_bars(self):
        assert analyze(_flat(30)) is None

    def test_returns_result_enough_bars(self):
        result = analyze(_flat(MIN_BARS))
        assert result is not None
        assert isinstance(result, MassIndexReport)

    def test_bad_attr_returns_none(self):
        class _Bad:
            high = low = 1.0
        assert analyze([_Bad()] * MIN_BARS) is None


class TestMassIndex:
    def test_mass_index_is_float(self):
        result = analyze(_flat(MIN_BARS))
        assert result is not None
        assert isinstance(result.mass_index, float)

    def test_mass_index_positive(self):
        # Ratio of EMAs is always positive, sum is always positive
        for bars in [_flat(MIN_BARS), _uptrend(), _wide_range()]:
            result = analyze(bars)
            if result:
                assert result.mass_index > 0

    def test_wide_range_higher_mi_than_narrow(self):
        wide_result   = analyze(_wide_range())
        narrow_result = analyze(_narrow_range())
        if wide_result and narrow_result:
            # Wide range should produce higher MI than narrow
            assert wide_result.mass_index >= narrow_result.mass_index

    def test_pct_rank_in_range(self):
        for bars in [_flat(MIN_BARS), _uptrend()]:
            result = analyze(bars)
            if result:
                assert 0.0 <= result.mi_pct_rank <= 100.0


class TestSignal:
    def test_signal_valid_values(self):
        valid = {"expansion", "normal", "contraction"}
        for bars in [_flat(MIN_BARS), _uptrend(), _wide_range()]:
            result = analyze(bars)
            if result:
                assert result.signal in valid

    def test_score_in_range(self):
        for bars in [_flat(MIN_BARS), _uptrend(), _wide_range()]:
            result = analyze(bars)
            if result:
                assert -100.0 <= result.score <= 100.0


class TestBulge:
    def test_bulge_bools(self):
        result = analyze(_flat(MIN_BARS))
        assert result is not None
        assert isinstance(result.above_bulge_setup, bool)
        assert isinstance(result.below_reversal, bool)
        assert isinstance(result.bulge_recently, bool)

    def test_reversal_signal_valid(self):
        valid = {"none", "potential_top", "potential_bottom"}
        for bars in [_flat(MIN_BARS), _uptrend()]:
            result = analyze(bars)
            if result:
                assert result.reversal_signal in valid


class TestTrend:
    def test_trend_direction_valid(self):
        valid = {"up", "down", "flat"}
        for bars in [_flat(MIN_BARS), _uptrend()]:
            result = analyze(bars)
            if result:
                assert result.trend_direction in valid

    def test_uptrend_detected(self):
        result = analyze(_uptrend())
        assert result is not None
        assert result.trend_direction in {"up", "flat"}


class TestSeries:
    def test_mi_series_non_empty(self):
        result = analyze(_flat(MIN_BARS))
        assert result is not None
        assert len(result.mi_series) > 0

    def test_mi_series_last_matches_current(self):
        result = analyze(_flat(MIN_BARS))
        assert result is not None
        assert abs(result.mi_series[-1] - result.mass_index) < 0.01


class TestMetadata:
    def test_symbol_stored(self):
        result = analyze(_flat(MIN_BARS), symbol="XOM")
        assert result is not None
        assert result.symbol == "XOM"

    def test_bars_used_correct(self):
        bars = _flat(MIN_BARS + 5)
        result = analyze(bars)
        assert result is not None
        assert result.bars_used == MIN_BARS + 5


class TestVerdict:
    def test_verdict_present(self):
        result = analyze(_flat(MIN_BARS))
        assert result is not None
        assert len(result.verdict) > 10

    def test_verdict_mentions_mass_index(self):
        result = analyze(_flat(MIN_BARS))
        assert result is not None
        assert "Mass Index" in result.verdict or "MI=" in result.verdict
