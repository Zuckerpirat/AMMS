"""Tests for amms.analysis.price_target_consensus."""

from __future__ import annotations

import pytest

from amms.analysis.price_target_consensus import (
    ConsensusZone,
    PriceTargetConsensusReport,
    TargetLevel,
    analyze,
)


class _Bar:
    def __init__(self, close, high=None, low=None):
        self.close = close
        self.high = high if high is not None else close + 1.0
        self.low = low if low is not None else close - 1.0


def _trending_bars(n: int = 40, start: float = 100.0) -> list[_Bar]:
    bars = []
    for i in range(n):
        price = start + i * 0.5
        bars.append(_Bar(price, high=price + 1.0, low=price - 0.5))
    return bars


def _range_bars(n: int = 40, center: float = 100.0, amplitude: float = 5.0) -> list[_Bar]:
    bars = []
    import math
    for i in range(n):
        price = center + amplitude * math.sin(i * 0.3)
        bars.append(_Bar(price, high=price + 0.5, low=price - 0.5))
    return bars


class TestEdgeCases:
    def test_returns_none_empty(self):
        assert analyze([]) is None

    def test_returns_none_too_few(self):
        bars = [_Bar(100.0) for _ in range(19)]
        assert analyze(bars) is None

    def test_returns_result(self):
        bars = _trending_bars(30)
        result = analyze(bars)
        assert result is not None
        assert isinstance(result, PriceTargetConsensusReport)


class TestTargets:
    def test_all_targets_nonempty(self):
        bars = _trending_bars(30)
        result = analyze(bars)
        assert result is not None
        assert len(result.all_targets) > 0

    def test_targets_have_upside_and_downside(self):
        bars = _range_bars(40)
        result = analyze(bars)
        assert result is not None
        assert len(result.upside_targets) > 0
        assert len(result.downside_targets) > 0

    def test_upside_targets_above_price(self):
        bars = _trending_bars(30)
        result = analyze(bars)
        assert result is not None
        for t in result.upside_targets:
            assert t.pct_from_current > 0

    def test_downside_targets_below_price(self):
        bars = _trending_bars(30)
        result = analyze(bars)
        assert result is not None
        for t in result.downside_targets:
            assert t.pct_from_current < 0

    def test_targets_have_source(self):
        bars = _trending_bars(30)
        result = analyze(bars)
        assert result is not None
        for t in result.all_targets:
            assert len(t.source) > 0
            assert t.direction in ("up", "down")

    def test_target_pct_consistent_with_price(self):
        bars = _trending_bars(30)
        result = analyze(bars)
        assert result is not None
        for t in result.all_targets:
            expected_pct = (t.price - result.current_price) / result.current_price * 100
            assert abs(t.pct_from_current - expected_pct) < 0.1


class TestConsensusZones:
    def test_best_upside_zone_set(self):
        bars = _range_bars(40)
        result = analyze(bars)
        assert result is not None
        if result.upside_targets:
            assert result.best_upside_zone is not None

    def test_best_downside_zone_set(self):
        bars = _range_bars(40)
        result = analyze(bars)
        assert result is not None
        if result.downside_targets:
            assert result.best_downside_zone is not None

    def test_upside_zone_above_price(self):
        bars = _range_bars(40)
        result = analyze(bars)
        assert result is not None
        if result.best_upside_zone:
            assert result.best_upside_zone.center > result.current_price

    def test_downside_zone_below_price(self):
        bars = _range_bars(40)
        result = analyze(bars)
        assert result is not None
        if result.best_downside_zone:
            assert result.best_downside_zone.center < result.current_price

    def test_zone_low_lte_center_lte_high(self):
        bars = _range_bars(40)
        result = analyze(bars)
        assert result is not None
        for zone in [result.best_upside_zone, result.best_downside_zone]:
            if zone:
                assert zone.low <= zone.center <= zone.high

    def test_zone_n_sources_positive(self):
        bars = _range_bars(40)
        result = analyze(bars)
        assert result is not None
        if result.best_upside_zone:
            assert result.best_upside_zone.n_sources >= 1

    def test_zone_contains_targets(self):
        bars = _range_bars(40)
        result = analyze(bars)
        assert result is not None
        if result.best_upside_zone:
            assert len(result.best_upside_zone.targets) >= 1


class TestConsensusScore:
    def test_consensus_pct_in_range(self):
        bars = _range_bars(40)
        result = analyze(bars)
        assert result is not None
        assert 0.0 <= result.upside_consensus_pct <= 100.0
        assert 0.0 <= result.downside_consensus_pct <= 100.0


class TestMetadata:
    def test_symbol_stored(self):
        bars = _trending_bars(30)
        result = analyze(bars, symbol="AAPL")
        assert result is not None
        assert result.symbol == "AAPL"

    def test_bars_used_correct(self):
        bars = _trending_bars(35)
        result = analyze(bars)
        assert result is not None
        assert result.bars_used == 35

    def test_current_price_matches_last_bar(self):
        bars = _trending_bars(30)
        result = analyze(bars)
        assert result is not None
        assert result.current_price == pytest.approx(bars[-1].close, abs=0.01)


class TestVerdict:
    def test_verdict_present(self):
        bars = _trending_bars(30)
        result = analyze(bars)
        assert result is not None
        assert len(result.verdict) > 10
