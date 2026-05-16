"""Tests for amms.analysis.gap_classifier."""

from __future__ import annotations

import pytest

from amms.analysis.gap_classifier import Gap, GapReport, analyze


class _Bar:
    def __init__(self, open_: float, high: float, low: float, close: float, volume: float = 100_000):
        self.open = open_
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume


def _flat(n: int = 30, price: float = 100.0) -> list[_Bar]:
    return [_Bar(price, price + 1.0, price - 1.0, price)] * n


def _gap_up_bars(n: int = 30, base: float = 100.0) -> list[_Bar]:
    """Flat then gap up."""
    bars = _flat(n - 3, price=base)
    # Day before gap: high=101
    bars.append(_Bar(base, base + 1.0, base - 1.0, base))
    # Gap day: low=105 > prior high=101
    bars.append(_Bar(105.0, 107.0, 105.0, 106.0))
    bars.append(_Bar(106.0, 108.0, 105.5, 107.0))
    return bars


def _gap_down_bars(n: int = 30, base: float = 100.0) -> list[_Bar]:
    """Flat then gap down."""
    bars = _flat(n - 3, price=base)
    # Day before gap: low=99
    bars.append(_Bar(base, base + 1.0, base - 1.0, base))
    # Gap day: high=95 < prior low=99
    bars.append(_Bar(95.0, 95.0, 93.0, 94.0))
    bars.append(_Bar(94.0, 95.0, 93.0, 94.5))
    return bars


def _gap_then_fill(n: int = 40, base: float = 100.0) -> list[_Bar]:
    """Gap up then price comes back down to fill."""
    bars = _flat(n - 6, price=base)
    bars.append(_Bar(base, base + 1.0, base - 1.0, base))
    bars.append(_Bar(105.0, 107.0, 105.0, 106.0))  # gap up
    bars.append(_Bar(106.0, 108.0, 105.5, 107.0))
    bars.append(_Bar(107.0, 108.0, 104.0, 104.5))  # starts falling
    bars.append(_Bar(104.5, 105.0, 100.5, 101.0))  # fills gap (low <= 101)
    bars.append(_Bar(101.0, 102.0, 99.0, 100.5))
    return bars


class TestEdgeCases:
    def test_returns_none_empty(self):
        assert analyze([]) is None

    def test_returns_none_too_few_bars(self):
        assert analyze(_flat(3)) is None

    def test_returns_result_enough_bars(self):
        result = analyze(_flat(10))
        assert result is not None
        assert isinstance(result, GapReport)

    def test_returns_none_no_ohlc(self):
        class _BadBar:
            close = 100.0
        assert analyze([_BadBar()] * 10) is None


class TestGapDetection:
    def test_gap_up_detected(self):
        result = analyze(_gap_up_bars(30))
        assert result is not None
        assert len(result.up_gaps) >= 1

    def test_gap_down_detected(self):
        result = analyze(_gap_down_bars(30))
        assert result is not None
        assert len(result.down_gaps) >= 1

    def test_no_gaps_for_flat(self):
        result = analyze(_flat(30))
        assert result is not None
        assert result.total_gaps == 0

    def test_gaps_are_gap_objects(self):
        result = analyze(_gap_up_bars(30))
        assert result is not None
        for g in result.all_gaps:
            assert isinstance(g, Gap)


class TestGapProperties:
    def test_up_gap_direction(self):
        result = analyze(_gap_up_bars(30))
        assert result is not None
        for g in result.up_gaps:
            assert g.direction == "up"

    def test_down_gap_direction(self):
        result = analyze(_gap_down_bars(30))
        assert result is not None
        for g in result.down_gaps:
            assert g.direction == "down"

    def test_gap_size_positive(self):
        result = analyze(_gap_up_bars(30))
        assert result is not None
        for g in result.all_gaps:
            assert g.gap_size > 0

    def test_gap_pct_positive(self):
        result = analyze(_gap_up_bars(30))
        assert result is not None
        for g in result.all_gaps:
            assert g.gap_pct > 0

    def test_gap_type_valid(self):
        result = analyze(_gap_up_bars(30))
        assert result is not None
        for g in result.all_gaps:
            assert g.gap_type in {"common", "breakaway", "runaway", "exhaustion", "island", "unknown"}

    def test_fill_probability_in_range(self):
        result = analyze(_gap_up_bars(30))
        assert result is not None
        for g in result.all_gaps:
            assert 0.0 <= g.fill_probability <= 1.0


class TestGapFill:
    def test_gap_fill_detected(self):
        result = analyze(_gap_then_fill(40))
        assert result is not None
        if result.up_gaps:
            filled = [g for g in result.up_gaps if g.filled]
            assert len(filled) >= 1

    def test_fill_rate_in_range(self):
        result = analyze(_gap_up_bars(30))
        assert result is not None
        assert 0.0 <= result.fill_rate <= 1.0

    def test_open_gaps_subset(self):
        result = analyze(_gap_up_bars(30))
        assert result is not None
        assert len(result.open_gaps) <= result.total_gaps

    def test_filled_gaps_count(self):
        result = analyze(_gap_then_fill(40))
        assert result is not None
        assert result.filled_gaps >= 0


class TestRecentGap:
    def test_recent_gap_is_gap(self):
        result = analyze(_gap_up_bars(30))
        assert result is not None
        if result.recent_gap is not None:
            assert isinstance(result.recent_gap, Gap)

    def test_no_recent_gap_for_flat(self):
        result = analyze(_flat(30))
        assert result is not None
        assert result.recent_gap is None


class TestMetadata:
    def test_bars_used_correct(self):
        bars = _flat(30)
        result = analyze(bars)
        assert result is not None
        assert result.bars_used == 30

    def test_current_price_positive(self):
        result = analyze(_flat(30, price=100.0))
        assert result is not None
        assert result.current_price > 0

    def test_symbol_stored(self):
        result = analyze(_flat(30), symbol="NVDA")
        assert result is not None
        assert result.symbol == "NVDA"

    def test_avg_gap_pct_zero_for_flat(self):
        result = analyze(_flat(30))
        assert result is not None
        assert result.avg_gap_pct == 0.0


class TestVerdict:
    def test_verdict_present(self):
        result = analyze(_flat(10))
        assert result is not None
        assert len(result.verdict) > 10

    def test_verdict_mentions_gaps(self):
        result = analyze(_flat(10))
        assert result is not None
        assert "gap" in result.verdict.lower()
