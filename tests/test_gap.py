"""Tests for amms.features.gap."""

from __future__ import annotations

import pytest

from amms.data.bars import Bar
from amms.features.gap import Gap, GapAnalysis, analyze_gaps


def _bar(close: float, open_: float | None = None, high: float | None = None,
         low: float | None = None, i: int = 0) -> Bar:
    o = open_ if open_ is not None else close
    h = high if high is not None else max(o, close) + 0.5
    l = low if low is not None else min(o, close) - 0.5
    return Bar("SYM", "1Day", f"2026-01-{1 + i % 28:02d}T10:00:00Z", o, h, l, close, 10_000)


class TestAnalyzeGaps:
    def test_returns_none_too_short(self):
        bars = [_bar(100.0, i=i) for i in range(2)]
        assert analyze_gaps(bars) is None

    def test_no_gaps_in_smooth_series(self):
        bars = [_bar(100.0 + i * 0.1, i=i) for i in range(20)]
        result = analyze_gaps(bars, min_gap_pct=0.5)
        assert result is not None
        assert len(result.gaps) == 0

    def test_detects_upward_gap(self):
        """Bar 5 opens significantly above previous close."""
        bars = [_bar(100.0, i=i) for i in range(5)]
        # Gap up: prev_close=100, open=105 (5%)
        bars.append(_bar(105.0, open_=105.0, i=5))
        bars += [_bar(105.0, i=i + 6) for i in range(5)]
        result = analyze_gaps(bars, min_gap_pct=0.3)
        assert result is not None
        up_gaps = [g for g in result.gaps if g.direction == "up"]
        assert len(up_gaps) >= 1
        assert up_gaps[0].gap_pct >= 0.3

    def test_detects_downward_gap(self):
        """Bar 5 opens significantly below previous close."""
        bars = [_bar(100.0, i=i) for i in range(5)]
        bars.append(_bar(94.0, open_=94.0, i=5))
        bars += [_bar(94.0, i=i + 6) for i in range(5)]
        result = analyze_gaps(bars, min_gap_pct=0.3)
        assert result is not None
        down_gaps = [g for g in result.gaps if g.direction == "down"]
        assert len(down_gaps) >= 1

    def test_gap_filled_when_price_returns(self):
        """Gap up that later gets filled (price dips back below open)."""
        bars = [_bar(100.0, i=i) for i in range(5)]
        # Gap up to 106
        bars.append(_bar(106.0, open_=106.0, high=108.0, low=105.0, i=5))
        # Later bar dips back to 100 → fills gap
        bars.append(_bar(100.0, open_=102.0, high=103.0, low=99.0, i=6))
        bars.append(_bar(101.0, i=7))
        result = analyze_gaps(bars, min_gap_pct=0.3)
        assert result is not None
        if result.gaps:
            # The gap should be marked as filled
            assert result.gaps[0].filled

    def test_unfilled_gap_appears_in_unfilled_list(self):
        """Gap that is never filled."""
        bars = [_bar(100.0, i=i) for i in range(5)]
        bars.append(_bar(110.0, open_=110.0, high=112.0, low=109.0, i=5))
        bars += [_bar(110.0 + i * 0.1, i=i + 6) for i in range(5)]
        result = analyze_gaps(bars, min_gap_pct=0.3)
        assert result is not None
        assert len(result.unfilled_gaps) >= 1

    def test_last_gap_is_most_recent(self):
        bars = [_bar(100.0, i=i) for i in range(5)]
        bars.append(_bar(106.0, open_=106.0, i=5))
        bars += [_bar(106.0, i=i + 6) for i in range(3)]
        result = analyze_gaps(bars, min_gap_pct=0.3)
        assert result is not None
        if result.last_gap:
            # Last gap should be the most recently detected one
            assert result.last_gap.bar_idx == max(g.bar_idx for g in result.gaps)

    def test_symbol_preserved(self):
        bars = [_bar(100.0, i=i) for i in range(10)]
        result = analyze_gaps(bars)
        assert result is not None
        assert result.symbol == "SYM"

    def test_gap_support_and_resistance(self):
        """Unfilled gap above price → gap resistance. Below → gap support."""
        # Start at 100, gap up to 110 (unfilled), then fall back to 105
        bars = [_bar(100.0, i=i) for i in range(5)]
        bars.append(_bar(110.0, open_=110.0, high=112.0, low=109.5, i=5))
        bars += [_bar(105.0, open_=105.0, high=106.0, low=104.0, i=i + 6) for i in range(5)]
        result = analyze_gaps(bars, min_gap_pct=0.3)
        assert result is not None
        # current price ~105, gap prev_close=100 → gap support at 100
        if result.nearest_gap_support is not None:
            assert result.nearest_gap_support <= result.current_price

    def test_bars_analyzed_correct(self):
        bars = [_bar(100.0, i=i) for i in range(20)]
        result = analyze_gaps(bars, lookback=15)
        assert result is not None
        assert result.bars_analyzed == 15

    def test_returns_analysis_instance(self):
        bars = [_bar(100.0 + i * 0.05, i=i) for i in range(20)]
        result = analyze_gaps(bars)
        assert isinstance(result, GapAnalysis)
