"""Tests for amms.analysis.gap_fill."""

from __future__ import annotations

import pytest

from amms.analysis.gap_fill import Gap, GapFillReport, analyze


class _Bar:
    def __init__(self, close, open_=None, high=None, low=None):
        self.close = close
        self.open = open_ if open_ is not None else close
        self.high = high if high is not None else close + 0.5
        self.low = low if low is not None else close - 0.5


def _no_gap_bars(n: int = 30) -> list[_Bar]:
    """Bars with negligible gaps."""
    return [_Bar(100.0 + i * 0.1) for i in range(n)]


def _up_gap_bars() -> list[_Bar]:
    """A series with a clear upside gap that gets filled."""
    bars = [_Bar(100.0) for _ in range(10)]
    # Gap up: open 103 vs prev close 100 (+3%)
    bars.append(_Bar(103.0, open_=103.0, high=104.0, low=102.0))
    # Fill the gap: next bar goes back down to 100
    bars.append(_Bar(100.5, open_=102.0, high=102.5, low=99.5))
    # More normal bars
    bars += [_Bar(101.0 + i * 0.1) for i in range(10)]
    return bars


def _down_gap_bars() -> list[_Bar]:
    """A series with a clear downside gap that gets filled."""
    bars = [_Bar(100.0) for _ in range(10)]
    # Gap down: open 97 vs prev close 100 (-3%)
    bars.append(_Bar(97.0, open_=97.0, high=98.0, low=96.0))
    # Fill: goes back up past 100
    bars.append(_Bar(100.5, open_=98.0, high=101.0, low=97.5))
    bars += [_Bar(100.0 + i * 0.1) for i in range(10)]
    return bars


def _unfilled_gap_bars() -> list[_Bar]:
    """Gap that never gets filled within max_fill_bars."""
    bars = [_Bar(100.0) for _ in range(10)]
    # Big gap up: never fills (price stays above)
    bars.append(_Bar(110.0, open_=110.0, high=111.0, low=109.0))
    bars += [_Bar(110.0 + i * 0.1) for i in range(12)]
    return bars


class TestEdgeCases:
    def test_returns_none_empty(self):
        assert analyze([]) is None

    def test_returns_none_too_few(self):
        bars = [_Bar(100.0) for _ in range(19)]
        assert analyze(bars) is None

    def test_returns_result_no_gaps(self):
        bars = _no_gap_bars(30)
        result = analyze(bars)
        assert result is not None
        assert isinstance(result, GapFillReport)

    def test_no_gaps_detected(self):
        bars = _no_gap_bars(30)
        result = analyze(bars)
        assert result is not None
        assert len(result.gaps) == 0


class TestGapDetection:
    def test_up_gap_detected(self):
        bars = _up_gap_bars()
        result = analyze(bars, gap_threshold_pct=0.5)
        assert result is not None
        assert result.n_up_gaps > 0

    def test_down_gap_detected(self):
        bars = _down_gap_bars()
        result = analyze(bars, gap_threshold_pct=0.5)
        assert result is not None
        assert result.n_down_gaps > 0

    def test_gaps_are_gap_objects(self):
        bars = _up_gap_bars()
        result = analyze(bars, gap_threshold_pct=0.5)
        assert result is not None
        for g in result.gaps:
            assert isinstance(g, Gap)

    def test_gap_kind_valid(self):
        bars = _up_gap_bars()
        result = analyze(bars, gap_threshold_pct=0.5)
        assert result is not None
        for g in result.gaps:
            assert g.kind in ("up", "down")

    def test_gap_pct_matches_kind(self):
        bars = _up_gap_bars()
        result = analyze(bars, gap_threshold_pct=0.5)
        assert result is not None
        for g in result.gaps:
            if g.kind == "up":
                assert g.gap_pct > 0
            else:
                assert g.gap_pct < 0


class TestFillRates:
    def test_fill_rate_in_range(self):
        bars = _up_gap_bars()
        result = analyze(bars, gap_threshold_pct=0.5)
        assert result is not None
        assert 0.0 <= result.fill_rate <= 100.0

    def test_up_gap_filled(self):
        bars = _up_gap_bars()
        result = analyze(bars, gap_threshold_pct=0.5, max_fill_bars=10)
        assert result is not None
        up_gaps = [g for g in result.gaps if g.kind == "up"]
        if up_gaps:
            assert any(g.filled for g in up_gaps)

    def test_down_gap_filled(self):
        bars = _down_gap_bars()
        result = analyze(bars, gap_threshold_pct=0.5, max_fill_bars=10)
        assert result is not None
        down_gaps = [g for g in result.gaps if g.kind == "down"]
        if down_gaps:
            assert any(g.filled for g in down_gaps)

    def test_unfilled_gap_not_filled(self):
        bars = _unfilled_gap_bars()
        result = analyze(bars, gap_threshold_pct=1.0, max_fill_bars=5)
        assert result is not None
        up_gaps = [g for g in result.gaps if g.kind == "up"]
        if up_gaps:
            assert all(not g.filled for g in up_gaps[:1])

    def test_filled_gaps_have_bars_to_fill(self):
        bars = _up_gap_bars()
        result = analyze(bars, gap_threshold_pct=0.5, max_fill_bars=10)
        assert result is not None
        for g in result.gaps:
            if g.filled:
                assert g.bars_to_fill is not None
                assert g.bars_to_fill >= 0


class TestSizeRates:
    def test_small_large_fill_in_range(self):
        bars = _up_gap_bars()
        result = analyze(bars, gap_threshold_pct=0.5)
        assert result is not None
        assert 0.0 <= result.small_fill_rate <= 100.0
        assert 0.0 <= result.large_fill_rate <= 100.0


class TestMetadata:
    def test_symbol_stored(self):
        bars = _up_gap_bars()
        result = analyze(bars, symbol="AMZN", gap_threshold_pct=0.5)
        assert result is not None
        assert result.symbol == "AMZN"

    def test_bars_analysed_correct(self):
        bars = _no_gap_bars(25)
        result = analyze(bars)
        assert result is not None
        assert result.bars_analysed == 25

    def test_threshold_stored(self):
        bars = _no_gap_bars(30)
        result = analyze(bars, gap_threshold_pct=0.5)
        assert result is not None
        assert result.gap_threshold_pct == 0.5


class TestVerdict:
    def test_verdict_present(self):
        bars = _no_gap_bars(30)
        result = analyze(bars)
        assert result is not None
        assert len(result.verdict) > 5

    def test_verdict_mentions_fill_rate(self):
        bars = _up_gap_bars()
        result = analyze(bars, gap_threshold_pct=0.5)
        assert result is not None
        assert "fill" in result.verdict.lower() or "gap" in result.verdict.lower()
