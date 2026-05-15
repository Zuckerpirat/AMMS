"""Tests for amms.analysis.overnight_risk."""

from __future__ import annotations

import pytest

from amms.data.bars import Bar
from amms.analysis.overnight_risk import OvernightRisk, analyze


def _bar(sym: str, open_: float, close: float, i: int = 0) -> Bar:
    high = max(open_, close) * 1.005
    low = min(open_, close) * 0.995
    return Bar(sym, "1Day", f"2026-01-{1 + i % 28:02d}T10:00:00Z",
               open_, high, low, close, 100_000)


def _no_gap_bars(sym: str, n: int, price: float = 100.0) -> list[Bar]:
    """Bars where open == previous close (no gap)."""
    bars = []
    for i in range(n):
        bars.append(_bar(sym, price, price, i))
    return bars


def _gappy_bars(sym: str, n: int, gap_pct: float = 2.0) -> list[Bar]:
    """Bars with consistent daily gap up."""
    bars = []
    price = 100.0
    for i in range(n):
        open_ = price * (1 + gap_pct / 100)
        close = open_ * 1.001
        bars.append(_bar(sym, open_, close, i))
        price = close
    return bars


class TestAnalyze:
    def test_returns_none_insufficient(self):
        bars = [_bar("AAPL", 100.0, 101.0, i) for i in range(3)]
        assert analyze(bars) is None

    def test_returns_result(self):
        bars = _no_gap_bars("AAPL", 20)
        result = analyze(bars)
        assert result is not None
        assert isinstance(result, OvernightRisk)

    def test_symbol_preserved(self):
        bars = _no_gap_bars("TSLA", 20)
        result = analyze(bars)
        assert result is not None
        assert result.symbol == "TSLA"

    def test_no_gaps_when_open_eq_prev_close(self):
        """Perfect no-gap bars → 0 gaps."""
        bars = _no_gap_bars("AAPL", 20)
        result = analyze(bars)
        assert result is not None
        assert result.n_gaps == 0
        assert result.gap_frequency_pct == pytest.approx(0.0, abs=0.1)

    def test_high_frequency_gappy(self):
        """Consistent gap every bar → high frequency."""
        bars = _gappy_bars("AAPL", 25, gap_pct=2.0)
        result = analyze(bars)
        assert result is not None
        assert result.gap_frequency_pct > 50.0

    def test_avg_gap_near_gap_size(self):
        bars = _gappy_bars("AAPL", 25, gap_pct=2.0)
        result = analyze(bars)
        assert result is not None
        assert result.avg_gap_pct == pytest.approx(2.0, abs=0.5)

    def test_risk_label_low_when_no_gaps(self):
        bars = _no_gap_bars("AAPL", 20)
        result = analyze(bars)
        assert result is not None
        assert result.risk_label == "low"

    def test_risk_label_high_for_large_gaps(self):
        bars = _gappy_bars("AAPL", 25, gap_pct=4.0)
        result = analyze(bars)
        assert result is not None
        assert result.risk_label in ("elevated", "high")

    def test_risk_score_range(self):
        bars = _gappy_bars("AAPL", 25, gap_pct=2.0)
        result = analyze(bars)
        assert result is not None
        assert 0 <= result.risk_score <= 100

    def test_max_gap_gte_avg_gap(self):
        bars = _gappy_bars("AAPL", 25, gap_pct=2.0)
        result = analyze(bars)
        assert result is not None
        if result.n_gaps > 0:
            assert result.max_gap_pct >= result.avg_gap_pct

    def test_gap_down_detected(self):
        """Bars gapping down → avg_gap_down_pct > 0."""
        bars = []
        price = 100.0
        for i in range(20):
            open_ = price * 0.98  # 2% gap down
            close = open_ * 1.001
            bars.append(_bar("AAPL", open_, close, i))
            price = close
        result = analyze(bars)
        assert result is not None
        assert result.avg_gap_down_pct > 0

    def test_bars_analyzed_correct(self):
        bars = _no_gap_bars("AAPL", 20)
        result = analyze(bars)
        assert result is not None
        assert result.bars_analyzed == 20

    def test_min_gap_threshold_respected(self):
        """Gaps < threshold are not counted."""
        bars = _gappy_bars("AAPL", 25, gap_pct=0.05)  # below 0.1% threshold
        result = analyze(bars, min_gap_pct=0.1)
        assert result is not None
        assert result.n_gaps == 0

    def test_higher_gap_higher_risk_score(self):
        small = analyze(_gappy_bars("A", 25, gap_pct=0.5))
        large = analyze(_gappy_bars("A", 25, gap_pct=4.0))
        assert small is not None and large is not None
        assert large.risk_score > small.risk_score
