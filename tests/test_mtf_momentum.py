"""Tests for amms.analysis.mtf_momentum."""

from __future__ import annotations

import pytest

from amms.analysis.mtf_momentum import MTFMomentumReport, MomentumWindow, analyze


class _Bar:
    def __init__(self, close: float):
        self.close = close


def _uptrend(n: int = 120, start: float = 100.0, step: float = 0.5) -> list[_Bar]:
    price = start
    bars = []
    for _ in range(n):
        bars.append(_Bar(price))
        price += step
    return bars


def _downtrend(n: int = 120, start: float = 300.0, step: float = 0.5) -> list[_Bar]:
    price = start
    bars = []
    for _ in range(n):
        bars.append(_Bar(price))
        price -= step
    return bars


def _flat(n: int = 120, price: float = 100.0) -> list[_Bar]:
    return [_Bar(price)] * n


def _diverging(n: int = 120) -> list[_Bar]:
    """Downtrend for first 80% then sharp upswing in last 20%."""
    bars = _downtrend(int(n * 0.8))
    # Sharp upswing: undo drop + more
    start = bars[-1].close
    for i in range(n - len(bars)):
        bars.append(_Bar(start + i * 2.0))
    return bars


class TestEdgeCases:
    def test_returns_none_empty(self):
        assert analyze([]) is None

    def test_returns_none_too_few_bars(self):
        bars = _flat(50)
        assert analyze(bars) is None

    def test_returns_none_zero_price(self):
        bars = [_Bar(0.0)] * 120
        assert analyze(bars) is None

    def test_returns_result_enough_bars(self):
        bars = _flat(110)
        result = analyze(bars)
        assert result is not None
        assert isinstance(result, MTFMomentumReport)


class TestWindows:
    def test_four_windows_returned(self):
        bars = _flat(110)
        result = analyze(bars)
        assert result is not None
        assert len(result.windows) == 4

    def test_window_bars_are_5_20_50_100(self):
        bars = _flat(110)
        result = analyze(bars)
        assert result is not None
        sizes = [w.bars for w in result.windows]
        assert sizes == [5, 20, 50, 100]

    def test_windows_are_momentum_window(self):
        bars = _uptrend(110)
        result = analyze(bars)
        assert result is not None
        for w in result.windows:
            assert isinstance(w, MomentumWindow)

    def test_window_direction_valid(self):
        bars = _uptrend(110)
        result = analyze(bars)
        assert result is not None
        for w in result.windows:
            assert w.direction in {"bullish", "bearish", "neutral"}

    def test_window_score_valid(self):
        bars = _uptrend(110)
        result = analyze(bars)
        assert result is not None
        for w in result.windows:
            assert w.score in {-1, 0, 1}


class TestTotalScore:
    def test_total_score_range(self):
        for bars in [_uptrend(110), _downtrend(110), _flat(110)]:
            result = analyze(bars)
            if result:
                assert -4 <= result.total_score <= 4

    def test_uptrend_positive_score(self):
        bars = _uptrend(120, step=1.0)
        result = analyze(bars)
        assert result is not None
        assert result.total_score > 0

    def test_downtrend_negative_score(self):
        bars = _downtrend(120, step=1.0)
        result = analyze(bars)
        assert result is not None
        assert result.total_score < 0


class TestRegime:
    def test_regime_valid(self):
        for bars in [_uptrend(110), _downtrend(110), _flat(110)]:
            result = analyze(bars)
            if result:
                assert result.regime in {"strong_bull", "bull", "neutral", "bear", "strong_bear"}

    def test_strong_uptrend_is_bull_regime(self):
        bars = _uptrend(120, step=2.0)
        result = analyze(bars)
        assert result is not None
        assert result.regime in {"strong_bull", "bull"}

    def test_strong_downtrend_is_bear_regime(self):
        bars = _downtrend(120, step=2.0)
        result = analyze(bars)
        assert result is not None
        assert result.regime in {"strong_bear", "bear"}


class TestAlignment:
    def test_aligned_for_strong_uptrend(self):
        bars = _uptrend(120, step=2.0)
        result = analyze(bars)
        assert result is not None
        assert result.aligned is True

    def test_aligned_for_strong_downtrend(self):
        bars = _downtrend(120, step=2.0)
        result = analyze(bars)
        assert result is not None
        assert result.aligned is True


class TestMetrics:
    def test_current_price_correct(self):
        bars = _flat(110, price=150.0)
        result = analyze(bars)
        assert result is not None
        assert abs(result.current_price - 150.0) < 0.01

    def test_bars_used_matches_input(self):
        bars = _flat(110)
        result = analyze(bars)
        assert result is not None
        assert result.bars_used == 110

    def test_symbol_stored(self):
        bars = _flat(110)
        result = analyze(bars, symbol="AAPL")
        assert result is not None
        assert result.symbol == "AAPL"


class TestVerdict:
    def test_verdict_present(self):
        bars = _flat(110)
        result = analyze(bars)
        assert result is not None
        assert len(result.verdict) > 10

    def test_verdict_mentions_momentum(self):
        bars = _uptrend(110)
        result = analyze(bars)
        assert result is not None
        assert "momentum" in result.verdict.lower() or "regime" in result.verdict.lower()
