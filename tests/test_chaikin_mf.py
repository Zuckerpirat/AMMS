"""Tests for amms.analysis.chaikin_mf."""

from __future__ import annotations

import pytest

from amms.analysis.chaikin_mf import CMFReport, CMFSnapshot, analyze


class _Bar:
    def __init__(self, close: float, high: float = None, low: float = None, volume: float = 100_000):
        self.close = close
        self.high = high if high is not None else close + 1.0
        self.low = low if low is not None else close - 1.0
        self.volume = volume


def _buying_bars(n: int = 40) -> list[_Bar]:
    """Closes near the high — buyers dominating."""
    bars = []
    for _ in range(n):
        low = 98.0
        high = 102.0
        close = 101.5  # near high → MFM ≈ +0.75
        bars.append(_Bar(close, high=high, low=low, volume=100_000))
    return bars


def _selling_bars(n: int = 40) -> list[_Bar]:
    """Closes near the low — sellers dominating."""
    bars = []
    for _ in range(n):
        low = 98.0
        high = 102.0
        close = 98.5  # near low → MFM ≈ -0.75
        bars.append(_Bar(close, high=high, low=low, volume=100_000))
    return bars


def _neutral_bars(n: int = 40) -> list[_Bar]:
    """Closes at midpoint — MFM ≈ 0."""
    bars = []
    for _ in range(n):
        low = 98.0
        high = 102.0
        close = 100.0  # midpoint → MFM = 0
        bars.append(_Bar(close, high=high, low=low, volume=100_000))
    return bars


def _flat(n: int = 40, price: float = 100.0) -> list[_Bar]:
    return [_Bar(price)] * n


class TestEdgeCases:
    def test_returns_none_empty(self):
        assert analyze([]) is None

    def test_returns_none_too_few_bars(self):
        assert analyze(_flat(15)) is None

    def test_returns_result_enough_bars(self):
        result = analyze(_flat(25))
        assert result is not None
        assert isinstance(result, CMFReport)

    def test_returns_none_no_volume(self):
        class _NoVol:
            close = high = low = 100.0
        assert analyze([_NoVol()] * 25) is None


class TestCMFValue:
    def test_cmf_in_range(self):
        for bars in [_buying_bars(40), _selling_bars(40), _neutral_bars(40)]:
            result = analyze(bars)
            if result:
                assert -1.0 <= result.cmf <= 1.0

    def test_buying_positive_cmf(self):
        result = analyze(_buying_bars(40))
        assert result is not None
        assert result.cmf > 0

    def test_selling_negative_cmf(self):
        result = analyze(_selling_bars(40))
        assert result is not None
        assert result.cmf < 0

    def test_neutral_near_zero(self):
        result = analyze(_neutral_bars(40))
        assert result is not None
        assert abs(result.cmf) < 0.1


class TestSignal:
    def test_signal_valid(self):
        for bars in [_buying_bars(40), _selling_bars(40), _neutral_bars(40)]:
            result = analyze(bars)
            if result:
                assert result.signal in {"strong_buy", "buy", "neutral", "sell", "strong_sell"}

    def test_buying_buy_signal(self):
        result = analyze(_buying_bars(40))
        assert result is not None
        assert result.signal in {"strong_buy", "buy"}

    def test_selling_sell_signal(self):
        result = analyze(_selling_bars(40))
        assert result is not None
        assert result.signal in {"strong_sell", "sell"}

    def test_signal_strength_in_range(self):
        for bars in [_buying_bars(40), _selling_bars(40)]:
            result = analyze(bars)
            if result:
                assert 0.0 <= result.signal_strength <= 1.0


class TestTrend:
    def test_cmf_trend_valid(self):
        for bars in [_buying_bars(40), _neutral_bars(40)]:
            result = analyze(bars)
            if result:
                assert result.cmf_trend in {"improving", "worsening", "stable"}

    def test_trend_bars_non_negative(self):
        result = analyze(_buying_bars(40))
        assert result is not None
        assert result.trend_bars >= 0


class TestStats:
    def test_above_zero_pct_in_range(self):
        result = analyze(_buying_bars(40))
        assert result is not None
        assert 0.0 <= result.above_zero_pct <= 100.0

    def test_buying_high_above_zero_pct(self):
        result = analyze(_buying_bars(40))
        assert result is not None
        assert result.above_zero_pct > 50.0

    def test_zero_crossings_non_negative(self):
        result = analyze(_flat(40))
        assert result is not None
        assert result.zero_crossings >= 0

    def test_avg_volume_positive(self):
        result = analyze(_buying_bars(40))
        assert result is not None
        assert result.avg_volume > 0


class TestHistory:
    def test_history_not_empty(self):
        result = analyze(_flat(30))
        assert result is not None
        assert len(result.history) > 0

    def test_history_items_are_snapshots(self):
        result = analyze(_flat(30))
        assert result is not None
        for s in result.history:
            assert isinstance(s, CMFSnapshot)

    def test_history_cmf_in_range(self):
        result = analyze(_buying_bars(40))
        assert result is not None
        for s in result.history:
            assert -1.0 <= s.cmf <= 1.0

    def test_history_mfm_in_range(self):
        result = analyze(_buying_bars(40))
        assert result is not None
        for s in result.history:
            assert -1.0 <= s.mfm <= 1.0


class TestMetadata:
    def test_bars_used_correct(self):
        bars = _flat(30)
        result = analyze(bars)
        assert result is not None
        assert result.bars_used == 30

    def test_current_price_correct(self):
        result = analyze(_flat(30, price=200.0))
        assert result is not None
        assert abs(result.current_price - 200.0) < 1.0

    def test_symbol_stored(self):
        result = analyze(_flat(30), symbol="META")
        assert result is not None
        assert result.symbol == "META"


class TestVerdict:
    def test_verdict_present(self):
        result = analyze(_flat(30))
        assert result is not None
        assert len(result.verdict) > 10

    def test_verdict_mentions_cmf(self):
        result = analyze(_flat(30))
        assert result is not None
        assert "cmf" in result.verdict.lower() or "money flow" in result.verdict.lower()
