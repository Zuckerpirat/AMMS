"""Tests for amms.analysis.dpo (Detrended Price Oscillator)."""

from __future__ import annotations

import math
import pytest

from amms.analysis.dpo import DPOReport, analyze


class _Bar:
    def __init__(self, close: float):
        self.close = close


# period=20, displacement=11, history=40, min_bars = 20+40+11+5 = 76
MIN_BARS = 80


def _flat(n: int = MIN_BARS, price: float = 100.0) -> list[_Bar]:
    return [_Bar(price)] * n


def _uptrend(n: int = MIN_BARS, start: float = 50.0, step: float = 1.0) -> list[_Bar]:
    return [_Bar(start + i * step) for i in range(n)]


def _downtrend(n: int = MIN_BARS, start: float = 200.0, step: float = 1.0) -> list[_Bar]:
    return [_Bar(max(start - i * step, 1.0)) for i in range(n)]


def _cyclic(n: int = MIN_BARS, center: float = 100.0, amp: float = 5.0, cycle: int = 20) -> list[_Bar]:
    """Sinusoidal price to test cycle detection."""
    return [_Bar(center + amp * math.sin(2 * math.pi * i / cycle)) for i in range(n)]


class TestEdgeCases:
    def test_returns_none_empty(self):
        assert analyze([]) is None

    def test_returns_none_too_few_bars(self):
        assert analyze(_flat(20)) is None

    def test_returns_result_enough_bars(self):
        result = analyze(_flat(MIN_BARS))
        assert result is not None
        assert isinstance(result, DPOReport)

    def test_bad_attr_returns_none(self):
        class _Bad:
            pass
        assert analyze([_Bad()] * MIN_BARS) is None


class TestDPOComponents:
    def test_dpo_is_float(self):
        result = analyze(_flat(MIN_BARS))
        assert result is not None
        assert isinstance(result.dpo, float)

    def test_dpo_pct_rank_in_range(self):
        for bars in [_flat(MIN_BARS), _uptrend(), _downtrend()]:
            result = analyze(bars)
            if result:
                assert 0.0 <= result.dpo_pct_rank <= 100.0

    def test_displacement_equals_period_half_plus_one(self):
        result = analyze(_flat(MIN_BARS), period=20)
        assert result is not None
        assert result.displacement == 11  # 20//2 + 1

    def test_period_stored(self):
        result = analyze(_flat(MIN_BARS), period=14)
        assert result is not None
        assert result.period == 14

    def test_dpo_positive_bool(self):
        result = analyze(_flat(MIN_BARS))
        assert result is not None
        assert isinstance(result.dpo_positive, bool)


class TestSignal:
    def test_signal_valid_values(self):
        valid = {"overbought", "high", "neutral", "low", "oversold"}
        for bars in [_flat(MIN_BARS), _uptrend(), _downtrend()]:
            result = analyze(bars)
            if result:
                assert result.signal in valid

    def test_score_in_range(self):
        for bars in [_flat(MIN_BARS), _uptrend(), _downtrend()]:
            result = analyze(bars)
            if result:
                assert -100.0 <= result.score <= 100.0


class TestCycleAnalysis:
    def test_cyclic_data_estimates_cycle(self):
        result = analyze(_cyclic(n=200), period=10)
        if result and result.estimated_cycle is not None:
            # Cycle should be roughly 20 bars
            assert 10 <= result.estimated_cycle <= 40

    def test_recent_peak_and_trough_types(self):
        result = analyze(_cyclic(n=150))
        assert result is not None
        # These may be None if no clear peaks/troughs
        if result.recent_peak is not None:
            assert isinstance(result.recent_peak, float)
        if result.recent_trough is not None:
            assert isinstance(result.recent_trough, float)


class TestSeries:
    def test_dpo_series_non_empty(self):
        result = analyze(_flat(MIN_BARS))
        assert result is not None
        assert len(result.dpo_series) > 0

    def test_dpo_series_last_matches_current(self):
        result = analyze(_flat(MIN_BARS))
        assert result is not None
        assert abs(result.dpo_series[-1] - result.dpo) < 1e-4


class TestMetadata:
    def test_symbol_stored(self):
        result = analyze(_flat(MIN_BARS), symbol="NVDA")
        assert result is not None
        assert result.symbol == "NVDA"

    def test_bars_used_correct(self):
        bars = _flat(90)
        result = analyze(bars)
        assert result is not None
        assert result.bars_used == 90

    def test_current_price_correct(self):
        result = analyze(_flat(MIN_BARS, price=250.0))
        assert result is not None
        assert abs(result.current_price - 250.0) < 1.0


class TestVerdict:
    def test_verdict_present(self):
        result = analyze(_flat(MIN_BARS))
        assert result is not None
        assert len(result.verdict) > 10

    def test_verdict_mentions_dpo(self):
        result = analyze(_flat(MIN_BARS))
        assert result is not None
        assert "DPO" in result.verdict
