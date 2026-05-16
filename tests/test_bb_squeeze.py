"""Tests for amms.analysis.bb_squeeze."""

from __future__ import annotations

import pytest

from amms.analysis.bb_squeeze import BBSqueeze, BBSqueezeReport, analyze


class _Bar:
    def __init__(self, close: float, spread: float = 0.5):
        self.close = close
        self.high = close + spread
        self.low = close - spread


def _volatile(n: int = 80) -> list[_Bar]:
    """Volatile bars with large spreads."""
    bars = []
    price = 100.0
    for i in range(n):
        spread = 3.0 + (i % 5)
        bars.append(_Bar(price, spread))
        price += 0.1 * (1 if i % 2 == 0 else -1)
    return bars


def _squeezed(n: int = 80) -> list[_Bar]:
    """Very tight bars — low bandwidth."""
    bars = []
    price = 100.0
    for i in range(n):
        spread = 0.05
        bars.append(_Bar(price, spread))
        price += 0.01 * (1 if i % 2 == 0 else -1)
    return bars


def _expanding_then_squeeze(n: int = 100) -> list[_Bar]:
    """Mostly volatile bars, then tight last 15% — so squeeze is rare in lookback."""
    bars = []
    price = 100.0
    squeeze_start = int(n * 0.85)
    for i in range(n):
        if i < squeeze_start:
            spread = 3.0
            price += 0.3 * (1 if i % 2 == 0 else -1)
        else:
            spread = 0.02
            price += 0.005 * (1 if i % 2 == 0 else -1)
        bars.append(_Bar(price, spread))
    return bars


class TestEdgeCases:
    def test_returns_none_empty(self):
        assert analyze([]) is None

    def test_returns_none_too_few_bars(self):
        bars = _squeezed(30)
        assert analyze(bars) is None

    def test_returns_result_enough_bars(self):
        bars = _squeezed(80)
        result = analyze(bars)
        assert result is not None
        assert isinstance(result, BBSqueezeReport)


class TestBandwidth:
    def test_bandwidth_positive(self):
        bars = _volatile(80)
        result = analyze(bars)
        assert result is not None
        assert result.current_bandwidth >= 0.0

    def test_tight_bars_lower_bandwidth(self):
        vol_result = analyze(_volatile(80))
        sq_result = analyze(_squeezed(80))
        if vol_result and sq_result:
            assert sq_result.current_bandwidth < vol_result.current_bandwidth

    def test_bandwidth_percentile_in_range(self):
        bars = _volatile(80)
        result = analyze(bars)
        assert result is not None
        assert 0.0 <= result.bandwidth_percentile <= 100.0


class TestSqueezeScore:
    def test_squeeze_score_in_range(self):
        bars = _volatile(80)
        result = analyze(bars)
        assert result is not None
        assert 0.0 <= result.current_squeeze_score <= 100.0

    def test_tight_bars_high_squeeze_score(self):
        # Volatile history, then squeeze at the end → current BW is below most history
        bars = _expanding_then_squeeze(100)
        result = analyze(bars)
        assert result is not None
        assert result.current_squeeze_score > 50.0

    def test_squeeze_score_is_complement_of_percentile(self):
        bars = _volatile(80)
        result = analyze(bars)
        assert result is not None
        assert abs(result.current_squeeze_score + result.bandwidth_percentile - 100.0) < 0.1


class TestSqueezedFlag:
    def test_squeezed_true_for_tight_bars(self):
        # Need volatile history first, then squeeze — so current BW is below 20th pct
        bars = _expanding_then_squeeze(100)
        result = analyze(bars)
        assert result is not None
        assert result.is_squeezed is True

    def test_squeezed_active_bars_positive(self):
        bars = _expanding_then_squeeze(100)
        result = analyze(bars)
        assert result is not None
        assert result.squeeze_active_bars >= 0


class TestBBValues:
    def test_upper_ge_middle_ge_lower(self):
        bars = _volatile(80)
        result = analyze(bars)
        assert result is not None
        assert result.upper >= result.middle >= result.lower

    def test_price_position_in_range(self):
        bars = _volatile(80)
        result = analyze(bars)
        assert result is not None
        assert 0.0 <= result.price_position <= 1.0


class TestDirectionBias:
    def test_direction_valid(self):
        bars = _volatile(80)
        result = analyze(bars)
        assert result is not None
        assert result.direction_bias in {"bullish", "bearish", "neutral"}


class TestHistory:
    def test_history_not_empty(self):
        bars = _volatile(80)
        result = analyze(bars)
        assert result is not None
        assert len(result.history) > 0

    def test_history_items_are_bb_squeeze(self):
        bars = _volatile(80)
        result = analyze(bars)
        assert result is not None
        for h in result.history:
            assert isinstance(h, BBSqueeze)

    def test_history_max_history_bars(self):
        bars = _volatile(80)
        result = analyze(bars, history_bars=20)
        assert result is not None
        assert len(result.history) <= 20


class TestSymbol:
    def test_symbol_stored(self):
        bars = _volatile(80)
        result = analyze(bars, symbol="AAPL")
        assert result is not None
        assert result.symbol == "AAPL"

    def test_bars_used_correct(self):
        bars = _volatile(80)
        result = analyze(bars)
        assert result is not None
        assert result.bars_used == 80


class TestVerdict:
    def test_verdict_present(self):
        bars = _volatile(80)
        result = analyze(bars)
        assert result is not None
        assert len(result.verdict) > 10

    def test_verdict_mentions_squeeze(self):
        bars = _volatile(80)
        result = analyze(bars)
        assert result is not None
        assert "squeeze" in result.verdict.lower() or "bb" in result.verdict.lower()

    def test_squeezed_verdict_says_squeeze(self):
        bars = _squeezed(80)
        result = analyze(bars)
        assert result is not None
        assert "squeeze" in result.verdict.lower()
