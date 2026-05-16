"""Tests for amms.analysis.connors_rsi."""

from __future__ import annotations

import pytest

from amms.analysis.connors_rsi import CRSIReport, analyze


class _Bar:
    def __init__(self, close: float):
        self.close = close
        self.high = close + 0.5
        self.low = close - 0.5
        self.open = close


def _up_bars(n: int = 150, start: float = 100.0, step: float = 1.0) -> list[_Bar]:
    return [_Bar(start + i * step) for i in range(n)]


def _down_bars(n: int = 150, start: float = 250.0, step: float = 1.0) -> list[_Bar]:
    return [_Bar(max(1.0, start - i * step)) for i in range(n)]


def _flat_bars(n: int = 150, price: float = 100.0) -> list[_Bar]:
    return [_Bar(price) for _ in range(n)]


def _oscillating_bars(n: int = 150) -> list[_Bar]:
    bars = []
    for i in range(n):
        bars.append(_Bar(100.0 + 5.0 * (1 if i % 4 < 2 else -1)))
    return bars


class TestEdgeCases:
    def test_empty_returns_none(self):
        assert analyze([]) is None

    def test_too_few_bars_returns_none(self):
        assert analyze(_up_bars(20)) is None

    def test_sufficient_bars_returns_report(self):
        result = analyze(_up_bars(150))
        assert result is not None
        assert isinstance(result, CRSIReport)

    def test_no_close_attr_returns_none(self):
        class _Bad:
            pass
        assert analyze([_Bad()] * 150) is None

    def test_zero_price_returns_none(self):
        assert analyze([_Bar(0.0)] * 150) is None


class TestCRSIBounds:
    def test_crsi_in_range(self):
        for bars in [_up_bars(150), _down_bars(150), _flat_bars(150), _oscillating_bars(150)]:
            result = analyze(bars)
            if result:
                assert 0.0 <= result.crsi <= 100.0

    def test_components_in_range(self):
        result = analyze(_up_bars(150))
        assert result is not None
        assert 0.0 <= result.rsi_price <= 100.0
        assert 0.0 <= result.rsi_streak <= 100.0
        assert 0.0 <= result.percentile <= 100.0

    def test_score_in_range(self):
        for bars in [_up_bars(150), _down_bars(150), _flat_bars(150)]:
            result = analyze(bars)
            if result:
                assert -100.0 <= result.score <= 100.0

    def test_crsi_is_average_of_components(self):
        result = analyze(_up_bars(150))
        assert result is not None
        expected = (result.rsi_price + result.rsi_streak + result.percentile) / 3.0
        assert abs(result.crsi - expected) < 0.01


class TestMeanReversionSignal:
    def test_signal_valid_values(self):
        valid = {"strong_buy", "buy", "neutral", "sell", "strong_sell"}
        for bars in [_up_bars(150), _down_bars(150), _flat_bars(150), _oscillating_bars(150)]:
            result = analyze(bars)
            if result:
                assert result.signal in valid

    def test_oversold_signal_buy(self):
        # Down trend → low RSI → oversold → buy signal
        result = analyze(_down_bars(150))
        assert result is not None
        if result.oversold:
            assert result.signal == "strong_buy"
        if result.os_soft:
            assert result.signal in {"strong_buy", "buy"}

    def test_overbought_signal_sell(self):
        result = analyze(_up_bars(150))
        assert result is not None
        if result.overbought:
            assert result.signal == "strong_sell"
        if result.ob_soft:
            assert result.signal in {"strong_sell", "sell"}

    def test_score_inversely_related_to_crsi(self):
        # Score is mean-reversion: low CRSI → positive score (buy)
        result = analyze(_down_bars(150))
        assert result is not None
        # If CRSI < 50, score should be positive
        if result.crsi < 50:
            assert result.score > 0


class TestFlags:
    def test_overbought_consistent(self):
        result = analyze(_up_bars(150))
        assert result is not None
        assert result.overbought == (result.crsi > 90.0)

    def test_oversold_consistent(self):
        result = analyze(_down_bars(150))
        assert result is not None
        assert result.oversold == (result.crsi < 10.0)

    def test_ob_soft_consistent(self):
        result = analyze(_up_bars(150))
        assert result is not None
        assert result.ob_soft == (result.crsi > 80.0)

    def test_os_soft_consistent(self):
        result = analyze(_down_bars(150))
        assert result is not None
        assert result.os_soft == (result.crsi < 20.0)


class TestStreak:
    def test_streak_is_int(self):
        result = analyze(_up_bars(150))
        assert result is not None
        assert isinstance(result.streak, int)

    def test_uptrend_positive_streak(self):
        result = analyze(_up_bars(150))
        assert result is not None
        assert result.streak > 0

    def test_downtrend_negative_streak(self):
        result = analyze(_down_bars(150))
        assert result is not None
        assert result.streak < 0

    def test_flat_streak_zero(self):
        result = analyze(_flat_bars(150))
        assert result is not None
        assert result.streak == 0

    def test_pct_change_is_float(self):
        result = analyze(_up_bars(150))
        assert result is not None
        assert isinstance(result.pct_change, float)


class TestHistory:
    def test_history_length_default(self):
        result = analyze(_up_bars(150))
        assert result is not None
        assert len(result.history) == 20

    def test_history_length_custom(self):
        result = analyze(_up_bars(150), history=10)
        assert result is not None
        assert len(result.history) == 10

    def test_history_in_range(self):
        result = analyze(_up_bars(150))
        assert result is not None
        for v in result.history:
            assert 0.0 <= v <= 100.0

    def test_last_history_matches_crsi(self):
        result = analyze(_up_bars(150))
        assert result is not None
        assert result.history[-1] == result.crsi


class TestMetadata:
    def test_symbol_stored(self):
        result = analyze(_up_bars(150), symbol="SPY")
        assert result is not None
        assert result.symbol == "SPY"

    def test_bars_used_correct(self):
        bars = _up_bars(160)
        result = analyze(bars)
        assert result is not None
        assert result.bars_used == 160


class TestVerdict:
    def test_verdict_present(self):
        result = analyze(_up_bars(150))
        assert result is not None
        assert len(result.verdict) > 10

    def test_verdict_mentions_crsi(self):
        result = analyze(_up_bars(150))
        assert result is not None
        assert "CRSI" in result.verdict

    def test_verdict_mentions_streak(self):
        result = analyze(_up_bars(150))
        assert result is not None
        assert "Streak" in result.verdict or "streak" in result.verdict.lower()
