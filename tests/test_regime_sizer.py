"""Tests for amms.analysis.regime_sizer."""

from __future__ import annotations

import pytest

from amms.analysis.regime_sizer import RegimeSizerReport, analyze


class _Bar:
    def __init__(self, close, high=None, low=None):
        self.close = close
        self.high = high if high is not None else close + 1.0
        self.low = low if low is not None else close - 1.0


def _bull_bars(n: int = 40) -> list[_Bar]:
    return [_Bar(100.0 + i * 0.3, high=100.0 + i * 0.3 + 0.5, low=100.0 + i * 0.3 - 0.2) for i in range(n)]


def _bear_bars(n: int = 40) -> list[_Bar]:
    return [_Bar(140.0 - i * 0.3, high=140.0 - i * 0.3 + 0.2, low=140.0 - i * 0.3 - 0.5) for i in range(n)]


def _volatile_bars(n: int = 40) -> list[_Bar]:
    """High ATR — wide bars."""
    bars = []
    import random
    rng = random.Random(7)
    price = 100.0
    for _ in range(n):
        move = rng.uniform(-3.0, 3.0)
        price = max(1.0, price + move)
        bars.append(_Bar(price, high=price + 2.5, low=price - 2.5))
    return bars


BASE = dict(win_rate=55.0, avg_win_pct=3.0, avg_loss_pct=1.5)


class TestEdgeCases:
    def test_returns_none_empty(self):
        assert analyze([], **BASE) is None

    def test_returns_none_too_few(self):
        bars = [_Bar(100.0) for _ in range(4)]
        assert analyze(bars, **BASE) is None

    def test_returns_result(self):
        bars = _bull_bars(30)
        result = analyze(bars, **BASE)
        assert result is not None
        assert isinstance(result, RegimeSizerReport)

    def test_invalid_win_rate_zero(self):
        bars = _bull_bars(30)
        assert analyze(bars, win_rate=0, avg_win_pct=3.0, avg_loss_pct=1.5) is None

    def test_invalid_win_rate_100(self):
        bars = _bull_bars(30)
        assert analyze(bars, win_rate=100, avg_win_pct=3.0, avg_loss_pct=1.5) is None

    def test_invalid_avg_win_zero(self):
        bars = _bull_bars(30)
        assert analyze(bars, win_rate=55.0, avg_win_pct=0.0, avg_loss_pct=1.5) is None


class TestKelly:
    def test_kelly_pct_nonnegative(self):
        bars = _bull_bars(30)
        result = analyze(bars, **BASE)
        assert result is not None
        assert result.kelly_pct >= 0

    def test_kelly_pct_capped(self):
        bars = _bull_bars(30)
        result = analyze(bars, **BASE)
        assert result is not None
        assert result.kelly_pct <= 25.0

    def test_half_kelly_lt_full_kelly(self):
        bars = _bull_bars(30)
        result = analyze(bars, **BASE)
        assert result is not None
        assert result.half_kelly_pct <= result.kelly_pct

    def test_payoff_ratio_correct(self):
        bars = _bull_bars(30)
        result = analyze(bars, win_rate=60.0, avg_win_pct=3.0, avg_loss_pct=1.5)
        assert result is not None
        assert result.payoff_ratio == pytest.approx(2.0, abs=0.01)


class TestRegime:
    def test_regime_valid(self):
        bars = _bull_bars(30)
        result = analyze(bars, **BASE)
        assert result is not None
        valid = {"calm_bull", "calm_neutral", "calm_bear", "hot_bull", "hot_neutral", "hot_bear", "extreme_vol"}
        assert result.regime in valid

    def test_bull_bars_get_bull_or_neutral_regime(self):
        bars = _bull_bars(40)
        result = analyze(bars, **BASE)
        assert result is not None
        assert "bull" in result.regime or "neutral" in result.regime

    def test_bear_bars_get_bear_or_neutral_regime(self):
        bars = _bear_bars(40)
        result = analyze(bars, **BASE)
        assert result is not None
        assert "bear" in result.regime or "neutral" in result.regime

    def test_volatile_bars_get_hot_or_extreme_regime(self):
        bars = _volatile_bars(40)
        result = analyze(bars, **BASE, high_atr=0.5)  # lower threshold to trigger
        assert result is not None
        assert "hot" in result.regime or "extreme" in result.regime

    def test_regime_multiplier_in_range(self):
        bars = _bull_bars(30)
        result = analyze(bars, **BASE)
        assert result is not None
        assert 0.0 < result.regime_multiplier <= 1.0


class TestAdjustedSize:
    def test_adjusted_pct_lte_half_kelly(self):
        bars = _bull_bars(30)
        result = analyze(bars, **BASE)
        assert result is not None
        assert result.adjusted_pct <= result.half_kelly_pct + 0.01  # allow rounding

    def test_adjusted_pct_nonnegative(self):
        bars = _bull_bars(30)
        result = analyze(bars, **BASE)
        assert result is not None
        assert result.adjusted_pct >= 0

    def test_max_loss_pct_positive(self):
        bars = _bull_bars(30)
        result = analyze(bars, **BASE)
        assert result is not None
        assert result.max_loss_pct >= 0

    def test_extreme_vol_minimal_size(self):
        bars = _volatile_bars(40)
        result = analyze(bars, **BASE, extreme_atr=0.1)  # force extreme regime
        assert result is not None
        assert result.regime_multiplier <= 0.15


class TestSuggestedShares:
    def test_suggested_shares_with_portfolio(self):
        bars = _bull_bars(30)
        result = analyze(bars, **BASE, portfolio_value=100_000.0, current_price=50.0)
        assert result is not None
        assert result.suggested_shares is not None
        assert result.suggested_shares >= 0

    def test_suggested_shares_none_without_portfolio(self):
        bars = _bull_bars(30)
        result = analyze(bars, **BASE)
        assert result is not None
        assert result.suggested_shares is None


class TestMetadata:
    def test_bars_used_correct(self):
        bars = _bull_bars(35)
        result = analyze(bars, **BASE)
        assert result is not None
        assert result.bars_used == 35

    def test_trend_direction_valid(self):
        bars = _bull_bars(40)
        result = analyze(bars, **BASE)
        assert result is not None
        assert result.trend_direction in ("up", "down", "flat")

    def test_atr_pct_nonnegative(self):
        bars = _bull_bars(30)
        result = analyze(bars, **BASE)
        assert result is not None
        assert result.atr_pct >= 0


class TestVerdict:
    def test_verdict_present(self):
        bars = _bull_bars(30)
        result = analyze(bars, **BASE)
        assert result is not None
        assert len(result.verdict) > 10

    def test_verdict_mentions_regime(self):
        bars = _bull_bars(30)
        result = analyze(bars, **BASE)
        assert result is not None
        assert "Regime" in result.verdict or "regime" in result.verdict
