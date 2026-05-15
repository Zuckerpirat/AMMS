"""Tests for position sizing strategies."""

from __future__ import annotations

from amms.risk.position_sizing import atr_based, fixed_fraction, kelly_criterion


class TestFixedFraction:
    def test_basic_sizing(self) -> None:
        result = fixed_fraction(100_000, 100.0, 2.0, risk_pct=1.0)
        assert result.shares == 200
        assert result.strategy == "fixed_fraction"

    def test_zero_price_returns_empty(self) -> None:
        result = fixed_fraction(100_000, 0.0, 2.0)
        assert result.shares == 0

    def test_zero_stop_returns_empty(self) -> None:
        result = fixed_fraction(100_000, 100.0, 0.0)
        assert result.shares == 0

    def test_respects_max_position_pct(self) -> None:
        result = fixed_fraction(100_000, 100.0, 0.1, risk_pct=1.0, max_position_pct=10.0)
        assert result.dollar_amount <= 10_100.0

    def test_pct_of_equity_reasonable(self) -> None:
        result = fixed_fraction(100_000, 100.0, 2.0)
        assert 0.0 <= result.pct_of_equity <= 100.0

    def test_risk_pct_equity_matches_intent(self) -> None:
        result = fixed_fraction(100_000, 100.0, 2.0, risk_pct=1.0)
        assert result.risk_pct_equity < 1.0  # capped by max_position_pct


class TestKellyCriterion:
    def test_basic_kelly(self) -> None:
        result = kelly_criterion(100_000, 100.0, 2.0, win_rate=0.55, avg_win_pct=3.0, avg_loss_pct=1.5)
        assert result.strategy == "kelly"
        assert result.shares >= 0

    def test_negative_edge_returns_zero(self) -> None:
        result = kelly_criterion(100_000, 100.0, 2.0, win_rate=0.2, avg_win_pct=1.0, avg_loss_pct=5.0)
        assert result.shares == 0

    def test_invalid_win_rate_returns_empty(self) -> None:
        result = kelly_criterion(100_000, 100.0, 2.0, win_rate=0.0)
        assert result.shares == 0

    def test_respects_max_position(self) -> None:
        result = kelly_criterion(100_000, 100.0, 2.0, win_rate=0.7, avg_win_pct=10.0, avg_loss_pct=1.0, max_position_pct=10.0)
        assert result.dollar_amount <= 10_100.0

    def test_fractional_kelly_smaller_than_full(self) -> None:
        r_full = kelly_criterion(100_000, 100.0, 2.0, win_rate=0.6, kelly_fraction=1.0)
        r_frac = kelly_criterion(100_000, 100.0, 2.0, win_rate=0.6, kelly_fraction=0.25)
        assert r_frac.shares <= r_full.shares


class TestAtrBased:
    def test_basic_atr_sizing(self) -> None:
        result = atr_based(100_000, 100.0, 2.0, atr_multiplier=2.0, risk_pct=1.0)
        assert result.shares == 200  # capped by max_position_pct=20%
        assert result.strategy == "atr_based"

    def test_zero_atr_returns_empty(self) -> None:
        result = atr_based(100_000, 100.0, 0.0)
        assert result.shares == 0

    def test_notes_include_stop_price(self) -> None:
        result = atr_based(100_000, 100.0, 2.0)
        assert "stop" in result.notes.lower() or "ATR" in result.notes

    def test_respects_max_position(self) -> None:
        result = atr_based(100_000, 100.0, 0.1, atr_multiplier=0.5, max_position_pct=10.0)
        assert result.dollar_amount <= 10_100.0
