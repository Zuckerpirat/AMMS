"""Tests for amms.analysis.position_sizer."""

from __future__ import annotations

import pytest

from amms.analysis.position_sizer import PositionSizeResult, SizeLevel, compute


class TestEdgeCases:
    def test_returns_none_stop_above_entry(self):
        assert compute("AAPL", 100.0, 110.0, 10_000) is None

    def test_returns_none_stop_equal_entry(self):
        assert compute("AAPL", 100.0, 100.0, 10_000) is None

    def test_returns_none_zero_capital(self):
        assert compute("AAPL", 100.0, 95.0, 0) is None

    def test_returns_none_negative_entry(self):
        assert compute("AAPL", -100.0, 90.0, 10_000) is None

    def test_returns_result(self):
        result = compute("AAPL", 100.0, 95.0, 10_000)
        assert result is not None
        assert isinstance(result, PositionSizeResult)


class TestRiskPerShare:
    def test_risk_per_share_correct(self):
        result = compute("AAPL", 100.0, 95.0, 10_000)
        assert result is not None
        assert result.risk_per_share == pytest.approx(5.0, abs=0.01)

    def test_risk_per_share_pct_correct(self):
        result = compute("AAPL", 100.0, 95.0, 10_000)
        assert result is not None
        assert result.risk_per_share_pct == pytest.approx(5.0, abs=0.1)

    def test_tight_stop(self):
        result = compute("AAPL", 100.0, 99.5, 10_000)
        assert result is not None
        assert result.risk_per_share_pct < 1.0
        assert "tight" in result.note.lower()

    def test_wide_stop(self):
        result = compute("AAPL", 100.0, 90.0, 10_000)
        assert result is not None
        assert result.risk_per_share_pct > 5.0
        assert "wide" in result.note.lower()


class TestSharesFormula:
    def test_shares_correct_1pct_risk(self):
        """capital=$10000, risk=1%, entry=$100, stop=$95 → risk=$100, risk/share=$5 → 20 shares."""
        result = compute("AAPL", 100.0, 95.0, 10_000)
        assert result is not None
        level_1pct = next(l for l in result.levels if l.risk_pct == 1.0)
        assert level_1pct.shares == 20

    def test_shares_correct_2pct_risk(self):
        """Same but 2% risk → $200 risk / $5 = 40 shares."""
        result = compute("AAPL", 100.0, 95.0, 10_000)
        assert result is not None
        level_2pct = next(l for l in result.levels if l.risk_pct == 2.0)
        assert level_2pct.shares == 40

    def test_max_loss_matches_risk_amount(self):
        result = compute("AAPL", 100.0, 95.0, 10_000)
        assert result is not None
        for level in result.levels:
            expected_loss = 10_000 * level.risk_pct / 100
            assert level.max_loss == pytest.approx(expected_loss, abs=0.01)

    def test_shares_nonnegative(self):
        result = compute("AAPL", 100.0, 95.0, 100)  # tiny capital
        assert result is not None
        for level in result.levels:
            assert level.shares >= 0

    def test_position_value_shares_times_entry(self):
        result = compute("AAPL", 100.0, 95.0, 10_000)
        assert result is not None
        for level in result.levels:
            assert level.position_value == pytest.approx(level.shares * 100.0, abs=0.01)


class TestTargets:
    def test_target_1r_correct(self):
        """Entry=100, stop=95 → 1R target = 100 + 5 = 105."""
        result = compute("AAPL", 100.0, 95.0, 10_000)
        assert result is not None
        level = result.levels[0]
        assert level.target_1r == pytest.approx(105.0, abs=0.01)

    def test_target_2r_correct(self):
        """2R target = 100 + 2×5 = 110."""
        result = compute("AAPL", 100.0, 95.0, 10_000)
        assert result is not None
        level = result.levels[0]
        assert level.target_2r == pytest.approx(110.0, abs=0.01)

    def test_target_3r_correct(self):
        """3R target = 100 + 3×5 = 115."""
        result = compute("AAPL", 100.0, 95.0, 10_000)
        assert result is not None
        level = result.levels[0]
        assert level.target_3r == pytest.approx(115.0, abs=0.01)

    def test_targets_ascending(self):
        result = compute("AAPL", 100.0, 95.0, 10_000)
        assert result is not None
        for level in result.levels:
            assert level.target_1r < level.target_2r < level.target_3r


class TestLevels:
    def test_default_three_levels(self):
        result = compute("AAPL", 100.0, 95.0, 10_000)
        assert result is not None
        assert len(result.levels) == 3

    def test_custom_levels(self):
        result = compute("AAPL", 100.0, 95.0, 10_000, risk_levels_pct=[0.25, 0.5, 1.0, 2.0])
        assert result is not None
        assert len(result.levels) == 4

    def test_higher_risk_more_shares(self):
        result = compute("AAPL", 100.0, 95.0, 10_000)
        assert result is not None
        shares = [l.shares for l in result.levels]
        assert shares == sorted(shares)

    def test_symbol_stored(self):
        result = compute("TSLA", 200.0, 190.0, 50_000)
        assert result is not None
        assert result.symbol == "TSLA"

    def test_entry_and_stop_stored(self):
        result = compute("AAPL", 100.0, 95.0, 10_000)
        assert result is not None
        assert result.entry_price == 100.0
        assert result.stop_price == 95.0

    def test_note_present(self):
        result = compute("AAPL", 100.0, 95.0, 10_000)
        assert result is not None
        assert len(result.note) > 5

    def test_position_weight_increases_with_risk(self):
        result = compute("AAPL", 100.0, 95.0, 10_000)
        assert result is not None
        weights = [l.position_weight_pct for l in result.levels]
        assert weights == sorted(weights)
