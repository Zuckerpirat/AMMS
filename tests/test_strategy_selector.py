"""Tests for regime-based strategy selector."""

from __future__ import annotations

from amms.analysis.strategy_selector import StrategyRecommendation, recommend


class TestStrategySelector:
    def test_returns_recommendation(self) -> None:
        result = recommend("bull")
        assert isinstance(result, StrategyRecommendation)

    def test_bull_regime_primary_set(self) -> None:
        result = recommend("bull")
        assert result.primary in ("sma_cross", "composite", "rsi_reversal")

    def test_bear_regime_low_risk_multiplier(self) -> None:
        result = recommend("bear")
        assert result.risk_multiplier == 0.5

    def test_bull_regime_full_risk_multiplier(self) -> None:
        result = recommend("bull")
        assert result.risk_multiplier == 1.0

    def test_neutral_regime_moderate_risk(self) -> None:
        result = recommend("neutral")
        assert result.risk_multiplier == 0.75

    def test_unknown_regime_conservative(self) -> None:
        result = recommend("unknown")
        assert result.risk_multiplier == 0.5

    def test_reasoning_is_nonempty_list(self) -> None:
        result = recommend("bull")
        assert isinstance(result.reasoning, list)
        assert len(result.reasoning) > 0

    def test_avoid_is_list(self) -> None:
        result = recommend("bear")
        assert isinstance(result.avoid, list)

    def test_high_vol_influences_bull(self) -> None:
        r1 = recommend("bull", vix_proxy=30.0)
        r2 = recommend("bull", vix_proxy=10.0)
        # High vol should push toward rsi_reversal, not sma_cross
        assert r1.primary in ("rsi_reversal", "composite")

    def test_strong_trend_in_neutral(self) -> None:
        result = recommend("neutral", adx=30.0)
        assert result.primary == "breakout"

    def test_sector_rotation_in_reasoning(self) -> None:
        result = recommend("bull", top_sector="XLK")
        assert any("XLK" in r for r in result.reasoning)
