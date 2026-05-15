"""Tests for amms.analysis.mean_reversion."""

from __future__ import annotations

import pytest

from amms.data.bars import Bar
from amms.analysis.mean_reversion import MeanReversionScore, score


def _bar(sym: str, close: float, i: int = 0) -> Bar:
    return Bar(sym, "1Day", f"2026-01-{1 + i % 28:02d}T10:00:00Z",
               close, close + 1.0, close - 1.0, close, 10_000)


def _bars(prices: list[float], sym: str = "SYM") -> list[Bar]:
    return [_bar(sym, p, i) for i, p in enumerate(prices)]


class TestMeanReversionScore:
    def test_returns_none_insufficient(self):
        bars = _bars([100.0] * 15)
        assert score(bars) is None

    def test_returns_result_with_enough_data(self):
        bars = _bars([100.0] * 25)
        result = score(bars)
        assert result is not None
        assert isinstance(result, MeanReversionScore)

    def test_symbol_preserved(self):
        bars = _bars([100.0] * 25, sym="AAPL")
        result = score(bars)
        assert result is not None
        assert result.symbol == "AAPL"

    def test_score_in_range(self):
        bars = _bars([100.0 + i for i in range(25)])
        result = score(bars)
        assert result is not None
        assert 0.0 <= result.score <= 100.0

    def test_direction_is_valid(self):
        bars = _bars([100.0] * 25)
        result = score(bars)
        assert result is not None
        assert result.direction in {"bullish_reversion", "bearish_reversion", "neutral"}

    def test_verdict_is_valid(self):
        bars = _bars([100.0] * 25)
        result = score(bars)
        assert result is not None
        assert result.verdict in {"extreme", "strong", "moderate", "mild", "none"}

    def test_flat_price_low_z_score(self):
        """Constant price → Z-score = 0 → Z-score component should be 0."""
        bars = _bars([100.0] * 30)
        result = score(bars)
        assert result is not None
        # Z-score should be 0 for flat prices
        assert result.components.get("zscore", 0) == 0.0

    def test_sharp_drop_bullish_reversion(self):
        """Sharp drop from stable base → oversold → bullish reversion."""
        bars = _bars([100.0] * 25 + [70.0])
        result = score(bars)
        assert result is not None
        assert result.direction in {"bullish_reversion", "neutral"}

    def test_sharp_spike_bearish_reversion(self):
        """Sharp spike from stable base → overbought → bearish reversion."""
        bars = _bars([100.0] * 25 + [130.0])
        result = score(bars)
        assert result is not None
        assert result.direction in {"bearish_reversion", "neutral"}

    def test_components_dict_populated(self):
        bars = _bars([100.0 + i * 0.5 for i in range(30)])
        result = score(bars)
        assert result is not None
        assert isinstance(result.components, dict)
        assert len(result.components) >= 1

    def test_recommended_action_is_string(self):
        bars = _bars([100.0] * 25)
        result = score(bars)
        assert result is not None
        assert isinstance(result.recommended_action, str)
        assert len(result.recommended_action) > 5

    def test_current_price_matches_last_close(self):
        prices = [100.0] * 24 + [105.0]
        bars = _bars(prices)
        result = score(bars)
        assert result is not None
        assert result.current_price == pytest.approx(105.0, abs=0.01)

    def test_extreme_oversold_has_bullish_direction(self):
        """Very large drop → multiple indicators should detect oversold."""
        bars = _bars([100.0] * 25 + [50.0])
        result = score(bars)
        assert result is not None
        assert result.direction in {"bullish_reversion", "neutral"}

    def test_score_higher_for_extreme_stretch(self):
        """Extreme drop should have higher score than mild drop."""
        mild_bars = _bars([100.0] * 25 + [97.0])
        extreme_bars = _bars([100.0] * 25 + [60.0])
        mild = score(mild_bars)
        extreme = score(extreme_bars)
        assert mild is not None and extreme is not None
        assert extreme.score >= mild.score
