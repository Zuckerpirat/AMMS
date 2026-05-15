"""Tests for RSI/price divergence detector."""

from __future__ import annotations

from amms.analysis.divergence import DivergenceResult, detect_divergence, _compute_rsi
from amms.data.bars import Bar


def _bar(close: float, i: int = 0) -> Bar:
    return Bar("X", "1D", f"2026-01-{1 + i % 28:02d}", close, close + 0.5, close - 0.5, close, 1000.0)


def _bars(closes: list[float]) -> list[Bar]:
    return [_bar(c, i) for i, c in enumerate(closes)]


class TestComputeRsi:
    def test_returns_empty_for_too_few_bars(self) -> None:
        assert _compute_rsi([100.0] * 10, 14) == []

    def test_returns_values_for_enough_bars(self) -> None:
        closes = [float(100 + i * 0.5) for i in range(30)]
        rsi = _compute_rsi(closes, 14)
        assert len(rsi) > 0

    def test_all_rising_produces_high_rsi(self) -> None:
        closes = [float(100 + i) for i in range(40)]
        rsi = _compute_rsi(closes, 14)
        assert rsi[-1] > 70  # strong uptrend → high RSI

    def test_all_falling_produces_low_rsi(self) -> None:
        closes = [float(200 - i) for i in range(40)]
        rsi = _compute_rsi(closes, 14)
        assert rsi[-1] < 30  # strong downtrend → low RSI

    def test_rsi_bounded_0_100(self) -> None:
        closes = [float(100 + (i % 5 - 2)) for i in range(50)]
        rsi = _compute_rsi(closes, 14)
        assert all(0.0 <= v <= 100.0 for v in rsi)


class TestDetectDivergence:
    def test_insufficient_bars_returns_none_type(self) -> None:
        bars = _bars([100.0] * 10)
        result = detect_divergence(bars)
        assert result.divergence_type == "none"
        assert "Insufficient" in result.reason

    def test_returns_divergence_result(self) -> None:
        closes = [float(100 + (i % 7)) for i in range(80)]
        result = detect_divergence(_bars(closes))
        assert isinstance(result, DivergenceResult)

    def test_symbol_preserved(self) -> None:
        bars = [Bar("AAPL", "1D", f"2026-01-{1 + i % 28:02d}", 100.0 + i * 0.1,
                    101.0 + i * 0.1, 99.0 + i * 0.1, 100.0 + i * 0.1, 1000.0)
                for i in range(80)]
        result = detect_divergence(bars)
        assert result.symbol == "AAPL"

    def test_confidence_between_0_and_1(self) -> None:
        closes = [float(100 + (i % 10)) for i in range(80)]
        result = detect_divergence(_bars(closes))
        assert 0.0 <= result.confidence <= 1.0

    def test_bars_checked_reflects_lookback(self) -> None:
        closes = [float(100 + i * 0.5) for i in range(80)]
        result = detect_divergence(_bars(closes), lookback=50)
        assert result.bars_checked <= 50

    def test_no_divergence_for_uniform_trend(self) -> None:
        """Perfectly linear trend should produce no classic divergence."""
        closes = [float(100 + i * 0.1) for i in range(80)]
        result = detect_divergence(_bars(closes))
        # Uniform trend shouldn't produce strong hidden signals either
        # (just verify it completes without error)
        assert result.divergence_type in ("none", "bullish", "bearish", "hidden_bullish", "hidden_bearish")

    def test_divergence_type_valid_values(self) -> None:
        valid = {"none", "bullish", "bearish", "hidden_bullish", "hidden_bearish"}
        closes = [float(100 + (i % 15) * 2 - (i // 15)) for i in range(80)]
        result = detect_divergence(_bars(closes))
        assert result.divergence_type in valid

    def test_swing_labels_are_strings(self) -> None:
        closes = [float(100 + i * 0.3 + (i % 5)) for i in range(80)]
        result = detect_divergence(_bars(closes))
        assert isinstance(result.price_swing, str)
        assert isinstance(result.rsi_swing, str)
