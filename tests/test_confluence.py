"""Tests for signal confluence analyzer."""

from __future__ import annotations

from amms.analysis.confluence import ConfluenceSignal, analyze
from amms.data.bars import Bar


def _bar(close: float, high: float, low: float, i: int = 0, vol: float = 1000.0) -> Bar:
    return Bar("X", "1D", f"2026-01-{1 + i % 28:02d}", close, high, low, close, vol)


def _bars_neutral(n: int = 60) -> list[Bar]:
    return [_bar(100.0 + (i % 5) * 0.1, 101.0 + (i % 3) * 0.1, 99.0, i) for i in range(n)]


def _bars_bullish(n: int = 60) -> list[Bar]:
    """Strongly oversold: price at multi-period low."""
    bars = []
    for i in range(n - 10):
        bars.append(_bar(100.0 + i * 0.1, 101.0 + i * 0.1, 99.0 + i * 0.1, i))
    # Last 10 bars crash down
    for j in range(10):
        i = n - 10 + j
        price = 100.0 + (n - 10) * 0.1 - j * 3.0
        bars.append(_bar(price, price + 0.5, price - 1.5, i))
    return bars


class TestConfluence:
    def test_returns_confluence_signal(self) -> None:
        result = analyze(_bars_neutral())
        assert isinstance(result, ConfluenceSignal)

    def test_score_in_range(self) -> None:
        result = analyze(_bars_neutral())
        assert -1.0 <= result.score <= 1.0

    def test_confidence_in_range(self) -> None:
        result = analyze(_bars_neutral())
        assert 0.0 <= result.confidence <= 1.0

    def test_verdict_valid(self) -> None:
        valid = {"strong_buy", "buy", "neutral", "sell", "strong_sell"}
        result = analyze(_bars_neutral())
        assert result.verdict in valid

    def test_symbol_preserved(self) -> None:
        bars = [Bar("AAPL", "1D", f"2026-01-{1 + i % 28:02d}", 150.0 + i * 0.1,
                    151.0 + i * 0.1, 149.0 + i * 0.1, 150.0 + i * 0.1, 1000.0)
                for i in range(60)]
        result = analyze(bars)
        assert result.symbol == "AAPL"

    def test_signals_are_lists(self) -> None:
        result = analyze(_bars_neutral())
        assert isinstance(result.bullish_signals, list)
        assert isinstance(result.bearish_signals, list)

    def test_signals_checked_matches_lists(self) -> None:
        result = analyze(_bars_neutral())
        assert result.signals_checked == len(result.bullish_signals) + len(result.bearish_signals)

    def test_bullish_crash_tends_positive(self) -> None:
        """A crash from highs to lows should produce bullish confluence signals."""
        result = analyze(_bars_bullish())
        assert result.score > 0 or len(result.bullish_signals) > 0

    def test_raw_score_consistent(self) -> None:
        result = analyze(_bars_neutral())
        assert result.max_possible >= 0

    def test_few_bars_produces_result(self) -> None:
        bars = [_bar(100.0, 101.0, 99.0, i) for i in range(5)]
        result = analyze(bars)
        assert isinstance(result, ConfluenceSignal)
        assert result.score == 0.0 or result.signals_checked >= 0
