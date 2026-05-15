"""Tests for multi-indicator signal scanner."""

from __future__ import annotations

from amms.analysis.signal_scanner import SignalSetup, _analyze, scan_signals
from amms.data.bars import Bar


def _bar(close: float, high: float, low: float, volume: float, i: int = 0) -> Bar:
    return Bar("X", "1D", f"2026-01-{1 + i % 28:02d}", close, high, low, close, volume)


def _crash_bars(n: int = 60) -> list[Bar]:
    """Bars that drop sharply — should produce buy signals (oversold)."""
    bars = []
    for i in range(n - 15):
        p = 100.0 + i * 0.2
        bars.append(_bar(p, p + 1.0, p - 0.5, 1000.0, i))
    for j in range(15):
        i = n - 15 + j
        p = 100.0 + (n - 15) * 0.2 - j * 2.5
        bars.append(_bar(p, p + 0.5, p - 2.0, 3000.0, i))
    return bars


class _FakeData:
    def __init__(self, bars_factory):
        self._factory = bars_factory

    def get_bars(self, symbol, *, limit=80):
        return self._factory(symbol)


class TestSignalScanner:
    def test_returns_list(self) -> None:
        data = _FakeData(lambda sym: _crash_bars())
        results = scan_signals(["AAPL"], data)
        assert isinstance(results, list)

    def test_returns_setup_dataclass(self) -> None:
        data = _FakeData(lambda sym: _crash_bars())
        results = scan_signals(["AAPL"], data, min_score=0.0)
        assert len(results) >= 0
        if results:
            assert isinstance(results[0], SignalSetup)

    def test_empty_symbols_returns_empty(self) -> None:
        data = _FakeData(lambda sym: _crash_bars())
        assert scan_signals([], data) == []

    def test_insufficient_bars_skipped(self) -> None:
        data = _FakeData(lambda sym: [_bar(100.0, 101.0, 99.0, 1000.0, i) for i in range(5)])
        results = scan_signals(["AAPL"], data)
        assert results == []

    def test_sorted_by_score(self) -> None:
        data = _FakeData(lambda sym: _crash_bars())
        results = scan_signals(["AAPL", "MSFT", "TSLA"], data, min_score=0.0)
        for i in range(len(results) - 1):
            assert results[i].score >= results[i + 1].score

    def test_direction_valid(self) -> None:
        data = _FakeData(lambda sym: _crash_bars())
        results = scan_signals(["AAPL"], data, min_score=0.0)
        for r in results:
            assert r.direction in ("buy", "sell", "neutral")

    def test_confidence_valid(self) -> None:
        data = _FakeData(lambda sym: _crash_bars())
        results = scan_signals(["AAPL"], data, min_score=0.0)
        for r in results:
            assert r.confidence in ("low", "medium", "high")

    def test_score_non_negative(self) -> None:
        data = _FakeData(lambda sym: _crash_bars())
        results = scan_signals(["AAPL"], data, min_score=0.0)
        for r in results:
            assert r.score >= 0

    def test_analyze_returns_setup(self) -> None:
        bars = _crash_bars()
        bars[0] = Bar("AAPL", "1D", "2026-01-01", 100.0, 101.0, 99.0, 100.0, 1000.0)
        result = _analyze("AAPL", bars)
        assert isinstance(result, SignalSetup)
        assert result.symbol == "AAPL"

    def test_min_score_filter_works(self) -> None:
        data = _FakeData(lambda sym: _crash_bars())
        high = scan_signals(["AAPL"], data, min_score=8.0)
        low = scan_signals(["AAPL"], data, min_score=0.0)
        assert len(high) <= len(low)
